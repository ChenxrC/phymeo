"""Independent CUDA workspace for frozen-J, final-normal smooth friction.

This experiment does not update q or production TYX contact state. FP32 device
qualification is distinct from the stricter I3q HOST reference. HOST checks
substitute the actual device velocity into the law; they never polish it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from threading import RLock
from time import perf_counter
from types import MappingProxyType

import numpy as np

from phymeo.backends.mujoco_tyx._resources import DeviceResources

from .coupled_friction_reference import CoupledFrictionProblem

MAX_WORLDS = 32
MAX_DOF = 6
MAX_SAMPLES = 8
TRACE_CAPACITY = 290


def _owned(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _f32(value):
    with np.errstate(over="raise", invalid="raise"):
        try:
            array = np.asarray(value, dtype=np.float32)
        except (OverflowError, FloatingPointError, ValueError) as error:
            raise ValueError("input cannot be represented as finite FP32") from error
    if not np.all(np.isfinite(array)):
        raise ValueError("input cannot be represented as finite FP32")
    return _owned(array)


def _positive_conversion(original, packed, name):
    if np.any((np.asarray(original) > 0) & (np.asarray(packed) <= 0)):
        raise ValueError(f"positive {name} underflows in FP32")


@dataclass(frozen=True)
class PackedFrictionSample:
    point_world: np.ndarray
    normal_world: np.ndarray
    jacobian_a: np.ndarray
    jacobian_b: np.ndarray
    stiffness_n_m: np.float32
    phi_m: np.float32
    mu: np.float32
    slip_speed_m_s: np.float32


@dataclass(frozen=True)
class PackedFrictionProblem:
    mass_matrix: np.ndarray
    damping: np.ndarray
    input_velocity: np.ndarray
    applied_force: np.ndarray
    h_s: np.float32
    samples: tuple[PackedFrictionSample, ...]


def _pack_problem(problem):
    if type(problem) is not CoupledFrictionProblem:
        raise ValueError("every problem must be CoupledFrictionProblem")
    original = CoupledFrictionProblem(*(getattr(problem, field.name) for field in fields(problem)))
    mass = _f32(original.mass_matrix)
    damping = _f32(original.damping)
    velocity = _f32(original.input_velocity)
    force = _f32(original.applied_force)
    h = _f32(original.h_s)[()]
    _positive_conversion(original.h_s, h, "h")
    _positive_conversion(original.damping, damping, "damping")
    try:
        factor = np.linalg.cholesky(mass.astype(np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError("FP32 mass matrix is not positive definite") from error
    if not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0):
        raise ValueError("FP32 mass matrix has invalid factorization")
    packed_samples = []
    for sample in original.samples:
        values = _f32((sample.stiffness_n_m, sample.phi_m, sample.mu, sample.slip_speed_m_s))
        _positive_conversion(
            (sample.stiffness_n_m, sample.mu, sample.slip_speed_m_s),
            values[[0, 2, 3]],
            "contact parameters",
        )
        # Preserve exactly the uploaded normal, including its FP32 length.
        packed_samples.append(
            PackedFrictionSample(
                _f32(sample.point_world),
                _f32(sample.normal_world),
                _f32(sample.jacobian_a),
                _f32(sample.jacobian_b),
                values[0],
                values[1],
                values[2],
                values[3],
            )
        )
    return original, PackedFrictionProblem(mass, damping, velocity, force, h, tuple(packed_samples))


@dataclass(frozen=True)
class DeviceFrictionTrace:
    iteration: int
    backtrack: int
    phase: str
    decision: str
    alpha: float
    residual_norm: float
    tolerance: float
    velocity: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class TyxCoupledFrictionResult:
    original_problem: CoupledFrictionProblem
    packed_problem: PackedFrictionProblem
    velocity: np.ndarray
    free_velocity: np.ndarray
    normal_force: np.ndarray
    tangential_force: np.ndarray
    force_on_a: np.ndarray
    force_on_b: np.ndarray
    torque_on_a_world_origin: np.ndarray
    torque_on_b_world_origin: np.ndarray
    generalized_force: np.ndarray
    residual: np.ndarray
    residual_jacobian: np.ndarray
    contact_data: np.ndarray
    response_matrix: np.ndarray
    trace: tuple[DeviceFrictionTrace, ...]
    diagnostics: Mapping
    energy: Mapping

    def to_dict(self):
        return _plain(self)


class TyxCoupledFrictionFailure(RuntimeError):
    """A rejected batch, with no partially published result."""

    def __init__(self, reason, raw):
        super().__init__(reason)
        self._diagnostics = MappingProxyType({name: _owned(value) for name, value in raw.items()})

    @property
    def diagnostics(self):
        return self._diagnostics

    def to_dict(self):
        # Hex preserves even nonfinite rejected device words without emitting
        # nonstandard JSON NaN/Infinity or silently replacing their payload.
        return {
            "reason": str(self),
            "raw": {
                name: {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "hex": value.tobytes().hex(),
                }
                for name, value in self._diagnostics.items()
            },
        }


def _certificate(problem, velocity):
    """Independent FP64 substitution, not a HOST contact solve."""
    mass = np.asarray(problem.mass_matrix, dtype=np.float64)
    damping = np.asarray(problem.damping, dtype=np.float64)
    v0 = np.asarray(problem.input_velocity, dtype=np.float64)
    external = np.asarray(problem.applied_force, dtype=np.float64)
    h = float(problem.h_s)
    response = mass + h * np.diag(damping)
    factor = np.linalg.cholesky(response)
    free = np.linalg.solve(response, mass @ v0 + h * external)
    velocity = np.asarray(velocity, dtype=np.float64)
    total = np.zeros(len(velocity))
    contacts = []
    normal_work = tangent_work = spring0 = spring1 = 0.0
    for sample in problem.samples:
        n = np.asarray(sample.normal_world, dtype=np.float64)
        g = np.asarray(sample.jacobian_a, dtype=np.float64) - np.asarray(
            sample.jacobian_b, dtype=np.float64
        )
        relative = g @ velocity
        vn = float(n @ relative)
        tangent = np.cross(n, np.cross(relative, n)) / float(n @ n)
        gap = float(sample.phi_m) + h * vn
        fn = max(0.0, -float(sample.stiffness_n_m) * gap)
        ft = (
            -float(sample.mu)
            * fn
            * tangent
            / np.sqrt(float(tangent @ tangent) + float(sample.slip_speed_m_s) ** 2)
        )
        force = fn * n + ft
        moment = np.cross(np.asarray(sample.point_world, dtype=np.float64), force)
        total += g.T @ force
        contacts.append((fn, tangent, ft, force, moment))
        normal_work += h * fn * vn
        tangent_work += h * float(ft @ tangent)
        spring0 += 0.5 * float(sample.stiffness_n_m) * float(sample.phi_m) ** 2
        spring1 += 0.5 * float(sample.stiffness_n_m) * min(gap, 0.0) ** 2
    residual = response @ velocity - mass @ v0 - h * (external + total)
    norm = float(np.linalg.norm(np.linalg.solve(factor, residual)))
    scale = 1.0 + float(np.linalg.norm(factor.T @ free))
    dv = velocity - v0
    kinetic0 = 0.5 * float(v0 @ mass @ v0)
    kinetic1 = 0.5 * float(velocity @ mass @ velocity)
    work = h * float(external @ velocity)
    loss = h * float((damping * velocity) @ velocity)
    numerical_loss = 0.5 * float(dv @ mass @ dv)
    energy = dict(
        kinetic_entry_j=kinetic0,
        kinetic_exit_j=kinetic1,
        spring_entry_j=spring0,
        spring_exit_j=spring1,
        external_work_j=work,
        damping_loss_j=loss,
        normal_work_j=normal_work,
        tangent_work_j=tangent_work,
        kinetic_numerical_loss_j=numerical_loss,
        kinetic_balance_residual_j=kinetic1
        - kinetic0
        - work
        + loss
        - normal_work
        - tangent_work
        + numerical_loss,
        normal_spring_balance_j=spring1 - spring0 + normal_work,
    )
    if (
        not np.all(np.isfinite(residual))
        or not np.isfinite(norm + scale)
        or not all(np.isfinite(v) for v in energy.values())
    ):
        raise ValueError("nonfinite independent HOST certificate")
    return norm, scale, contacts, residual, total, energy


class TyxCoupledFrictionWorkspace:
    """Persistent bounded CUDA buffers; atomic HOST-owned result publication."""

    def __init__(self, *, capacity=MAX_WORLDS, mode="cuda"):
        if type(capacity) is not int or not 1 <= capacity <= MAX_WORLDS:
            raise ValueError("capacity must be an integer in 1..32")
        if type(mode) is not str or mode != "cuda":
            raise ValueError("this experimental workspace supports only cuda")
        import tyx as tx

        from ._tyx_coupled_friction_kernels import solve_batch

        start = perf_counter()
        self._capacity = capacity
        self._tx, self._kernel = tx, solve_batch
        self._lock = RLock()
        self._resources = DeviceResources()
        self._buffers = {}
        self._last_results = ()
        self._faulted = False
        self._busy = False
        self._stats = dict.fromkeys(
            (
                "launches",
                "batch_calls",
                "uploaded_bytes",
                "downloaded_bytes",
                "live_bytes",
                "peak_bytes",
            ),
            0,
        )
        self._stats.update(
            dict.fromkeys(
                (
                    "setup_seconds",
                    "upload_seconds",
                    "kernel_seconds",
                    "download_seconds",
                    "prepare_seconds",
                    "validation_seconds",
                    "total_seconds",
                ),
                0.0,
            )
        )
        self._layouts = {
            "sizes": ((capacity, 4), np.int32),
            "mass": ((capacity, 6, 6), np.float32),
            "inputs": ((capacity, 3, 6), np.float32),
            "steps": ((capacity,), np.float32),
            "samples": ((capacity, 8, 10), np.float32),
            "ja": ((capacity, 8, 3, 6), np.float32),
            "jb": ((capacity, 8, 3, 6), np.float32),
            "matrices": ((capacity, 4, 6, 6), np.float32),
            "vectors": ((capacity, 8, 6), np.float32),
            "contact": ((capacity, 8, 17), np.float32),
            "meta": ((capacity, 5), np.int32),
            "scalars": ((capacity, 3), np.float32),
            "trace_meta": ((capacity, TRACE_CAPACITY, 4), np.int32),
            "trace": ((capacity, TRACE_CAPACITY, 15), np.float32),
        }
        try:
            self._runtime = tx.Runtime(backend="cuda")
            self._resources.set_runtime(self._runtime)
            with tx.use_runtime(self._runtime):
                for name, (shape, dtype) in self._layouts.items():
                    buffer = tx.cuda_empty(shape, dtype)
                    self._resources.add_buffer(buffer)
                    self._buffers[name] = buffer
                    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    self._stats["live_bytes"] += size
                    self._stats["peak_bytes"] += size
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "workspace construction and cleanup failed", [error, cleanup]
                ) from None
            raise
        finally:
            self._stats["setup_seconds"] = perf_counter() - start

    @property
    def capacity(self):
        return self._capacity

    @property
    def closed(self):
        return self._resources.closed

    @property
    def faulted(self):
        return self._faulted

    @property
    def last_results(self):
        return self._last_results

    @property
    def stats(self):
        with self._lock:
            return MappingProxyType(
                dict(self._stats, mode="cuda", closed=self.closed, faulted=self.faulted)
            )

    def _prepare(self, problems, max_iterations, max_backtracks):
        if type(problems) is not tuple or not 1 <= len(problems) <= self.capacity:
            raise ValueError("problems must be a nonempty tuple within workspace capacity")
        if type(max_iterations) is not int or not 0 <= max_iterations <= 24:
            raise ValueError("max_iterations must be an integer in 0..24")
        if type(max_backtracks) is not int or not 0 <= max_backtracks <= 12:
            raise ValueError("max_backtracks must be an integer in 0..12")
        # Complete admission of every world precedes upload, not just launch.
        prepared = tuple(_pack_problem(problem) for problem in problems)
        names = ("sizes", "mass", "inputs", "steps", "samples", "ja", "jb")
        arrays = {name: np.zeros(*self._layouts[name]) for name in names}
        for world, (_, problem) in enumerate(prepared):
            nv, ns = len(problem.input_velocity), len(problem.samples)
            arrays["sizes"][world] = (nv, ns, max_iterations, max_backtracks)
            arrays["mass"][world, :nv, :nv] = problem.mass_matrix
            arrays["inputs"][world, 0, :nv] = problem.damping
            arrays["inputs"][world, 1, :nv] = problem.input_velocity
            arrays["inputs"][world, 2, :nv] = problem.applied_force
            arrays["steps"][world] = problem.h_s
            for index, sample in enumerate(problem.samples):
                arrays["samples"][world, index, :3] = sample.point_world
                arrays["samples"][world, index, 3:6] = sample.normal_world
                arrays["samples"][world, index, 6:] = (
                    sample.stiffness_n_m,
                    sample.phi_m,
                    sample.mu,
                    sample.slip_speed_m_s,
                )
                arrays["ja"][world, index, :, :nv] = sample.jacobian_a
                arrays["jb"][world, index, :, :nv] = sample.jacobian_b
        return prepared, arrays

    def _download(self, name):
        result = self._buffers[name].to_numpy()
        self._stats["downloaded_bytes"] += result.nbytes
        return result

    def solve_many(self, problems, *, max_iterations=24, max_backtracks=12):
        with self._lock:
            if self._resources.state != "OPEN" or self._faulted:
                raise RuntimeError("workspace is closed, faulted, or release is unresolved")
            if self._busy:
                raise RuntimeError("recursive workspace solve is unsupported")
            self._busy = True
            start = perf_counter()
            self._stats["batch_calls"] += 1
            try:
                phase = perf_counter()
                try:
                    prepared, arrays = self._prepare(problems, max_iterations, max_backtracks)
                finally:
                    self._stats["prepare_seconds"] += perf_counter() - phase
                try:
                    with self._tx.use_runtime(self._runtime):
                        phase = perf_counter()
                        try:
                            for name, array in arrays.items():
                                self._buffers[name].copy_from(array)
                                self._stats["uploaded_bytes"] += array.nbytes
                            self._runtime.sync()
                        finally:
                            self._stats["upload_seconds"] += perf_counter() - phase
                        phase = perf_counter()
                        try:
                            self._stats["launches"] += 1
                            self._kernel(
                                self.capacity,
                                len(prepared),
                                *(self._buffers[name] for name in self._layouts),
                                mode="cuda",
                            )
                            self._runtime.sync()
                        finally:
                            self._stats["kernel_seconds"] += perf_counter() - phase
                        phase = perf_counter()
                        try:
                            raw = {
                                name: self._download(name)
                                for name in (
                                    "meta",
                                    "scalars",
                                    "vectors",
                                    "matrices",
                                    "contact",
                                    "trace_meta",
                                    "trace",
                                )
                            }
                        finally:
                            self._stats["download_seconds"] += perf_counter() - phase
                except BaseException:
                    self._faulted = True
                    raise
                phase = perf_counter()
                try:
                    for name, array in raw.items():
                        shape, dtype = self._layouts[name]
                        if (
                            not isinstance(array, np.ndarray)
                            or array.shape != shape
                            or array.dtype != np.dtype(dtype)
                        ):
                            raise TyxCoupledFrictionFailure("invalid device readback layout", raw)
                    raw = {name: _owned(array) for name, array in raw.items()}
                    if any(np.any(array[len(prepared) :] != 0) for array in raw.values()):
                        raise TyxCoupledFrictionFailure("inactive world padding is not zero", raw)
                    statuses = raw["meta"][: len(prepared), 0]
                    if np.any(statuses != 0):
                        raise TyxCoupledFrictionFailure("device numerical/status rejection", raw)
                    results = tuple(
                        self._result(original, packed, index, raw)
                        for index, (original, packed) in enumerate(prepared)
                    )
                    self._last_results = results
                    return results
                finally:
                    self._stats["validation_seconds"] += perf_counter() - phase
            finally:
                self._stats["total_seconds"] += perf_counter() - start
                self._busy = False

    def _result(self, original, packed, index, raw):
        nv, ns = len(packed.input_velocity), len(packed.samples)
        meta = raw["meta"][index]
        trace_count = int(meta[2])
        if not (
            0 <= meta[1] <= 24
            and 2 <= trace_count <= TRACE_CAPACITY
            and 0 <= meta[3] <= 288
            and 0 <= meta[4] <= 24 * 6
        ):
            raise TyxCoupledFrictionFailure("invalid device iteration metadata", raw)
        for name in ("scalars", "vectors", "matrices", "contact", "trace"):
            if not np.all(np.isfinite(raw[name][index])):
                raise TyxCoupledFrictionFailure("nonfinite device output", raw)
        if (
            np.any(raw["vectors"][index, :, nv:] != 0)
            or np.any(raw["matrices"][index, :, nv:, :] != 0)
            or np.any(raw["matrices"][index, :, :, nv:] != 0)
            or np.any(raw["contact"][index, ns:] != 0)
            or np.any(raw["trace"][index, trace_count:] != 0)
            or np.any(raw["trace_meta"][index, trace_count:] != 0)
            or np.any(raw["trace"][index, :trace_count, 3 + nv : 9] != 0)
            or np.any(raw["trace"][index, :trace_count, 9 + nv :] != 0)
        ):
            raise TyxCoupledFrictionFailure("effective-size padding is not zero", raw)
        velocity = _owned(raw["vectors"][index, 1, :nv])
        contact = raw["contact"][index, :ns]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                norm, scale, expected, residual, total, energy = _certificate(packed, velocity)
                original_norm, original_scale, _, _, _, _ = _certificate(original, velocity)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise TyxCoupledFrictionFailure(
                f"independent HOST verification failed: {error}", raw
            ) from error
        device_norm, device_tolerance = (float(v) for v in raw["scalars"][index, :2])
        if device_norm < 0 or device_tolerance <= 0 or device_norm > device_tolerance:
            raise TyxCoupledFrictionFailure("device residual gate failed", raw)
        if norm > 2e-5 * scale:
            raise TyxCoupledFrictionFailure("independent FP64 residual gate failed", raw)
        actual_total = np.zeros(nv)
        normal_work = tangent_work = 0.0
        h = float(packed.h_s)
        for sample, actual, check in zip(packed.samples, contact, expected, strict=True):
            fn, tangent, ft, force, moment = check
            if actual[0] < 0 or abs(float(actual[0]) - fn) > 0.002 + 0.002 * abs(fn):
                raise TyxCoupledFrictionFailure("normal load does not match actual velocity", raw)
            for actual_vector, expected_vector, absolute in (
                (actual[6:9], ft, 0.002),
                (actual[9:12], force, 0.002),
                (actual[12:15], moment, 1e-5),
            ):
                if np.any(
                    np.abs(actual_vector.astype(np.float64) - expected_vector)
                    > absolute + 0.002 * np.abs(expected_vector)
                ):
                    raise TyxCoupledFrictionFailure(
                        "load or wrench does not match actual velocity", raw
                    )
            g = sample.jacobian_a.astype(np.float64) - sample.jacobian_b.astype(np.float64)
            relative = g @ velocity.astype(np.float64)
            actual_force = actual[9:12].astype(np.float64)
            actual_total += g.T @ actual_force
            normal_work += (
                h * float(actual[0]) * float(sample.normal_world.astype(np.float64) @ relative)
            )
            actual_ft = actual[6:9].astype(np.float64)
            tangent_work += h * float(actual_ft @ tangent)
            if np.linalg.norm(actual_ft) > float(sample.mu) * float(actual[0]) + 1e-5:
                raise TyxCoupledFrictionFailure("final-normal friction capacity exceeded", raw)
            if float(actual_ft @ tangent) > 1e-14:
                raise TyxCoupledFrictionFailure("friction has positive relative power", raw)
        # The certificate above checks the constitutive law. This second
        # residual and energy ledger use the stored device forces themselves.
        mass = packed.mass_matrix.astype(np.float64)
        response = mass + h * np.diag(packed.damping.astype(np.float64))
        stored_residual = (
            response @ velocity.astype(np.float64)
            - mass @ packed.input_velocity.astype(np.float64)
            - h * (packed.applied_force.astype(np.float64) + actual_total)
        )
        stored_norm = float(
            np.linalg.norm(np.linalg.solve(np.linalg.cholesky(response), stored_residual))
        )
        if not np.isfinite(stored_norm) or stored_norm > 2e-5 * scale:
            raise TyxCoupledFrictionFailure("stored device force/motion residual gate failed", raw)
        energy["normal_work_j"] = normal_work
        energy["tangent_work_j"] = tangent_work
        energy["kinetic_balance_residual_j"] = (
            energy["kinetic_exit_j"]
            - energy["kinetic_entry_j"]
            - energy["external_work_j"]
            + energy["damping_loss_j"]
            - normal_work
            - tangent_work
            + energy["kinetic_numerical_loss_j"]
        )
        energy["normal_spring_balance_j"] = (
            energy["spring_exit_j"] - energy["spring_entry_j"] + normal_work
        )
        if not all(np.isfinite(value) for value in energy.values()):
            raise TyxCoupledFrictionFailure("nonfinite actual-device-force energy ledger", raw)
        phases = {0: "initial", 1: "trial", 2: "final"}
        decisions = {0: "rejected", 1: "accepted", 2: "converged"}
        traces = []
        for row in range(trace_count):
            metadata = raw["trace_meta"][index, row]
            values = raw["trace"][index, row]
            if int(metadata[2]) not in phases or int(metadata[3]) not in decisions:
                raise TyxCoupledFrictionFailure("invalid accepted trace metadata", raw)
            traces.append(
                DeviceFrictionTrace(
                    int(metadata[0]),
                    int(metadata[1]),
                    phases[int(metadata[2])],
                    decisions[int(metadata[3])],
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    _owned(values[3 : 3 + nv]),
                    _owned(values[9 : 9 + nv]),
                )
            )
        if (
            traces[-1].phase != "final"
            or traces[-1].decision != "converged"
            or not np.array_equal(traces[-1].velocity, velocity)
        ):
            raise TyxCoupledFrictionFailure("final trace does not bind the published velocity", raw)
        diagnostics = MappingProxyType(
            dict(
                device_residual_norm=device_norm,
                device_tolerance=device_tolerance,
                host_residual_norm=norm,
                host_tolerance=2e-5 * scale,
                stored_force_residual_norm=stored_norm,
                host_reference_gate=bool(norm <= 1e-10 * scale),
                host_reference_tolerance=1e-10 * scale,
                original_residual_norm=original_norm,
                original_reference_gate=bool(original_norm <= 1e-10 * original_scale),
                iterations=int(meta[1]),
                backtracks=int(meta[3]),
                line_search_trials=int(meta[3]),
                pivot_swaps=int(meta[4]),
                qualification="FP32_DEVICE_REFERENCE",
                energy_basis="actual-uploaded-inputs-frozen-linear-gap",
            )
        )
        return TyxCoupledFrictionResult(
            original,
            packed,
            velocity,
            _owned(raw["vectors"][index, 0, :nv]),
            _owned(contact[:, 0]),
            _owned(contact[:, 6:9]),
            _owned(contact[:, 9:12]),
            _owned(-contact[:, 9:12]),
            _owned(contact[:, 12:15]),
            _owned(-contact[:, 12:15]),
            _owned(raw["vectors"][index, 7, :nv]),
            _owned(raw["vectors"][index, 3, :nv]),
            _owned(raw["matrices"][index, 2, :nv, :nv]),
            _owned(contact),
            _owned(raw["matrices"][index, 0, :nv, :nv]),
            tuple(traces),
            diagnostics,
            MappingProxyType(energy),
        )

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close during workspace solve")
            self._resources.close()
            if self._resources.closed:
                self._buffers.clear()
                self._stats["live_bytes"] = 0

    def __enter__(self):
        if self._resources.state != "OPEN" or self.faulted:
            raise RuntimeError("workspace is not open")
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup(
                    "workspace operation and cleanup failed", [error, cleanup]
                ) from None
            raise
        return False


def _plain(value):
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("nonfinite accepted array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
