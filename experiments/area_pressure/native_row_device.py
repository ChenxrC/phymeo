"""HOST-staged frozen native rows evaluated by the existing TYX working-set kernel.

This experiment does not compile a model, assemble contacts, integrate a state, or
relax a production admission gate. Each evaluation uploads a HOST load and
downloads its outputs; this is not a device-resident coupled solve. Acceleration
and forces use the selected backend; float64 returns contain actual FP32 output.
"""

from threading import RLock

import numpy as np

_KERNEL_FIELDS = (
    "mass",
    "bias",
    "passive",
    "applied",
    "efc_data",
    "efc_J",
    "counts",
    "physical_factor",
    "mass_work",
    "smooth_acc",
    "face_matrix",
    "face_factor",
    "face_rhs",
    "face_forward",
    "trial_acc",
    "trial_force",
    "working",
    "previous_working",
    "previous_force",
    "free_ids",
    "constraint",
    "total_applied",
    "physical_acc",
    "diagnostics",
    "status",
)


def _fp32(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all() or np.any(
        (raw != 0) & (np.abs(result) < np.finfo(np.float32).tiny)
    ):
        raise ValueError(f"{name} must be representable as normal FP32 or zero")
    return result


class NativeRowSolveError(RuntimeError):
    """A rejected device attempt, with its diagnostic packet still available."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            f"native row solve rejected: status={diagnostics['status']}, "
            f"iterations={diagnostics['iterations']}, "
            f"KKT={diagnostics['kkt_scaled_residual']}"
        )


class NativeRowDevice:
    """Evaluate fixed unilateral rows on one to eight physical velocity DOFs.

    Rows obey ``f >= 0`` and ``J @ acceleration + R*f - aref >= 0`` with
    complementarity. ``load`` is the complete unconstrained generalized force,
    including any caller-computed passive, bias, and area trial contributions.
    This wrapper therefore supplies zero separate passive/bias buffers.

    Each call is a cold working-set solve. ``diagnostics`` includes ``iterations``,
    ``kkt_scaled_residual``, ``kkt_absolute_residual``, ``mass_scaled_residual``,
    ``working_set`` (0=lower, 1=free, 2=upper), and positive-force ``active_rows``.
    Nonconvergence raises :class:`NativeRowSolveError`; no failed solution returns.
    """

    def __init__(
        self,
        mass,
        jacobian,
        regularization,
        reference_acceleration,
        *,
        budget=128,
        tolerance=1e-5,
        mode="cuda",
    ):
        if type(budget) is not int or not 1 <= budget <= 512:
            raise ValueError("budget must be an integer in 1..512")
        if type(mode) is not str or mode not in ("cuda", "interpret"):
            raise ValueError("mode must be 'cuda' or 'interpret'")
        tolerance32 = _fp32(tolerance, "tolerance", ())
        if tolerance32 <= 0:
            raise ValueError("tolerance must be positive")
        physical_mass = _fp32(mass, "mass")
        if (
            physical_mass.ndim != 2
            or physical_mass.shape[0] != physical_mass.shape[1]
            or not 1 <= physical_mass.shape[0] <= 8
        ):
            raise ValueError("mass must be a square matrix with 1..8 velocity DOFs")
        if not np.array_equal(physical_mass, physical_mass.T):
            raise ValueError("mass must be symmetric in FP32")
        # Input validation only. All accelerations and forces come from solve().
        try:
            np.linalg.cholesky(physical_mass.astype(np.float64))
        except np.linalg.LinAlgError as error:
            raise ValueError("mass must be positive definite in FP32") from error
        nv = physical_mass.shape[0]
        jac = _fp32(jacobian, "jacobian")
        if jac.ndim != 2 or jac.shape[1] != nv or jac.shape[0] > 512:
            raise ValueError(f"jacobian must have shape (0..512, {nv})")
        nrow = jac.shape[0]
        reg = _fp32(regularization, "regularization", (nrow,))
        aref = _fp32(reference_acceleration, "reference_acceleration", (nrow,))
        if np.any(reg <= 0):
            raise ValueError("regularization must be strictly positive")
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            inverse_reg = np.float32(1.0) / reg
        if not np.isfinite(inverse_reg).all() or np.any(inverse_reg < np.finfo(np.float32).tiny):
            raise ValueError("inverse regularization must be positive normal FP32")

        import tyx as tx

        from phymeo.backends.mujoco_tyx._constraint_solver import solve
        from phymeo.backends.mujoco_tyx._resources import DeviceResources

        self._tx, self._solve = tx, solve
        self._nv, self._nrow = nv, nrow
        self._budget, self._tolerance, self._mode = budget, float(tolerance32), mode
        self._lock = RLock()
        self._resources = DeviceResources()
        self._buffers = {}
        width, storage = 8, max(1, nrow)
        shapes = {
            "mass": (1, width, width),
            "efc_data": (1, storage, 14),
            "efc_J": (1, storage, width),
            "counts": (1, 3),
            "physical_factor": (1, width, width),
            "face_matrix": (1, width, width),
            "face_factor": (1, width, width),
            "trial_force": (1, storage),
            "working": (1, storage),
            "previous_working": (1, storage),
            "previous_force": (1, storage),
            "diagnostics": (1, 10),
            "status": (1,),
        }
        integer_fields = {"counts", "working", "previous_working", "free_ids", "status"}
        packet = {
            name: np.zeros(
                shapes.get(name, (1, width)),
                dtype=np.int32 if name in integer_fields else np.float32,
            )
            for name in _KERNEL_FIELDS
        }
        packet["mass"][0, :nv, :nv] = physical_mass
        packet["efc_J"][0, :nrow, :nv] = jac
        packet["efc_data"][0, :nrow, 3] = aref
        packet["efc_data"][0, :nrow, 4] = reg
        packet["efc_data"][0, :nrow, 5] = inverse_reg
        packet["efc_data"][0, :nrow, 11] = 1.0
        packet["counts"][0, 2] = nrow
        for value in packet.values():
            value.setflags(write=False)
        self._packet = packet
        try:
            self._runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu")
            self._resources.set_runtime(self._runtime)
            with tx.use_runtime(self._runtime):
                for name in _KERNEL_FIELDS:
                    value = packet[name]
                    if mode == "interpret":
                        self._buffers[name] = value.copy()
                    else:
                        buffer = tx.cuda_array(value)
                        self._resources.add_buffer(buffer)
                        self._buffers[name] = buffer
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "native row construction and cleanup failed", [error, cleanup]
                ) from None
            raise

    def _put(self, name, value):
        if self._mode == "interpret":
            np.copyto(self._buffers[name], value)
        else:
            self._buffers[name].copy_from(value)

    def _get(self, name):
        buffer = self._buffers[name]
        return buffer.copy() if self._mode == "interpret" else buffer.to_numpy()

    def evaluate(self, load):
        with self._lock:
            if self._resources.state != "OPEN":
                raise RuntimeError("native row device is not open")
            values = _fp32(load, "load", (self._nv,))
            applied = np.zeros((1, 8), dtype=np.float32)
            applied[0, : self._nv] = values
            with self._tx.use_runtime(self._runtime):
                self._put("applied", applied)
                self._put("status", self._packet["status"])
                self._solve(
                    1,
                    self._nv,
                    self._nrow,
                    self._budget,
                    self._tolerance,
                    *(self._buffers[name] for name in _KERNEL_FIELDS),
                    mode=self._mode,
                )
                status = int(self._get("status")[0])
                raw = self._get("diagnostics")[0]
                acceleration = self._get("physical_acc")[0, : self._nv].astype(np.float64)
                rows = self._get("efc_data")[0, : self._nrow]
                forces = rows[:, 9].astype(np.float64)
                working = self._get("working")[0, : self._nrow]
            diagnostics = {
                "status": status,
                "converged": status == 0 and raw[6] == 1.0,
                "iterations": int(raw[0]),
                "kkt_scaled_residual": float(raw[1]),
                "kkt_absolute_residual": float(raw[2]),
                "bound_violation": float(raw[3]),
                "nrow": int(raw[4]),
                "free_count": int(raw[5]),
                "budget": int(raw[7]),
                "mass_scaled_residual": float(raw[8]),
                "kkt_scale": float(raw[9]),
                "tolerance": self._tolerance,
                "mode": self._mode,
                "working_set": working.tolist(),
                "active_rows": np.flatnonzero(forces > 0).tolist(),
                "row_gradients": rows[:, 10].astype(np.float64).tolist(),
            }
            diagnostics["converged"] = bool(diagnostics["converged"])
            if (
                not diagnostics["converged"]
                or not np.isfinite(raw).all()
                or not np.isfinite(acceleration).all()
                or not np.isfinite(forces).all()
                or np.any(forces < 0)
            ):
                raise NativeRowSolveError(diagnostics)
            return acceleration, forces, diagnostics

    def close(self):
        with self._lock:
            self._resources.close()
            if self._resources.closed:
                self._buffers.clear()

    def __enter__(self):
        if self._resources.state != "OPEN":
            raise RuntimeError("native row device is not open")
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup(
                    "native row operation and cleanup failed", [error, cleanup]
                ) from None
            raise
        return False
