"""Joint device projection with independent host gates and whole-frame rollback."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from math import fsum

import numpy as np

from ..constraints.coloring import color_edges
from ..constraints.distance import distance_residual
from ..constraints.particle_sphere import (
    ContactTransfer,
    particle_sphere_residual,
    particle_sphere_velocity_residual,
)
from ..core.capabilities import SolverFailure
from ..core.coupled import CoupledState, CoupledSubstepReport
from ..core.rigid import RigidBodyState
from ..core.state import ParticleState
from ..geometry.particle_sphere import particle_sphere_geometry
from ..solvers.coupled_xpbd import CoupledXPBDSolver, _impulse_ledger
from .tyx import TyxSemiImplicitSolver
from .tyx_rigid import TyxRigidSemiImplicitSolver


@dataclass(frozen=True, eq=False)
class _DeviceFrame(CoupledXPBDSolver):
    executor: object = None

    def _substep(self, initial, inputs, h):
        return self.executor._substep(initial, inputs, h)


def _assigned_loads(pair_loads, particle_count, body_count, kind):
    """Account for already assigned device loads; never modify velocities here."""
    rows = np.asarray(pair_loads, dtype=np.float64).reshape(particle_count, body_count, 3)
    particle = np.zeros((particle_count, 3))
    rigid = np.zeros((body_count, 3))
    for i in range(particle_count):
        for axis in range(3):
            particle[i, axis] = fsum(rows[i, :, axis])
    for j in range(body_count):
        for axis in range(3):
            rigid[j, axis] = -fsum(rows[:, j, axis])
    return ContactTransfer(particle, rigid, kind)


class TyxCoupledXPBDSolver:
    """FP32 prediction, colored distance and serial shared-body contact on tyx.

    Every iteration synchronizes device positions/multipliers for independent
    FP64 residual checks. Buffers are per substep; free prediction also transfers
    through host snapshots. This first implementation makes no speedup claim.
    """

    MODES = ("interpret", "cpu_cpp", "cuda")

    def __init__(
        self,
        particle_model,
        rigid_model,
        constraints,
        spheres,
        *,
        mode="interpret",
        substeps=4,
        iterations=200,
        tolerance=1e-6,
        reference_length=1.0,
        reference_speed=1.0,
        record_substeps=False,
        cache_root=None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self._frame = _DeviceFrame(
            particle_model,
            rigid_model,
            constraints,
            spheres,
            substeps,
            iterations,
            tolerance,
            reference_length,
            reference_speed,
            record_substeps,
            executor=self,
        )
        if particle_model.inverse_mass.dtype != np.float32:
            raise ValueError("the tyx coupled adapter supports FP32 state only")
        indices = np.empty((0, 2), np.int64) if constraints is None else constraints.indices
        self._colors = color_edges(indices)
        counts = (
            particle_model.particle_count,
            rigid_model.body_count,
            len(indices),
            particle_model.particle_count * rigid_model.body_count,
        )
        if max(counts) > np.iinfo(np.int32).max:
            raise ValueError("coupled counts exceed the tyx i32 interface")
        self._rigid_solver = None
        self._particle_solver = TyxSemiImplicitSolver(mode, cache_root=cache_root)
        self._runtime, self._tx = self._particle_solver._runtime, self._particle_solver._tx
        self._mode = mode
        try:
            from ._tyx_coupled_kernels import (
                contact_geometry,
                project_contact_velocities,
                project_contacts,
                reconstruct_velocity,
            )
            from ._tyx_kernels import project_distance_color

            self._project_contacts = project_contacts
            self._contact_geometry = contact_geometry
            self._project_velocities = project_contact_velocities
            self._reconstruct = reconstruct_velocity
            self._project_distances = project_distance_color
            self._rigid_solver = TyxRigidSemiImplicitSolver(mode, cache_root=cache_root)
        except BaseException:
            self.close()
            raise

    @property
    def mode(self):
        return self._mode

    def __enter__(self):
        if self._runtime.closed:
            raise RuntimeError("tyx coupled solver is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    @property
    def color_count(self):
        return len(self._colors.offsets) - 1

    def close(self):
        with ExitStack() as cleanup:
            cleanup.callback(self._particle_solver.close)
            if self._rigid_solver is not None:
                cleanup.callback(self._rigid_solver.close)

    def step(self, state, inputs, dt):
        if self._runtime.closed:
            raise RuntimeError("tyx coupled solver is closed")
        result = self._frame.step(state, inputs, dt)
        return replace(result, report=replace(result.report, backend=f"tyx:{self.mode}"))

    def _substep(self, initial, inputs, h):
        config = self._frame
        iteration = velocity_iteration = 0
        distance = contact = velocity = float("inf")
        geometric = penetration = 0.0

        def reject(reason, *, numerical=False):
            residuals = (float("inf"),) * 3 if numerical else (distance, contact, velocity)
            return None, CoupledSubstepReport(
                False,
                iteration,
                velocity_iteration,
                *residuals,
                geometric,
                penetration,
                reason=reason,
            )

        try:
            p = self._particle_solver.step(
                config.particle_model, initial.particles, inputs.particles, h
            )
            b = self._rigid_solver.step(config.rigid_model, initial.rigid, inputs.rigid, h)
            if not p.report.accepted or not b.report.accepted:
                return reject("non-finite or underflowed coupled free prediction", numerical=True)
            np_count, nb_count = initial.particles.particle_count, initial.rigid.body_count
            pair_count = np_count * nb_count
            constraints = config.constraints
            edge_count = 0 if constraints is None else len(constraints.indices)
            alpha64 = np.zeros(edge_count)
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                if edge_count:
                    alpha64 = constraints.compliance.astype(np.float64) / float(h) / float(h)
                alpha32 = alpha64.astype(np.float32)
                band = config.tolerance * config.reference_length
                band32 = np.float32(band)
                if float(band32) < band:
                    band32 = np.nextafter(band32, np.float32(np.inf))
            if (
                not np.isfinite(alpha32).all()
                or np.any((alpha32 == 0) & (alpha64 > 0))
                or not np.isfinite(band32)
            ):
                return reject("scaled constraint parameters are outside FP32 range", numerical=True)
            position_transfers, velocity_transfers = [], []
            wp, wb = config.particle_model.inverse_mass, config.rigid_model.inverse_mass
            with (
                self._tx.use_runtime(self._runtime),
                ExitStack() as resources,
                np.errstate(all="ignore"),
            ):

                def buffer(value):
                    # No empty CUDA allocation or kernel argument: consumers skip zero counts.
                    if self.mode != "cuda" or value.size == 0:
                        return np.array(value, order="C", copy=True)
                    device = self._tx.cuda_array(value)
                    resources.callback(device.free)
                    return device

                def read(value):
                    return value if isinstance(value, np.ndarray) else value.to_numpy()

                pp, bp = buffer(p.state.position), buffer(b.state.position)
                pv, bv = buffer(p.state.velocity), buffer(b.state.velocity)
                p_initial, b_initial = (
                    buffer(initial.particles.position),
                    buffer(initial.rigid.position),
                )
                p_predicted, b_predicted = buffer(p.state.position), buffer(b.state.position)
                p_weights, b_weights = buffer(wp), buffer(wb)
                radii = buffer(config.spheres.radii)
                failed = buffer(
                    np.zeros(max(np_count, nb_count, pair_count, edge_count, 1), np.int32)
                )
                contact_lambda = buffer(np.zeros(pair_count, np.float32))
                pair_loads = buffer(np.zeros((pair_count, 3), np.float32))
                impulses = buffer(np.zeros(pair_count, np.float32))
                normals = buffer(np.zeros((pair_count, 3), np.float32))
                gaps = buffer(np.zeros(pair_count, np.float32))
                edge_lambda = buffer(np.zeros(edge_count, np.float32))
                if edge_count:
                    order = buffer(self._colors.order.astype(np.int32))
                    edges = buffer(constraints.indices.astype(np.int32))
                    rest = buffer(constraints.rest_length)
                    alpha = buffer(alpha32)
                resources.callback(self._runtime.sync)

                for iteration in range(1, config.iterations + 1):
                    if edge_count:
                        for begin, end in zip(
                            self._colors.offsets[:-1], self._colors.offsets[1:], strict=True
                        ):
                            self._project_distances(
                                int(end - begin),
                                int(begin),
                                order,
                                edges,
                                rest,
                                alpha,
                                p_weights,
                                pp,
                                edge_lambda,
                                failed,
                                mode=self.mode,
                            )
                    if pair_count:
                        self._project_contacts(
                            np_count,
                            nb_count,
                            pp,
                            bp,
                            p_weights,
                            b_weights,
                            radii,
                            contact_lambda,
                            pair_loads,
                            failed,
                            mode=self.mode,
                        )
                    self._runtime.sync()
                    host_p, host_b = read(pp), read(bp)
                    host_lambda = read(contact_lambda).astype(np.float64)
                    host_edge_lambda = read(edge_lambda).astype(np.float64)
                    if np.any(read(failed)) or not all(
                        np.isfinite(v).all()
                        for v in (host_p, host_b, host_lambda, host_edge_lambda)
                    ):
                        return reject("invalid device position projection", numerical=True)
                    position_transfers.append(
                        _assigned_loads(read(pair_loads), np_count, nb_count, "position")
                    )
                    distance, geometric = (
                        (0.0, 0.0)
                        if constraints is None
                        else distance_residual(
                            host_p, constraints, alpha64, host_edge_lambda, config.reference_length
                        )
                    )
                    contact = particle_sphere_residual(
                        host_p,
                        host_b,
                        wp,
                        wb,
                        config.spheres,
                        host_lambda,
                        reference_length=config.reference_length,
                    )
                    if not np.isfinite([distance, geometric, contact]).all():
                        return reject("invalid host position residual", numerical=True)
                    if max(distance, contact) <= config.tolerance:
                        break
                else:
                    return reject("coupled position iteration budget exhausted")

                if np_count:
                    self._reconstruct(
                        np_count,
                        h,
                        p_weights,
                        p_initial,
                        p_predicted,
                        pp,
                        pv,
                        failed,
                        mode=self.mode,
                    )
                if nb_count:
                    self._reconstruct(
                        nb_count,
                        h,
                        b_weights,
                        b_initial,
                        b_predicted,
                        bp,
                        bv,
                        failed,
                        mode=self.mode,
                    )
                if pair_count:
                    self._contact_geometry(
                        np_count, nb_count, pp, bp, radii, normals, gaps, failed, mode=self.mode
                    )
                self._runtime.sync()
                if np.any(read(failed)) or not all(
                    np.isfinite(v).all() for v in (read(pv), read(bv), read(normals), read(gaps))
                ):
                    return reject(
                        "invalid velocity reconstruction or device geometry", numerical=True
                    )
                geometry = particle_sphere_geometry(host_p, host_b, config.spheres)
                penetration = float(np.max(np.maximum(-geometry.gap, 0), initial=0))
                for velocity_iteration in range(1, config.iterations + 1):
                    if pair_count:
                        self._project_velocities(
                            np_count,
                            nb_count,
                            band32,
                            pv,
                            bv,
                            p_weights,
                            b_weights,
                            normals,
                            gaps,
                            impulses,
                            pair_loads,
                            failed,
                            mode=self.mode,
                        )
                    self._runtime.sync()
                    host_pv, host_bv = read(pv), read(bv)
                    host_impulse = read(impulses).astype(np.float64)
                    if np.any(read(failed)) or not all(
                        np.isfinite(v).all() for v in (host_pv, host_bv, host_impulse)
                    ):
                        return reject("invalid device velocity projection", numerical=True)
                    velocity_transfers.append(
                        _assigned_loads(read(pair_loads), np_count, nb_count, "velocity")
                    )
                    velocity = particle_sphere_velocity_residual(
                        host_pv,
                        host_bv,
                        wp,
                        wb,
                        geometry,
                        host_impulse,
                        contact_tolerance=band,
                        reference_speed=config.reference_speed,
                    )
                    if not np.isfinite(velocity):
                        return reject("invalid host velocity residual", numerical=True)
                    if velocity <= config.tolerance:
                        break
                else:
                    return reject("coupled velocity iteration budget exhausted")
                candidate = CoupledState(
                    ParticleState(host_p, host_pv),
                    RigidBodyState(host_b, b.state.orientation, host_bv, b.state.angular_velocity),
                )
                position_impulses = _impulse_ledger(
                    position_transfers, float(h), np_count, nb_count
                )
                velocity_impulses = _impulse_ledger(velocity_transfers, 1.0, np_count, nb_count)
            return candidate, CoupledSubstepReport(
                True,
                iteration,
                velocity_iteration,
                distance,
                contact,
                velocity,
                geometric,
                penetration,
                position_impulses,
                velocity_impulses,
            )
        except (SolverFailure, FloatingPointError, OverflowError) as error:
            return reject(str(error), numerical=True)
