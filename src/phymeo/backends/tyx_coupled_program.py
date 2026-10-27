"""Real tyx stage workspaces; the generic runtime owns time and acceptance."""

from contextlib import ExitStack
from functools import wraps

import numpy as np

from ..constraints.distance import distance_residual
from ..constraints.particle_sphere import (
    particle_sphere_residual,
    particle_sphere_velocity_residual,
)
from ..core.capabilities import SolverFailure
from ..core.rigid import RigidBodyState
from ..core.scene import SceneState, StageMetrics
from ..core.state import ParticleState
from ..geometry.particle_sphere import particle_sphere_geometry
from ..runtime.program import ProgramCapabilities
from ..solvers.coupled_program import coupled_configuration, coupled_scene, coupled_signature
from ..solvers.coupled_xpbd import _impulse_ledger
from .tyx_coupled import TyxCoupledXPBDSolver, _assigned_loads


def _device_arithmetic(function):
    @wraps(function)
    def call(*args, **kwargs):
        # Preserve the existing interpreter/device arithmetic and explicit failure flags.
        with np.errstate(all="ignore"):
            return function(*args, **kwargs)

    return call


class TyxCoupledProgram:
    history_fields = ()

    def __init__(
        self,
        particle_model,
        rigid_model,
        constraints,
        spheres,
        *,
        mode="interpret",
        cache_root=None,
    ):
        # Reuse toolchain/kernel ownership only. Neither step nor _substep is invoked.
        self.engine = TyxCoupledXPBDSolver(
            particle_model, rigid_model, constraints, spheres, mode=mode, cache_root=cache_root
        )
        try:
            self.scene = coupled_scene(self.engine._frame)
            self._configuration = coupled_configuration(self.engine._frame, self.engine._colors)
            self.capabilities = ProgramCapabilities(
                f"tyx:{mode}",
                {"float32"},
                {"particles", "rigid_bodies"},
                {"distance", "point_sphere", "rotation"},
            )
        except BaseException:
            self.engine.close()
            raise

    def problem_signature(self):
        if coupled_configuration(self.engine._frame, self.engine._colors) != self._configuration:
            raise ValueError(
                "coupled configuration changed physical problem or layout; bind a new program"
            )
        return coupled_signature(self.engine._frame)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self.engine._runtime.closed:
            raise RuntimeError("tyx coupled program is closed")
        self.problem_signature()
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("coupled program must use its bound physical models")
        return _DeviceWorkspace(self.engine, state, inputs, h, settings)

    def close(self):
        self.engine.close()


class _DeviceWorkspace:
    def __init__(self, engine, state, inputs, h, settings):
        self.engine, self.config = engine, engine._frame
        self.initial, self.inputs, self.h, self.settings = state, inputs, h, settings
        self.mode, self.runtime, self.tx = engine.mode, engine._runtime, engine._tx
        self._resources = ExitStack()
        self.position_transfers, self.velocity_transfers = [], []
        self.np_count = self.config.particle_model.particle_count
        self.nb_count = self.config.rigid_model.body_count
        self.pair_count = self.np_count * self.nb_count
        self.constraints = self.config.constraints
        self.edge_count = 0 if self.constraints is None else len(self.constraints.indices)
        self.wp, self.wb = (
            self.config.particle_model.inverse_mass,
            self.config.rigid_model.inverse_mass,
        )

    def _buffer(self, value):
        if self.mode != "cuda" or value.size == 0:
            return np.array(value, order="C", copy=True)
        device = self.tx.cuda_array(value)
        self._resources.callback(device.free)
        return device

    @staticmethod
    def _read(value):
        return value if isinstance(value, np.ndarray) else value.to_numpy()

    def _finite(self, *values):
        if np.any(self._read(self.failed)) or not all(np.isfinite(v).all() for v in values):
            raise SolverFailure("invalid device stage output")

    def predict(self):
        p = self.engine._particle_solver.step(
            self.config.particle_model,
            self.initial.domains["particles"],
            self.inputs.domains["particles"],
            self.h,
        )
        b = self.engine._rigid_solver.step(
            self.config.rigid_model,
            self.initial.domains["rigid"],
            self.inputs.domains["rigid"],
            self.h,
        )
        if not p.report.accepted or not b.report.accepted:
            raise SolverFailure("non-finite or underflowed coupled free prediction")
        self.p, self.b = p.state, b.state
        self.alpha64 = np.zeros(self.edge_count)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            if self.edge_count:
                self.alpha64 = (
                    self.constraints.compliance.astype(np.float64) / float(self.h) / float(self.h)
                )
            alpha32 = self.alpha64.astype(np.float32)
            self.band = self.settings.tolerance * self.settings.reference_length
            self.band32 = np.float32(self.band)
            if float(self.band32) < self.band:
                self.band32 = np.nextafter(self.band32, np.float32(np.inf))
        if (
            not np.isfinite(alpha32).all()
            or np.any((alpha32 == 0) & (self.alpha64 > 0))
            or not np.isfinite(self.band32)
        ):
            raise SolverFailure("scaled constraint parameters are outside FP32 range")
        # Entry is retained until close, including on partial allocation failure.
        self._resources.enter_context(self.tx.use_runtime(self.runtime))
        self.pp, self.bp = self._buffer(self.p.position), self._buffer(self.b.position)
        self.pv, self.bv = self._buffer(self.p.velocity), self._buffer(self.b.velocity)
        self.p_initial = self._buffer(self.initial.domains["particles"].position)
        self.b_initial = self._buffer(self.initial.domains["rigid"].position)
        self.p_predicted, self.b_predicted = (
            self._buffer(self.p.position),
            self._buffer(self.b.position),
        )
        self.p_weights, self.b_weights = self._buffer(self.wp), self._buffer(self.wb)
        self.radii = self._buffer(self.config.spheres.radii)
        self.failed = self._buffer(
            np.zeros(
                max(self.np_count, self.nb_count, self.pair_count, self.edge_count, 1), np.int32
            )
        )
        self.contact_lambda = self._buffer(np.zeros(self.pair_count, np.float32))
        self.pair_loads = self._buffer(np.zeros((self.pair_count, 3), np.float32))
        self.impulses = self._buffer(np.zeros(self.pair_count, np.float32))
        self.normals = self._buffer(np.zeros((self.pair_count, 3), np.float32))
        self.gaps = self._buffer(np.zeros(self.pair_count, np.float32))
        self.edge_lambda = self._buffer(np.zeros(self.edge_count, np.float32))
        if self.edge_count:
            self.order = self._buffer(self.engine._colors.order.astype(np.int32))
            self.edges = self._buffer(self.constraints.indices.astype(np.int32))
            self.rest = self._buffer(self.constraints.rest_length)
            self.alpha = self._buffer(alpha32)
        # LIFO: synchronize first, free every buffer, then leave the runtime context.
        self._resources.callback(self.runtime.sync)

    @_device_arithmetic
    def position_sweep(self):
        if self.edge_count:
            for begin, end in zip(
                self.engine._colors.offsets[:-1], self.engine._colors.offsets[1:], strict=True
            ):
                self.engine._project_distances(
                    int(end - begin),
                    int(begin),
                    self.order,
                    self.edges,
                    self.rest,
                    self.alpha,
                    self.p_weights,
                    self.pp,
                    self.edge_lambda,
                    self.failed,
                    mode=self.mode,
                )
        if self.pair_count:
            self.engine._project_contacts(
                self.np_count,
                self.nb_count,
                self.pp,
                self.bp,
                self.p_weights,
                self.b_weights,
                self.radii,
                self.contact_lambda,
                self.pair_loads,
                self.failed,
                mode=self.mode,
            )
        self.runtime.sync()
        self.host_p, self.host_b = self._read(self.pp), self._read(self.bp)
        self.host_lambda = self._read(self.contact_lambda).astype(np.float64)
        self.host_edge_lambda = self._read(self.edge_lambda).astype(np.float64)
        self._finite(self.host_p, self.host_b, self.host_lambda, self.host_edge_lambda)
        self.position_transfers.append(
            _assigned_loads(self._read(self.pair_loads), self.np_count, self.nb_count, "position")
        )

    @_device_arithmetic
    def position_metrics(self):
        distance, geometric = (
            (0.0, 0.0)
            if self.constraints is None
            else distance_residual(
                self.host_p,
                self.constraints,
                self.alpha64,
                self.host_edge_lambda,
                self.settings.reference_length,
            )
        )
        contact = particle_sphere_residual(
            self.host_p,
            self.host_b,
            self.wp,
            self.wb,
            self.config.spheres,
            self.host_lambda,
            reference_length=self.settings.reference_length,
        )
        if not np.isfinite([distance, geometric, contact]).all():
            raise SolverFailure("invalid host position residual")
        return StageMetrics({"distance": distance, "contact": contact}, geometric)

    @_device_arithmetic
    def prepare_velocity(self):
        if self.np_count:
            self.engine._reconstruct(
                self.np_count,
                self.h,
                self.p_weights,
                self.p_initial,
                self.p_predicted,
                self.pp,
                self.pv,
                self.failed,
                mode=self.mode,
            )
        if self.nb_count:
            self.engine._reconstruct(
                self.nb_count,
                self.h,
                self.b_weights,
                self.b_initial,
                self.b_predicted,
                self.bp,
                self.bv,
                self.failed,
                mode=self.mode,
            )
        if self.pair_count:
            self.engine._contact_geometry(
                self.np_count,
                self.nb_count,
                self.pp,
                self.bp,
                self.radii,
                self.normals,
                self.gaps,
                self.failed,
                mode=self.mode,
            )
        self.runtime.sync()
        self._finite(
            self._read(self.pv),
            self._read(self.bv),
            self._read(self.normals),
            self._read(self.gaps),
        )
        self.geometry = particle_sphere_geometry(self.host_p, self.host_b, self.config.spheres)
        self.penetration = float(np.max(np.maximum(-self.geometry.gap, 0), initial=0))

    @_device_arithmetic
    def velocity_sweep(self):
        if self.pair_count:
            self.engine._project_velocities(
                self.np_count,
                self.nb_count,
                self.band32,
                self.pv,
                self.bv,
                self.p_weights,
                self.b_weights,
                self.normals,
                self.gaps,
                self.impulses,
                self.pair_loads,
                self.failed,
                mode=self.mode,
            )
        self.runtime.sync()
        self.host_pv, self.host_bv = self._read(self.pv), self._read(self.bv)
        self.host_impulse = self._read(self.impulses).astype(np.float64)
        self._finite(self.host_pv, self.host_bv, self.host_impulse)
        self.velocity_transfers.append(
            _assigned_loads(self._read(self.pair_loads), self.np_count, self.nb_count, "velocity")
        )

    @_device_arithmetic
    def velocity_metrics(self):
        velocity = particle_sphere_velocity_residual(
            self.host_pv,
            self.host_bv,
            self.wp,
            self.wb,
            self.geometry,
            self.host_impulse,
            contact_tolerance=self.band,
            reference_speed=self.settings.reference_speed,
        )
        if not np.isfinite(velocity):
            raise SolverFailure("invalid host velocity residual")
        return StageMetrics({"velocity": velocity}, penetration=self.penetration)

    def result(self):
        return SceneState(
            {
                "particles": ParticleState(self.host_p, self.host_pv),
                "rigid": RigidBodyState(
                    self.host_b, self.b.orientation, self.host_bv, self.b.angular_velocity
                ),
            }
        )

    def diagnostics(self):
        return {
            "position_impulses": _impulse_ledger(
                self.position_transfers, float(self.h), self.np_count, self.nb_count
            ),
            "velocity_impulses": _impulse_ledger(
                self.velocity_transfers, 1.0, self.np_count, self.nb_count
            ),
        }

    def close(self):
        self._resources.close()
