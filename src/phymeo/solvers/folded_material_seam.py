"""Objective single-station contact across a small, changing material hinge."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.result import StepReport
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_step
from ..geometry.folded_material_seam import FoldedMaterialChart, FoldStation, _jacobian
from ._material_stick_law import return_traction
from .collision_loads import NormalLayer, _host_law
from .deformable_sliding import _owned_readonly, _tuple
from .semi_implicit import SemiImplicitSolver


@dataclass(frozen=True)
class FoldedMaterialHistory:
    key: tuple
    start_tick: int
    active_steps: int
    regime: str
    elastic_m: float
    world_elastic_m: tuple
    stored_energy_j: float
    plastic_slip_m: float
    tangent_work_j: float


@dataclass(frozen=True, eq=False)
class FoldedMaterialState:
    """Free particle positions/velocities in the caller's row order and one unfolded history."""

    tick: int
    history: FoldedMaterialHistory | None
    face: object
    weights: tuple
    _position: np.ndarray = field(repr=False)
    _velocity: np.ndarray = field(repr=False)
    _owner: object = field(repr=False)

    @property
    def particles(self):
        return ParticleState(self._position, self._velocity)


@dataclass(frozen=True, eq=False)
class FoldedMaterialLeg:
    face: object
    t0: float
    t1: float
    start: FoldStation
    end: FoldStation
    activation_gap_bounds_m: tuple
    nodal_impulses_ns: np.ndarray
    nodal_work_j: np.ndarray
    normal_work_j: float
    tangent_work_j: float
    integrated_angular_impulse_nms: tuple


@dataclass(frozen=True, eq=False)
class FoldedMaterialStep:
    state: FoldedMaterialState
    report: StepReport
    normal_backend: str
    normal_force_n: float
    tangent_force_scalar_n: float
    legs: tuple
    event_bracket: tuple | None
    site_iterations: int
    jacobian_residual: float
    quadrature_difference: float
    nodal_contact_impulse_ns: np.ndarray | None
    discrete_angular_impulse_nms: tuple
    normal_energy_j: tuple
    normal_work_j: float
    tangent_work_j: float
    plastic_loss_j: float
    implicit_loss_j: float


class FoldedMaterialSeamStepper:
    """Independent masses; one homogeneous pressure station on a transverse, two-face material chart.

    The bisector projection defines the address. Forces follow the selected face normal and
    its oriented transverse tangent. The pullback includes material-frame gradients at a
    frozen material address. Over the affine drift, the algorithm closes the resulting nodal
    impulses against the actual end state. These face and edge certificates have a finite
    domain; unloading, arbitrary along-edge slip or a turning hinge axis are outside it.
    """

    def __init__(
        self,
        model,
        faces,
        *,
        point_row,
        point_material,
        material_rows,
        layer,
        static_friction,
        tangent_stiffness_n_per_m,
        max_angle_rad=np.pi / 12,
        layer_operator=None,
    ):
        angle = finite_real(max_angle_rad, "max_angle_rad")
        if not 0 < angle <= np.pi / 6:
            raise ValueError("small-fold limit must lie in (0, pi/6]")
        chart = FoldedMaterialChart(faces, material_rows, point_row, angle)
        if (
            not isinstance(model, ParticleModel)
            or model.particle_count != 5
            or np.any(model.inverse_mass <= 0)
        ):
            raise ValueError("five independently moving positive-mass particles required")
        point = tuple(point_material)
        if (
            len(point) != 3
            or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in point)
            or point[0] != chart.body[0]
            or (point[:2] == chart.body and point[2] in chart.ids)
        ):
            raise ValueError("invalid material vertex identity in the registered world")
        if not isinstance(layer, NormalLayer) or layer.damping_s_per_m:
            raise ValueError("the folded pressure window uses an undamped NormalLayer")
        if layer_operator is not None and not isinstance(layer_operator, TyxNormalLayer):
            raise TypeError("expected a TyxNormalLayer")
        self._k = finite_real(tangent_stiffness_n_per_m, "tangent_stiffness_n_per_m")
        self._static = finite_real(static_friction, "static_friction")
        if self._k <= 0 or self._static < layer.friction:
            raise ValueError("invalid elastic shear calibration")
        self._model, self._chart, self._point = model, chart, tuple(map(int, point))
        self._layer, self._normal = layer, layer_operator
        self._solver, self._owner = SemiImplicitSolver(), object()

    def _state(self, state, tick, history=None, face=None, weights=()):
        return FoldedMaterialState(
            tick,
            history,
            face,
            weights,
            _owned_readonly(state.position),
            _owned_readonly(state.velocity),
            self._owner,
        )

    def initial_state(self, state):
        p, _, _ = validate_step(
            self._model, state, StepInputs.zeros(5, dtype=self._model.inverse_mass.dtype), 1.0
        )
        x = p.position[list(self._chart.rows)].astype(float)
        self._chart.trace(x, x, self._layer.width_m, float(np.finfo(p.position.dtype).eps))
        return self._state(p, 0)

    def _pressure(self, station, rate):
        law = self._layer
        n = np.asarray(station.normal)
        data = np.array(
            [
                [
                    *n,
                    *(n * rate),
                    law.width_m,
                    station.normal_gap_m,
                    law.stiffness_n_per_m,
                    1.0,
                    0.0,
                    0.0,
                    law.slip_speed_m_s,
                ]
            ]
        )
        measured = _host_law(data) if self._normal is None else self._normal.evaluate(data)
        value = np.asarray(measured, dtype=float)
        if (
            value.shape != (1, 5)
            or not np.isfinite(value).all()
            or value[0, 1] <= 0
            or not np.allclose(value[0, 2:], value[0, 1] * n, rtol=2e-6, atol=1e-36)
        ):
            raise SolverFailure("invalid positive folded material pressure")
        return float(value[0, 1])

    def _outrows(self, values):
        result = np.empty_like(values)
        result[list(self._chart.rows)] = values
        return _owned_readonly(result)

    def step(self, state, inputs, dt, *, particle_solver=None, max_iterations=32):
        if not isinstance(state, FoldedMaterialState) or state._owner is not self._owner:
            raise ValueError("folded checkpoint belongs to a different stepper")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, Integral)
            or not 1 <= max_iterations <= 128
        ):
            raise ValueError("folded address iteration budget must be between 1 and 128")
        p, external, hd = validate_step(self._model, state.particles, inputs, dt)
        h = float(hd)
        dtype = p.position.dtype
        eps = float(np.finfo(dtype).eps)
        rows = list(self._chart.rows)
        x = p.position[rows].astype(float)
        v = p.velocity[rows].astype(float)
        prior = state.history
        source = self._chart.start_side(x)
        entry = self._chart.station(x, source, spatial=False)
        if prior is not None and (
            state.face != entry.face
            or max(abs(w - dict(state.weights)[i]) for i, w in entry.material_weights)
            > max(2e-9, 128 * eps)
        ):
            raise SolverFailure("stale unfolded material history")
        J = _jacobian(x, source)
        P = self._pressure(entry, float(np.sum(J[1] * v)))
        old = 0.0 if prior is None else prior.elastic_m
        ext = external[rows].astype(float)
        inv = self._model.inverse_mass[rows].astype(float)
        gravity = self._model.gravity.astype(float)
        backend = "host" if self._normal is None else self._normal.mode
        for it in range(1, max_iterations + 1):
            vn = v + h * (gravity + inv[:, None] * (ext + J[1] * P))
            free = h * float(np.sum(J[0] * vn))
            mobility = float(np.sum(inv[:, None] * J[0] ** 2))
            returned, sliding = return_traction(
                np.array([old]),
                np.array([free]),
                mobility,
                h,
                P,
                prior,
                self._k,
                self._static,
                self._layer.friction,
            )
            shear = float(returned[0])
            traction = np.array([shear, P])
            fc = np.tensordot(traction, J, axes=1)
            force = (ext + fc).astype(dtype)
            if not np.isfinite(force).all() or np.any((force == 0) & (ext + fc != 0)):
                raise SolverFailure("folded force outside selected particle precision")
            vtrial = p.velocity[rows] + hd * (
                self._model.gravity + self._model.inverse_mass[rows, None] * force
            )
            xtrial = p.position[rows] + hd * vtrial
            if not np.isfinite(xtrial).all():
                raise SolverFailure("nonfinite folded particle trial")
            proposed = self._chart.trace(x, xtrial.astype(float), self._layer.width_m, eps)
            if np.max(abs(proposed.jacobian - J)) <= max(2e-12, 0.5 * eps):
                break
            J = proposed.jacobian
        else:
            raise SolverFailure("folded material force/address iteration did not converge")
        forces = np.empty_like(external)
        forces[rows] = force
        solver = self._solver if particle_solver is None else particle_solver
        result = solver.step(self._model, p, StepInputs(forces), h)
        if not result.report.accepted:
            return FoldedMaterialStep(
                state,
                result.report,
                backend,
                0.0,
                0.0,
                (),
                None,
                it,
                0.0,
                0.0,
                None,
                (0.0, 0.0, 0.0),
                (),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        final = result.state
        xf = final.position[rows].astype(float)
        vf = final.velocity[rows].astype(float)
        dx = xf - x
        vscale = np.linalg.norm(v, axis=1) + h * (
            np.linalg.norm(gravity)
            + inv * (np.linalg.norm(ext, axis=1) + np.linalg.norm(fc, axis=1))
        )
        pscale = np.linalg.norm(x, axis=1) + np.linalg.norm(xf, axis=1) + h * vscale
        if (
            not np.isfinite(xf).all()
            or not np.isfinite(vf).all()
            or np.any(np.linalg.norm(vf - vtrial, axis=1) > 64 * eps * vscale)
            or np.any(np.linalg.norm(dx - h * vf, axis=1) > 64 * eps * pscale)
        ):
            raise SolverFailure("particle solver did not realize the folded nodal motion")
        trace = self._chart.trace(x, xf, self._layer.width_m, eps)
        residual = float(np.max(abs(trace.jacobian - J)))
        tol = max(3e-10, 64 * eps)
        if residual + trace.quadrature_difference > tol:
            raise SolverFailure("folded material traction differs from actual particle geometry")
        relative = np.einsum("kir,ir->k", J, dx)
        du = float(relative[0])
        elastic = -shear / self._k if sliding else old + du
        plastic = old + du - elastic if sliding else 0.0
        U = 0.5 * self._k * elastic**2
        U0 = prior.stored_energy_j if prior else 0.0
        W = shear * du
        loss = -shear * plastic
        implicit = 0.5 * self._k * (elastic - old) ** 2
        etol = 96 * eps * (abs(shear) + self._k * abs(elastic - old)) * sum(pscale) + 96 * eps * (
            U + U0 + abs(W) + abs(loss) + implicit
        )
        if (
            not np.isfinite((W, loss, implicit, U, etol)).all()
            or loss < -etol
            or abs(W + U - U0 + loss + implicit) > etol
        ):
            raise SolverFailure("unfolded elastic work or plastic loss did not close")
        legs = []
        for leg in trace.intervals:
            a, b = leg.t0, leg.t1
            impulse = h * np.tensordot(traction, leg.rows, axes=1)
            first = h * np.tensordot(traction, leg.time_rows, axes=1)
            integrated_moment = (np.cross(x, impulse) + np.cross(dx, first)).sum(axis=0)
            nodal_work = np.sum(impulse * dx, axis=1) / h
            lw = traction * np.einsum("kir,ir->k", leg.rows, dx)
            legs.append(
                FoldedMaterialLeg(
                    self._chart.keys[leg.side],
                    a,
                    b,
                    self._chart.station(x + a * dx, leg.side),
                    self._chart.station(x + b * dx, leg.side),
                    leg.gap_bounds,
                    self._outrows(impulse),
                    self._outrows(nodal_work),
                    float(lw[1]),
                    float(lw[0]),
                    _tuple(integrated_moment),
                )
            )
        end = legs[-1].end
        gap_delta = end.normal_gap_m - entry.normal_gap_m
        nwork = P * float(relative[1])
        work_tol = (
            (tol + 128 * eps)
            * max(sum(np.linalg.norm(dx, axis=1)), self._layer.width_m)
            * max(P, 1.0)
            * 16
        )
        if abs(nwork - P * gap_delta) > work_tol:
            raise SolverFailure("folded normal work differs from the actual gap change")
        key = prior.key if prior else ("folded-section", self._point, self._chart.body, entry.face)
        hist = FoldedMaterialHistory(
            key,
            prior.start_tick if prior else state.tick,
            (prior.active_steps if prior else 0) + 1,
            "slide" if sliding else "stick",
            elastic,
            _tuple(elastic * np.asarray(end.tangent)),
            U,
            (prior.plastic_slip_m if prior else 0.0) + abs(plastic),
            (prior.tangent_work_j if prior else 0.0) + W,
        )
        closed = self._state(final, state.tick + 1, hist, end.face, end.material_weights)
        NU = tuple(
            0.5 * self._layer.stiffness_n_per_m * (self._layer.width_m - gap) ** 2
            for gap in (entry.normal_gap_m, end.normal_gap_m)
        )
        return FoldedMaterialStep(
            closed,
            result.report,
            backend,
            P,
            shear,
            tuple(legs),
            trace.event_bracket,
            it,
            residual,
            trace.quadrature_difference,
            self._outrows(h * fc),
            _tuple(h * np.cross(x, fc).sum(axis=0)),
            NU,
            nwork,
            W,
            max(loss, 0.0),
            implicit,
        )
