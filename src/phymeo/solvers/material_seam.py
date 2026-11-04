"""Persistent contact across one shared material edge of an explicitly translating planar patch."""

from dataclasses import dataclass, field
from math import hypot
from numbers import Integral

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.collision_shapes import CollisionKey
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.result import StepReport
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_step
from ..geometry.material_seam import TranslatingMaterialPatch
from ._material_stick_law import return_traction
from .collision_loads import NormalLayer, _host_law
from .deformable_sliding import _owned_readonly, _tuple
from .deformable_stiction import MaterialStickHistory, _in_plane


@dataclass(frozen=True, eq=False)
class MaterialSeamState:
    """Row 0 is the free point. Row 1 is the patch mass center's world translation."""

    tick: int
    history: MaterialStickHistory | None
    face: CollisionKey | None
    weights: tuple
    _position: np.ndarray = field(repr=False)
    _velocity: np.ndarray = field(repr=False)
    _owner: object = field(repr=False)

    @property
    def particles(self):
        return ParticleState(self._position, self._velocity)


@dataclass(frozen=True)
class MaterialFaceLoad:
    face: CollisionKey
    t0: float
    t1: float
    entry_weights: tuple
    exit_weights: tuple
    entry_point_on_vertex_m: tuple
    exit_point_on_vertex_m: tuple
    entry_point_on_patch_m: tuple
    exit_point_on_patch_m: tuple
    impulse_on_vertex_ns: tuple
    node_impulses_ns: tuple
    vertex_work_j: float
    node_work_j: tuple
    tangent_work_j: float


@dataclass(frozen=True, eq=False)
class MaterialSeamStep:
    state: MaterialSeamState
    report: StepReport
    normal_backend: str
    normal_force_n: float
    force_on_vertex_n: tuple
    segments: tuple[MaterialFaceLoad, ...]
    patch_constraint_angular_impulse_nms: tuple
    ended: MaterialStickHistory | None


class MaterialSeamStepper:
    """One independent material point and one possibly massive, translation-only homogeneous sheet.

    The discrete station force is constant through the accepted affine drift. Its exact material
    triangle intervals distribute impulses to chart vertices. The sheet's single translational
    inertia responds to their resultant. Its rotation constraint supplies the reported angular
    impulse about its mass center. Material chart vertices are not independent particle DOFs.
    """

    def __init__(
        self,
        patch,
        *,
        point_material,
        point_mass_kg,
        patch_mass_kg,
        layer,
        static_friction,
        tangent_stiffness_n_per_m,
        gravity=(0.0, 0.0, 0.0),
        dtype=np.float64,
        layer_operator=None,
    ):
        from .semi_implicit import SemiImplicitSolver

        if not isinstance(patch, TranslatingMaterialPatch) or not isinstance(layer, NormalLayer):
            raise TypeError("expected a translating material patch and calibrated normal layer")
        point = tuple(point_material)
        if len(point) != 3 or any(
            isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in point
        ):
            raise ValueError("point_material requires its world, material body and node ID")
        if point[0] != patch.material_body[0] or (
            point[:2] == patch.material_body and point[2] in patch.vertices
        ):
            raise ValueError(
                "point and patch require distinct material addresses in the same world"
            )
        if layer_operator is not None and not isinstance(layer_operator, TyxNormalLayer):
            raise TypeError("expected a TyxNormalLayer")
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("physical state requires FP32 or FP64")
        m = finite_real(point_mass_kg, "point_mass_kg")
        M = None if patch_mass_kg is None else finite_real(patch_mass_kg, "patch_mass_kg")
        if m <= 0 or (M is not None and M <= 0):
            raise ValueError("dynamic masses must be positive; use None for a fixed patch")
        inv = np.array([1 / m, 0 if M is None else 1 / M], dtype=dtype)
        if inv[0] == 0 or (M is not None and inv[1] == 0):
            raise ValueError("inverse mass is below state precision")
        self._model = ParticleModel(inv, np.array(gravity, dtype=dtype))
        self._patch = patch
        self._point = tuple(map(int, point))
        self._layer = layer
        self._normal_operator = layer_operator
        self._solver = SemiImplicitSolver()
        self._owner = object()
        self._static = finite_real(static_friction, "static_friction")
        self._k = finite_real(tangent_stiffness_n_per_m, "tangent_stiffness_n_per_m")
        if self._k <= 0 or self._static < layer.friction:
            raise ValueError("require positive tangent stiffness and static >= kinetic friction")

    def _state(self, particles, tick, history=None, face=None, weights=()):
        return MaterialSeamState(
            tick,
            history,
            face,
            weights,
            _owned_readonly(particles.position),
            _owned_readonly(particles.velocity),
            self._owner,
        )

    def initial_state(
        self,
        point_position,
        point_velocity=(0.0, 0.0, 0.0),
        *,
        patch_translation=(0.0, 0.0, 0.0),
        patch_velocity=(0.0, 0.0, 0.0),
    ):
        dtype = self._model.inverse_mass.dtype
        x = np.asarray([point_position, patch_translation], dtype=dtype)
        v = np.asarray([point_velocity, patch_velocity], dtype=dtype)
        if not self._model.inverse_mass[1] and np.any(v[1]):
            raise ValueError("a fixed patch must be stationary")
        return self._state(ParticleState(x, v), 0)

    def _normal_load(self, n, gap, relative):
        law = self._layer
        if gap >= law.width_m:
            return 0.0, np.zeros(3)
        row = np.asarray(
            [
                [
                    *n,
                    *relative,
                    law.width_m,
                    gap,
                    law.stiffness_n_per_m,
                    1.0,
                    law.damping_s_per_m,
                    0.0,
                    law.slip_speed_m_s,
                ]
            ],
            dtype=float,
        )
        output = (
            _host_law(row) if self._normal_operator is None else self._normal_operator.evaluate(row)
        )
        value = np.asarray(output, dtype=float)
        if value.shape != (1, 5) or not np.isfinite(value).all():
            raise SolverFailure("invalid normal-layer station output")
        mag = value[0, 1]
        if mag < 0 or not np.allclose(value[0, 2:], mag * n, rtol=2e-6, atol=1e-36):
            raise SolverFailure("invalid normal-layer direction")
        if mag == 0 and 1 - law.damping_s_per_m * float(relative @ n) > 0:
            raise SolverFailure("positive normal load below selected precision")
        return float(mag), value[0, 2:]

    def _split(self, trace, start, delta, h, force, shear):
        sections = []
        patch_moment = np.zeros(3)
        for segment in trace:
            a, b = segment.t0, segment.t1
            fraction = b - a
            impulse = h * fraction * force
            reactions = []
            work = []
            for (node, w0), (other, w1) in zip(
                segment.entry_weights, segment.exit_weights, strict=True
            ):
                if node != other:
                    raise SolverFailure("inconsistent interval address")
                weight = (w0 + w1) * 0.5
                ji = -weight * impulse
                reactions.append((node, _tuple(ji)))
                work.append((node, -fraction * weight * float(force @ delta[1])))
                patch_moment += np.cross(self._patch.vertices[node], ji)
            sections.append(
                MaterialFaceLoad(
                    segment.face,
                    a,
                    b,
                    segment.entry_weights,
                    segment.exit_weights,
                    _tuple(start[0] + a * delta[0]),
                    _tuple(start[0] + b * delta[0]),
                    _tuple(start[1] + a * delta[1] + segment.entry_local_m),
                    _tuple(start[1] + b * delta[1] + segment.exit_local_m),
                    _tuple(impulse),
                    tuple(reactions),
                    fraction * float(force @ delta[0]),
                    tuple(work),
                    fraction * float(shear @ (delta[0] - delta[1])),
                )
            )
        return tuple(sections), _tuple(-patch_moment)

    def step(self, state, inputs, dt, *, particle_solver=None):
        if not isinstance(state, MaterialSeamState) or state._owner is not self._owner:
            raise ValueError("seam state belongs to a different stepper")
        current, external, h = validate_step(self._model, state.particles, inputs, dt)
        h = float(h)
        x, v = current.position.astype(float), current.velocity.astype(float)
        inv = self._model.inverse_mass.astype(float)
        n = np.asarray(self._patch.normal)
        gap = self._patch.gap(*x)
        if gap <= 0:
            raise SolverFailure("seam point is on or behind its oriented support plane")
        mag, normal = self._normal_load(n, gap, v[0] - v[1])
        backend = "host" if self._normal_operator is None else self._normal_operator.mode
        forces = external.astype(float).copy()
        forces[0] += normal
        forces[1] -= normal
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            accel = self._model.gravity.astype(float) + inv[:, None] * forces
            accel[inv == 0] = 0
            free_v = v + h * accel
            prior = state.history
            old = np.zeros(3) if prior is None else np.asarray(prior.elastic_displacement_m)
            shear = np.zeros(3)
            sliding = False
            if mag:
                free = h * _in_plane(free_v[0] - free_v[1], n)
                shear, sliding = return_traction(
                    old,
                    free,
                    float(inv.sum()),
                    h,
                    mag,
                    prior,
                    self._k,
                    self._static,
                    self._layer.friction,
                )
            forces[0] += shear
            forces[1] -= shear
            total = forces.astype(current.position.dtype)
            if not np.isfinite(total).all() or np.any((total == 0) & (forces != 0)):
                raise SolverFailure("seam load outside particle precision")
            solver = self._solver if particle_solver is None else particle_solver
            result = solver.step(self._model, current, StepInputs(total), h)
            zero = (0.0, 0.0, 0.0)
            if not result.report.accepted:
                return MaterialSeamStep(state, result.report, backend, 0.0, zero, (), zero, None)
            final = result.state
            dx = final.position.astype(float) - x
            expected = free_v + h * inv[:, None] * np.stack((shear, -shear))
            eps = np.finfo(current.position.dtype).eps
            velocity = final.velocity.astype(float)
            vscale = (
                np.linalg.norm(v, axis=1)
                + h * np.linalg.norm(accel, axis=1)
                + h * inv * hypot(*shear)
            )
            xscale = (
                np.linalg.norm(x, axis=1)
                + np.linalg.norm(final.position.astype(float), axis=1)
                + h * vscale
            )
            if np.any(np.linalg.norm(velocity - expected, axis=1) > 48 * eps * vscale) or np.any(
                np.linalg.norm(dx - h * expected, axis=1) > 48 * eps * xscale
            ):
                raise SolverFailure(
                    "particle solver did not realize point and patch mass responses"
                )
            end_gap = self._patch.gap(*final.position)
            if end_gap <= 0:
                raise SolverFailure("seam point would cross the support plane")
            if not mag:
                return MaterialSeamStep(
                    self._state(final, state.tick + 1),
                    result.report,
                    backend,
                    0.0,
                    zero,
                    (),
                    zero,
                    prior,
                )
            trace = self._patch.trace(x[0], final.position[0], x[1], final.position[1])
            if prior is not None and (
                state.face != trace[0].face or state.weights != trace[0].entry_weights
            ):
                raise SolverFailure("previous material attachment differs from the active chart")
            if len(trace) > 1 and end_gap >= self._layer.width_m:
                raise SolverFailure(
                    "simultaneous unloading and material-edge transfer is unsupported"
                )
            du = _in_plane(dx[0] - dx[1], n)
            elastic = -shear / self._k if sliding else old + du
            plastic = old + du - elastic if sliding else np.zeros(3)
            work = float(shear @ du)
            energy = 0.5 * self._k * float(elastic @ elastic)
            loss = -float(shear @ plastic)
            implicit = 0.5 * self._k * float((elastic - old) @ (elastic - old))
            u0 = prior.stored_energy_j if prior else 0.0
            tol = (
                64
                * eps
                * (
                    (hypot(*shear) + self._k * hypot(*(elastic - old))) * sum(xscale)
                    + u0
                    + energy
                    + abs(work)
                    + abs(loss)
                    + implicit
                )
            )
            if loss < -tol or abs(work + energy - u0 + loss + implicit) > tol:
                raise SolverFailure("seam spring work, storage and plastic loss do not balance")
            attachment = (
                prior.key
                if prior
                else ("VT-chart", self._point, self._patch.material_body, trace[0].face)
            )
            history = MaterialStickHistory(
                attachment,
                prior.start_tick if prior else state.tick,
                (prior.active_steps if prior else 0) + 1,
                "slide" if sliding else "stick",
                _tuple(n),
                _tuple(elastic),
                energy,
                (prior.plastic_slip_m if prior else 0.0) + hypot(*plastic),
                (prior.tangent_work_j if prior else 0.0) + work,
            )
            full = normal + shear
            segs, torque = self._split(trace, x, dx, h, full, shear)
            if not np.isfinite(
                (history.tangent_work_j, history.plastic_slip_m, energy, *torque)
            ).all():
                raise SolverFailure("material seam history exceeds representable precision")
            accepted = self._state(
                final, state.tick + 1, history, trace[-1].face, trace[-1].exit_weights
            )
            return MaterialSeamStep(
                accepted, result.report, backend, mag, _tuple(full), segs, torque, None
            )
