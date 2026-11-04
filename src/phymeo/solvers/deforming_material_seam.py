"""Implicit material address/traction closure for two independently deforming coplanar triangles."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.cloth_material import ClothMaterial
from ..constraints.collision_shapes import CollisionKey
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.result import StepReport
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_step
from ..geometry.deforming_material_seam import DeformingSeamChart
from ._material_stick_law import return_traction
from .collision_loads import NormalLayer, _host_law
from .compliant_cloth_contact import _elastic
from .deformable_sliding import _owned_readonly, _tuple
from .deformable_stiction import MaterialStickHistory, _in_plane
from .material_seam import MaterialFaceLoad
from .semi_implicit import SemiImplicitSolver


@dataclass(frozen=True)
class DeformingFaceLoad(MaterialFaceLoad):
    integrated_weights: tuple


@dataclass(frozen=True, eq=False)
class DeformingSeamState:
    """Owned free-particle rows in the caller's ParticleModel order plus one material history."""

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


@dataclass(frozen=True, eq=False)
class DeformingSeamStep:
    state: DeformingSeamState
    report: StepReport
    normal_backend: str
    normal_force_n: float
    force_on_vertex_n: tuple
    segments: tuple
    event_bracket: tuple | None
    site_iterations: int
    weight_residual_l1: float
    quadrature_error_l1: float
    weight_tolerance: float
    nodal_contact_impulse_ns: np.ndarray | None
    structural_force_n: np.ndarray | None
    structural_energy_j: tuple | None
    plastic_loss_j: float
    implicit_loss_j: float
    iteration_residuals_l1: tuple = ()


class DeformingMaterialSeamStepper:
    """One positive-mass vertex and separate positive-mass rows for every support node.

    Physical states are never projected onto the reference plane. The local iteration uses
    trial in-plane addresses; accepted end coordinates must independently pass the planar,
    spin-free and material incidence checks. One actual particle solve uses the converged
    nodal forces. Inadequate geometric or mass closure publishes no new state.
    """

    def __init__(
        self,
        model,
        faces,
        *,
        material_rows,
        point_row,
        point_material,
        normal,
        layer,
        static_friction,
        tangent_stiffness_n_per_m,
        material=None,
        layer_operator=None,
    ):
        chart = DeformingSeamChart(faces, material_rows, point_row, normal)
        if (
            not isinstance(model, ParticleModel)
            or set(chart.rows) != set(range(model.particle_count))
            or np.any(model.inverse_mass <= 0)
        ):
            raise ValueError("the vertex and all material nodes need independent positive masses")
        p = tuple(point_material)
        if (
            len(p) != 3
            or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in p)
            or p[0] != chart.body[0]
            or (p[:2] == chart.body and p[2] in chart.ids)
        ):
            raise ValueError("expected a distinct material vertex in the same world")
        if not isinstance(layer, NormalLayer) or (
            layer_operator is not None and not isinstance(layer_operator, TyxNormalLayer)
        ):
            raise TypeError("expected a normal layer and an optional TYX normal operator")
        if material is not None:
            if not isinstance(material, ClothMaterial) or material.bending is not None:
                raise ValueError(
                    "this planar material supports finite-compliance distance elasticity"
                )
            d = material.distance
            if d is not None and (
                np.any(d.compliance <= 0) or not set(d.indices.ravel()).issubset(chart.rows[1:])
            ):
                raise ValueError(
                    "elastic links must address support nodes and have positive compliance"
                )
        self._model, self._chart, self._point = model, chart, tuple(map(int, p))
        self._layer, self._op, self._material = layer, layer_operator, material
        self._static = finite_real(static_friction, "static_friction")
        self._k = finite_real(tangent_stiffness_n_per_m, "tangent_stiffness_n_per_m")
        if self._k <= 0 or self._static < layer.friction:
            raise ValueError("invalid static/kinetic or tangent spring calibration")
        self._owner = object()
        self._solver = SemiImplicitSolver()

    def _state(self, p, tick, history=None, face=None, weights=()):
        return DeformingSeamState(
            tick,
            history,
            face,
            weights,
            _owned_readonly(p.position),
            _owned_readonly(p.velocity),
            self._owner,
        )

    def initial_state(self, particles):
        p, _, _ = validate_step(
            self._model,
            particles,
            StepInputs.zeros(self._model.particle_count, dtype=self._model.inverse_mass.dtype),
            1.0,
        )
        x = p.position[list(self._chart.rows)].astype(float)
        self._chart.validate_plane(x, x, np.finfo(p.position.dtype).eps)
        return self._state(p, 0)

    def _normal(self, gap, relative):
        cfg, n = self._layer, self._chart.normal
        if gap <= 0:
            raise SolverFailure("material vertex on or behind its reference plane")
        if gap >= cfg.width_m:
            return 0.0, np.zeros(3)
        data = np.array(
            [
                [
                    *n,
                    *relative,
                    cfg.width_m,
                    gap,
                    cfg.stiffness_n_per_m,
                    1.0,
                    cfg.damping_s_per_m,
                    0.0,
                    cfg.slip_speed_m_s,
                ]
            ]
        )
        output = _host_law(data) if self._op is None else self._op.evaluate(data)
        value = np.asarray(output, dtype=float)
        if (
            value.shape != (1, 5)
            or not np.isfinite(value).all()
            or value[0, 1] < 0
            or not np.allclose(value[0, 2:], value[0, 1] * n, rtol=2e-6, atol=1e-36)
        ):
            raise SolverFailure("invalid force on the deforming material station")
        if value[0, 1] == 0 and 1 - cfg.damping_s_per_m * float(relative @ n) > 0:
            raise SolverFailure("positive normal load below selected precision")
        return float(value[0, 1]), value[0, 2:]

    def _spring_energy(self, positions):
        d = None if self._material is None else self._material.distance
        if d is None:
            return 0.0
        i, j = d.indices.T
        extension = (
            np.linalg.norm(positions[j].astype(float) - positions[i], axis=1) - d.rest_length
        )
        return 0.5 * float(np.sum(extension**2 / d.compliance))

    def _segments(self, trace, x, dx, h, f, shear):
        result = []
        for part in trace.spans:
            a, b = part.t0, part.t1
            loc = list(part.local_rows)
            impulse = []
            works = []
            for node, row, w in zip(part.node_ids, loc, part.integral, strict=True):
                impulse.append((node, _tuple(-h * w * f)))
                works.append((node, -w * float(f @ dx[row])))
            at0 = np.asarray(part.entry) @ (x[loc] + a * dx[loc])
            at1 = np.asarray(part.exit) @ (x[loc] + b * dx[loc])
            material_delta = (b - a) * dx[0] - np.asarray(part.integral) @ dx[loc]
            result.append(
                DeformingFaceLoad(
                    part.face,
                    a,
                    b,
                    tuple(zip(part.node_ids, part.entry, strict=True)),
                    tuple(zip(part.node_ids, part.exit, strict=True)),
                    _tuple(x[0] + a * dx[0]),
                    _tuple(x[0] + b * dx[0]),
                    _tuple(at0),
                    _tuple(at1),
                    _tuple(h * (b - a) * f),
                    tuple(impulse),
                    (b - a) * float(f @ dx[0]),
                    tuple(works),
                    float(shear @ material_delta),
                    tuple(zip(part.node_ids, part.integral, strict=True)),
                )
            )
        return tuple(result)

    def step(self, state, inputs, dt, *, particle_solver=None, max_iterations=32):
        if not isinstance(state, DeformingSeamState) or state._owner is not self._owner:
            raise ValueError("deforming seam state belongs to another stepper")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, Integral)
            or not 1 <= max_iterations <= 128
        ):
            raise ValueError("max_iterations must be between 1 and 128")
        p, external, hd = validate_step(self._model, state.particles, inputs, dt)
        h = float(hd)
        dtype = p.position.dtype
        eps = float(np.finfo(dtype).eps)
        tolerance = max(1e-10, 64 * eps)
        rows = list(self._chart.rows)
        x = p.position[rows].astype(float)
        v = p.velocity[rows].astype(float)
        n = self._chart.normal
        self._chart.validate_plane(x, x, eps)
        gap = float((x[0] - x[1]) @ n)
        if gap <= 0:
            raise SolverFailure("initial point is on or behind the deforming plane")
        structural = (
            np.zeros_like(p.position, dtype=float)
            if self._material is None
            else _elastic(self._material, p.position.astype(float), p.velocity.astype(float), 0.0)
        )
        base = (external.astype(float) + structural)[rows]
        inv = self._model.inverse_mass[rows].astype(float)
        gravity = self._model.gravity.astype(float)
        start = self._chart.trace(x, x) if gap < self._layer.width_m else None
        w = np.zeros(len(rows) - 1) if start is None else start.weights
        if state.history is not None and start is not None:
            entry = start.spans[0]
            if (
                state.face != entry.face
                or max(
                    abs(dict(state.weights)[i] - a)
                    for i, a in zip(entry.node_ids, entry.entry, strict=True)
                )
                > tolerance
            ):
                raise SolverFailure("stale deforming material attachment")
        mag, normal = self._normal(gap, v[0] - w @ v[1:])
        prior = state.history
        old = np.zeros(3) if prior is None else np.asarray(prior.elastic_displacement_m)
        shear = np.zeros(3)
        sliding = False
        residuals = []
        for iteration in range(1, max_iterations + 1):
            c = np.r_[1.0, -w]
            free = v + h * (gravity + inv[:, None] * (base + c[:, None] * normal))
            if mag:
                shear, sliding = return_traction(
                    old,
                    h * _in_plane(c @ free, n),
                    float((c * c) @ inv),
                    h,
                    mag,
                    prior,
                    self._k,
                    self._static,
                    self._layer.friction,
                )
            force = normal + shear
            total = base + c[:, None] * force
            packed = total.astype(dtype)
            if not np.isfinite(packed).all() or np.any((packed == 0) & (total != 0)):
                raise SolverFailure("deforming force outside selected particle precision")
            vp = p.velocity[rows] + hd * (
                self._model.gravity + self._model.inverse_mass[rows, None] * packed
            )
            xp = p.position[rows] + hd * vp
            if not np.isfinite(xp).all():
                raise SolverFailure("nonfinite deforming material trial")
            if not mag:
                break
            trial = self._chart.trace(x, xp.astype(float))
            residuals.append(float(np.linalg.norm(trial.weights - w, ord=1)))
            if residuals[-1] <= max(4e-14, 0.5 * eps):
                break
            w = trial.weights
        else:
            raise SolverFailure("deforming material address iteration did not converge")
        supplied = np.empty_like(external)
        supplied[rows] = packed
        solver = self._solver if particle_solver is None else particle_solver
        physical = solver.step(self._model, p, StepInputs(supplied), h)
        backend = "host" if self._op is None else self._op.mode
        if not physical.report.accepted:
            return DeformingSeamStep(
                state,
                physical.report,
                backend,
                0.0,
                (0.0, 0.0, 0.0),
                (),
                None,
                iteration,
                0.0,
                0.0,
                tolerance,
                None,
                None,
                None,
                0.0,
                0.0,
            )
        final = physical.state
        xf = final.position[rows].astype(float)
        vf = final.velocity[rows].astype(float)
        if not np.isfinite(xf).all() or not np.isfinite(vf).all():
            raise SolverFailure("nonfinite independent particle candidate")
        dx = xf - x
        vscale = np.linalg.norm(v, axis=1) + h * (
            np.linalg.norm(gravity)
            + inv * (np.linalg.norm(base, axis=1) + abs(c) * np.linalg.norm(force))
        )
        xscale = np.linalg.norm(x, axis=1) + np.linalg.norm(xf, axis=1) + h * vscale
        if np.any(np.linalg.norm(vf - vp.astype(float), axis=1) > 64 * eps * vscale) or np.any(
            np.linalg.norm(dx - h * vf, axis=1) > 64 * eps * xscale
        ):
            raise SolverFailure("particle solver did not realize independent nodal inertia")
        self._chart.validate_plane(x, xf, eps)
        end_gap = float((xf[0] - xf[1]) @ n)
        if end_gap <= 0:
            raise SolverFailure("independent point crosses the deforming material plane")
        contact = np.zeros_like(x)
        contact[rows] = h * c[:, None] * force
        structure = _owned_readonly(structural)
        energies = (self._spring_energy(p.position), self._spring_energy(final.position))
        if not np.isfinite(energies).all():
            raise SolverFailure("structural energy exceeds precision")
        if not mag:
            return DeformingSeamStep(
                self._state(final, state.tick + 1),
                physical.report,
                backend,
                0.0,
                (0.0, 0.0, 0.0),
                (),
                None,
                iteration,
                0.0,
                0.0,
                tolerance,
                _owned_readonly(contact),
                structure,
                energies,
                0.0,
                0.0,
            )
        trace = self._chart.trace(x, xf)
        residual = float(np.linalg.norm(trace.weights - w, ord=1))
        if residual + trace.error > tolerance:
            raise SolverFailure("accepted material addresses exceed impulse closure tolerance")
        if trace.bracket is not None and end_gap >= self._layer.width_m:
            raise SolverFailure("simultaneous normal unloading and material-edge transfer")
        material_delta = _in_plane(c @ dx, n)
        elastic = -shear / self._k if sliding else old + material_delta
        plastic = old + material_delta - elastic if sliding else np.zeros(3)
        energy = 0.5 * self._k * float(elastic @ elastic)
        loss = -float(shear @ plastic)
        implicit = 0.5 * self._k * float((elastic - old) @ (elastic - old))
        work = float(shear @ material_delta)
        u0 = prior.stored_energy_j if prior else 0.0
        etol = (
            64
            * eps
            * (
                (np.linalg.norm(shear) + self._k * np.linalg.norm(elastic - old)) * sum(xscale)
                + abs(work)
                + energy
                + u0
                + abs(loss)
                + implicit
            )
        )
        if not np.isfinite((work, energy, u0, loss, implicit, etol)).all():
            raise SolverFailure("unrepresentable deforming material ledger")
        if loss < -etol or abs(work + energy - u0 + loss + implicit) > etol:
            raise SolverFailure("deforming spring work and stored energy fail to close")
        key = (
            prior.key
            if prior
            else ("deforming-seam", self._point, self._chart.body, trace.spans[0].face)
        )
        history = MaterialStickHistory(
            key,
            prior.start_tick if prior else state.tick,
            (prior.active_steps if prior else 0) + 1,
            "slide" if sliding else "stick",
            _tuple(n),
            _tuple(elastic),
            energy,
            (prior.plastic_slip_m if prior else 0.0) + np.linalg.norm(plastic),
            (prior.tangent_work_j if prior else 0.0) + work,
        )
        legs = self._segments(trace, x, dx, h, force, shear)
        closed = self._state(final, state.tick + 1, history, legs[-1].face, legs[-1].exit_weights)
        return DeformingSeamStep(
            closed,
            physical.report,
            backend,
            mag,
            _tuple(force),
            legs,
            trace.bracket,
            iteration,
            residual,
            trace.error,
            tolerance,
            _owned_readonly(contact),
            structure,
            energies,
            max(0.0, loss),
            implicit,
            tuple(residuals),
        )
