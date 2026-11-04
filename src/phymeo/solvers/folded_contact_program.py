"""Whole-particle folded cloth contact with geometry-selected, resumable material episodes."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..backends.tyx_normal_layer import TyxNormalLayer
from ..constraints.cloth_material import ClothMaterial
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.result import StepReport
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_step
from ..geometry.folded_contact_sites import FoldedSiteSelector, FoldSiteKey, FoldSiteSelection
from ..geometry.folded_material_seam import _integrate, _jacobian
from ._material_stick_law import return_traction
from .collision_loads import NormalLayer, _host_law
from .compliant_cloth_contact import _elastic
from .deformable_sliding import _owned_readonly, _tuple
from .folded_material_seam import FoldedMaterialLeg
from .semi_implicit import SemiImplicitSolver


@dataclass(frozen=True)
class FoldSiteHistory:
    key: FoldSiteKey
    start_tick: int
    start_fraction: float
    active_steps: int
    regime: str
    elastic_m: float
    stored_energy_j: float
    plastic_slip_m: float
    tangent_work_j: float
    last_face: object
    last_weights: tuple


@dataclass(frozen=True, eq=False)
class FoldedContactState:
    """An owned global particle checkpoint and only its currently pressurized site episodes."""

    tick: int
    active: tuple[FoldSiteHistory, ...]
    _position: np.ndarray = field(repr=False)
    _velocity: np.ndarray = field(repr=False)
    _owner: object = field(repr=False)

    @property
    def particles(self):
        return ParticleState(self._position, self._velocity)


@dataclass(frozen=True)
class FoldSiteEvent:
    kind: str
    key: FoldSiteKey
    tick: int
    fraction: float
    bracket: tuple | None
    history: FoldSiteHistory


@dataclass(frozen=True, eq=False)
class FoldSiteContact:
    key: FoldSiteKey
    phase: str
    rows: tuple
    t0: float
    t1: float
    sample_fraction: float
    sample_gap_m: float
    normal_force_n: float
    shear_n: float
    normal_work_j: float
    tangent_work_j: float
    plastic_loss_j: float
    implicit_loss_j: float
    history: FoldSiteHistory
    legs: tuple[FoldedMaterialLeg, ...]


@dataclass(frozen=True, eq=False)
class FoldedContactStep:
    state: FoldedContactState
    report: StepReport
    selection: FoldSiteSelection
    normal_backend: str
    contacts: tuple[FoldSiteContact, ...]
    events: tuple[FoldSiteEvent, ...]
    nodal_contact_impulse_ns: np.ndarray | None
    structural_force_n: np.ndarray | None
    discrete_angular_impulse_nms: tuple
    integrated_angular_impulse_nms: tuple
    iterations: int
    force_residual_n: float

    @property
    def released_spring_energy_j(self):
        return sum((e.history.stored_energy_j for e in self.events if e.kind == "release"), 0.0)


@dataclass
class _Attempt:
    selection: FoldSiteSelection
    bands: list
    normal_rows: np.ndarray


class FoldedContactProgram:
    """Discover candidate particles/mesh hinges each step and jointly advance the full ParticleModel.

    Existing CompliantClothContactProgram distance/bending forces can enter through its
    ClothMaterial. All active station reactions contribute to the same nodal momentum update.
    The original entry pressure is used for uninterrupted windows. A truncated interval
    uses its interior midpoint pressure and applies traction only for its actual duration.
    """

    def __init__(
        self,
        model,
        geometry,
        bindings,
        *,
        layer,
        static_friction,
        tangent_stiffness_n_per_m,
        material=None,
        layer_operator=None,
        max_fold_angle_rad=np.pi / 12,
    ):
        if (
            not isinstance(model, ParticleModel)
            or not isinstance(layer, NormalLayer)
            or layer.damping_s_per_m
        ):
            raise ValueError(
                "folded contact program requires a particle model and an undamped NormalLayer"
            )
        if layer_operator is not None and not isinstance(layer_operator, TyxNormalLayer):
            raise TypeError("expected a TyxNormalLayer")
        self.selector = FoldedSiteSelector(
            geometry, bindings, width_m=layer.width_m, max_fold_angle_rad=max_fold_angle_rad
        )
        if self.selector.row_count > model.particle_count:
            raise ValueError("folded material rows exceed particle model")
        if material is not None:
            if not isinstance(material, ClothMaterial):
                raise TypeError("expected ClothMaterial")
            for kind in material.order:
                law = getattr(material, kind)
                if law is not None and (
                    np.any(law.indices >= model.particle_count) or np.any(law.compliance <= 0)
                ):
                    raise ValueError(
                        "cloth elasticity requires registered nodes and positive compliance"
                    )
        self._k = finite_real(tangent_stiffness_n_per_m, "tangent_stiffness_n_per_m")
        self._static = finite_real(static_friction, "static_friction")
        if self._k <= 0 or self._static < layer.friction:
            raise ValueError("invalid folded shear calibration")
        self._model, self._material, self._layer = model, material, layer
        self._operator, self._owner, self._solver = layer_operator, object(), SemiImplicitSolver()

    def _state(self, p, tick, active=()):
        return FoldedContactState(
            tick,
            tuple(active),
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
        if np.any(p.velocity[self._model.inverse_mass == 0]):
            raise ValueError("fixed cloth particles must be stationary")
        return self._state(p, 0)

    def _prepare(self, x, xf, h, capacity, detailed):
        selection = self.selector.select(x, xf, capacity=capacity, detailed=detailed)
        items = []
        data = []
        for site in selection.sites:
            idx = list(site.rows)
            local = x[idx].astype(float)
            delta = xf[idx].astype(float) - local
            for band in site.intervals:
                when = (
                    band.t0
                    if band.t0 == 0 and band.entry_bracket is None and band.exit_bracket is None
                    else (band.t0 + band.t1) / 2
                )
                sample_leg = next(leg for leg in band.legs if leg.t0 <= when <= leg.t1)
                q = local + when * delta
                sample = site.chart.station(q, sample_leg.side, spatial=False)
                n = np.asarray(sample.normal)
                rate = float(np.sum(_jacobian(q, sample_leg.side)[1] * delta)) / h
                law = self._layer
                data.append(
                    [
                        *n,
                        *(rate * n),
                        law.width_m,
                        sample.normal_gap_m,
                        law.stiffness_n_per_m,
                        1.0,
                        0.0,
                        0.0,
                        law.slip_speed_m_s,
                    ]
                )
                parts = []
                total = np.zeros((2, 5, 3))
                for leg in band.legs:
                    J, error = _integrate(local, delta, leg.side, leg.t0, leg.t1)
                    parts.append((leg, J, error))
                    total += J[0]
                items.append((site, band, when, sample, local, delta, parts, total))
        return _Attempt(selection, items, np.asarray(data, dtype=float).reshape(-1, 13))

    def _pressures(self, attempt, operator):
        if not len(attempt.normal_rows):
            return np.zeros(0)
        expected = _host_law(attempt.normal_rows)
        values = (
            expected
            if operator is None
            else np.asarray(operator.evaluate(attempt.normal_rows), dtype=float)
        )
        if (
            values.shape != expected.shape
            or not np.isfinite(values).all()
            or np.any(values[:, 1] <= 0)
            or not np.allclose(values[:, 1:], expected[:, 1:], rtol=3e-5, atol=2e-6)
        ):
            raise SolverFailure("folded site sample is unresolved at the selected normal precision")
        return values[:, 1]

    def _loads(self, attempt, pressures, state, h, *, publish):
        old = {p.key: p for p in state.active}
        active = {}
        events = []
        records = []
        nodal = np.zeros(state._position.shape, dtype=float)
        torque = np.zeros(3)
        for P, item in zip(pressures, attempt.bands, strict=True):
            site, band, when, sample, local, dx, parts, J = item
            prior = old.pop(site.key, None)
            if prior is not None and (band.t0 != 0 or band.entry_bracket is not None):
                raise SolverFailure("existing material history has no covered entry interval")
            d_t, d_n = np.einsum("kir,ir->k", J, dx)
            old_e = prior.elastic_m if prior is not None else 0.0
            ft, sliding = return_traction(
                np.array([old_e]),
                np.array([d_t]),
                0.0,
                h,
                float(P),
                prior,
                self._k,
                self._static,
                self._layer.friction,
            )
            shear = float(ft[0])
            traction = np.array([shear, P])
            force = np.tensordot(traction, J, axes=1)
            nodal[list(site.rows)] += force
            if not publish:
                continue
            start = band.start
            end = band.end
            assert start is not None and end is not None
            if prior is None:
                initial = FoldSiteHistory(
                    site.key,
                    state.tick,
                    band.t0,
                    0,
                    "stick",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    start.face,
                    start.material_weights,
                )
                events.append(
                    FoldSiteEvent(
                        "enter", site.key, state.tick, band.t0, band.entry_bracket, initial
                    )
                )
            else:
                if (
                    prior.last_face != start.face
                    or max(abs(dict(prior.last_weights)[i] - v) for i, v in start.material_weights)
                    > 3e-7
                ):
                    raise SolverFailure(
                        "folded material attachment did not meet the previous accepted state"
                    )
                initial = prior
            elastic = -shear / self._k if sliding else old_e + d_t
            plastic = old_e + d_t - elastic if sliding else 0.0
            U = 0.5 * self._k * elastic**2
            W = float(shear * d_t)
            loss = -float(shear * plastic)
            implicit = 0.5 * self._k * (elastic - old_e) ** 2
            tol = (
                128
                * np.finfo(float).eps
                * max(1.0, abs(W), U, initial.stored_energy_j, loss, implicit)
            )
            if (
                not np.isfinite((U, W, loss, implicit)).all()
                or loss < -tol
                or abs(W + U - initial.stored_energy_j + loss + implicit) > tol
            ):
                raise SolverFailure("folded contact episode work and energy did not close")
            history = FoldSiteHistory(
                site.key,
                initial.start_tick,
                initial.start_fraction,
                initial.active_steps + 1,
                "slide" if sliding else "stick",
                float(elastic),
                float(U),
                initial.plastic_slip_m + abs(plastic),
                initial.tangent_work_j + W,
                end.face,
                end.material_weights,
            )
            legs = []
            for leg, moments, _ in parts:
                impulse = h * np.tensordot(traction, moments[0], axes=1)
                first = h * np.tensordot(traction, moments[1], axes=1)
                moment = (np.cross(local, impulse) + np.cross(dx, first)).sum(axis=0)
                torque += moment
                work = traction * np.einsum("kir,ir->k", moments[0], dx)
                # Positive gap and compression over the full leg were checked by the selector.
                legs.append(
                    FoldedMaterialLeg(
                        leg.start.face,
                        leg.t0,
                        leg.t1,
                        leg.start,
                        leg.end,
                        (0.0, self._layer.width_m),
                        _owned_readonly(impulse),
                        _owned_readonly(np.sum(impulse * dx, axis=1) / h),
                        float(work[1]),
                        float(work[0]),
                        _tuple(moment),
                    )
                )
            records.append(
                FoldSiteContact(
                    site.key,
                    "continued" if prior is not None else "entered",
                    site.rows,
                    band.t0,
                    band.t1,
                    when,
                    sample.normal_gap_m,
                    float(P),
                    shear,
                    float(P * d_n),
                    W,
                    max(0.0, loss),
                    float(implicit),
                    history,
                    tuple(legs),
                )
            )
            if band.t1 != 1 or band.exit_bracket is not None:
                events.append(
                    FoldSiteEvent(
                        "release", site.key, state.tick, band.t1, band.exit_bracket, history
                    )
                )
            else:
                active[site.key] = history
        if old:
            raise SolverFailure("a previously loaded site disappeared without a witnessed release")
        events.sort(key=lambda e: (e.fraction, e.kind != "release", e.key))
        return (
            nodal,
            tuple(records),
            tuple(events),
            tuple(active[k] for k in sorted(active)),
            torque,
        )

    def step(self, state, inputs, dt, *, capacity=4096, particle_solver=None, max_iterations=48):
        if not isinstance(state, FoldedContactState) or state._owner is not self._owner:
            raise ValueError("folded contact checkpoint belongs to another program")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, Integral)
            or not 1 <= max_iterations <= 128
        ):
            raise ValueError("max_iterations must be between 1 and 128")
        initial, external, hd = validate_step(self._model, state.particles, inputs, dt)
        x, v = initial.position, initial.velocity
        h = float(hd)
        eps = float(np.finfo(x.dtype).eps)
        structural = (
            np.zeros_like(x, dtype=float)
            if self._material is None
            else _elastic(self._material, x.astype(float), v.astype(float), 0.0)
        )
        base = external.astype(float) + structural
        inv = self._model.inverse_mass[:, None]
        dynamic = self._model.inverse_mass > 0

        def predict(force):
            r = base + force
            packed = r.astype(x.dtype)
            if not np.isfinite(packed).all() or np.any((packed == 0) & (r != 0)):
                raise SolverFailure("folded global load exceeds particle precision")
            u = np.zeros_like(v)
            u[dynamic] = v[dynamic] + hd * (self._model.gravity + inv[dynamic] * packed[dynamic])
            out = x + hd * u
            if not np.isfinite(out).all():
                raise SolverFailure("nonfinite folded particle predictor")
            return packed, u, out

        _, _, guess = predict(np.zeros_like(base))
        for it in range(1, max_iterations + 1):
            plan = self._prepare(x, guess, h, capacity, False)
            P = self._pressures(plan, None)
            f = self._loads(plan, P, state, h, publish=False)[0]
            _, _, candidate = predict(f)
            if np.max(abs(candidate.astype(float) - guess), initial=0.0) <= max(2e-13, eps * 0.5):
                break
            guess = candidate
        else:
            raise SolverFailure("global folded contact selection and forces did not converge")
        P = self._pressures(plan, self._operator)
        force = self._loads(plan, P, state, h, publish=False)[0]
        packed, velocity, predicted = predict(force)
        solver = self._solver if particle_solver is None else particle_solver
        result = solver.step(self._model, initial, StepInputs(packed), h)
        backend = "host" if self._operator is None else self._operator.mode
        if not result.report.accepted:
            return FoldedContactStep(
                state,
                result.report,
                plan.selection,
                backend,
                (),
                (),
                None,
                None,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                it,
                0.0,
            )
        x1 = result.state.position
        v1 = result.state.velocity
        if (
            not np.isfinite(x1).all()
            or not np.isfinite(v1).all()
            or np.any(v1[~dynamic])
            or not np.array_equal(x[~dynamic], x1[~dynamic])
        ):
            raise SolverFailure("invalid global folded particle result")
        vscale = np.linalg.norm(v.astype(float), axis=1) + h * (
            np.linalg.norm(self._model.gravity)
            + self._model.inverse_mass
            * (np.linalg.norm(base, axis=1) + np.linalg.norm(force, axis=1))
        )
        if np.any(
            np.linalg.norm(v1.astype(float) - velocity, axis=1) > 64 * eps * vscale
        ) or np.any(
            np.linalg.norm(x1.astype(float) - predicted, axis=1)
            > 64 * eps * (np.linalg.norm(x, axis=1) + h * vscale)
        ):
            raise SolverFailure("particle solver did not realize the global folded response")
        actual = self._prepare(x, x1, h, capacity, True)

        def layout(a):
            return tuple(
                (
                    s.key,
                    tuple(leg.side for leg in b.legs),
                    b.entry_bracket is not None,
                    b.exit_bracket is not None,
                )
                for s, b, *_ in a.bands
            )

        if layout(plan) != layout(actual):
            raise SolverFailure("folded active intervals differ on the accepted path")
        geometric_pressure = self._pressures(actual, None)
        if not np.allclose(
            geometric_pressure, P, rtol=max(3e-9, 256 * eps), atol=max(1e-9, 64 * eps)
        ):
            raise SolverFailure("finite material pressure differs on the accepted path")
        reconstructed, records, events, current, moment = self._loads(
            actual, P, state, h, publish=True
        )
        residual = float(np.max(abs(reconstructed - force), initial=0.0))
        if residual > max(2e-9, 128 * eps) * max(1.0, np.max(abs(force), initial=0.0)):
            raise SolverFailure("folded global traction differs from the accepted geometry")
        return FoldedContactStep(
            self._state(result.state, state.tick + 1, current),
            result.report,
            actual.selection,
            backend,
            records,
            events,
            _owned_readonly(h * force),
            _owned_readonly(structural),
            _tuple(h * np.cross(x.astype(float), force).sum(axis=0)),
            _tuple(moment),
            it,
            residual,
        )
