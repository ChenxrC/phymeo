"""Synchronous held/terminal world iterations, certification, and owned history."""

from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import SceneState, StageMetrics, fingerprint
from ..dynamics.robot_dynamics import mass_matrix
from ..geometry._prism_exact import Quadratic as Q
from ..geometry._prism_exact import cross, dot, scale, sub, vector
from ..geometry.prism_patch_sweep import candidate_fingerprint, triangle_prism_patch_sweep
from ..runtime.program import own_diagnostics
from ..validation._material_path_math import q
from ..validation.material_path import MaterialPathLedger
from ..validation.terminal_impulse import _PositiveMass
from ..validation.thick_contact_mapping import (
    paired_moment_residual,
    phase_audit,
    phase_capacity,
    row_scales,
)
from ._dynamic_friction_law import projected
from ._dynamic_normal_geometry import normal_projection_bounds
from ._material_contact_balance import MaterialContactBalance
from ._robot_replay_ledger import FrozenRobotLedger, upper_float
from ._thick_contact_geometry import discover, held_feature_gap, manifolds, maps, relative, response
from ._thick_contact_replay import ThickReplay, require_selected_pre

ZERO = F(0)


class ThickContactWorkspace(ThickReplay):
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.inputs, self.h = program, state, inputs, h
        self.settings, self.inner, self.context = settings, inner, context
        self.domain, self.model = program.robot_domain, program.model
        self.models = {d.name: d.model for d in program.scene.domains}
        self.node_domains = tuple(program.materials)
        self.exact_gap_scale = min(q(settings.reference_length), q(h) * q(settings.reference_speed))
        self.bound = q(settings.tolerance) * self.exact_gap_scale
        if any(q(s.half_thickness_m) <= self.bound for s in program.surfaces):
            raise SolverFailure("half thickness must exceed penetration budget")
        self.anchors = ()
        self.initial_manifolds = None
        self.initial_manifolds, added = discover(self, state)
        self.anchors = tuple(sorted(added, key=lambda a: a.token))
        if any(m.exact_gap_bounds[0] < -self.bound for m in self.initial_manifolds):
            raise SolverFailure("initial whole-prism penetration exceeds budget")
        fields = program._own_fields
        self.old_active = state.history[fields[0].name].copy()
        active = state.history[fields[1].name]
        mat = state.history[fields[2].name]
        phases = state.history[fields[3].name]
        if (
            not np.isfinite(mat).all()
            or not np.isfinite(phases).all()
            or np.any(mat[~active] != 0)
            or np.any(phases[~active] != 0)
            or np.any(active[~self.old_active])
        ):
            raise SolverFailure("invalid or dirty thick history")
        self.ledger = FrozenRobotLedger(
            self.model, state.domains[self.domain], inputs.domains[self.domain], h, settings, 1.0
        )
        self.balance = MaterialContactBalance(
            self.models, program.materials, state, inputs, self.domain, h, program.budgets
        )
        self.node_ledgers = {
            name: MaterialPathLedger(
                self.models[name],
                program.materials[name],
                state.domains[name],
                inputs.domains[name],
                h,
                inner,
                context,
            )
            for name in self.node_domains
        }
        self.base_state = SceneState(
            state.domains, {f.name: state.history[f.name] for f in program.base.history_fields}
        )
        self.channels = {
            self.domain: ("effort",),
            **{name: ("force",) for name in self.node_domains},
        }
        self.held_replays = self.terminal_replays = self.line_search_replays = self.rebuilds = (
            self.ps
        ) = self.vs = 0
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.trial = self.held_trial = None
        self.ready = self.pdone = self.vdone = False
        self.native_audit = self.material_reports = self.physical_report = self.mapping_report = {}
        self.ccd = ()
        self._initialize_anchors()

    def _initialize_anchors(self):
        self.j0, self.points0, self.local0, self.rj0, samples, self.mapping0 = maps(
            self, self.initial
        )
        self.robot_points0 = np.array([s.world_point for s in samples]).reshape(-1, 3)
        self.indices = np.array(
            [self.program.pairs[a.pair][1].link_index for a in self.anchors], int
        )
        self.initial_closed = np.array(
            [s.exact_distance_bounds[1] <= self.bound for s in samples], bool
        )
        self.n0 = np.array(
            [
                self.initial_manifolds[a.pair].normal if closed else s.normal
                for a, closed, s in zip(self.anchors, self.initial_closed, samples, strict=True)
            ],
            float,
        ).reshape(-1, 3)
        if not np.isfinite(self.n0).all():
            raise SolverFailure("unknown legal initial anchor normal")
        self.mu = np.array(
            [
                self.program.friction_by_collider[self.program.pairs[a.pair][1].key]
                for a in self.anchors
            ]
        )
        self.w0 = response(self, self.j0, self.rj0, self.ledger.inverse)
        self.k0, self.kappa0 = self._scales(self.w0, self.n0)
        self.alpha0 = self._alpha(self.w0)
        self.p = np.zeros(len(self.anchors))
        self.t = np.zeros((len(self.anchors), 3))
        self.eta = self.p.copy()
        self.tau = self.t.copy()

    def _scales(self, w, n):
        values = [row_scales(w[i, i], normal) for i, normal in enumerate(n)]
        return (np.array([v[0] for v in values]), np.array([v[1] for v in values]))

    def _alpha(self, w):
        if not len(w):
            return 0.0
        lipschitz = float(np.max(np.sum(np.sqrt(np.sum(w * w, axis=(2, 3))), axis=1)))
        if not np.isfinite(lipschitz) or lipschitz <= 0:
            raise SolverFailure("invalid full-world synchronous response")
        return self.program.relaxation / lipschitz

    def predict(self):
        self._evaluate()
        _, added = discover(self, self.trial.result.state)
        if added:
            self.anchors = tuple(sorted((*self.anchors, *added), key=lambda a: a.token))
            self._initialize_anchors()
            self._evaluate()

    def _current(self, state):
        current = maps(self, state)
        blocks, points, local, rjac, samples, _ = current
        full = manifolds(self, state)
        gaps = [
            held_feature_gap(self, a, p, full[a.pair], sample.exact_distance_bounds)
            for a, p, sample in zip(self.anchors, points, samples, strict=True)
        ]
        return current, tuple(gaps)

    def _friction(self, terminal=False):
        state = (
            self.trial.result.state if terminal or not self.ready else self.held_trial.result.state
        )
        blocks, jac = (
            (self.j1, self.rj1) if terminal else (maps(self, state)[0], maps(self, state)[3])
        )
        rel = relative(self, state, blocks, jac)
        normals = self.n1 if terminal else self.n0
        worlds = self._world(self.eta, True) if terminal else self._world(self.p)
        kappas = self.kappa1 if terminal else self.kappa0
        residual = dict(cone=0.0, dissipation=0.0, positive_work=0.0)
        for i, (n, s, u) in enumerate(zip(normals, worlds, rel, strict=True)):
            if not terminal and not self.initial_closed[i] and np.any(self.t[i] != 0):
                raise SolverFailure("initially open held tangent is nonzero")
            cap = phase_capacity(
                self.n0[i],
                self._world(self.p)[i],
                self.n1[i] if self.ready else self.n0[i],
                self._world(self.eta, True)[i] if self.ready else np.zeros(3),
                self.mu[i],
                self.initial_closed[i],
                terminal,
            )
            values, _, _ = phase_audit(n, s, u, cap, kappas[i], self.settings.reference_speed)
            for key, v in values.items():
                residual[key] = max(residual[key], v)
        return residual, np.array(rel, float)

    def _position_metrics(self):
        state = self.held_trial.result.state if self.ready else self.trial.result.state
        current, gaps = self._current(state)
        full = manifolds(self, state)
        penetration = max((max(-m.exact_gap_bounds[0], ZERO) for m in full), default=ZERO)
        residual = dict(
            native=self.trial.result.report.max_residual,
            final_penetration=upper_float(penetration / self.exact_gap_scale),
            held_normal=0.0,
        )
        residual.update({key: upper_float(value) for key, value in self.native_audit.items()})
        for k, n, s, g in zip(self.k0, self.n0, self._world(self.p), gaps, strict=True):
            projection = normal_projection_bounds(n, s)
            if projection[0] < 0:
                raise SolverFailure("negative actual held normal")
            value = max(abs(min(q(k) * p, delta / q(self.h))) for p in projection for delta in g)
            residual["held_normal"] = max(
                residual["held_normal"], upper_float(value / q(self.settings.reference_speed))
            )
        friction, _ = self._friction()
        residual.update({"held_" + key: v for key, v in friction.items()})
        return StageMetrics(residual, penetration=upper_float(penetration)), gaps, full

    def position_metrics(self):
        metrics, _, _ = self._position_metrics()
        self.pdone = metrics.max_residual <= self.settings.tolerance and (
            metrics.max_residual <= self.settings.tolerance / 8
            or self.ps >= self.settings.position_iterations
        )
        return (
            metrics
            if self.pdone
            else StageMetrics(
                {**metrics.residuals, "refinement_pending": 2 * self.settings.tolerance},
                penetration=metrics.penetration,
            )
        )

    def _improve(self, terminal=False):
        oldp = (self.eta if terminal else self.p).copy()
        oldt = (self.tau if terminal else self.t).copy()
        if terminal:
            rel = np.array(relative(self, self.trial.result.state, self.j1, self.rj1), float)
            normal = np.einsum("ij,ij->i", self.n1, rel)
            norms = self.n1
            alpha = self.alpha1
            oldscore = self._velocity_metrics().max_residual
        else:
            current, gaps = self._current(self.trial.result.state)
            rel = np.array(relative(self, self.trial.result.state, current[0], current[3]), float)
            normal = np.array([float((a + b) / 2) / self.h for a, b in gaps])
            norms = self.n0
            alpha = self.alpha0
            oldscore = self._position_metrics()[0].max_residual
        if oldscore <= self.settings.tolerance / 8:
            return
        saved = (self.trial, self.native_audit, self.material_reports, self.mapping_report)
        for power in range(13):
            a = alpha / (2**power)
            newp = np.maximum(0.0, oldp - a * normal)
            newt = np.zeros_like(oldt)
            for i, n in enumerate(norms):
                if terminal and not self.active[i]:
                    newp[i] = 0.0
                    continue
                if not terminal and not self.initial_closed[i]:
                    continue
                cap = self.mu[i] * (
                    newp[i] + (self.p[i] if terminal and not self.initial_closed[i] else 0.0)
                )
                newt[i] = projected(oldt[i] - a * rel[i], n, cap)
            if terminal:
                self.eta, self.tau = newp, newt
            else:
                self.p, self.t = newp, newt
            self._evaluate(terminal)
            if power:
                self.line_search_replays += 1
            if not terminal:
                _, added = discover(self, self.trial.result.state, penetrating_only=True)
                if added:
                    self._rebuild(added)
                    return
            score = (
                self._velocity_metrics().max_residual
                if terminal
                else self._position_metrics()[0].max_residual
            )
            if score < oldscore or score <= self.settings.tolerance:
                return
        if terminal:
            self.eta, self.tau = oldp, oldt
        else:
            self.p, self.t = oldp, oldt
        self.trial, self.native_audit, self.material_reports, self.mapping_report = saved

    def _rebuild(self, added):
        self.rebuilds += 1
        if self.rebuilds > self.program.max_manifold_rebuilds:
            raise SolverFailure("manifold rebuild budget exhausted")
        self.anchors = tuple(sorted((*self.anchors, *added), key=lambda a: a.token))
        self._initialize_anchors()
        self._evaluate()

    def position_sweep(self):
        self.ps += 1
        self._improve()
        _, added = discover(self, self.trial.result.state, penetrating_only=True)
        if added:
            self._rebuild(added)

    def prepare_velocity(self):
        if not self.pdone:
            raise SolverFailure("held stage not converged")
        self.held_trial = self.trial
        full, added = discover(self, self.held_trial.result.state, True)
        if added:
            old = {a.token: (self.p[i], self.t[i].copy()) for i, a in enumerate(self.anchors)}
            self.anchors = tuple(sorted((*self.anchors, *added), key=lambda a: a.token))
            self._initialize_anchors()
            for i, a in enumerate(self.anchors):
                if a.token in old:
                    self.p[i], self.t[i] = old[a.token]
            self._evaluate()
            require_selected_pre(
                self.trial.result.report.substeps[0].diagnostics["terminal_impulse"],
                self.held_trial.result.report.substeps[0].diagnostics["terminal_impulse"],
                self.trial.effective_inputs,
                self.held_trial.effective_inputs,
            )
            self.held_trial = self.trial
        self.j1, self.points1, self.local1, self.rj1, samples, self.mapping1 = maps(
            self, self.held_trial.result.state
        )
        self.robot_points1 = np.array([s.world_point for s in samples]).reshape(-1, 3)
        self.active = np.array(
            [
                s.exact_distance_bounds[1] <= self.bound and full[a.pair].status == "contact"
                for a, s in zip(self.anchors, samples, strict=True)
            ]
        )
        self.n1 = np.array([full[a.pair].normal for a in self.anchors]).reshape(-1, 3)
        for i, (active, p, t) in enumerate(zip(self.active, self.p, self.t, strict=True)):
            if not active and (
                self.k0[i] * abs(p) / self.settings.reference_speed > self.settings.tolerance
                or np.linalg.norm(t) * self.kappa0[i] / self.settings.reference_speed
                > self.settings.tolerance
            ):
                raise SolverFailure("nonzero held load on disappeared anchor")
        self.terminal_mass = _PositiveMass(
            mass_matrix(self.model, self.held_trial.result.state.domains[self.domain].q)
        )
        inverse = np.array(
            [
                self.terminal_mass.solve([F(i == j) for i in range(self.model.dof_count)])
                for j in range(self.model.dof_count)
            ],
            float,
        ).T
        self.w1 = response(self, self.j1, self.rj1, inverse)
        self.k1, self.kappa1 = self._scales(self.w1, self.n1)
        self.alpha1 = self._alpha(self.w1)
        self.eta = np.zeros(len(self.anchors))
        self.tau = np.zeros((len(self.anchors), 3))
        self._require_ccd()
        self.ready = True
        self._evaluate(True)

    def velocity_sweep(self):
        self.vs += 1
        self._improve(True)

    def _velocity_metrics(self):
        position = self._position_metrics()[0]
        residual = dict(position.residuals, terminal_normal=0.0)
        rel = relative(self, self.trial.result.state, self.j1, self.rj1)
        for active, k, n, s, u in zip(
            self.active, self.k1, self.n1, self._world(self.eta, True), rel, strict=True
        ):
            if not active:
                if np.any(s != 0):
                    raise SolverFailure("nonzero inactive terminal impulse")
                continue
            et = normal_projection_bounds(n, s)
            speed = normal_projection_bounds(n, u)
            value = max(max(ZERO, -v, abs(min(q(k) * e, v))) for e in et for v in speed)
            residual["terminal_normal"] = max(
                residual["terminal_normal"], upper_float(value / q(self.settings.reference_speed))
            )
        friction, _ = self._friction(True)
        residual.update({"terminal_" + k: v for k, v in friction.items()})
        physical, self.physical_report, _ = self.balance.require_metrics(
            self.trial.result.state,
            self.ledger,
            self._loads()[2],
            self.settings.tolerance,
            terminal=self._node_terminal(),
        )
        residual["paired_held_moment"] = paired_moment_residual(
            self.points0,
            self.robot_points0,
            self._world(self.p),
            self.settings.tolerance,
            self.program.budgets.max_material_angular_defect_nms,
        )
        residual["paired_terminal_moment"] = paired_moment_residual(
            self.points1,
            self.robot_points1,
            self._world(self.eta, True),
            self.settings.tolerance,
            self.program.budgets.max_material_angular_defect_nms,
        )
        residual.update(physical)
        residual.update(self._terminal_energy())
        residual["ideal_extreme_velocity"] = self._ideal_extremes()
        return StageMetrics(residual, penetration=position.penetration)

    def velocity_metrics(self):
        metrics = self._velocity_metrics()
        self.vdone = metrics.max_residual <= self.settings.tolerance and (
            metrics.max_residual <= self.settings.tolerance / 8
            or self.vs >= self.settings.velocity_iterations
        )
        return (
            metrics
            if self.vdone
            else StageMetrics(
                {**metrics.residuals, "refinement_pending": 2 * self.settings.tolerance},
                penetration=metrics.penetration,
            )
        )

    def _ideal_extremes(self):
        final = self.trial.result.state
        worst = ZERO
        for pair, m in enumerate(manifolds(self, final)):
            if m.status != "contact":
                continue
            s, c = self.program.pairs[pair]
            x = tuple(vector(p) for p in final.domains[s.domain].position[list(s.node_indices)])
            v = tuple(vector(p) for p in final.domains[s.domain].velocity[list(s.node_indices)])
            e1, e2 = sub(x[1], x[0]), sub(x[2], x[0])
            cv = cross(e1, e2)
            length = Q(0, 1, dot(cv, cv).a)
            n = scale(cv, 1 / length)
            cd = tuple(
                a + b
                for a, b in zip(cross(sub(v[1], v[0]), e2), cross(e1, sub(v[2], v[0])), strict=True)
            )
            nd = scale(sub(cd, scale(n, dot(n, cd))), 1 / length)
            ids = [i for i, a in enumerate(self.anchors) if a.pair == pair]
            if not ids and m.points:
                raise SolverFailure("ideal manifold has no material coverage")
            rvel = tuple(
                sum(
                    (
                        Q(q(self.rj1[ids[0], k, j])) * q(final.domains[self.domain].v[j])
                        for j in range(self.model.dof_count)
                    ),
                    Q(0),
                )
                for k in range(3)
            )
            for p in m.points:
                b0, b1, b2, offset = map(Q.coerce, p.material)
                u = tuple(
                    b0 * v[0][k] + b1 * v[1][k] + b2 * v[2][k] + offset * nd[k] - rvel[k]
                    for k in range(3)
                )
                # Ideal support normal normalization uses the exact single-prism field.
                axis = m.selected_axis
                raw = dot(axis.direction, u)
                low = raw.bounds()[0]
                nlo, nhi = axis.norm_squared.bounds()
                from ._contact_replay_audit import sqrt_bounds

                norm = sqrt_bounds(nlo)[0]
                if low < 0:
                    worst = max(worst, -low / norm / q(self.settings.reference_speed))
        return upper_float(worst)

    def _candidate_signature(self, state):
        return fingerprint(
            (
                self.initial.domains,
                self.inputs,
                tuple((name, state.domains[name].position) for name in self.node_domains),
                state.domains[self.domain].q,
            )
        )

    def _require_ccd(self):
        final = self.held_trial.result.state
        results = []
        for s, c in self.program.pairs:
            result = triangle_prism_patch_sweep(
                self.model,
                self.initial.domains[self.domain].q,
                final.domains[self.domain].q,
                c,
                self.initial.domains[s.domain].position[list(s.node_indices)],
                final.domains[s.domain].position[list(s.node_indices)],
                half_thickness_m=s.half_thickness_m,
                penetration_budget_m=self.bound,
                max_intervals=self.program.max_ccd_intervals,
                max_depth=self.program.max_ccd_depth,
            )
            expected = candidate_fingerprint(
                self.model,
                self.initial.domains[self.domain].q,
                final.domains[self.domain].q,
                c,
                self.initial.domains[s.domain].position[list(s.node_indices)],
                final.domains[s.domain].position[list(s.node_indices)],
                s.half_thickness_m,
                self.bound,
                self.program.max_ccd_intervals,
                self.program.max_ccd_depth,
            )
            if result.input_fingerprint != expected:
                raise SolverFailure("whole-prism CCD input fingerprint mismatch")
            if result.status != "certified":
                raise SolverFailure("whole-prism CCD " + result.status + ": " + result.reason)
            results.append(result)
        self.ccd = tuple(results)
        self.certified_candidate = self._candidate_signature(final)

    def result(self):
        if not self.vdone:
            raise SolverFailure("terminal stage not converged")
        state = self.trial.result.state
        if self._candidate_signature(state) != self.certified_candidate:
            raise SolverFailure("CCD candidate fingerprint mismatch")
        k, r = len(self.program.pairs), self.program.max_anchors_per_pair
        pair = np.zeros(k, bool)
        active = np.zeros((k, r), bool)
        material = np.zeros((k, r, 4))
        phases = np.zeros((k, r, 2, 3))
        slots = [0] * k
        for i, a in enumerate(self.anchors):
            slot = slots[a.pair]
            slots[a.pair] += 1
            if self.active[i]:
                pair[a.pair] = True
                active[a.pair, slot] = True
                material[a.pair, slot] = a.material
                phases[a.pair, slot] = [self._world(self.p)[i], self._world(self.eta, True)[i]]
        fields = self.program._own_fields
        return SceneState(
            state.domains,
            {
                **state.history,
                **{
                    f.name: v for f, v in zip(fields, (pair, active, material, phases), strict=True)
                },
            },
        )

    def diagnostics(self):
        self._velocity_metrics()
        state = self.result()
        active = state.history[self.program._own_fields[0].name]
        # Public exact geometry retains immutable Fraction/Q values. Runtime diagnostics
        # use explicit exact rational tokens because the old fingerprint is intentionally narrow.
        ccd = tuple(
            dict(
                status=r.status,
                input_fingerprint=r.input_fingerprint,
                checked_intervals=r.checked_intervals,
                reason=r.reason,
                segments=tuple((str(s.start), str(s.end), s.gap_lower_m) for s in r.segments),
            )
            for r in self.ccd
        )
        ideal_manifolds = tuple(
            dict(
                pair_keys=self.program.pair_keys[i],
                status=m.status,
                normal=m.normal,
                gap_bounds=m.gap_bounds,
                prism_points=np.array([p.prism_point for p in m.points]).reshape(-1, 3),
                robot_points=np.array([p.robot_point for p in m.points]).reshape(-1, 3),
                material_exact=tuple(tuple(str(v) for v in p.material) for p in m.points),
                features=tuple((p.prism_feature, p.patch_feature) for p in m.points),
            )
            for i, m in enumerate(manifolds(self, self.trial.result.state))
        )
        return own_diagnostics(
            dict(
                dynamic_robot_thick_contact=dict(
                    pair_keys=self.program.pair_keys,
                    anchor_features=tuple(a.feature for a in self.anchors),
                    representation_error_m=np.array(
                        [a.representation_error_m for a in self.anchors]
                    ),
                    ideal_manifolds=ideal_manifolds,
                    anchor_material=np.array([a.material for a in self.anchors]),
                    initial_closed=self.initial_closed,
                    terminal_only=np.array([a.terminal_only for a in self.anchors]),
                    anchor_active=self.active,
                    held_normal=self.n0,
                    terminal_normal=self.n1,
                    held_impulses=self.p,
                    terminal_impulses=self.eta,
                    held_tangent=self.t,
                    terminal_tangent=self.tau,
                    held_world=self._world(self.p),
                    terminal_world=self._world(self.eta, True),
                    held_node_impulses={
                        name: np.array(value, float) for name, value in self._loads()[2].items()
                    },
                    terminal_node_impulses={
                        name: np.array(value, float)
                        for name, value in self._node_terminal().items()
                    },
                    held_robot_impulse=np.array(self._loads()[1], float),
                    held_state=self.held_trial.result.state,
                    material_reports=self.material_reports,
                    physical_budgets=self.physical_report,
                    terminal_energy=self.terminal_energy,
                    mapping=self.mapping_report,
                    terminal_ledger=self.trial.result.report.substeps[0].diagnostics[
                        "terminal_impulse"
                    ],
                    ccd=ccd,
                    held_replays=self.held_replays,
                    terminal_replays=self.terminal_replays,
                    line_search_replays=self.line_search_replays,
                    rebuilds=self.rebuilds,
                    transitions=tuple(
                        "persist" if a and b else "exit" if a else "enter" if b else "inactive"
                        for a, b in zip(self.old_active, active, strict=True)
                    ),
                ),
                execution=self.execution,
            )
        )

    def close(self):
        pass
