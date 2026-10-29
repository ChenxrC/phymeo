"""Automatic finite convex patch contact through two complete native phases."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction as F
from types import MappingProxyType

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..constraints.robot_finite_patch_collider import RobotFinitePatchCollider
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import HistoryField, SceneState, StageMetrics, _name, fingerprint
from ..dynamics.robot_dynamics import mass_matrix
from ..dynamics.robot_kinematics import point_jacobians
from ..geometry._patch_interval import validate_domain
from ..geometry.robot_patch_sweep import _positive_integer, robot_patch_sweep
from ..runtime.program import own_diagnostics
from ..validation._material_path_math import interval_float
from ..validation.material_path import MaterialPathLedger
from ..validation.terminal_impulse import _PositiveMass
from ._finite_contact_geometry import (
    RotorLedger,
    axis_metrics,
    geometry,
    normal_projection_bounds,
    paired_moment,
    relative_bounds,
)
from ._finite_contact_replay import FiniteReplay
from ._material_contact_balance import MaterialContactBalance
from ._robot_replay_ledger import rational as q
from ._robot_replay_ledger import upper_float
from .initial_robot_material_contact import MaterialContactBudgets
from .robot_plane_contact import RobotPlaneContactProgram

ZERO = F(0)


class DynamicRobotFiniteContactProgram(RobotPlaneContactProgram):
    """Positive-radius, two-sided patches; all-prismatic or one fixed COM0 rotor."""

    def __init__(
        self,
        scene,
        base_program,
        colliders,
        *,
        materials,
        budgets,
        robot_domain="robot",
        registry=None,
        inner_settings=None,
        relaxation=1.0,
        reference_angular_speed=1.0,
        max_ccd_intervals=4096,
        max_ccd_depth=32,
        history_prefix="finite_contact",
    ):
        super().__init__(
            scene,
            base_program,
            (),
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
            reference_angular_speed=reference_angular_speed,
        )
        if type(reference_angular_speed) is not float:
            raise ValueError("reference_angular_speed must be a positive builtin float")
        self.robot_kind = validate_domain(self.model)
        if "terminal_point_impulses" not in base_program.capabilities.features:
            raise ValueError("base lacks terminal_point_impulses capability")
        self.max_ccd_intervals = _positive_integer(max_ccd_intervals, "max_ccd_intervals")
        self.max_ccd_depth = _positive_integer(max_ccd_depth, "max_ccd_depth")
        colliders = tuple(colliders)
        if not colliders or any(not isinstance(c, RobotFinitePatchCollider) for c in colliders):
            raise ValueError("at least one RobotFinitePatchCollider required")
        if len({c.key for c in colliders}) != len(colliders) or any(
            c.link_index >= self.model.dof_count for c in colliders
        ):
            raise ValueError("unique collider keys and valid links required")
        self.colliders = tuple(sorted((replace(c) for c in colliders), key=lambda c: c.key))
        if not isinstance(materials, Mapping) or not materials:
            raise ValueError("materials must explicitly select nonempty particle domains")
        bindings = {d.name: d for d in scene.domains}
        owned = {}
        for name, material in sorted(materials.items()):
            if name not in bindings or bindings[name].family != "particles":
                raise ValueError("materials must select registered particle domains")
            model = bindings[name].model
            if (
                not isinstance(model, ParticleModel)
                or model.inverse_mass.dtype != np.float64
                or np.any(model.inverse_mass <= 0)
            ):
                raise ValueError("selected nodes must be FP64 and fully dynamic")
            family = self.registry.get("particles")
            if (
                not {"position", "velocity"} <= set(family.state_fields)
                or "force" not in family.input_fields
            ):
                raise ValueError("particle fields must be registered")
            if material is not None:
                if (
                    not isinstance(material, DistanceConstraints)
                    or material.rest_length.dtype != np.float64
                ):
                    raise TypeError("materials require FP64 DistanceConstraints or None")
                if np.any(material.indices >= model.particle_count):
                    raise ValueError("material edge outside selected domain")
                material = replace(material)
            owned[name] = material
        if not isinstance(budgets, MaterialContactBudgets):
            raise TypeError("explicit MaterialContactBudgets required")
        self.materials, self.budgets = MappingProxyType(owned), replace(budgets)
        _name(history_prefix, "history prefix")
        self.history_prefix = history_prefix
        self.row_keys = tuple(
            (name, i, c.key)
            for name in owned
            for i in range(bindings[name].model.particle_count)
            for c in self.colliders
        )
        count = len(self.row_keys)
        self._own_fields = (
            HistoryField(history_prefix + "/active", (count,), bool),
            HistoryField(history_prefix + "/feature", (count, 2), np.int64),
            HistoryField(history_prefix + "/phase_world", (count, 2, 3), np.float64),
        )
        if any(
            f.name == history_prefix or f.name.startswith(history_prefix + "/")
            for f in base_program.history_fields
        ):
            raise ValueError("finite contact history prefix conflicts with base")

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            history="explicit",
            dtypes=frozenset({"float64"}),
            features=frozenset({"dynamic_robot_finite_split"}),
        )

    @property
    def history_fields(self):
        return tuple(self.base.history_fields) + self._own_fields

    def initial_history(self):
        return {f.name: np.zeros(f.shape, f.dtype) for f in self._own_fields}

    def problem_signature(self):
        return dict(
            super().problem_signature(),
            finite_law="finite-two-stage-v1",
            colliders=self.colliders,
            robot_kind=self.robot_kind,
            materials=self.materials,
            budgets=self.budgets,
            max_ccd_intervals=self.max_ccd_intervals,
            max_ccd_depth=self.max_ccd_depth,
            history=tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields),
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self._closed:
            raise RuntimeError("finite program closed")
        self.scene.assert_unchanged()
        if (
            scene.identity != self.scene.identity
            or settings.tolerance <= 0
            or not np.isfinite(h)
            or h <= 0
        ):
            raise ValueError("bound scene and positive h/tolerance required")
        inner = (
            replace(settings, substeps=1, record_substeps=False)
            if self.inner_settings is None
            else self.inner_settings
        )
        if any(
            getattr(inner, k) != getattr(settings, k)
            for k in ("tolerance", "reference_length", "reference_speed")
        ):
            raise ValueError("inner tolerance and scales must match outer")
        return _FiniteWorkspace(self, state, inputs, h, settings, inner, context)


class _FiniteWorkspace(FiniteReplay):
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.inputs, self.h = program, state, inputs, h
        self.settings, self.inner, self.context = settings, inner, context
        self.domain, self.model = program.robot_domain, program.model
        self.models = {d.name: d.model for d in program.scene.domains}
        self.node_domains = tuple(program.materials)
        self.rows = tuple(
            (name, i, c)
            for name in self.node_domains
            for i in range(self.models[name].particle_count)
            for c in program.colliders
        )
        self.indices = np.array([r[2].link_index for r in self.rows], int)
        self.exact_gap_scale = min(q(settings.reference_length), q(h) * q(settings.reference_speed))
        self.bound = q(settings.tolerance) * self.exact_gap_scale
        if any(q(c.radius_m) <= self.bound for _, _, c in self.rows):
            raise SolverFailure("patch radius must exceed original penetration budget")
        self.points0, self.j0, self.normals, self.gaps0, self.initial_bounds = geometry(self, state)
        if any(s.normal is None for s in self.samples) or any(
            lo < -self.bound for lo, _ in self.initial_bounds
        ):
            raise SolverFailure(
                "initial finite patch zero distance or penetration exceeds tolerance"
            )
        self.local = np.array([s.local_point for s in self.samples])
        self.initial_feature = np.array([s.feature for s in self.samples], np.int64)
        self.initial_sides = np.array(
            [
                1 if s.signed_support[0] > 0 else -1 if s.signed_support[1] < 0 else 0
                for s in self.samples
            ]
        )
        self.old_active = state.history[program._own_fields[0].name].copy()
        self.old_feature = state.history[program._own_fields[1].name].copy()
        old = state.history[program._own_fields[2].name]
        if (
            np.any(~np.isfinite(old))
            or np.any(old[~self.old_active] != 0)
            or np.any(self.old_feature[~self.old_active] != 0)
        ):
            raise SolverFailure("invalid or dirty finite history")
        if any(
            active and lo > self.bound
            for active, (lo, _) in zip(self.old_active, self.initial_bounds, strict=True)
        ):
            raise SolverFailure("active history is geometrically separated")
        self.ledger = RotorLedger(
            self.model,
            state.domains[self.domain],
            inputs.domains[self.domain],
            h,
            settings,
            program.reference_angular_speed,
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
        self.k = np.array(
            [
                n
                @ (
                    jac @ self.ledger.inverse @ jac.T
                    + self.models[name].inverse_mass[i] * np.eye(3)
                )
                @ n
                for (name, i, _), jac, n in zip(self.rows, self.j0, self.normals, strict=True)
            ]
        )
        if np.any(~np.isfinite(self.k)) or np.any(self.k <= 0):
            raise SolverFailure("invalid frozen positive response")
        self.base_state = SceneState(
            state.domains, {f.name: state.history[f.name] for f in program.base.history_fields}
        )
        self.channels = {
            self.domain: ("effort",),
            **{name: ("force",) for name in self.node_domains},
        }
        self.p, self.eta = np.zeros(len(self.rows)), np.zeros(len(self.rows))
        self.held_replays = self.terminal_replays = self.line_search_replays = self.ps = self.vs = 0
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.trial = self.held_trial = None
        self.ready = self.pdone = self.vdone = False
        self.pbest = self.vbest = None
        self.native_audit = self.material_reports = self.physical_report = {}
        self.ccd = None

    def predict(self):
        if self.trial is not None:
            raise RuntimeError("finite contact already predicted")
        self._evaluate()

    def _generator(self):
        geometry(self, self.trial.result.state)
        return np.array(
            [
                float(side * sum(s.signed_support) / 2 - q(c.radius_m))
                if side and s.projection_inside
                else float(sum(s.exact_distance_bounds) / 2 - q(c.radius_m))
                for side, s, (_, _, c) in zip(
                    self.initial_sides, self.samples, self.rows, strict=True
                )
            ]
        )

    def _position_metrics(self):
        _, _, _, gaps, bounds = geometry(self, self.trial.result.state)
        penetration = max((max(-lo, ZERO) for lo, _ in bounds), default=ZERO)
        residual = {k: upper_float(q(v)) for k, v in self.native_audit.items()}
        residual.update(
            native=self.trial.result.report.max_residual,
            final_penetration=upper_float(penetration / self.exact_gap_scale),
            held_normal=0.0,
            held_direction_mapping=self._world_mapping(self.p),
        )
        for k, p, n, world, interval in zip(
            self.k, self.p, self.normals, self._world(self.p), bounds, strict=True
        ):
            if p < 0:
                raise SolverFailure("negative held impulse")
            err = max(
                abs(min(q(k) * impulse, g / q(self.h)))
                for g in interval
                for impulse in normal_projection_bounds(n, world)
            )
            residual["held_normal"] = max(
                residual["held_normal"], upper_float(err / q(self.settings.reference_speed))
            )
        self.held_moment, moment = paired_moment(
            self, self.initial, self.local, self._world(self.p)
        )
        residual["held_paired_moment"] = upper_float(moment)
        return StageMetrics(residual, penetration=upper_float(penetration)), gaps, bounds

    def position_sweep(self):
        self.ps += 1
        if self.pdone:
            return
        for i, k in enumerate(self.k):
            gaps = self._generator()
            old = self.p[i]
            if old == 0 and gaps[i] >= -float(self.bound):
                continue
            target = max(0.0, old - self.program.relaxation * gaps[i] / (self.h * k))
            if old == target:
                continue
            original = self.trial, self.native_audit, self.material_reports
            oldscore = float(np.max(np.abs(np.minimum(self.h * self.k * self.p, gaps))))
            accepted = False
            for power in range(13):
                self.p[i] = old + (target - old) / (2**power)
                self._evaluate()
                if power:
                    self.line_search_replays += 1
                score = float(
                    np.max(np.abs(np.minimum(self.h * self.k * self.p, self._generator())))
                )
                if score < oldscore or score <= float(self.bound):
                    accepted = True
                    break
            if not accepted:
                self.p[i] = old
                self.trial, self.native_audit, self.material_reports = original

    def _select(self, metrics, velocity=False):
        best = self.vbest if velocity else self.pbest
        score = (
            metrics.residuals["terminal_normal"]
            if velocity
            else max(metrics.residuals["held_normal"], metrics.residuals["final_penetration"])
        )
        values = self.eta if velocity else self.p
        iterations = self.vs if velocity else self.ps
        budget = (
            self.settings.velocity_iterations if velocity else self.settings.position_iterations
        )
        if metrics.max_residual <= self.settings.tolerance:
            if best is None or score < best[0]:
                best = (score, values.copy(), self.trial, self.native_audit, self.material_reports)
                if velocity:
                    self.vbest = best
                else:
                    self.pbest = best
                if score == 0 or iterations >= budget:
                    if velocity:
                        self.vdone = True
                    else:
                        self.pdone = True
                    return metrics
                return StageMetrics(
                    {
                        **metrics.residuals,
                        "iteration_refinement_pending": 2 * self.settings.tolerance,
                    },
                    penetration=metrics.penetration,
                )
        if best is not None:
            _, values, self.trial, self.native_audit, self.material_reports = best
            if velocity:
                self.eta, self.vdone = values, True
                return self._velocity_metrics()
            self.p, self.pdone = values, True
            return self._position_metrics()[0]
        if (
            velocity
            and iterations >= budget
            and metrics.residuals["terminal_normal"] <= self.settings.tolerance
        ):
            failed = tuple(k for k, v in metrics.residuals.items() if v > self.settings.tolerance)
            raise SolverFailure("finite physical budget exceeded: " + ", ".join(failed))
        return metrics

    def position_metrics(self):
        metrics = self._position_metrics()[0]
        return metrics if self.pdone else self._select(metrics)

    def prepare_velocity(self):
        if self.ready or self._position_metrics()[0].max_residual > self.settings.tolerance:
            raise SolverFailure("finite held stage not converged")
        self.held_trial = self.trial
        _, _, bounds = self._position_metrics()
        closed = []
        for p, (lo, hi) in zip(self.p, bounds, strict=True):
            if lo >= -self.bound and hi <= self.bound:
                closed.append(True)
            elif lo > self.bound:
                closed.append(False)
            else:
                raise SolverFailure("uncertain finite closure interval")
            if p > 0 and not closed[-1]:
                raise SolverFailure("positive held impulse on open finite row")
        if any(s.normal is None for s in self.samples):
            raise SolverFailure("accepted finite distance must be positive")
        self.terminal_local = np.array([s.local_point for s in self.samples])
        self.terminal_normals = np.array([s.normal for s in self.samples])
        self.final_feature = np.array([s.feature for s in self.samples], np.int64)
        self.j1 = point_jacobians(
            self.model,
            self.held_trial.result.state.domains[self.domain].q,
            self.indices,
            self.terminal_local,
        )[0]
        self.terminal_mass = _PositiveMass(
            mass_matrix(self.model, self.held_trial.result.state.domains[self.domain].q)
        )
        n = self.model.dof_count
        inverse = np.array(
            [self.terminal_mass.solve([F(i == j) for i in range(n)]) for j in range(n)], float
        ).T
        self.k1 = np.array(
            [
                normal
                @ (jac @ inverse @ jac.T + self.models[name].inverse_mass[i] * np.eye(3))
                @ normal
                for (name, i, _), jac, normal in zip(
                    self.rows, self.j1, self.terminal_normals, strict=True
                )
            ]
        )
        if np.any(~np.isfinite(self.k1)) or np.any(self.k1 <= 0):
            raise SolverFailure("invalid current terminal response")
        self.active = np.array(closed, bool)
        self._require_ccd()
        self.ready = True
        self._evaluate(terminal=True)

    def velocity_sweep(self):
        if not self.ready:
            raise RuntimeError("finite velocity not prepared")
        self.vs += 1
        if self.vdone:
            return
        for i, k in enumerate(self.k1):
            if not self.active[i]:
                continue
            a, b = relative_bounds(self, self.trial.result.state)[i]
            u = float((a + b) / 2)
            old = self.eta[i]
            if old == 0 and u >= -self.settings.tolerance * self.settings.reference_speed:
                continue
            self.eta[i] = max(0.0, old - self.program.relaxation * u / k)
            if old != self.eta[i]:
                self._evaluate(terminal=True)

    def _velocity_metrics(self):
        metrics = self._position_metrics()[0]
        residual = dict(
            metrics.residuals,
            terminal_normal=0.0,
            terminal_direction_mapping=self._world_mapping(self.eta, True),
        )
        for active, k, eta, n, world, bounds in zip(
            self.active,
            self.k1,
            self.eta,
            self.terminal_normals,
            self._world(self.eta, True),
            relative_bounds(self, self.trial.result.state),
            strict=True,
        ):
            if eta < 0 or (not active and eta != 0):
                raise SolverFailure("invalid terminal impulse")
            if active:
                err = max(
                    max(-u, ZERO, abs(min(q(k) * p, u)))
                    for u in bounds
                    for p in normal_projection_bounds(n, world)
                )
                residual["terminal_normal"] = max(
                    residual["terminal_normal"], upper_float(err / q(self.settings.reference_speed))
                )
        physical, self.physical_report, _ = self.balance.require_metrics(
            self.trial.result.state,
            self.ledger,
            self._loads()[2],
            self.settings.tolerance,
            terminal=self._node_terminal(),
        )
        _, moment = paired_moment(
            self, self.held_trial.result.state, self.terminal_local, self._world(self.eta, True)
        )
        residual["terminal_paired_moment"] = upper_float(moment)
        residual.update(physical)
        residual.update(self._terminal_energy())
        residual.update(axis_metrics(self))
        return StageMetrics(residual, penetration=metrics.penetration)

    def velocity_metrics(self):
        metrics = self._velocity_metrics()
        return metrics if self.vdone else self._select(metrics, True)

    def _candidate_signature(self, state):
        return fingerprint(
            (
                self.initial.domains[self.domain].q,
                state.domains[self.domain].q,
                tuple(
                    (
                        name,
                        i,
                        self.initial.domains[name].position[i],
                        state.domains[name].position[i],
                    )
                    for name, i, c in self.rows
                ),
            )
        )

    def _require_ccd(self):
        final = self.held_trial.result.state
        results = []
        for name, i, c in self.rows:
            exact = q(c.radius_m) - self.bound
            clearance = float(exact)
            if q(clearance) < exact:
                clearance = float(np.nextafter(clearance, np.inf))
            result = robot_patch_sweep(
                self.model,
                self.initial.domains[self.domain].q,
                final.domains[self.domain].q,
                c,
                self.initial.domains[name].position[i],
                final.domains[name].position[i],
                clearance_m=clearance,
                max_intervals=self.program.max_ccd_intervals,
                max_depth=self.program.max_ccd_depth,
            )
            expected = fingerprint(
                (
                    self.model,
                    self.initial.domains[self.domain].q,
                    final.domains[self.domain].q,
                    c,
                    self.initial.domains[name].position[i],
                    final.domains[name].position[i],
                    clearance,
                    self.program.max_ccd_intervals,
                    self.program.max_ccd_depth,
                )
            )
            if result.input_fingerprint != expected:
                raise SolverFailure("finite CCD input fingerprint mismatch")
            if result.status != "certified":
                raise SolverFailure("finite curved FK CCD " + result.status)
            results.append(result)
        self.ccd = tuple(results)
        self.certified_candidate = self._candidate_signature(final)

    def result(self):
        if not self.ready:
            raise RuntimeError("finite contact not ready")
        state = self.trial.result.state
        if self._candidate_signature(state) != self.certified_candidate:
            raise SolverFailure("CCD certificate not bound to selected candidate")
        phases = np.stack((self._world(self.p), self._world(self.eta, True)), axis=1)
        phases[~self.active] = 0.0
        features = self.final_feature.copy()
        features[~self.active] = 0
        return SceneState(
            state.domains,
            {
                **state.history,
                self.program._own_fields[0].name: self.active.copy(),
                self.program._own_fields[1].name: features,
                self.program._own_fields[2].name: phases,
            },
        )

    def diagnostics(self):
        self._velocity_metrics()
        final = geometry(self, self.trial.result.state)
        events = tuple(
            "persist" if a and b else "exit" if a else "enter" if b else "inactive"
            for a, b in zip(self.old_active, self.active, strict=True)
        )
        return own_diagnostics(
            dict(
                dynamic_robot_finite_contact=dict(
                    row_keys=self.program.row_keys,
                    initial_feature=self.initial_feature,
                    final_feature=self.final_feature,
                    held_local_points=self.local,
                    terminal_local_points=self.terminal_local,
                    held_normal=self.normals,
                    terminal_normal=self.terminal_normals,
                    held_impulses=self.p,
                    terminal_impulses=self.eta,
                    held_world=self._world(self.p),
                    terminal_world=self._world(self.eta, True),
                    gap_bounds=np.array([interval_float(b) for b in final[4]]),
                    relative_normal_bounds=np.array(
                        [interval_float(b) for b in relative_bounds(self, self.trial.result.state)]
                    ),
                    transitions=events,
                    feature_changed=self.old_active
                    & self.active
                    & np.any(self.old_feature != self.final_feature, axis=1),
                    held_replays=self.held_replays,
                    terminal_replays=self.terminal_replays,
                    line_search_replays=self.line_search_replays,
                    held_state=self.held_trial.result.state,
                    physical_budgets=self.physical_report,
                    axis_balance=self.axis_report,
                    terminal_ledger=self.trial.result.report.substeps[0].diagnostics[
                        "terminal_impulse"
                    ],
                    terminal_energy=self.terminal_energy,
                    material_reports=self.material_reports,
                    native_audit={k: upper_float(v) for k, v in self.native_audit.items()},
                    ccd=self.ccd,
                    input_integrals=self.trial.integrated_loads,
                ),
                execution=self.execution,
            )
        )

    def close(self):
        pass
