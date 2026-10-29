"""Native replay workspace for the phasewise friction law.

Reuse the normal program's native transaction and independent ledgers; own all
friction unknowns and phase selection. No state is computed by the outer solver.
"""

from fractions import Fraction

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import SceneState, StageMetrics
from ..runtime.program import own_diagnostics
from ..runtime.terminal_impulse import PointVelocityImpulses
from ..validation._material_path_math import interval_float
from ..validation.material_path import MaterialPathLedger
from ._dynamic_friction_law import audit, capacity, combined_capacity, nonnegative_world, projected
from ._dynamic_normal_geometry import geometry, project_local
from ._material_contact_balance import MaterialContactBalance
from ._robot_replay_ledger import FrozenRobotLedger, dot, exact_float, rational, upper_float
from .dynamic_robot_normal_contact import _DynamicNormalWorkspace

ZERO = Fraction(0)


class _DynamicFrictionWorkspace(_DynamicNormalWorkspace):
    def __init__(self, program, state, inputs, h, settings, inner, context):
        self.program, self.initial, self.inputs, self.h = program, state, inputs, h
        self.settings, self.inner, self.context = settings, inner, context
        self.domain, self.model = program.robot_domain, program.model
        self.models = {d.name: d.model for d in program.scene.domains}
        self.node_domains = tuple(program.materials)
        self.rows = [
            (name, i, c)
            for name in self.node_domains
            for i in range(self.models[name].particle_count)
            for c in program.colliders
        ]
        self.indices = np.array([r[2].link_index for r in self.rows], int)
        self.anchors = np.array([r[2].local_point for r in self.rows]).reshape(-1, 3)
        self.local_normals = np.array([r[2].local_normal for r in self.rows]).reshape(-1, 3)
        self.local = project_local(self, state)
        self.exact_gap_scale = min(
            rational(settings.reference_length), rational(h) * rational(settings.reference_speed)
        )
        self.gap_scale = exact_float(self.exact_gap_scale)
        self.points0, self.j0, self.normals, self.gaps0, initial_bounds = geometry(self, state)
        self.initial_bounds = initial_bounds
        self.bound = rational(settings.tolerance) * self.exact_gap_scale
        if any(lo < -self.bound for lo, _ in initial_bounds):
            raise SolverFailure("initial plane penetration exceeds tolerance")
        self.old_active = state.history[program._own_fields[0].name].copy()
        self.initial_closed = self._classify(initial_bounds)
        self.mu = np.array([program.friction_by_collider[row[2].key] for row in self.rows])
        if any(
            active and not closed
            for active, closed in zip(self.old_active, self.initial_closed, strict=True)
        ):
            raise SolverFailure("active contact history is geometrically separated")
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
        self.k = np.array(
            [
                float(
                    n
                    @ (
                        jac @ self.ledger.inverse @ jac.T
                        + self.models[row[0]].inverse_mass[row[1]] * np.eye(3)
                    )
                    @ n
                )
                for row, jac, n in zip(self.rows, self.j0, self.normals, strict=True)
            ]
        )
        if np.any(~np.isfinite(self.k)) or np.any(self.k <= 0):
            raise SolverFailure("invalid normal mass response")
        self.base_state = SceneState(
            state.domains, {f.name: state.history[f.name] for f in program.base.history_fields}
        )
        self.channels = {
            self.domain: ("effort",),
            **{name: ("force",) for name in self.node_domains},
        }
        self.p, self.eta = np.zeros(len(self.rows)), np.zeros(len(self.rows))
        self.held_replays = self.terminal_replays = self.ps = self.vs = 0
        self.execution = dict(launches=0, completed_launches=0, uploads=0, downloads=0)
        self.trial = self.held_trial = None
        self.ready, self.pdone, self.vdone = False, False, False
        self.pbest = self.vbest = None
        self.material_reports = self.native_audit = self.physical_report = {}

        self.kappa = np.array(
            [
                np.trace(
                    jac @ self.ledger.inverse @ jac.T
                    + self.models[row[0]].inverse_mass[row[1]] * np.eye(3)
                )
                for row, jac in zip(self.rows, self.j0, strict=True)
            ]
        )
        self.t, self.tau = np.zeros((len(self.rows), 3)), np.zeros((len(self.rows), 3))
        self._validate_history(state)

    def _classify(self, bounds):
        result = []
        for lo, hi in bounds:
            if lo >= -self.bound and hi <= self.bound:
                result.append(True)
            elif lo > self.bound:
                result.append(False)
            else:
                raise SolverFailure("uncertain plane closure interval")
        return np.array(result, dtype=bool)

    def _validate_history(self, state):
        fields = self.program._own_fields
        closed, worlds = state.history[fields[1].name], state.history[fields[2].name]
        if (
            not np.isfinite(worlds).all()
            or np.any(worlds[~self.old_active] != 0)
            or np.any(closed[~self.old_active])
        ):
            raise SolverFailure("invalid or dirty friction history")
        for i, n in enumerate(self.normals):
            p, e = worlds[i]
            ch = capacity(n, p, self.mu[i]) if closed[i] else (ZERO, ZERO)
            cv = capacity(n, e, self.mu[i]) if closed[i] else combined_capacity(n, p, e, self.mu[i])
            for s, c in ((p, ch), (e, cv)):
                metrics, _, _ = audit(
                    n, s, [ZERO] * 3, c, self.kappa[i], self.settings.reference_speed
                )
                if max(metrics.values()) > self.settings.tolerance:
                    raise SolverFailure("friction history exceeds cone")

    def _world(self, values):
        if values is self.p:
            tangent = self.t
        elif values is self.eta:
            tangent = self.tau
        else:
            raise SolverFailure("unknown friction phase")
        raw = self.normals * values[:, None] + tangent
        return np.array(
            [nonnegative_world(n, s) for n, s in zip(self.normals, raw, strict=True)]
        ).reshape(-1, 3)

    def _world_mapping(self, values):
        tangent = self.t if values is self.p else self.tau
        node = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        generalized = [ZERO] * self.model.dof_count
        orthogonal = ZERO
        for i, (row, jac, n, p, s) in enumerate(
            zip(self.rows, self.j0, self.normals, values, self._world(values), strict=True)
        ):
            error = [
                rational(s[k]) - rational(n[k]) * rational(p) - rational(tangent[i, k])
                for k in range(3)
            ]
            orthogonal = max(
                orthogonal,
                rational(self.kappa[i])
                * abs(dot(n, tangent[i]))
                / rational(self.settings.reference_speed),
            )
            for k in range(3):
                node[row[0]][row[1]][k] += error[k]
            for j in range(self.model.dof_count):
                generalized[j] -= sum((rational(jac[k, j]) * error[k] for k in range(3)), ZERO)
        residual = max(orthogonal, self.ledger.joint_error(self.ledger.solve(generalized)))
        for name, errors in node.items():
            for inverse, error in zip(self.models[name].inverse_mass, errors, strict=True):
                residual = max(
                    residual,
                    max(
                        (
                            rational(inverse) * abs(v) / rational(self.settings.reference_speed)
                            for v in error
                        ),
                        default=ZERO,
                    ),
                )
        return upper_float(residual)

    def _requests(self):
        world = self._world(self.eta)
        result = []
        nonzero = np.any(world != 0, axis=1)
        for name in self.node_domains:
            ids = [i for i, r in enumerate(self.rows) if r[0] == name and nonzero[i]]
            if ids:
                result.append(
                    PointVelocityImpulses(
                        name, np.array([self.rows[i][1] for i in ids]), world[ids]
                    )
                )
        ids = np.flatnonzero(nonzero)
        if len(ids):
            result.append(
                PointVelocityImpulses(
                    self.domain, self.indices[ids], -world[ids], self.terminal_local[ids]
                )
            )
        return tuple(result)

    def _relative_vectors(self, state):
        return np.array(
            [
                [
                    rational(state.domains[row[0]].velocity[row[1], k])
                    - dot(jac[k], state.domains[self.domain].v)
                    for k in range(3)
                ]
                for row, jac in zip(self.rows, self.j0, strict=True)
            ],
            dtype=object,
        ).reshape(-1, 3)

    def _friction_metrics(self, terminal):
        state = (
            self.trial.result.state
            if terminal
            else (self.held_trial.result.state if self.ready else self.trial.result.state)
        )
        world = self._world(self.eta if terminal else self.p)
        residual = dict(cone=0.0, dissipation=0.0, positive_work=0.0)
        reports, caps, works = [], [], []
        for i, (n, s, u) in enumerate(
            zip(self.normals, world, self._relative_vectors(state), strict=True)
        ):
            if terminal:
                c = (
                    capacity(n, s, self.mu[i])
                    if self.initial_closed[i]
                    else combined_capacity(n, self._world(self.p)[i], s, self.mu[i])
                )
            elif self.initial_closed[i]:
                c = capacity(n, s, self.mu[i])
            else:
                if np.any(self.t[i] != 0):
                    raise SolverFailure("initially open held tangent must be zero")
                c = (ZERO, ZERO)
            if terminal and not self.active[i] and np.any(s != 0):
                raise SolverFailure("open row has terminal friction")
            metrics, c, work = audit(n, s, u, c, self.kappa[i], self.settings.reference_speed)
            reports.append(metrics)
            caps.append(interval_float(c))
            works.append(exact_float(work))
            for key in residual:
                residual[key] = max(residual[key], metrics[key])
        return residual, np.array(caps).reshape(-1, 2), np.array(works), tuple(reports)

    def _position_metrics(self):
        # Terminal's velocity jump must never change the held friction audit.
        current = self.trial
        if self.ready:
            self.trial = self.held_trial
        try:
            metrics, gaps, bounds = super()._position_metrics()
            residual = dict(metrics.residuals)
            friction, _, _, _ = self._friction_metrics(False)
            residual.update({"held_" + k: v for k, v in friction.items()})
            return StageMetrics(residual, penetration=metrics.penetration), gaps, bounds
        finally:
            self.trial = current

    def position_sweep(self):
        self.ps += 1
        if self.pdone:
            return
        for i, k in enumerate(self.k):
            gaps = geometry(self, self.trial.result.state)[3]
            old = self.p[i]
            if old != 0 or gaps[i] < -float(self.bound):
                self.p[i] = max(0.0, old - self.program.relaxation * gaps[i] / (self.h * k))
                if self.p[i] != old:
                    self._evaluate()
            u = np.array(
                [exact_float(v) for v in self._relative_vectors(self.trial.result.state)[i]]
            )
            new = projected(
                self.t[i] - self.program.relaxation * u / self.kappa[i],
                self.normals[i],
                self.mu[i] * self.p[i] if self.initial_closed[i] else 0.0,
            )
            if not np.array_equal(new, self.t[i]):
                self.t[i] = new
                self._evaluate()

    def _refine(self, terminal):
        metrics = self._velocity_metrics() if terminal else self._position_metrics()[0]
        done, best, sweeps, budget = (
            (self.vdone, self.vbest, self.vs, self.settings.velocity_iterations)
            if terminal
            else (self.pdone, self.pbest, self.ps, self.settings.position_iterations)
        )
        if done:
            return metrics
        keys = (
            ("terminal_normal", "terminal_cone", "terminal_dissipation")
            if terminal
            else ("held_normal", "final_penetration", "held_cone", "held_dissipation")
        )
        score = max(metrics.residuals[k] for k in keys)
        if metrics.max_residual <= self.settings.tolerance and (best is None or score < best[0]):
            best = (
                score,
                (self.eta if terminal else self.p).copy(),
                (self.tau if terminal else self.t).copy(),
                self.trial,
                self.native_audit,
                self.material_reports,
            )
            if terminal:
                self.vbest = best
            else:
                self.pbest = best
            if score != 0 and sweeps < budget:
                return StageMetrics(
                    {
                        **metrics.residuals,
                        "iteration_refinement_pending": 2 * self.settings.tolerance,
                    },
                    penetration=metrics.penetration,
                )
        elif best is None:
            return metrics
        _, normal, tangent, self.trial, self.native_audit, self.material_reports = best
        if terminal:
            self.eta, self.tau, self.vdone = normal, tangent, True
            return self._velocity_metrics()
        self.p, self.t, self.pdone = normal, tangent, True
        return self._position_metrics()[0]

    def position_metrics(self):
        return self._refine(False)

    def velocity_sweep(self):
        if not self.ready:
            raise RuntimeError("velocity is not prepared")
        self.vs += 1
        if self.vdone:
            return
        for i, k in enumerate(self.k):
            if not self.active[i]:
                continue
            u = np.array(
                [exact_float(v) for v in self._relative_vectors(self.trial.result.state)[i]]
            )
            old = self.eta[i]
            un = float(self.normals[i] @ u)
            if old != 0 or un < -self.settings.tolerance * self.settings.reference_speed:
                self.eta[i] = max(0.0, old - self.program.relaxation * un / k)
                if self.eta[i] != old:
                    self._evaluate(terminal=True)
            u = np.array(
                [exact_float(v) for v in self._relative_vectors(self.trial.result.state)[i]]
            )
            c = self.mu[i] * (self.eta[i] + (0.0 if self.initial_closed[i] else self.p[i]))
            new = projected(
                self.tau[i] - self.program.relaxation * u / self.kappa[i], self.normals[i], c
            )
            if not np.array_equal(new, self.tau[i]):
                self.tau[i] = new
                self._evaluate(terminal=True)

    def _velocity_metrics(self):
        metrics = super()._velocity_metrics()
        residual = dict(metrics.residuals)
        friction, _, _, _ = self._friction_metrics(True)
        residual.update({"terminal_" + k: v for k, v in friction.items()})
        return StageMetrics(residual, penetration=metrics.penetration)

    def velocity_metrics(self):
        return self._refine(True)

    def result(self):
        if not self.ready:
            raise RuntimeError("dynamic friction not ready")
        fields = self.program._own_fields
        worlds = np.stack((self._world(self.p), self._world(self.eta)), axis=1)
        worlds[~self.active] = 0.0
        return SceneState(
            self.trial.result.state.domains,
            {
                **self.trial.result.state.history,
                fields[0].name: self.active.copy(),
                fields[1].name: self.initial_closed & self.active,
                fields[2].name: worlds,
            },
        )

    def diagnostics(self):
        inherited = super().diagnostics()
        d = dict(inherited["dynamic_robot_normal_contact"])
        for key in (
            "held_impulses",
            "terminal_impulses",
            "total_impulses",
            "held_world_impulses",
            "terminal_world_impulses",
        ):
            d.pop(key)
        hm, hc, hw, hr = self._friction_metrics(False)
        vm, vc, vw, vr = self._friction_metrics(True)
        d.update(
            law="dynamic-robot-friction-split-v1",
            initial_closed=self.initial_closed,
            active=self.active,
            held_normal=self.p,
            held_tangent=self.t,
            terminal_normal=self.eta,
            terminal_tangent=self.tau,
            held_world=self._world(self.p),
            terminal_world=self._world(self.eta),
            total_world=self._world(self.p) + self._world(self.eta),
            capacities_ns=np.stack((hc, vc), axis=1),
            held_capacities_ns=hc,
            terminal_capacities_ns=vc,
            held_tangential_work=hw,
            terminal_tangential_work=vw,
            held_friction_residuals=hr,
            terminal_friction_residuals=vr,
            pre_relative_velocity=np.asarray(
                self._relative_vectors(self.held_trial.result.state), float
            ),
            post_relative_velocity=np.asarray(
                self._relative_vectors(self.trial.result.state), float
            ),
        )
        return own_diagnostics(
            dict(
                dynamic_robot_friction_contact=d,
                execution={
                    **self.execution,
                    "backend": self.context.backend,
                    "compute_dtype": "float64" if self.context.backend == "numpy" else "float32",
                },
            )
        )
