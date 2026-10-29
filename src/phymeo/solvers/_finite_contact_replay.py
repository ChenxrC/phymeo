"""Complete replay operations and independent phase response audits."""

from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import fingerprint
from ..runtime.finite_contact_terminal import FiniteContactTerminalProgram
from ..runtime.load_replay import SceneLoadReplay
from ..runtime.terminal_impulse import PointVelocityImpulses
from ._robot_replay_ledger import exact_float, rational, upper_float
from .particle_plane_contact import _BorrowedProgram

ZERO = F(0)


def q(value):
    return value if isinstance(value, F) else rational(value)


def dot(a, b):
    return sum((q(x) * q(y) for x, y in zip(a, b, strict=True)), ZERO)


class FiniteReplay:
    def _world(self, values, terminal=False):
        normals = self.terminal_normals if terminal else self.normals
        return normals * values[:, None]

    def _node_values(self, world):
        result = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        for (name, i, _), p in zip(self.rows, world, strict=True):
            for k in range(3):
                result[name][i][k] += q(p[k])
        return result

    def _node_terminal(self):
        return self._node_values(self._world(self.eta, True))

    def _loads(self):
        node = self._node_values(self._world(self.p))
        generalized = [ZERO] * self.model.dof_count
        for jac, p in zip(self.j0, self._world(self.p), strict=True):
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], p)
        loads = {self.domain: {"effort": np.array([exact_float(v) for v in generalized])}}
        loads.update(
            {
                name: {"force": np.array(values, dtype=float).reshape(-1, 3)}
                for name, values in node.items()
            }
        )
        return loads, generalized, node

    def _requests(self):
        world = self._world(self.eta, True)
        result = []
        for name in self.node_domains:
            ids = [i for i, row in enumerate(self.rows) if row[0] == name and self.eta[i] != 0]
            if ids:
                result.append(
                    PointVelocityImpulses(
                        name, np.array([self.rows[i][1] for i in ids]), world[ids]
                    )
                )
        ids = np.flatnonzero(self.eta)
        if len(ids):
            result.append(
                PointVelocityImpulses(
                    self.domain, self.indices[ids], -world[ids], self.terminal_local[ids]
                )
            )
        return tuple(result)

    def _require_compute_storage(self, state):
        if self.context.backend == "numpy":
            return
        for name, fields in (
            (self.domain, ("q", "v")),
            *((name, ("position", "velocity")) for name in self.node_domains),
        ):
            for field in fields:
                value = getattr(state.domains[name], field)
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    promoted = value.astype(np.float32).astype(np.float64)
                if not np.isfinite(promoted).all() or not np.array_equal(value, promoted):
                    raise SolverFailure(f"native {name}.{field} is not stored FP32 output")

    def _evaluate(self, terminal=False):
        loads, generalized, node = self._loads()
        adapter = FiniteContactTerminalProgram(
            _BorrowedProgram(self.program.base),
            self._requests() if terminal else (),
            registry=self.program.registry,
            reference_angular_speed=self.program.reference_angular_speed,
        )
        with SceneLoadReplay(
            self.program.scene,
            adapter,
            self.base_state,
            self.inputs,
            self.h,
            channels=self.channels,
            settings=self.inner,
            context=self.context,
            registry=self.program.registry,
        ) as replay:
            trial = replay.evaluate(loads)
        if not trial.result.report.accepted:
            raise SolverFailure("native finite replay failed: " + str(trial.result.report.reason))
        sub = trial.result.report.substeps[0]
        evidence = sub.diagnostics["terminal_impulse"]
        pre = evidence["pre_impulse_state"]
        self._require_compute_storage(pre)
        self._require_compute_storage(trial.result.state)
        audit = self.ledger.trial_audit(
            pre.domains[self.domain], trial.effective_inputs.domains[self.domain], generalized
        )
        if max(audit.values()) > q(self.settings.tolerance):
            raise SolverFailure("native held robot audit exceeds tolerance")
        reports = {}
        for name, ledger in self.node_ledgers.items():
            reports[name] = ledger.require(
                pre.domains[name],
                trial.effective_inputs.domains[name],
                np.asarray(node[name], dtype=object).reshape(-1, 3),
                evidence["pre_impulse_diagnostics"].get("participants", {}).get(name),
                sweeps=evidence["pre_position_iterations"],
            )
            audit["material_" + name] = reports[name]["max_residual"]
        if terminal:
            selected = self.held_trial.result.report.substeps[0].diagnostics["terminal_impulse"]
            for key in (
                "pre_impulse_state",
                "pre_impulse_diagnostics",
                "pre_position_iterations",
                "pre_velocity_iterations",
                "pre_position_metrics",
                "pre_velocity_metrics",
            ):
                if fingerprint(evidence[key]) != fingerprint(selected[key]):
                    raise SolverFailure(
                        "terminal replay changed complete held pre evidence: " + key
                    )
            if fingerprint(trial.effective_inputs) != fingerprint(self.held_trial.effective_inputs):
                raise SolverFailure("terminal replay changed original complete inputs")
            self.terminal_replays += 1
        else:
            self.held_replays += 1
        for key in self.execution:
            self.execution[key] += sub.diagnostics.get("execution", {}).get(key, 0)
        self.trial, self.native_audit, self.material_reports = trial, audit, reports

    def _world_mapping(self, values, terminal=False):
        normals = self.terminal_normals if terminal else self.normals
        jacobians = self.j1 if terminal else self.j0
        generalized = [ZERO] * self.model.dof_count
        node = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        for (name, i, _), jac, n, p, world in zip(
            self.rows, jacobians, normals, values, self._world(values, terminal), strict=True
        ):
            errors = [q(world[k]) - q(n[k]) * q(p) for k in range(3)]
            for k in range(3):
                node[name][i][k] += errors[k]
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], errors)
        response = (
            self.terminal_mass.solve(generalized) if terminal else self.ledger.solve(generalized)
        )
        residual = self.ledger.joint_error(response)
        for name, values in node.items():
            for inverse, errors in zip(self.models[name].inverse_mass, values, strict=True):
                residual = max(
                    residual,
                    max(
                        (q(inverse) * abs(v) / q(self.settings.reference_speed) for v in errors),
                        default=ZERO,
                    ),
                )
        return upper_float(residual)

    def _kinetic(self, state):
        result = self.ledger.kinetic(state.domains[self.domain].v)
        for name in self.node_domains:
            for row, inverse in zip(
                state.domains[name].velocity, self.models[name].inverse_mass, strict=True
            ):
                result += sum((q(v) ** 2 / (2 * q(inverse)) for v in row), ZERO)
        return result

    def _terminal_energy(self):
        pre, post = self.held_trial.result.state, self.trial.result.state
        change = self._kinetic(post) - self._kinetic(pre)
        linear = ZERO
        generalized = [ZERO] * self.model.dof_count
        for (name, i, _), jac, p in zip(
            self.rows, self.j1, self._world(self.eta, True), strict=True
        ):
            relative = [
                q(pre.domains[name].velocity[i, k]) - dot(jac[k], pre.domains[self.domain].v)
                for k in range(3)
            ]
            linear += dot(p, relative)
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], p)
        response = self.terminal_mass.solve(generalized)
        quadratic = dot(generalized, response) / 2
        for name, values in self._node_terminal().items():
            for i, p in enumerate(values):
                quadratic += (
                    q(self.models[name].inverse_mass[i]) * sum((v * v for v in p), ZERO) / 2
                )
        defect = abs(change - linear - quadratic)
        limit = q(self.program.budgets.max_energy_gain_j)
        self.terminal_energy = dict(
            change_j=float(change),
            linear_work_j=float(linear),
            quadratic_response_j=float(quadratic),
            identity_defect_j=float(defect),
            limit_j=float(limit),
        )
        return {
            "terminal_energy": upper_float(
                q(self.settings.tolerance) * max(max(change, ZERO), defect) / limit
            )
        }
