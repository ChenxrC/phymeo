"""Native two-phase replay glue, with independent full-node aggregation."""

from fractions import Fraction as F

import numpy as np

from ..runtime.terminal_impulse import PointVelocityImpulses
from ..validation._material_path_math import q
from ._dynamic_friction_law import nonnegative_world
from ._finite_contact_replay import FiniteReplay
from ._robot_replay_ledger import upper_float

ZERO = F(0)


class ThickReplay(FiniteReplay):
    def _world(self, values, terminal=False):
        tangent = self.tau if terminal else self.t
        normals = self.n1 if terminal else self.n0
        return np.array(
            [
                nonnegative_world(n, n * p + t)
                for n, p, t in zip(normals, values, tangent, strict=True)
            ]
        ).reshape(-1, 3)

    def _node_values(self, world, terminal=False):
        blocks = self.j1 if terminal else self.j0
        result = {
            name: np.full((self.models[name].particle_count, 3), ZERO, object)
            for name in self.node_domains
        }
        for a, j, s in zip(self.anchors, blocks, world, strict=True):
            surface = self.program.pairs[a.pair][0]
            for i, node in enumerate(surface.node_indices):
                for k in range(3):
                    result[surface.domain][node, k] += sum(
                        (q(j[i, component, k]) * q(s[component]) for component in range(3)), ZERO
                    )
        return result

    def _node_terminal(self):
        return self._node_values(self._world(self.eta, True), True)

    def _generalized(self, world, terminal=False):
        jac = self.rj1 if terminal else self.rj0
        return [
            -sum(
                (q(j[k, d]) * q(s[k]) for j, s in zip(jac, world, strict=True) for k in range(3)),
                ZERO,
            )
            for d in range(self.model.dof_count)
        ]

    def _loads(self):
        world = self._world(self.p)
        node = self._node_values(world)
        generalized = self._generalized(world)
        loads = {self.domain: {"effort": np.array(generalized, float)}}
        loads.update({name: {"force": np.array(values, float)} for name, values in node.items()})
        return loads, generalized, node

    def _requests(self):
        world = self._world(self.eta, True)
        node = self._node_terminal()
        result = []
        for name, values in node.items():
            actual = np.array(values, float)
            ids = np.flatnonzero(np.any(actual != 0, axis=1))
            if len(ids):
                result.append(PointVelocityImpulses(name, ids, actual[ids]))
        ids = np.flatnonzero(np.any(world != 0, axis=1))
        if len(ids):
            result.append(
                PointVelocityImpulses(self.domain, self.indices[ids], -world[ids], self.local1[ids])
            )
        return tuple(result)

    def _evaluate(self, terminal=False):
        super()._evaluate(terminal)
        phase = self.mapping1 if terminal else self.mapping0
        blocks = self.j1 if terminal else self.j0
        state = self.held_trial.result.state if terminal else self.initial
        world = self._world(self.eta, True) if terminal else self._world(self.p)
        node = self._node_values(world, terminal)
        report = {}
        for name, (ids, ledger) in phase.items():
            current = ledger.require(
                blocks[ids],
                world[ids],
                np.array(node[name], float),
                state.domains[name].velocity,
                tolerance=self.settings.tolerance,
                speed=self.settings.reference_speed,
                length=self.settings.reference_length,
            )
            report.update({name + "/" + k: v for k, v in current.items()})
        self.mapping_report = report

    def _terminal_energy(self):
        pre, post = self.held_trial.result.state, self.trial.result.state
        change = self._kinetic(post) - self._kinetic(pre)
        node = self._node_terminal()
        generalized = self._generalized(self._world(self.eta, True), True)
        linear = sum(
            (p * q(v) for p, v in zip(generalized, pre.domains[self.domain].v, strict=True)), ZERO
        )
        quadratic = (
            sum(
                (
                    a * b
                    for a, b in zip(generalized, self.terminal_mass.solve(generalized), strict=True)
                ),
                ZERO,
            )
            / 2
        )
        for name, values in node.items():
            for i, p in enumerate(values):
                linear += sum(
                    (v * q(u) for v, u in zip(p, pre.domains[name].velocity[i], strict=True)), ZERO
                )
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
