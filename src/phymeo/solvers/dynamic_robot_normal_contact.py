"""Automatic all-node robot-plane contact with two independently audited stages."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..constraints.robot_plane_collider import RobotPlaneCollider
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.scene import HistoryField, SceneState, StageMetrics, _name, fingerprint
from ..dynamics.robot_kinematics import point_jacobians
from ..runtime.load_replay import SceneLoadReplay
from ..runtime.program import own_diagnostics
from ..runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram
from ..validation._material_path_math import interval_float
from ..validation.material_path import MaterialPathLedger
from ._dynamic_normal_geometry import (
    geometry,
    normal_projection_bounds,
    paired_moment,
    project_local,
    relative,
    relative_bounds,
)
from ._material_contact_balance import MaterialContactBalance
from ._robot_replay_ledger import FrozenRobotLedger, dot, exact_float, rational, upper_float
from .initial_robot_material_contact import MaterialContactBudgets
from .particle_plane_contact import _BorrowedProgram
from .robot_plane_contact import RobotPlaneContactProgram

ZERO = Fraction(0)


class DynamicRobotNormalContactProgram(RobotPlaneContactProgram):
    """Normal-only infinite planes, all-prismatic robot and dynamic distance nodes."""

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
        history_prefix="normal_contact",
    ):
        super().__init__(
            scene,
            base_program,
            (),
            robot_domain=robot_domain,
            registry=registry,
            inner_settings=inner_settings,
            relaxation=relaxation,
        )
        if "terminal_point_impulses" not in base_program.capabilities.features:
            raise ValueError("base lacks terminal_point_impulses capability")
        if any(kind != "prismatic" for kind in self.model.joint_types):
            raise ValueError("dynamic normal contact requires the entire robot all-prismatic")
        if self.model.mass.dtype != np.float64:
            raise TypeError("dynamic normal contact requires FP64 robot storage")
        colliders = tuple(colliders)
        if not colliders or any(not isinstance(c, RobotPlaneCollider) for c in colliders):
            raise ValueError("at least one RobotPlaneCollider is required")
        if len({c.key for c in colliders}) != len(colliders):
            raise ValueError("collider keys must be unique")
        if any(c.link_index >= self.model.dof_count for c in colliders):
            raise ValueError("collider link outside robot")
        self.colliders = tuple(sorted((replace(c) for c in colliders), key=lambda c: c.key))
        if not isinstance(materials, Mapping) or not materials:
            raise ValueError("materials must explicitly select at least one node domain")
        bindings = {d.name: d for d in scene.domains}
        owned = {}
        for name, material in sorted(materials.items()):
            if name not in bindings or bindings[name].family != "particles":
                raise ValueError("materials must select registered particle domains")
            model = bindings[name].model
            if not isinstance(model, ParticleModel) or model.inverse_mass.dtype != np.float64:
                raise TypeError("selected nodes require FP64 ParticleModel")
            if np.any(model.inverse_mass <= 0):
                raise ValueError("selected nodes must all be dynamic")
            family = self.registry.get("particles")
            if (
                not {"position", "velocity"} <= set(family.state_fields)
                or "force" not in family.input_fields
            ):
                raise ValueError("selected node fields are not registered")
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
        self._own_fields = (
            HistoryField(history_prefix + "/active", (len(self.row_keys),), bool),
            HistoryField(history_prefix + "/phase_impulses", (len(self.row_keys), 2), np.float64),
        )
        if any(
            f.name == history_prefix or f.name.startswith(history_prefix + "/")
            for f in base_program.history_fields
        ):
            raise ValueError("contact history prefix conflicts with base history")

    @property
    def capabilities(self):
        return replace(
            self.base.capabilities,
            history="explicit",
            dtypes=frozenset({"float64"}),
            features=frozenset({"dynamic_robot_normal_split"}),
        )

    @property
    def history_fields(self):
        return tuple(self.base.history_fields) + self._own_fields

    def initial_history(self):
        return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in self._own_fields}

    def problem_signature(self):
        result = super().problem_signature()
        result.update(
            law="dynamic-robot-normal-split-v1",
            colliders=self.colliders,
            materials=self.materials,
            budgets=self.budgets,
            history=tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields),
        )
        return result

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if self._closed:
            raise RuntimeError("dynamic normal program is closed")
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
            raise ValueError("inner tolerance and reference scales must match outer")
        return _DynamicNormalWorkspace(self, state, inputs, h, settings, inner, context)


class _DynamicNormalWorkspace:
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
        old = state.history[program._own_fields[1].name]
        if np.any(~np.isfinite(old)) or np.any(old < 0) or np.any(old[~self.old_active] != 0):
            raise SolverFailure("invalid or dirty contact impulse history")
        if any(
            active and lo > self.bound
            for active, (lo, _) in zip(self.old_active, initial_bounds, strict=True)
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

    def _world(self, values):
        return self.normals * values[:, None]

    def _world_mapping(self, values):
        """Audit evaluated n*p against the original products in speed units."""
        node = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        generalized = [ZERO] * self.model.dof_count
        for row, jac, n, p, world in zip(
            self.rows, self.j0, self.normals, values, self._world(values), strict=True
        ):
            error = [rational(world[k]) - rational(n[k]) * rational(p) for k in range(3)]
            for k in range(3):
                node[row[0]][row[1]][k] += error[k]
            for j in range(self.model.dof_count):
                generalized[j] -= sum((rational(jac[k, j]) * error[k] for k in range(3)), ZERO)
        residual = self.ledger.joint_error(self.ledger.solve(generalized))
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

    def _loads(self):
        node = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        generalized = [ZERO] * self.model.dof_count
        for row, jac, p in zip(self.rows, self.j0, self._world(self.p), strict=True):
            for k in range(3):
                node[row[0]][row[1]][k] += rational(p[k])
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], p)
        loads = {self.domain: {"effort": np.array([exact_float(v) for v in generalized])}}
        loads.update(
            {
                name: {
                    "force": np.array([[exact_float(v) for v in row] for row in values]).reshape(
                        -1, 3
                    )
                }
                for name, values in node.items()
            }
        )
        return loads, generalized, node

    def _requests(self):
        world = self._world(self.eta)
        result = []
        for name in self.node_domains:
            ids = [i for i, r in enumerate(self.rows) if r[0] == name and self.eta[i] != 0]
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
        adapter = TerminalImpulseProgram(
            _BorrowedProgram(self.program.base),
            self._requests() if terminal else (),
            registry=self.program.registry,
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
            try:
                trial = replay.evaluate(loads)
            except (TypeError, ValueError) as error:
                raise SolverFailure(f"invalid native dynamic replay: {error}") from error
        if not trial.result.report.accepted:
            raise SolverFailure("native dynamic replay failed: " + str(trial.result.report.reason))
        sub = trial.result.report.substeps[0]
        evidence = sub.diagnostics["terminal_impulse"]
        pre = evidence["pre_impulse_state"]
        self._require_compute_storage(pre)
        self._require_compute_storage(trial.result.state)
        audit = self.ledger.trial_audit(
            pre.domains[self.domain], trial.effective_inputs.domains[self.domain], generalized
        )
        if max(audit.values()) > rational(self.settings.tolerance):
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
            if fingerprint(pre) != fingerprint(selected["pre_impulse_state"]):
                raise SolverFailure("terminal replay changed held pre-state or base history")
            if fingerprint(evidence["pre_impulse_diagnostics"]) != fingerprint(
                selected["pre_impulse_diagnostics"]
            ):
                raise SolverFailure("terminal replay changed complete pre-native diagnostics")
            for name in self.node_domains:
                a = evidence["pre_impulse_diagnostics"]["participants"][name]["material_path"]
                b = selected["pre_impulse_diagnostics"]["participants"][name]["material_path"]
                if fingerprint(a) != fingerprint(b):
                    raise SolverFailure("terminal replay changed native material path")
            if (
                evidence["pre_position_iterations"],
                evidence["pre_velocity_iterations"],
                fingerprint(evidence["pre_position_metrics"]),
                fingerprint(evidence["pre_velocity_metrics"]),
            ) != (
                selected["pre_position_iterations"],
                selected["pre_velocity_iterations"],
                fingerprint(selected["pre_position_metrics"]),
                fingerprint(selected["pre_velocity_metrics"]),
            ):
                raise SolverFailure("terminal replay changed native stage convergence")
            self.terminal_replays += 1
        else:
            self.held_replays += 1
        for key in self.execution:
            self.execution[key] += sub.diagnostics.get("execution", {}).get(key, 0)
        self.trial, self.native_audit, self.material_reports = trial, audit, reports

    def predict(self):
        if self.trial is not None:
            raise RuntimeError("dynamic contact already predicted")
        self._evaluate()

    def _position_metrics(self):
        state = self.trial.result.state
        _, jac, normals, gaps, bounds = geometry(self, state)
        if not np.array_equal(jac, self.j0) or not np.array_equal(normals, self.normals):
            raise SolverFailure("all-prismatic contact geometry changed Jacobian or normal")
        penetration = max((max(-lo, ZERO) for lo, _ in bounds), default=ZERO)
        residual = {k: upper_float(rational(v)) for k, v in self.native_audit.items()}
        residual.update(
            native=self.trial.result.report.max_residual,
            final_penetration=upper_float(penetration / self.exact_gap_scale),
            held_normal=0.0,
        )
        residual["held_direction_mapping"] = self._world_mapping(self.p)
        for k, p, n, world, (lo, hi) in zip(
            self.k, self.p, self.normals, self._world(self.p), bounds, strict=True
        ):
            if p < 0:
                raise SolverFailure("negative held impulse")
            projected = normal_projection_bounds(n, world)
            err = max(
                abs(min(rational(k) * impulse, g / rational(self.h)))
                for g in (lo, hi)
                for impulse in projected
            )
            residual["held_normal"] = max(
                residual["held_normal"], upper_float(err / rational(self.settings.reference_speed))
            )
        self.held_moment, moment_residual = paired_moment(
            self, self.initial, self.local, self._world(self.p)
        )
        residual["held_paired_moment"] = upper_float(moment_residual)
        return StageMetrics(residual, penetration=upper_float(penetration)), gaps, bounds

    def position_sweep(self):
        self.ps += 1
        if self.pdone:
            return
        for i, k in enumerate(self.k):
            _, gaps, _ = self._position_metrics()
            old = self.p[i]
            # A zero-load already-admissible free path stays genuinely unloaded.
            if old == 0 and gaps[i] >= -float(self.bound):
                continue
            self.p[i] = max(0.0, old - self.program.relaxation * gaps[i] / (self.h * k))
            if self.p[i] != old:
                self._evaluate()

    def position_metrics(self):
        metrics, _, _ = self._position_metrics()
        if self.pdone:
            return metrics
        score = max(metrics.residuals["held_normal"], metrics.residuals["final_penetration"])
        if metrics.max_residual <= self.settings.tolerance:
            if self.pbest is None or score < self.pbest[0]:
                self.pbest = (
                    score,
                    self.p.copy(),
                    self.trial,
                    self.native_audit,
                    self.material_reports,
                )
                if score == 0 or self.ps >= self.settings.position_iterations:
                    self.pdone = True
                    return metrics
            else:
                _, self.p, self.trial, self.native_audit, self.material_reports = self.pbest
                self.pdone = True
                return self._position_metrics()[0]
            return StageMetrics(
                {**metrics.residuals, "iteration_refinement_pending": 2 * self.settings.tolerance},
                penetration=metrics.penetration,
            )
        if self.pbest is not None:
            _, self.p, self.trial, self.native_audit, self.material_reports = self.pbest
            self.pdone = True
            return self._position_metrics()[0]
        return metrics

    def prepare_velocity(self):
        if self.ready or self._position_metrics()[0].max_residual > self.settings.tolerance:
            raise SolverFailure("dynamic held stage not converged")
        self.held_trial = self.trial
        _, _, bounds = self._position_metrics()
        closed = []
        for p, (lo, hi) in zip(self.p, bounds, strict=True):
            if lo >= -self.bound and hi <= self.bound:
                closed.append(True)
            elif lo > self.bound:
                closed.append(False)
            else:
                raise SolverFailure("uncertain plane closure interval")
            if p > 0 and not closed[-1]:
                raise SolverFailure("positive held impulse on open row")
        self.terminal_local = project_local(self, self.held_trial.result.state)
        terminal_jac = point_jacobians(
            self.model,
            self.held_trial.result.state.domains[self.domain].q,
            self.indices,
            self.terminal_local,
        )[0]
        if not np.array_equal(terminal_jac, self.j0):
            raise SolverFailure(
                "current terminal Jacobian differs from all-prismatic held Jacobian"
            )
        self.active = np.array(closed, dtype=bool)
        self.ready = True
        self._evaluate(terminal=True)

    def velocity_sweep(self):
        if not self.ready:
            raise RuntimeError("velocity is not prepared")
        self.vs += 1
        if self.vdone:
            return
        for i, k in enumerate(self.k):
            if not self.active[i]:
                continue
            u = relative(self, self.trial.result.state)[i]
            old = self.eta[i]
            if old == 0 and u >= -rational(self.settings.tolerance) * rational(
                self.settings.reference_speed
            ):
                continue
            self.eta[i] = max(0.0, old - self.program.relaxation * exact_float(u) / k)
            if old != self.eta[i]:
                self._evaluate(terminal=True)

    def _velocity_metrics(self):
        metrics, _, _ = self._position_metrics()
        residual = dict(metrics.residuals)
        residual["terminal_normal"] = 0.0
        residual["terminal_direction_mapping"] = self._world_mapping(self.eta)
        for active, k, eta, n, world, u_bounds in zip(
            self.active,
            self.k,
            self.eta,
            self.normals,
            self._world(self.eta),
            relative_bounds(self, self.trial.result.state),
            strict=True,
        ):
            if eta < 0 or (not active and eta != 0):
                raise SolverFailure("invalid terminal impulse")
            if active:
                impulses = normal_projection_bounds(n, world)
                err = max(
                    max(max(-u, ZERO), abs(min(rational(k) * impulse, u)))
                    for u in u_bounds
                    for impulse in impulses
                )
                residual["terminal_normal"] = max(
                    residual["terminal_normal"],
                    upper_float(err / rational(self.settings.reference_speed)),
                )
        physical, self.physical_report, _ = self.balance.require_metrics(
            self.trial.result.state,
            self.ledger,
            self._loads()[2],
            self.settings.tolerance,
            terminal=self._node_terminal(),
        )
        self.terminal_moment, moment_residual = paired_moment(
            self, self.held_trial.result.state, self.terminal_local, self._world(self.eta)
        )
        residual["terminal_paired_moment"] = upper_float(moment_residual)
        residual.update(physical)
        residual.update(self._terminal_energy())
        return StageMetrics(residual, penetration=metrics.penetration)

    def _node_terminal(self):
        result = {
            name: [[ZERO] * 3 for _ in range(self.models[name].particle_count)]
            for name in self.node_domains
        }
        for row, p in zip(self.rows, self._world(self.eta), strict=True):
            for k in range(3):
                result[row[0]][row[1]][k] += rational(p[k])
        return result

    def _kinetic(self, state):
        return self.ledger.kinetic(state.domains[self.domain].v) + sum(
            (
                sum(
                    (
                        rational(v) ** 2 / (2 * rational(w))
                        for row, w in zip(
                            state.domains[name].velocity,
                            self.models[name].inverse_mass,
                            strict=True,
                        )
                        for v in row
                    ),
                    ZERO,
                )
                for name in self.node_domains
            ),
            ZERO,
        )

    def _terminal_energy(self):
        pre, post = self.held_trial.result.state, self.trial.result.state
        change = self._kinetic(post) - self._kinetic(pre)
        linear = ZERO
        generalized = [ZERO] * self.model.dof_count
        for row, jac, p in zip(self.rows, self.j0, self._world(self.eta), strict=True):
            node_velocity = pre.domains[row[0]].velocity[row[1]]
            relative_velocity = [
                rational(node_velocity[k]) - dot(jac[k], pre.domains[self.domain].v)
                for k in range(3)
            ]
            linear += sum((rational(p[k]) * relative_velocity[k] for k in range(3)), ZERO)
            for j in range(self.model.dof_count):
                generalized[j] -= dot(jac[:, j], p)
        response = self.ledger.solve(generalized)
        quadratic = sum((p * v for p, v in zip(generalized, response, strict=True)), ZERO) / 2
        for name, values in self._node_terminal().items():
            for i, p in enumerate(values):
                quadratic += (
                    rational(self.models[name].inverse_mass[i]) * sum((v * v for v in p), ZERO) / 2
                )
        defect = abs(change - linear - quadratic)
        limit = rational(self.program.budgets.max_energy_gain_j)
        self.terminal_energy = dict(
            change_j=exact_float(change),
            linear_work_j=exact_float(linear),
            quadratic_response_j=exact_float(quadratic),
            identity_defect_j=exact_float(defect),
            limit_j=float(limit),
        )
        return dict(
            terminal_energy=upper_float(
                rational(self.settings.tolerance) * max(max(change, ZERO), defect) / limit
            )
        )

    def velocity_metrics(self):
        metrics = self._velocity_metrics()
        if self.vdone:
            return metrics
        score = metrics.residuals["terminal_normal"]
        if metrics.max_residual <= self.settings.tolerance:
            if self.vbest is None or score < self.vbest[0]:
                self.vbest = (
                    score,
                    self.eta.copy(),
                    self.trial,
                    self.native_audit,
                    self.material_reports,
                )
                if score == 0 or self.vs >= self.settings.velocity_iterations:
                    self.vdone = True
                    return metrics
            else:
                _, self.eta, self.trial, self.native_audit, self.material_reports = self.vbest
                self.vdone = True
                return self._velocity_metrics()
            return StageMetrics(
                {**metrics.residuals, "iteration_refinement_pending": 2 * self.settings.tolerance},
                penetration=metrics.penetration,
            )
        if self.vbest is not None:
            _, self.eta, self.trial, self.native_audit, self.material_reports = self.vbest
            self.vdone = True
            return self._velocity_metrics()
        return metrics

    def result(self):
        if not self.ready:
            raise RuntimeError("dynamic contact not ready")
        state = self.trial.result.state
        phases = np.column_stack((self.p, self.eta))
        phases[~self.active] = 0.0
        return SceneState(
            state.domains,
            {
                **state.history,
                self.program._own_fields[0].name: self.active.copy(),
                self.program._own_fields[1].name: phases,
            },
        )

    def diagnostics(self):
        self._velocity_metrics()
        final_geometry = geometry(self, self.trial.result.state)

        def normal_values(state):
            return np.array([exact_float((lo + hi) / 2) for lo, hi in relative_bounds(self, state)])

        events = tuple(
            "persist"
            if before and after
            else "exit"
            if before
            else "enter"
            if after
            else "inactive"
            for before, after in zip(self.old_active, self.active, strict=True)
        )
        return own_diagnostics(
            dict(
                dynamic_robot_normal_contact=dict(
                    row_keys=self.program.row_keys,
                    held_impulses=self.p,
                    terminal_impulses=self.eta,
                    total_impulses=self.p + self.eta,
                    held_world_impulses=self._world(self.p),
                    terminal_world_impulses=self._world(self.eta),
                    initial_gaps=self.gaps0,
                    final_gaps=final_geometry[3],
                    initial_gap_bounds=np.array(
                        [interval_float(b) for b in self.initial_bounds]
                    ).reshape(-1, 2),
                    final_gap_bounds=np.array(
                        [interval_float(b) for b in final_geometry[4]]
                    ).reshape(-1, 2),
                    initial_relative_normal_velocity=normal_values(self.initial),
                    held_relative_normal_velocity=normal_values(self.held_trial.result.state),
                    relative_normal_velocity=normal_values(self.trial.result.state),
                    events=events,
                    held_replays=self.held_replays,
                    terminal_replays=self.terminal_replays,
                    held_state=self.held_trial.result.state,
                    terminal_evidence=self.trial.result.report.substeps[0].diagnostics[
                        "terminal_impulse"
                    ],
                    native_audit={
                        k: upper_float(rational(v)) for k, v in self.native_audit.items()
                    },
                    material_reports=self.material_reports,
                    physical_budgets=self.physical_report,
                    terminal_energy=self.terminal_energy,
                    held_paired_moment_nms=exact_float(self.held_moment),
                    terminal_paired_moment_nms=exact_float(self.terminal_moment),
                    input_integrals=self.trial.integrated_loads,
                    native_backend=self.program.capabilities.backend,
                    coefficient_backend="canonical_host_float64",
                ),
                execution=self.execution,
            )
        )

    def close(self):
        pass
