"""Isolated finite-contact current-configuration impulse equations."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.robot import RobotModel
from ..core.scene import BackendContext, StepSettings
from ..coupling.ports import ParticlePointPort
from ..coupling.robot_port import RobotPointPort
from ..dynamics.robot_dynamics import mass_matrix
from ..dynamics.robot_kinematics import point_jacobians
from ..geometry._patch_interval import validate_domain
from ..runtime.program import own_diagnostics
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.terminal_impulse import _batch_requests, _own_requests
from ..solvers._robot_replay_ledger import rational, upper_float
from .terminal_impulse import _compute, _PositiveMass, _same


class FiniteContactTerminalLedger:
    """Audit original requests in speed units; never derive a replacement state.

    The current host mass/Jacobians support the bounded finite-contact domain. Device
    candidates must additionally be exact FP32 promotions in FP64 storage.
    """

    def __init__(
        self,
        scene,
        before,
        requests,
        settings,
        context=BackendContext(),
        *,
        reference_angular_speed=1.0,
        registry=None,
        history_fields=(),
    ):
        if not isinstance(settings, StepSettings) or settings.tolerance <= 0:
            raise ValueError("terminal ledger requires positive StepSettings tolerance")
        if (
            not isinstance(context, BackendContext)
            or context.dtype != np.float64
            or context.backend not in ("numpy", "tyx:interpret", "tyx:cuda", "tyx:cpu_cpp")
        ):
            raise ValueError("unsupported terminal ledger context/precision")
        if registry is None:
            registry = StateRegistry()
            registry.register(robot_family())
        if not isinstance(registry, StateRegistry):
            raise TypeError("terminal ledger requires StateRegistry")
        if (
            type(reference_angular_speed) is not float
            or not np.isfinite(reference_angular_speed)
            or reference_angular_speed <= 0
        ):
            raise ValueError("reference_angular_speed must be a finite positive builtin float")
        self._angular_speed = rational(reference_angular_speed)
        self._registry = registry.copy()
        self._scene, self._context, self._schema = scene, context, tuple(history_fields)
        self._before = own_diagnostics(
            self._registry.clone_state(scene, before, context, self._schema)
        )
        self.requests = _own_requests(requests)
        self._speed, self._tol = rational(settings.reference_speed), rational(settings.tolerance)
        self._models, self._mass, self._expected, self._mapped = {}, {}, {}, {}
        bindings = {binding.name: binding for binding in scene.domains}
        for request in self.requests:
            if request.domain not in bindings:
                raise ValueError("unknown terminal impulse domain")
            binding = bindings[request.domain]
            model, state = binding.model, self._before.domains[request.domain]
            if binding.family == "particles" and isinstance(model, ParticleModel):
                if model.inverse_mass.dtype != np.float64 or np.any(model.inverse_mass <= 0):
                    raise ValueError("terminal particle domain must be FP64 and fully dynamic")
                ParticlePointPort(model, state, request.indices, domain=request.domain)
                if request.local_points is not None and np.any(request.local_points != 0):
                    raise ValueError("particle points cannot have local offsets")
            elif binding.family == "robot" and isinstance(model, RobotModel):
                validate_domain(model)
                if model.joint_types == ("revolute",) and np.any(np.abs(state.q) > 4):
                    raise ValueError("rotor q must remain in [-4,4]")
                RobotPointPort(
                    model, state, request.indices, request.local_points, domain=request.domain
                )
            else:
                raise ValueError("unsupported requested terminal state family")
            if request.domain not in self._models:
                model = replace(model)
                self._models[request.domain] = model
                count = (
                    model.particle_count * 3 if binding.family == "particles" else model.dof_count
                )
                self._expected[request.domain] = [F(0)] * count
                self._mapped[request.domain] = [F(0)] * count
                if binding.family == "robot":
                    raw = mass_matrix(model, state.q)
                    self._mass[request.domain] = _PositiveMass(raw)
                    if model.joint_types == ("revolute",):
                        axis = int(np.flatnonzero(model.joint_axis[0])[0])
                        inertia = rational(model.inertia_diagonal[0, axis])
                        if abs(rational(raw[0, 0]) - inertia) / inertia > self._tol:
                            raise SolverFailure(
                                "current rotor mass differs from analytic constant inertia"
                            )
            self._accumulate(request, request.impulses, self._expected[request.domain])
        for batch in _batch_requests(self.requests):
            values = (
                batch.impulses
                if context.backend == "numpy"
                else _compute(batch.impulses, input_value=True)
            )
            self._accumulate(batch, values, self._mapped[batch.domain])

    def _accumulate(self, request, values, target):
        model = self._models[request.domain]
        if isinstance(model, ParticleModel):
            for index, row in zip(request.indices, values, strict=True):
                for k in range(3):
                    target[int(index) * 3 + k] += rational(row[k])
        else:
            state = self._before.domains[request.domain]
            jac = point_jacobians(model, state.q, request.indices, request.local_points)[0]
            for i in range(len(values)):
                for j in range(model.dof_count):
                    target[j] += sum(
                        (rational(jac[i, k, j]) * rational(values[i, k]) for k in range(3)), F(0)
                    )

    def require(self, after):
        try:
            return self._require(after)
        except SolverFailure:
            raise
        except (
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            ArithmeticError,
        ) as error:
            raise SolverFailure(f"invalid terminal impulse evidence: {error}") from error

    def _require(self, after):
        candidate = self._registry.validate_candidate(
            self._scene, self._before, after, self._context, self._schema
        )
        if self._context.backend != "numpy":
            for binding in self._scene.domains:
                for state in (self._before, candidate):
                    part = state.domains[binding.name]
                    for field in (
                        ("q", "v") if binding.family == "robot" else ("position", "velocity")
                    ):
                        value = getattr(part, field)
                        if not np.array_equal(value, _compute(value)):
                            raise SolverFailure("terminal native state is not stored FP32")
        for name in self._before.history:
            if not _same(self._before.history[name], candidate.history[name]):
                raise SolverFailure("terminal impulse changed base history")
        result = {}
        for binding in self._scene.domains:
            name = binding.name
            before, post = self._before.domains[name], candidate.domains[name]
            family = self._registry.get(binding.family)
            if name not in self._models:
                if any(
                    not _same(getattr(before, f), getattr(post, f)) for f in family.state_fields
                ):
                    raise SolverFailure("terminal impulse changed an unrequested domain")
                continue
            model = self._models[name]
            particle = isinstance(model, ParticleModel)
            config, velocity = ("position", "velocity") if particle else ("q", "v")
            if not _same(getattr(before, config), getattr(post, config)):
                raise SolverFailure("terminal impulse changed configuration")
            v0, v1 = getattr(before, velocity), getattr(post, velocity)
            if self._context.backend != "numpy" and not np.array_equal(v1, _compute(v1)):
                raise SolverFailure("terminal device velocity is not stored FP32")
            dv = [rational(b) - rational(a) for a, b in zip(v0.flat, v1.flat, strict=True)]
            expected, mapped = self._expected[name], self._mapped[name]
            if particle:
                weights = [rational(w) for w in np.repeat(model.inverse_mass, 3)]
                error = [d - w * p for d, w, p in zip(dv, weights, expected, strict=True)]
                mapping = [w * (m - p) for w, m, p in zip(weights, mapped, expected, strict=True)]
            else:
                mass = self._mass[name]
                residual = [
                    sum((mass.matrix[i][j] * dv[j] for j in range(mass.n)), F(0)) - expected[i]
                    for i in range(mass.n)
                ]
                error = mass.solve(residual)
                mapping = mass.solve([m - p for m, p in zip(mapped, expected, strict=True)])
            scales = (
                [self._speed] * len(error)
                if particle
                else [
                    self._angular_speed if kind == "revolute" else self._speed
                    for kind in model.joint_types
                ]
            )
            errors = {
                "velocity": max(
                    (abs(e) / s for e, s in zip(error, scales, strict=True)), default=F(0)
                ),
                "input_mapping": max(
                    (abs(e) / s for e, s in zip(mapping, scales, strict=True)), default=F(0)
                ),
            }
            for kind, value in errors.items():
                if value > self._tol:
                    raise SolverFailure(f"terminal {name} {kind} equation exceeds speed tolerance")
            result[name] = {
                **{key: upper_float(value) for key, value in errors.items()},
                "configuration_preserved": True,
            }
        maximum = max(
            (max(value["velocity"], value["input_mapping"]) for value in result.values()),
            default=0.0,
        )
        return own_diagnostics(
            dict(
                max_residual=maximum,
                by_domain=result,
                configuration_preserved=True,
                history_preserved=True,
                compute_precision="float64" if self._context.backend == "numpy" else "float32",
                requests=self.requests,
            )
        )
