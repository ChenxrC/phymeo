"""Owned requests and live native terminal velocity impulses."""

from dataclasses import dataclass

import numpy as np

from ..core._arrays import float_array
from ..core.scene import _name


@dataclass(frozen=True, eq=False)
class PointVelocityImpulses:
    """A point batch in world N s, with optional local attachment positions."""

    domain: str
    indices: np.ndarray
    impulses: np.ndarray
    local_points: np.ndarray | None = None

    def __post_init__(self):
        _name(self.domain, "impulse domain")
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("point indices must be rank-one integers")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
            raise ValueError("point indices must fit nonnegative int64")
        indices = np.array(indices, dtype=np.int64, copy=True)
        indices.flags.writeable = False
        object.__setattr__(self, "indices", indices)
        for name in ("impulses", "local_points"):
            value = getattr(self, name)
            if name == "local_points" and value is None:
                continue
            value = float_array(value, name, ndim=2, columns=3)
            if value.dtype != np.float64 or value.shape != (len(indices), 3):
                raise ValueError(f"{name} must be FP64 with one row per index")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _own_requests(requests):
    values = tuple(requests)
    if any(not isinstance(value, PointVelocityImpulses) for value in values):
        raise TypeError("terminal requests must be PointVelocityImpulses")
    return tuple(
        PointVelocityImpulses(v.domain, v.indices, v.impulses, v.local_points) for v in values
    )


def _batch_requests(requests):
    """Keep row order and duplicate points; the real port sums before one solve."""
    groups = {}
    for value in requests:
        groups.setdefault(value.domain, []).append(value)
    result = []
    for domain, values in groups.items():
        local = None
        if any(v.local_points is not None for v in values):
            local = np.concatenate(
                [
                    np.zeros_like(v.impulses) if v.local_points is None else v.local_points
                    for v in values
                ]
            )
        result.append(
            PointVelocityImpulses(
                domain,
                np.concatenate([v.indices for v in values]),
                np.concatenate([v.impulses for v in values]),
                local,
            )
        )
    return tuple(result)


class TerminalImpulseProgram:
    """Borrow a base factory and apply prescribed impulses before each native close.

    Closing this adapter does not close the borrowed base factory. The owner of
    that factory closes it after all adapters/replays have finished.
    """

    def __init__(self, base_program, requests, *, registry):
        from dataclasses import replace

        from .registry import StateRegistry

        if not isinstance(registry, StateRegistry):
            raise TypeError("terminal program requires StateRegistry")
        if "terminal_point_impulses" not in base_program.capabilities.features:
            raise ValueError("base program lacks terminal_point_impulses capability")
        self.base = base_program
        self.requests = _own_requests(requests)
        self.registry = registry.copy()
        self.history_fields = tuple(base_program.history_fields)
        self.capabilities = replace(
            base_program.capabilities,
            features=(base_program.capabilities.features - {"terminal_point_impulses"})
            | {"terminal_point_impulse_pass"},
        )

    def problem_signature(self):
        from ..solvers.articulated_coupling import _capability_signature

        return dict(
            law="prescribed-terminal-point-impulse-v1",
            base=self.base.problem_signature(),
            base_capabilities=_capability_signature(self.base.capabilities),
            requests=self.requests,
            history=tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields),
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        native = self.base.create_workspace(scene, state, inputs, h, settings, context)
        try:
            if not callable(getattr(native, "apply_point_velocity_impulses", None)):
                raise ValueError("base workspace lacks callable terminal point impulse operation")
            return _TerminalWorkspace(self, native, scene, settings, context)
        except BaseException:
            native.close()
            raise

    def close(self):
        pass


class _TerminalWorkspace:
    def __init__(self, owner, native, scene, settings, context):
        self.owner, self.native, self.scene = owner, native, scene
        self.settings, self.context = settings, context
        self.closed = self.prepared = self.applied = False
        self.pi = self.vi = 0
        self.evidence = self.post = self.metric = self.position_metric = None

    def _open(self):
        if self.closed:
            raise RuntimeError("terminal workspace is closed")

    def predict(self):
        self._open()
        self.native.predict()

    def position_sweep(self):
        self._open()
        self.native.position_sweep()
        self.pi += 1

    def position_metrics(self):
        self._open()
        self.position_metric = self.native.position_metrics()
        return self.position_metric

    def prepare_velocity(self):
        self._open()
        self.native.prepare_velocity()
        self.prepared = True

    def velocity_sweep(self):
        self._open()
        if not self.prepared:
            raise RuntimeError("terminal velocity requires prepare_velocity")
        if not self.applied:
            self.native.velocity_sweep()
            self.vi += 1

    def _snapshot(self):
        from .program import own_diagnostics

        state = self.owner.registry.clone_state(
            self.scene, self.native.result(), self.context, self.owner.history_fields
        )
        return own_diagnostics(state)

    def velocity_metrics(self):
        from ..core.capabilities import SolverFailure
        from ..core.scene import StageMetrics
        from .program import own_diagnostics

        self._open()
        if not self.prepared:
            raise RuntimeError("terminal velocity requires prepare_velocity")
        if self.applied:
            return self.metric
        metric = self.native.velocity_metrics()
        if not isinstance(metric, StageMetrics):
            raise SolverFailure("invalid base velocity metrics")
        if metric.max_residual > self.settings.tolerance or not self.vi:
            return metric
        before = self._snapshot()
        diagnostics = own_diagnostics(self.native.diagnostics())
        from ..validation.terminal_impulse import TerminalImpulseLedger

        ledger = TerminalImpulseLedger(
            self.scene,
            before,
            self.owner.requests,
            self.settings,
            self.context,
            registry=self.owner.registry,
            history_fields=self.owner.history_fields,
        )
        self.native.apply_point_velocity_impulses(self.owner.requests)
        self.post = self._snapshot()
        certificate = ledger.require(self.post)
        after_diag = own_diagnostics(self.native.diagnostics())
        self.post_diagnostics = after_diag
        delta = {}
        for name, item in after_diag.get("participants", {}).items():
            old = diagnostics.get("participants", {}).get(name, {}).get("execution", {})
            execution = item.get("execution", {})
            delta[name] = {
                key: value - old.get(key, 0)
                for key, value in execution.items()
                if type(value) is int and key in old
            }
        self.evidence = own_diagnostics(
            dict(
                pre_impulse_state=before,
                pre_impulse_diagnostics=diagnostics,
                pre_position_iterations=self.pi,
                pre_velocity_iterations=self.vi,
                pre_position_metrics=self.position_metric,
                pre_velocity_metrics=metric,
                requests=self.owner.requests,
                batches=_batch_requests(self.owner.requests),
                post_execution_delta=delta,
                applications=1,
                ledger=certificate,
            )
        )
        self.metric = (
            StageMetrics(
                {
                    **{f"base/{key}": value for key, value in metric.residuals.items()},
                    "terminal_impulse": certificate["max_residual"],
                },
                geometric_error=metric.geometric_error,
                penetration=metric.penetration,
            )
            if self.owner.requests
            else metric
        )
        self.applied = True
        return self.metric

    def result(self):
        self._open()
        if not self.applied:
            raise RuntimeError("terminal stage is not complete")
        return self.post

    def diagnostics(self):
        self._open()
        if not self.applied:
            raise RuntimeError("terminal stage is not complete")
        return dict(self.post_diagnostics, terminal_impulse=self.evidence)

    def close(self):
        if not self.closed:
            self.closed = True
            self.native.close()
