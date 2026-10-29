"""Live finite-contact terminal adapter, isolated from the old prismatic ledger."""

import numpy as np

from .terminal_impulse import TerminalImpulseProgram, _batch_requests


class FiniteContactTerminalProgram(TerminalImpulseProgram):
    def __init__(self, base_program, requests, *, registry, reference_angular_speed=1.0):
        if (
            type(reference_angular_speed) is not float
            or not np.isfinite(reference_angular_speed)
            or reference_angular_speed <= 0
        ):
            raise ValueError("reference_angular_speed must be a finite positive builtin float")
        super().__init__(base_program, requests, registry=registry)
        self.reference_angular_speed = reference_angular_speed

    def problem_signature(self):
        return dict(
            super().problem_signature(),
            finite_law="finite-current-configuration-v1",
            reference_angular_speed=self.reference_angular_speed,
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        native = self.base.create_workspace(scene, state, inputs, h, settings, context)
        try:
            if not callable(getattr(native, "apply_point_velocity_impulses", None)):
                raise ValueError("base workspace lacks callable terminal point impulse operation")
            return _FiniteTerminalWorkspace(self, native, scene, settings, context)
        except BaseException:
            native.close()
            raise


class _FiniteTerminalWorkspace:
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
        from ..validation.finite_contact_terminal import FiniteContactTerminalLedger

        ledger = FiniteContactTerminalLedger(
            self.scene,
            before,
            self.owner.requests,
            self.settings,
            self.context,
            registry=self.owner.registry,
            history_fields=self.owner.history_fields,
            reference_angular_speed=self.owner.reference_angular_speed,
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
