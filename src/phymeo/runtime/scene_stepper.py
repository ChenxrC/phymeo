"""One generic schedule and outer transaction for registered physical domains."""

from collections.abc import Mapping
from contextlib import contextmanager
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.scene import (
    BackendContext,
    HistoryField,
    SceneModel,
    StageMetrics,
    StepSettings,
    fingerprint,
)
from ..core.validation import finite_real, validate_dt
from ..core.verification import VerificationReport
from .program import (
    ProgramCapabilities,
    SceneStepReport,
    SceneStepResult,
    SceneSubstepReport,
    own_diagnostics,
)
from .registry import StateRegistry


@contextmanager
def _timed(values, name):
    start = perf_counter()
    try:
        yield
    finally:
        values[name] = values.get(name, 0.0) + perf_counter() - start


class SceneStepper:
    """Return a committed owned state only after every substep and close succeeds.

    SolverFailure and arithmetic failures reject the entire frame. Configuration
    and programming errors propagate after cleanup. Programs are trusted code;
    external hidden physical history and side effects are outside this contract.
    """

    def __init__(self, scene, program, settings=None, context=None, registry=None, verifier=None):
        if not isinstance(scene, SceneModel):
            raise TypeError("expected SceneModel")
        self._scene, self._program = scene, program
        self._settings = StepSettings() if settings is None else settings
        self._context = BackendContext() if context is None else context
        if not isinstance(self._settings, StepSettings) or not isinstance(
            self._context, BackendContext
        ):
            raise TypeError("expected StepSettings and BackendContext")
        self._registry = StateRegistry() if registry is None else registry.copy()
        self._verifier = verifier
        self._scene_identity = scene.identity
        self._signature = fingerprint(program.problem_signature())
        self._capabilities = program.capabilities
        self._schema = self._read_schema()
        self._active = self._closed = False
        self._check_bound()

    @property
    def scene(self):
        return self._scene

    @property
    def settings(self):
        return self._settings

    @property
    def context(self):
        return self._context

    def _read_schema(self):
        schema = tuple(self._program.history_fields)
        if any(not isinstance(f, HistoryField) for f in schema):
            raise TypeError("history schema requires HistoryField values")
        if len({f.name for f in schema}) != len(schema):
            raise ValueError("duplicate history field")
        return schema

    def _check_bound(self):
        self.scene.assert_unchanged()
        if (
            self.scene.identity != self._scene_identity
            or fingerprint(self._program.problem_signature()) != self._signature
        ):
            raise ValueError("bound physical problem changed")
        caps = self._program.capabilities
        if not isinstance(caps, ProgramCapabilities):
            raise TypeError("program must declare ProgramCapabilities")
        if caps != self._capabilities or self._read_schema() != self._schema:
            raise ValueError("bound capabilities or history schema changed")
        if caps.backend != self.context.backend or self.context.dtype.name not in caps.dtypes:
            raise ValueError("backend context is unsupported by the program")
        for domain in self.scene.domains:
            self._registry.get(domain.family)
            if domain.family not in caps.families:
                raise ValueError("program does not support a scene state family")
        if caps.history == "none" and self._schema:
            raise ValueError("history schema contradicts program capability")
        if caps.provider_kind == "learned" and self._verifier is None:
            raise ValueError("learned providers require independent verification")
        if (
            self._verifier is not None
            and fingerprint(self._verifier.problem_signature()) != self._signature
        ):
            raise ValueError("verifier must retain the same physical problem")

    def _clone(self, state):
        return self._registry.clone_state(self.scene, state, self.context, self._schema)

    def _snapshot(self, state, inputs):
        return self._registry.snapshot(self.scene, state, inputs, self.context, self._schema)

    @staticmethod
    def _metrics(value):
        try:
            if not isinstance(value, StageMetrics):
                raise TypeError("expected StageMetrics")
            return StageMetrics(value.residuals, value.geometric_error, value.penetration)
        except (TypeError, ValueError) as error:
            raise SolverFailure(f"invalid stage metrics: {error}") from error

    def _verify(self, before, inputs, h, candidate):
        if self._verifier is None:
            return
        initial, forces = self._snapshot(before, inputs)
        result = self._verifier.verify(self.scene, initial, forces, h, self._clone(candidate))
        try:
            if not isinstance(result, VerificationReport) or not isinstance(
                result.accepted, (bool, np.bool_)
            ):
                raise ValueError("malformed verification report")
            residual = finite_real(result.max_residual, "verification residual")
            geometric = finite_real(result.max_constraint_error, "verification geometry")
            if (
                residual < 0
                or geometric < 0
                or residual > self.settings.tolerance
                or not result.accepted
            ):
                raise ValueError(result.reason or "physical verification rejected candidate")
        except (TypeError, ValueError) as error:
            raise SolverFailure(f"independent verification: {error}") from error
        return VerificationReport(True, residual, geometric, result.reason)

    def _substep(self, before, inputs, h, index):
        workspace = None
        position = velocity = None
        verification = None
        pi = vi = 0
        candidate = None
        diagnostics, times = {}, {}
        reason = None
        try:
            try:
                with _timed(times, "setup"):
                    initial, forces = self._snapshot(before, inputs)
                    workspace = self._program.create_workspace(
                        self.scene, initial, forces, h, self.settings, self.context
                    )
                with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                    with _timed(times, "predict"):
                        workspace.predict()
                    with _timed(times, "position"):
                        for pi in range(1, self.settings.position_iterations + 1):
                            workspace.position_sweep()
                            position = self._metrics(workspace.position_metrics())
                            if position.max_residual <= self.settings.tolerance:
                                break
                        else:
                            raise SolverFailure("position iteration budget exhausted")
                    with _timed(times, "prepare_velocity"):
                        workspace.prepare_velocity()
                    with _timed(times, "velocity"):
                        for vi in range(1, self.settings.velocity_iterations + 1):
                            workspace.velocity_sweep()
                            velocity = self._metrics(workspace.velocity_metrics())
                            if velocity.max_residual <= self.settings.tolerance:
                                break
                        else:
                            raise SolverFailure("velocity iteration budget exhausted")
                    with _timed(times, "validation"):
                        proposed = workspace.result()
                        try:
                            candidate = self._registry.validate_candidate(
                                self.scene, before, proposed, self.context, self._schema
                            )
                        except (TypeError, ValueError) as error:
                            raise SolverFailure(f"invalid candidate: {error}") from error
                    with _timed(times, "verification"):
                        verification = self._verify(before, inputs, h, candidate)
                    with _timed(times, "diagnostics"):
                        proposed_diagnostics = workspace.diagnostics()
                        try:
                            if not isinstance(proposed_diagnostics, Mapping):
                                raise TypeError("expected diagnostic mapping")
                            diagnostics = own_diagnostics(proposed_diagnostics)
                        except (TypeError, ValueError) as error:
                            raise SolverFailure(f"invalid diagnostics: {error}") from error
            finally:
                if workspace is not None:
                    with _timed(times, "close"):
                        workspace.close()
        except (SolverFailure, FloatingPointError, OverflowError) as error:
            candidate, diagnostics = None, {}
            reason = str(error) or "numerical program failure"
        # Changes during a callback are configuration errors, even if it failed.
        self._check_bound()
        report = SceneSubstepReport(
            candidate is not None,
            index,
            float(h),
            pi,
            vi,
            position,
            velocity,
            MappingProxyType(dict(diagnostics)),
            MappingProxyType(times),
            reason,
            verification,
        )
        return candidate, report

    def step(self, state, inputs, dt):
        if self._closed:
            raise RuntimeError("scene stepper is closed")
        if self._active:
            raise RuntimeError("scene stepper already active")
        self._active = True
        try:
            self._check_bound()
            h = validate_dt(
                float(validate_dt(dt, self.context.dtype)) / self.settings.substeps,
                self.context.dtype,
            )
            original, forces = self._snapshot(state, inputs)
            current, reports, snapshots = self._clone(original), [], []
            for index in range(self.settings.substeps):
                current, report = self._substep(current, forces, h, index)
                reports.append(report)
                if not report.accepted:
                    return SceneStepResult(
                        original,
                        SceneStepReport(False, self.context.backend, tuple(reports), report.reason),
                    )
                if self.settings.record_substeps:
                    snapshots.append(self._clone(current))
            return SceneStepResult(
                current,
                SceneStepReport(True, self.context.backend, tuple(reports)),
                tuple(snapshots),
            )
        finally:
            self._active = False

    def close(self):
        if self._active:
            raise RuntimeError("scene stepper already active")
        if not self._closed:
            self._closed = True
            self._program.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("scene stepper is closed")
        return self

    def __exit__(self, *_):
        self.close()
