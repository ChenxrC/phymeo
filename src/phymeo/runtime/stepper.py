"""Transactional particle or rigid stepping with one optional same-problem fallback."""

from dataclasses import replace
from numbers import Real

import numpy as np

from ..core.capabilities import ProblemSignature, Solver, SolverCapabilities, SolverFailure
from ..core.result import StepReport, StepResult
from .state_contract import state_contract


class Stepper:
    """Commit valid owned candidates, or return the unchanged original snapshot.

    This v0 gate independently checks shape, precision, finiteness, unit rigid
    quaternions and fixed-state invariants. Other physics residuals come from the solver. A future
    learned solver needs independent physical validation, not just a matching signature.
    Hidden-state solvers are rejected until a snapshot/restore contract exists.
    """

    def __init__(
        self,
        primary: Solver,
        fallback: Solver | None = None,
        *,
        required_features=(),
        gradient="unsupported",
        max_residual=1e-5,
    ):
        if (
            isinstance(max_residual, (bool, np.bool_))
            or not isinstance(max_residual, Real)
            or not np.isfinite(max_residual)
            or max_residual < 0
        ):
            raise ValueError("max_residual must be finite and non-negative")
        self._solvers = (primary,) if fallback is None else (primary, fallback)
        self._required_features = frozenset(required_features)
        self._gradient = gradient
        self._max_residual = float(max_residual)
        capabilities = primary.capabilities
        if not isinstance(capabilities, SolverCapabilities):
            raise TypeError("solver must declare SolverCapabilities")
        self._contract = state_contract(capabilities.state_kind)
        self._signature = primary.problem_signature()
        if not isinstance(self._signature, ProblemSignature):
            raise TypeError("solver must return a ProblemSignature")
        self._check_solvers()

    def _check_solvers(self, dtype=None):
        for solver in self._solvers:
            if solver.problem_signature() != self._signature:
                raise ValueError("primary and fallback must retain the same physical problem")
            capabilities = solver.capabilities
            if not isinstance(capabilities, SolverCapabilities):
                raise TypeError("solver must declare SolverCapabilities")
            if capabilities.state_kind != self._contract.kind:
                raise ValueError("primary and fallback must retain the same state capability")
            if not self._required_features <= capabilities.features:
                raise ValueError("required capabilities are not supported by every solver")
            if self._gradient != "unsupported" and capabilities.gradient != self._gradient:
                raise ValueError("requested gradient capability is unavailable")
            if capabilities.history != "none":
                raise ValueError(
                    "solver history requires a snapshot/restore protocol not yet supported"
                )
            if dtype is not None and dtype.name not in capabilities.dtypes:
                raise ValueError("state dtype is not supported by every solver")

    def step(self, model, state_in, inputs, dt) -> StepResult:
        snapshot, owned_inputs, h = self._contract.snapshot(model, state_in, inputs, dt)
        self._check_solvers(snapshot.position.dtype)
        primary_failure = None
        for index, solver in enumerate(self._solvers):
            candidate, report = self._attempt(solver, model, snapshot, owned_inputs, h)
            report = replace(report, fallback_used=index == 1, primary_failure=primary_failure)
            if candidate is not None:
                return StepResult(candidate, report)
            if index == 0:
                primary_failure = report.reason
        return StepResult(snapshot, report)

    def _attempt(self, solver, model, snapshot, inputs, dt):
        try:
            result = solver.step(model, snapshot.copy(), inputs.copy(), dt)
        except SolverFailure as error:
            return self._reject(str(error) or "solver rejected the step")
        if not isinstance(result, StepResult) or not isinstance(result.report, StepReport):
            return self._reject("malformed solver result")
        report = result.report
        if not isinstance(report.accepted, (bool, np.bool_)):
            return self._reject("invalid acceptance flag")
        if not report.accepted:
            return None, replace(
                report, reason=report.reason or "solver did not accept its candidate"
            )
        if (
            isinstance(report.iterations, (bool, np.bool_))
            or not isinstance(report.iterations, (int, np.integer))
            or report.iterations < 0
        ):
            return self._reject("invalid iteration count")
        for value in (report.max_residual, report.max_constraint_error):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                return self._reject("invalid residual or geometric error")
        if report.max_residual > self._max_residual:
            return None, replace(
                report, accepted=False, reason="residual exceeds acceptance threshold"
            )
        candidate, reason = self._contract.candidate(result.state, snapshot, model)
        if candidate is None:
            return self._reject(reason)
        return candidate, report

    @staticmethod
    def _reject(reason):
        return None, StepReport(accepted=False, max_residual=float("inf"), reason=reason)
