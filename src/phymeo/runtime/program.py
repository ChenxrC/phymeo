"""Typed phased programs; every persistent physical field belongs to SceneState."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ..core.scene import HistoryField, SceneState, StageMetrics, _name, fingerprint
from ..core.verification import VerificationReport


@dataclass(frozen=True)
class ProgramCapabilities:
    backend: str
    dtypes: frozenset[str]
    families: frozenset[str]
    features: frozenset[str] = frozenset()
    history: str = "none"
    provider_kind: str = "mathematical"
    gradient: str = "unsupported"

    def __post_init__(self):
        _name(self.backend, "backend")
        for name in ("dtypes", "families", "features"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise TypeError(f"{name} must be a collection of names")
            values = frozenset(_name(v, name) for v in values)
            object.__setattr__(self, name, values)
        if not self.dtypes or not self.dtypes <= {"float32", "float64"} or not self.families:
            raise ValueError("declare supported state precision and families")
        if self.history not in ("none", "explicit"):
            raise ValueError("history must be absent or explicit in SceneState")
        if self.provider_kind not in ("mathematical", "learned"):
            raise ValueError("unknown provider kind")
        if self.gradient not in ("unsupported", "exact", "approximate"):
            raise ValueError("unknown gradient capability")


class SubstepWorkspace(Protocol):
    """Temporary candidate and resources; construction cleans partial failures.

    The runtime calls predict once, then sweeps without advancing time again.
    close must release resources on success and failure, without external commits.
    """

    def predict(self) -> None: ...
    def position_sweep(self) -> None: ...
    def position_metrics(self) -> StageMetrics: ...
    def prepare_velocity(self) -> None: ...
    def velocity_sweep(self) -> None: ...
    def velocity_metrics(self) -> StageMetrics: ...
    def result(self) -> SceneState: ...
    def diagnostics(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


class SceneProgram(Protocol):
    capabilities: ProgramCapabilities
    history_fields: tuple[HistoryField, ...]

    def problem_signature(self) -> object: ...
    def create_workspace(self, scene, state, inputs, h, settings, context) -> SubstepWorkspace: ...
    def close(self) -> None: ...


class SceneVerifier(Protocol):
    def problem_signature(self) -> object: ...
    def verify(self, scene, before, inputs, h, after) -> VerificationReport: ...


def own_diagnostics(value):
    """Copy only explicit finite data, retaining typed load ledgers for audit clients."""
    fingerprint(value)

    def clone(item):
        if isinstance(item, np.ndarray):
            result = item.copy()
            result.flags.writeable = False
            return result
        if isinstance(item, Mapping):
            return MappingProxyType({k: clone(v) for k, v in item.items()})
        if is_dataclass(item) and not isinstance(item, type):
            return type(item)(**{f.name: clone(getattr(item, f.name)) for f in fields(item)})
        if isinstance(item, (tuple, list)):
            return tuple(clone(v) for v in item)
        return item

    return clone(value)


@dataclass(frozen=True)
class SceneSubstepReport:
    accepted: bool
    index: int
    dt: float
    position_iterations: int = 0
    velocity_iterations: int = 0
    position: StageMetrics | None = None
    velocity: StageMetrics | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    timings: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None
    verification: VerificationReport | None = None

    @property
    def max_residual(self):
        if not self.accepted:
            return float("inf")
        return max(
            self.position.max_residual,
            self.velocity.max_residual,
            0.0 if self.verification is None else self.verification.max_residual,
        )


@dataclass(frozen=True)
class SceneStepReport:
    accepted: bool
    backend: str
    substeps: tuple[SceneSubstepReport, ...] = ()
    reason: str | None = None

    @property
    def completed_substeps(self):
        """Successful computations; all are rolled back if accepted is false."""
        return sum(item.accepted for item in self.substeps)

    @property
    def max_residual(self):
        return max((r.max_residual for r in self.substeps), default=0.0)


@dataclass(frozen=True)
class SceneStepResult:
    """Runtime-owned result; direct dataclass construction borrows its arguments."""

    state: SceneState
    report: SceneStepReport
    substep_states: tuple[SceneState, ...] = ()
