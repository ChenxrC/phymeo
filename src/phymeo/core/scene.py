"""Typed scene boundaries; registered families own numerical copies at runtime entry."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .validation import finite_real


def _name(value, label="name"):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty unpadded string")
    return value


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return MappingProxyType({_name(k, label): v for k, v in value.items()})


def fingerprint(value):
    """Hash explicit numerical configuration, rejecting opaque mutable objects."""

    def encode(item):
        if isinstance(item, (np.ndarray, np.generic)):
            array = np.asarray(item)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise TypeError("fingerprints require finite numerical arrays")
            return [
                "array",
                array.dtype.str,
                array.shape,
                np.ascontiguousarray(array).tobytes().hex(),
            ]
        if is_dataclass(item) and not isinstance(item, type):
            cls = type(item)
            return [
                "dataclass",
                cls.__module__,
                cls.__qualname__,
                [(f.name, encode(getattr(item, f.name))) for f in fields(item)],
            ]
        if item is None or type(item) in (str, bool, int):
            return [type(item).__name__, item]
        if type(item) is float:
            return ["float", finite_real(item, "configuration")]
        if isinstance(item, Mapping):
            for key in item:
                _name(key, "configuration key")
            return ["mapping", [(k, encode(item[k])) for k in sorted(item)]]
        if isinstance(item, (tuple, list)):
            return [type(item).__name__, [encode(v) for v in item]]
        raise TypeError(f"unsupported configuration type: {type(item).__name__}")

    return hashlib.sha256(json.dumps(encode(value), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class DomainBinding:
    name: str
    family: str
    model: object

    def __post_init__(self):
        _name(self.name, "domain name")
        _name(self.family, "state family")


@dataclass(frozen=True)
class SceneModel:
    domains: tuple[DomainBinding, ...]
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        domains = tuple(self.domains)
        if any(not isinstance(d, DomainBinding) for d in domains):
            raise TypeError("scene domains must be DomainBinding instances")
        if len({d.name for d in domains}) != len(domains):
            raise ValueError("duplicate scene domain")
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "_identity", fingerprint(domains))

    @property
    def identity(self):
        return self._identity

    def assert_unchanged(self):
        if fingerprint(self.domains) != self._identity:
            raise ValueError("bound scene model changed; build a new scene")


@dataclass(frozen=True)
class SceneState:
    """Domain values are borrowed here; StateRegistry creates the canonical owned snapshot."""

    domains: Mapping[str, object]
    history: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "domains", _mapping(self.domains, "state domains"))
        object.__setattr__(self, "history", _mapping(self.history, "history"))


@dataclass(frozen=True)
class SceneInputs:
    domains: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "domains", _mapping(self.domains, "input domains"))


@dataclass(frozen=True)
class HistoryField:
    name: str
    shape: tuple[int, ...]
    dtype: object

    def __post_init__(self):
        _name(self.name, "history field")
        shape = tuple(self.shape)
        if any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in shape
        ):
            raise ValueError("history dimensions must be non-negative integers")
        dtype = np.dtype(self.dtype)
        if dtype not in tuple(
            map(np.dtype, (np.float32, np.float64, np.int32, np.int64, np.bool_))
        ):
            raise ValueError("unsupported numerical history dtype")
        object.__setattr__(self, "shape", tuple(map(int, shape)))
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True)
class StepSettings:
    substeps: int = 4
    position_iterations: int = 200
    velocity_iterations: int = 200
    tolerance: float = 1e-7
    reference_length: float = 1.0
    reference_speed: float = 1.0
    record_substeps: bool = False

    def __post_init__(self):
        for name in ("substeps", "position_iterations", "velocity_iterations"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("tolerance", "reference_length", "reference_speed"):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (name != "tolerance" and value == 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not isinstance(self.record_substeps, (bool, np.bool_)):
            raise TypeError("record_substeps must be boolean")


@dataclass(frozen=True)
class BackendContext:
    backend: str = "numpy"
    dtype: object = np.float64

    def __post_init__(self):
        _name(self.backend, "backend")
        dtype = np.dtype(self.dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("state precision must be float32 or float64")
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True)
class StageMetrics:
    """Dimensionless residuals; geometry diagnostics remain in metres."""

    residuals: Mapping[str, float] = field(default_factory=dict)
    geometric_error: float = 0.0
    penetration: float = 0.0

    def __post_init__(self):
        residuals = _mapping(self.residuals, "residual name")
        values = {k: finite_real(v, k) for k, v in residuals.items()}
        for name in ("geometric_error", "penetration"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if min((*values.values(), self.geometric_error, self.penetration)) < 0:
            raise ValueError("residuals and geometry diagnostics must be non-negative")
        object.__setattr__(self, "residuals", MappingProxyType(values))

    @property
    def max_residual(self):
        return max(self.residuals.values(), default=0.0)
