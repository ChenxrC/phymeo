"""Complete owned observations. Observations are evidence, never an ideal oracle."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.scene import fingerprint
from ..core.validation import finite_real
from .bending_trace import _owned_float64

SCHEMA = "xpbd-composite-material-path-v1"
EXTERNAL_PURPOSE = "standalone-prescribed-external-impulses-v1"


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _token(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, eq=False)
class MaterialBatch:
    kind: str
    node_indices: np.ndarray
    positions_in: np.ndarray
    compute_start: np.ndarray
    compute_end: np.ndarray
    positions_out: np.ndarray
    lambda_in: np.ndarray
    lambda_compute_start: np.ndarray
    lambda_compute_end: np.ndarray
    lambda_out: np.ndarray
    compute_weights: np.ndarray
    compute_rest: np.ndarray
    compute_alpha: np.ndarray
    rows: np.ndarray
    status: np.ndarray
    launches: int = 0
    completed_launches: int = 0

    def __post_init__(self):
        if self.kind not in ("distance", "bending"):
            raise ValueError("unknown material block")
        width, wire = (2, 18) if self.kind == "distance" else (4, 45)
        ids = np.asarray(self.node_indices)
        if ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[1] != width:
            raise ValueError("invalid ordered material identities")
        ids = ids.copy()
        if np.any(ids < 0):
            raise ValueError("negative material identity")
        ids.flags.writeable = False
        object.__setattr__(self, "node_indices", ids)
        n, m = len(self.positions_in), len(ids)
        for name in ("positions_in", "compute_start", "compute_end", "positions_out"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, (n, 3)))
        if np.any(ids >= n):
            raise ValueError("material identity outside observation")
        for name in ("lambda_in", "lambda_compute_start", "lambda_compute_end", "lambda_out",
                     "compute_rest", "compute_alpha"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, (m,)))
        object.__setattr__(self, "compute_weights", _owned_float64(
            self.compute_weights, "compute_weights", (n,)))
        object.__setattr__(self, "rows", _owned_float64(self.rows, "rows", (m, wire)))
        status = np.asarray(self.status)
        if status.dtype != np.int64 or status.shape != (m,) or np.any(status != 0):
            raise ValueError("only successful complete rows may be published")
        status = status.copy()
        status.flags.writeable = False
        object.__setattr__(self, "status", status)
        for name in ("launches", "completed_launches"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True)
class CompositeMaterialSweep:
    sweep_index: int
    blocks: tuple[MaterialBatch, MaterialBatch]

    def __post_init__(self):
        object.__setattr__(self, "sweep_index", _integer(self.sweep_index, "sweep_index"))
        blocks = tuple(self.blocks)
        if len(blocks) != 2 or any(not isinstance(b, MaterialBatch) for b in blocks):
            raise TypeError("each sweep requires both typed material blocks")
        if tuple(b.kind for b in blocks) != ("distance", "bending"):
            raise ValueError("material order must be distance then bending")
        object.__setattr__(self, "blocks", blocks)


@dataclass(frozen=True, eq=False)
class CompositeComputeBoundary:
    name: str
    original: np.ndarray
    computed: np.ndarray

    def __post_init__(self):
        _token(self.name, "compute boundary")
        shape = np.asarray(self.original).shape
        for name in ("original", "computed"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, shape))


@dataclass(frozen=True, eq=False)
class CompositeMaterialPath:
    material_fingerprint: str
    policy_fingerprint: str
    input_fingerprint: str
    initial_fingerprint: str
    backend: str
    duration: float
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    distance_alpha: np.ndarray
    bending_alpha: np.ndarray
    sweeps: tuple[CompositeMaterialSweep, ...]
    prediction_launches: int
    reconstruction_launches: int
    schema: str = SCHEMA
    compute_boundaries: tuple[CompositeComputeBoundary, ...] = ()
    alpha_launches: tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.schema != SCHEMA or self.backend not in ("numpy", "tyx:interpret", "tyx:cpu_cpp", "tyx:cuda"):
            raise ValueError("unsupported composite observation schema/backend")
        for name in ("material_fingerprint", "policy_fingerprint", "input_fingerprint", "initial_fingerprint"):
            _token(getattr(self, name), name)
        h = finite_real(self.duration, "duration")
        if h <= 0:
            raise ValueError("duration must be positive")
        object.__setattr__(self, "duration", h)
        boundaries = tuple(self.compute_boundaries)
        if any(not isinstance(b, CompositeComputeBoundary) for b in boundaries):
            raise TypeError("typed compute boundaries required")
        if len({b.name for b in boundaries}) != len(boundaries):
            raise ValueError("duplicate compute boundary")
        object.__setattr__(self, "compute_boundaries", boundaries)
        launches = tuple(_integer(v, "alpha launches") for v in self.alpha_launches)
        if len(launches) != 2:
            raise ValueError("both alpha operations required")
        object.__setattr__(self, "alpha_launches", launches)
        n = len(self.predicted_position)
        for name in ("predicted_position", "predicted_velocity"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, (n, 3)))
        for name in ("distance_alpha", "bending_alpha"):
            value = getattr(self, name)
            object.__setattr__(self, name, _owned_float64(value, name, (len(value),)))
        sweeps = tuple(self.sweeps)
        if any(not isinstance(s, CompositeMaterialSweep) or s.sweep_index != i for i, s in enumerate(sweeps)):
            raise ValueError("nonconsecutive composite sweep sequence")
        object.__setattr__(self, "sweeps", sweeps)
        for name in ("prediction_launches", "reconstruction_launches"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True, eq=False)
class CompositeExternalImpulses:
    held: np.ndarray
    terminal: np.ndarray
    duration: float
    original_input_fingerprint: str
    selected_pre_fingerprint: str
    final_state_fingerprint: str
    purpose: str = EXTERNAL_PURPOSE

    def __post_init__(self):
        n = len(self.held)
        for name in ("held", "terminal"):
            object.__setattr__(self, name, _owned_float64(getattr(self, name), name, (n, 3)))
        h = finite_real(self.duration, "duration")
        if h <= 0 or self.purpose != EXTERNAL_PURPOSE:
            raise ValueError("invalid explicit external power boundary")
        object.__setattr__(self, "duration", h)
        for name in ("original_input_fingerprint", "selected_pre_fingerprint", "final_state_fingerprint"):
            _token(getattr(self, name), name)

    @classmethod
    def bind(cls, held, terminal, h, original_inputs, selected_pre, final):
        return cls(held, terminal, h, fingerprint(original_inputs),
                   fingerprint(selected_pre), fingerprint(final))


