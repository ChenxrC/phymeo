"""Owned v2 representation evidence and finite exact binary input codecs.

These codecs convert immutable inputs and decode actual native words. They do
not evaluate material geometry, gradients, angles, or replacement solutions.
"""

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
import json
import math
import struct

import numpy as np

from ..core.capabilities import SolverFailure

SCHEMA = "xpbd-composite-material-path-v2"
KIND = "reference-displacement-f32-v2"
CODEC_BITS = 4096
PIN = "99f6df7c0fd8fdeef77f97df83914da27b64b7ee"


def exact(value):
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value).bit_length() > CODEC_BITS:
            raise SolverFailure("representation codec resource exhausted")
        result = Fraction(int(value))
    else:
        value = float(value)
        if not math.isfinite(value):
            raise SolverFailure("representation requires finite input")
        result = Fraction(value)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > CODEC_BITS:
        raise SolverFailure("representation codec resource exhausted")
    return result


def round_binary(value, bits):
    """Integer nearest/ties-even rounding, with bounded rational operands."""
    value = exact(value)
    if bits not in (32, 64):
        raise ValueError("binary codec width must be 32 or 64")
    if not value:
        return 0.0
    precision, minimum, maximum = (24, -126, 127) if bits == 32 else (53, -1022, 1023)
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator < denominator << exponent) if exponent >= 0 else (numerator << -exponent < denominator):
        exponent -= 1
    quantum = max(exponent - precision + 1, minimum - precision + 1)
    if exponent > maximum:
        raise SolverFailure("representation codec overflow")
    scaled_n = numerator << max(-quantum, 0)
    scaled_d = denominator << max(quantum, 0)
    word, remainder = divmod(scaled_n, scaled_d)
    if 2 * remainder > scaled_d or (2 * remainder == scaled_d and word % 2):
        word += 1
    if bits == 32 and (not word or quantum + word.bit_length() - 1 < minimum):
        raise SolverFailure("representation native word requires nonzero subnormal/zero")
    try:
        result = math.ldexp(float(word), quantum)
    except OverflowError as error:
        raise SolverFailure("representation codec overflow") from error
    if not math.isfinite(result):
        raise SolverFailure("representation codec overflow")
    return (-result if value < 0 else result) if result else 0.0


def _array(value, dtype, shape, name):
    source = np.asarray(value)
    if source.dtype != np.dtype(dtype) or source.shape != shape or not np.isfinite(source).all():
        raise ValueError(f"invalid {name} dtype/shape/finite values")
    result = np.array(source, copy=True, order="C")
    result.flags.writeable = False
    return result


def _count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("nonnegative integer required")
    return int(value)


def _token(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("nonempty identity token required")
    return value


@dataclass(frozen=True)
class NativeCompositeRepresentation:
    kind: str = KIND
    codec_bits: int = CODEC_BITS
    decoder: str = "binary64-nearest-even-v1"

    def __post_init__(self):
        if (self.kind, self.codec_bits, self.decoder) != (KIND, CODEC_BITS, "binary64-nearest-even-v1"):
            raise ValueError("unsupported native composite representation")


@dataclass(frozen=True)
class RoundingWitness:
    numerator: int
    denominator: int
    bits: int
    word: float

    def __post_init__(self):
        if type(self.numerator) is not int or type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("exact integer rounding witness required")
        if max(abs(self.numerator).bit_length(), self.denominator.bit_length()) > CODEC_BITS:
            raise SolverFailure("representation codec resource exhausted")
        value = exact(Fraction(self.numerator, self.denominator))
        expected = round_binary(value, self.bits)
        if self.word != expected or (self.word == 0 and math.copysign(1, self.word) < 0):
            raise ValueError("incorrect canonical rounding witness")

    @property
    def source(self):
        return Fraction(self.numerator, self.denominator)

    @property
    def residual(self):
        return exact(self.word) - self.source


def encode_values(values, shape, bits=32):
    words, witnesses = [], []
    for value in values:
        value = exact(value)
        word = round_binary(value, bits)
        words.append(word)
        witnesses.append(RoundingWitness(value.numerator, value.denominator, bits, word))
    result = np.array(words, dtype=np.float32 if bits == 32 else np.float64).reshape(shape)
    return result, tuple(witnesses)


def decode(reference, displacement):
    reference, displacement = np.asarray(reference), np.asarray(displacement)
    if reference.dtype != np.float64 or displacement.dtype != np.float32 or reference.shape != displacement.shape:
        raise ValueError("decode requires matched FP64 reference/FP32 displacement")
    return encode_values((exact(r) + exact(d) for r, d in zip(reference.flat, displacement.flat, strict=True)), reference.shape, 64)


@dataclass(frozen=True, eq=False)
class RepresentationBoundary:
    name: str
    computed: np.ndarray
    witnesses: tuple[RoundingWitness, ...]

    def __post_init__(self):
        _token(self.name)
        value = np.asarray(self.computed)
        object.__setattr__(self, "computed", _array(value, np.float32, value.shape, self.name))
        witnesses = tuple(self.witnesses)
        if len(witnesses) != value.size or any(not isinstance(w, RoundingWitness) or w.bits != 32 or w.word != float(v) for w, v in zip(witnesses, value.flat, strict=True)):
            raise ValueError("boundary rounding witness mismatch")
        object.__setattr__(self, "witnesses", witnesses)


@dataclass(frozen=True, eq=False)
class RepresentedMaterialBatch:
    kind: str
    node_indices: np.ndarray
    reference_fingerprint: str
    displacement_in: np.ndarray
    displacement_out: np.ndarray
    positions_in: np.ndarray
    positions_out: np.ndarray
    lambda_in: np.ndarray
    lambda_out: np.ndarray
    compute_weights: np.ndarray
    compute_rest: np.ndarray
    compute_alpha: np.ndarray
    rows: np.ndarray
    products: np.ndarray
    geometry_before: np.ndarray
    geometry_after: np.ndarray
    world_before: np.ndarray
    world_after: np.ndarray
    decode_before: tuple[RoundingWitness, ...]
    decode_after: tuple[RoundingWitness, ...]
    status: np.ndarray
    launches: int = 0
    completed_launches: int = 0

    def __post_init__(self):
        if self.kind not in ("distance", "bending"):
            raise ValueError("unknown represented material kind")
        arity, width, vectors = (2, 18, 1) if self.kind == "distance" else (4, 45, 3)
        ids = np.asarray(self.node_indices)
        if ids.ndim != 2:
            raise ValueError("ordered indices required")
        n, m = len(self.displacement_in), len(ids)
        object.__setattr__(self, "node_indices", _array(ids, np.int64, (m, arity), "identities"))
        if np.any(ids < 0) or np.any(ids >= n):
            raise ValueError("represented identity outside state")
        _token(self.reference_fingerprint)
        for name in ("displacement_in", "displacement_out"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float32, (n, 3), name))
        for name in ("positions_in", "positions_out"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float64, (n, 3), name))
        for name in ("lambda_in", "lambda_out", "compute_rest", "compute_alpha"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float32, (m,), name))
        for name, shape in (("compute_weights", (n,)), ("rows", (m, width)), ("products", (m, arity, 3)), ("geometry_before", (m, vectors, 3)), ("geometry_after", (m, vectors, 3))):
            object.__setattr__(self, name, _array(getattr(self, name), np.float32, shape, name))
        for name in ("world_before", "world_after"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float64, (m, arity, 3), name))
        for name in ("decode_before", "decode_after"):
            values = tuple(getattr(self, name))
            if len(values) != m * arity * 3 or any(not isinstance(w, RoundingWitness) or w.bits != 64 for w in values):
                raise ValueError("complete row decoder witnesses required")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "status", _array(self.status, np.int32, (m,), "status"))
        if np.any(self.status):
            raise ValueError("only successful complete represented rows may publish")
        for name in ("launches", "completed_launches"):
            object.__setattr__(self, name, _count(getattr(self, name)))


@dataclass(frozen=True)
class RepresentedMaterialSweep:
    sweep_index: int
    blocks: tuple[RepresentedMaterialBatch, RepresentedMaterialBatch]

    def __post_init__(self):
        object.__setattr__(self, "sweep_index", _count(self.sweep_index))
        blocks = tuple(self.blocks)
        if len(blocks) != 2 or any(not isinstance(b, RepresentedMaterialBatch) for b in blocks) or tuple(b.kind for b in blocks) != ("distance", "bending"):
            raise ValueError("complete ordered represented sweep required")
        object.__setattr__(self, "blocks", blocks)

@dataclass(frozen=True, eq=False)
class NativeRepresentationOperation:
    name: str
    inputs: tuple[np.ndarray, ...]
    outputs: tuple[np.ndarray, ...]
    launches: int
    completed_launches: int

    def __post_init__(self):
        _token(self.name)
        for name in ("inputs", "outputs"):
            arrays = []
            for value in getattr(self, name):
                value = np.asarray(value)
                dtype = np.int32 if value.dtype == np.int32 else np.float32
                arrays.append(_array(value, dtype, value.shape, name))
            object.__setattr__(self, name, tuple(arrays))
        for name in ("launches", "completed_launches"):
            object.__setattr__(self, name, _count(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class RepresentedMaterialPath:
    material_fingerprint: str
    policy_fingerprint: str
    input_fingerprint: str
    initial_fingerprint: str
    backend: str
    duration: float
    representation: NativeCompositeRepresentation
    domain: str
    reference: np.ndarray
    reference_fingerprint: str
    predicted_displacement: np.ndarray
    final_displacement: np.ndarray
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    reconstructed_velocity: np.ndarray
    distance_alpha: np.ndarray
    bending_alpha: np.ndarray
    sweeps: tuple[RepresentedMaterialSweep, ...]
    compute_boundaries: tuple[RepresentationBoundary, ...]
    prediction_operation: NativeRepresentationOperation
    reconstruction_operation: NativeRepresentationOperation
    alpha_operations: tuple[NativeRepresentationOperation, ...]
    schema: str = SCHEMA
    backend_pin: str = PIN

    def __post_init__(self):
        if self.schema != SCHEMA or self.backend_pin != PIN or self.backend not in ("tyx:interpret", "tyx:cuda", "tyx:cpu_cpp"):
            raise ValueError("unsupported represented schema/backend/pin")
        if not isinstance(self.representation, NativeCompositeRepresentation):
            raise TypeError("explicit represented configuration required")
        for name in ("material_fingerprint", "policy_fingerprint", "input_fingerprint", "initial_fingerprint", "reference_fingerprint"):
            _token(getattr(self, name))
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("positive represented duration required")
        _token(self.domain)
        n = len(self.reference)
        for name in ("reference", "predicted_position", "predicted_velocity"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float64, (n, 3), name))
        for name in ("predicted_displacement", "final_displacement", "reconstructed_velocity"):
            object.__setattr__(self, name, _array(getattr(self, name), np.float32, (n, 3), name))
        for name in ("distance_alpha", "bending_alpha"):
            value = getattr(self, name)
            object.__setattr__(self, name, _array(value, np.float32, (len(value),), name))
        sweeps = tuple(self.sweeps)
        if any(not isinstance(s, RepresentedMaterialSweep) or s.sweep_index != k for k, s in enumerate(sweeps)):
            raise ValueError("consecutive complete represented sweeps required")
        object.__setattr__(self, "sweeps", sweeps)
        boundaries = tuple(self.compute_boundaries)
        if any(not isinstance(b, RepresentationBoundary) for b in boundaries) or len({b.name for b in boundaries}) != len(boundaries):
            raise ValueError("unique typed representation boundaries required")
        object.__setattr__(self, "compute_boundaries", boundaries)
        for name in ("prediction_operation", "reconstruction_operation"):
            if not isinstance(getattr(self, name), NativeRepresentationOperation):
                raise TypeError("typed native operation required")
        operations = tuple(self.alpha_operations)
        if any(not isinstance(o, NativeRepresentationOperation) for o in operations):
            raise TypeError("typed alpha operations required")
        object.__setattr__(self, "alpha_operations", operations)

    @property
    def prediction_launches(self):
        return self.prediction_operation.completed_launches

    @property
    def reconstruction_launches(self):
        return self.reconstruction_operation.completed_launches


def serialize_evidence(value):
    """Exact typed byte serialization, with no pickle or executable payloads."""
    def encode(item):
        if isinstance(item, np.ndarray):
            return {"array": item.dtype.str, "shape": list(item.shape), "hex": item.tobytes(order="C").hex()}
        if is_dataclass(item):
            return {"type": type(item).__name__, "fields": {f.name: encode(getattr(item, f.name)) for f in fields(item)}}
        if isinstance(item, tuple):
            return {"tuple": [encode(v) for v in item]}
        if isinstance(item, float):
            return {"float64": struct.pack(">d", item).hex()}
        if type(item) in (str, int, bool) or item is None:
            return item
        raise TypeError("unsupported representation evidence value")
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def deserialize_evidence(payload):
    types = {c.__name__: c for c in (NativeCompositeRepresentation, RoundingWitness, RepresentationBoundary, RepresentedMaterialBatch, RepresentedMaterialSweep, NativeRepresentationOperation, RepresentedMaterialPath)}
    def decode_item(item):
        if not isinstance(item, dict):
            if type(item) in (str, int, bool) or item is None:
                return item
            raise ValueError("untyped representation scalar")
        if set(item) == {"array", "shape", "hex"}:
            dtype = np.dtype(item["array"])
            if dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int32"), np.dtype("int64")):
                raise ValueError("unsupported representation array dtype")
            return np.frombuffer(bytes.fromhex(item["hex"]), dtype=dtype).reshape(tuple(item["shape"])).copy()
        if set(item) == {"float64"}:
            return struct.unpack(">d", bytes.fromhex(item["float64"]))[0]
        if set(item) == {"tuple"}:
            return tuple(decode_item(v) for v in item["tuple"])
        if set(item) == {"type", "fields"} and item["type"] in types:
            return types[item["type"]](**{k: decode_item(v) for k, v in item["fields"].items()})
        raise ValueError("unknown representation evidence schema")
    return decode_item(json.loads(payload))
