"""Explicit binary32 delivery of already-generated contact loads.

This boundary is for device-generated contact aggregates only. It is not an
external-input coercion, a contact law, or a state update. Every finite normal
value and both signed zeros are preserved bit for bit. An IEEE-754 binary32
subnormal is replaced by a zero with the same sign, with its exact binary64
raw-minus-delivered loss recorded. No caller-selected epsilon is accepted.

The returned arrays are backed by immutable bytes, including the raw evidence.
A frozen dataclass alone would not prevent a caller from re-enabling writes to
an owning ndarray. Empty load axes are allowed, but the caller must declare the
complete non-scalar shape. There is no claim of physically lossless delivery.
"""

from dataclasses import dataclass, field

import numpy as np


_SCHEMA = "generated-contact-fp32-delivery-v1"
_TINY = float.fromhex("0x1.0p-126")
_SIGN = np.uint32(0x80000000)
_EXPONENT = np.uint32(0x7F800000)
_FRACTION = np.uint32(0x007FFFFF)


def _immutable(array):
    """Copy every logical element into a bytes owner that cannot become writable."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class ContactFP32Delivery:
    """A complete, immutable record of one generated contact-load conversion.

    loss = float64(raw) - float64(delivered) is exact because binary64 can
    represent every binary32 value, including all subnormals. loss_bound is
    the fixed smallest positive normal binary32 value at converted elements,
    and zero elsewhere. The strict actual magnitude is less than that bound.
    The same validation is applied when this class is constructed directly.
    """

    raw: np.ndarray
    expected_shape: tuple[int, ...]
    schema: str = field(default=_SCHEMA, init=False)
    delivered: np.ndarray = field(init=False)
    subnormal_mask: np.ndarray = field(init=False)
    loss: np.ndarray = field(init=False)
    loss_bound: np.ndarray = field(init=False)

    def __post_init__(self):
        shape = self.expected_shape
        if type(shape) is not tuple:
            raise TypeError("expected_shape must be a builtin tuple")
        if not shape:
            raise ValueError("a generated contact load requires a non-scalar shape")
        if any(type(size) is not int or size < 0 for size in shape):
            raise TypeError("expected_shape dimensions must be nonnegative builtin ints")
        raw = self.raw
        if type(raw) is not np.ndarray or raw.dtype != np.dtype(np.float32):
            raise TypeError("generated contact load must already be a native float32 ndarray")
        if raw.shape != shape:
            raise ValueError(f"generated contact shape {raw.shape} differs from {shape}")
        if not np.isfinite(raw).all():
            raise ValueError("generated contact load must be finite")

        # Snapshot the complete source before deriving any delivered field.
        saved = _immutable(raw)
        bits = saved.view(np.uint32)
        subnormal = ((bits & _EXPONENT) == 0) & ((bits & _FRACTION) != 0)
        delivered_bits = bits.copy()
        delivered_bits[subnormal] &= _SIGN
        delivered = delivered_bits.view(np.float32)
        # Decode the subnormal significand as an integer before any floating
        # arithmetic. Its 23-bit integer times 2^-149 is exact binary64 and does
        # not depend on the host's handling of a float32 subnormal conversion.
        loss = np.zeros(shape, dtype=np.float64)
        magnitude = (bits[subnormal] & _FRACTION).astype(np.float64) * float.fromhex("0x1.0p-149")
        loss[subnormal] = np.where((bits[subnormal] & _SIGN) != 0, -magnitude, magnitude)
        bound = np.zeros(shape, dtype=np.float64)
        bound[subnormal] = _TINY
        # These comparisons assert this fixed IEEE rule, not a fitted tolerance.
        if np.any(loss[~subnormal] != 0) or np.any(np.abs(loss[subnormal]) >= _TINY):
            raise ArithmeticError("generated contact delivery violates its IEEE loss contract")

        object.__setattr__(self, "raw", saved)
        object.__setattr__(self, "delivered", _immutable(delivered))
        object.__setattr__(self, "subnormal_mask", _immutable(subnormal))
        object.__setattr__(self, "loss", _immutable(loss))
        object.__setattr__(self, "loss_bound", _immutable(bound))


def normalize_contact_fp32(raw, *, expected_shape):
    """Normalize only subnormal generated loads; preserve source and exact loss.

    The explicit expected_shape must come from the existing physical load
    layout. This helper neither broadens that layout nor changes its capacity.
    External declarations must continue through their original input guard.
    """
    return ContactFP32Delivery(raw=raw, expected_shape=expected_shape)
