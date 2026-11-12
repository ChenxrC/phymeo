"""Fixed IEEE bit-pattern controls for generated contact-load delivery.

Authoring only: no TYX, solver, material, or native physics oracle is used.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phymeo.backends._contact_fp32_delivery import (
    ContactFP32Delivery,
    normalize_contact_fp32,
)

BITS = (
    0x00000001,
    0x007FFFFF,
    0x80000001,
    0x807FFFFF,
    0x00000000,
    0x80000000,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x800000AC,
)
DELIVERED = (
    0x00000000,
    0x00000000,
    0x80000000,
    0x80000000,
    0x00000000,
    0x80000000,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x80000000,
)
MASK = (True, True, True, True, False, False, False, False, False, False, False, False, True)


def raw_bits(bits=BITS):
    return np.array(bits, dtype=np.uint32).view(np.float32)


def test_fixed_ieee_boundary_patterns_keep_raw_and_exact_signed_loss():
    raw = raw_bits()
    receipt = normalize_contact_fp32(raw, expected_shape=(len(BITS),))
    assert isinstance(receipt, ContactFP32Delivery)
    assert receipt.schema == "generated-contact-fp32-delivery-v1"
    assert receipt.expected_shape == (len(BITS),)
    np.testing.assert_array_equal(receipt.raw.view(np.uint32), BITS)
    np.testing.assert_array_equal(receipt.delivered.view(np.uint32), DELIVERED)
    np.testing.assert_array_equal(receipt.subnormal_mask, MASK)
    expected_loss = np.array(
        [
            2.0**-149,
            (2**23 - 1) * 2.0**-149,
            -(2.0**-149),
            -(2**23 - 1) * 2.0**-149,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -172 * 2.0**-149,
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(receipt.loss, expected_loss)
    np.testing.assert_array_equal(
        receipt.loss_bound,
        np.array([2.0**-126 if active else 0 for active in MASK], dtype=np.float64),
    )
    assert receipt.raw.dtype == receipt.delivered.dtype == np.dtype(np.float32)
    assert receipt.subnormal_mask.dtype == np.dtype(np.bool_)
    assert receipt.loss.dtype == receipt.loss_bound.dtype == np.dtype(np.float64)
    assert np.all(abs(receipt.loss[receipt.subnormal_mask]) < 2.0**-126)
    np.testing.assert_array_equal(receipt.loss[~receipt.subnormal_mask], 0)
    # In particular the observed bit pattern 0x800000ac remains visible as raw
    # and its delivered result retains the negative sign bit.
    assert receipt.raw.view(np.uint32)[-1] == 0x800000AC
    assert receipt.delivered.view(np.uint32)[-1] == 0x80000000


@pytest.mark.parametrize(
    "shape", ((3,), (1, 3), (2, 3), (2, 1, 3), (1, 2, 1, 3), (0, 3), (2, 0, 6))
)
def test_explicit_shapes_including_empty_axes_preserve_complete_layout(shape):
    size = int(np.prod(shape))
    bits = np.resize(np.array([0x00000001, 0x80000001, 0x3F800000], np.uint32), size)
    raw = bits.reshape(shape).view(np.float32)
    result = normalize_contact_fp32(raw, expected_shape=shape)
    expected = np.resize(np.array([0x00000000, 0x80000000, 0x3F800000], np.uint32), size)
    for name in ("raw", "delivered", "subnormal_mask", "loss", "loss_bound"):
        assert getattr(result, name).shape == shape
    np.testing.assert_array_equal(result.raw.view(np.uint32), bits.reshape(shape))
    np.testing.assert_array_equal(result.delivered.view(np.uint32), expected.reshape(shape))


def test_noncontiguous_array_is_owned_without_reordering_values():
    owner = raw_bits()
    raw = owner[::2]
    assert not raw.flags.c_contiguous
    result = normalize_contact_fp32(raw, expected_shape=raw.shape)
    np.testing.assert_array_equal(result.raw.view(np.uint32), BITS[::2])
    np.testing.assert_array_equal(result.delivered.view(np.uint32), DELIVERED[::2])
    owner[:] = 2
    np.testing.assert_array_equal(result.raw.view(np.uint32), BITS[::2])


def test_all_receipt_arrays_are_bytes_backed_and_cannot_be_made_writable():
    original = raw_bits()
    result = normalize_contact_fp32(original, expected_shape=original.shape)
    original[:] = 3
    np.testing.assert_array_equal(result.raw.view(np.uint32), BITS)
    for name in ("raw", "delivered", "subnormal_mask", "loss", "loss_bound"):
        value = getattr(result, name)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
        with pytest.raises(ValueError):
            value.flat[0] = 0
        base = value
        while isinstance(base, np.ndarray) and base.base is not None:
            base = base.base
        assert isinstance(base, bytes)
    with pytest.raises(FrozenInstanceError):
        result.schema = "changed"
    with pytest.raises(FrozenInstanceError):
        result.raw = original


@pytest.mark.parametrize("bits", (0x7F800000, 0xFF800000, 0x7FC00001))
def test_nonfinite_input_is_rejected_not_flushed(bits):
    with pytest.raises(ValueError, match="finite"):
        normalize_contact_fp32(raw_bits((bits,)), expected_shape=(1,))


@pytest.mark.parametrize(
    "raw",
    (
        np.array([1], np.float64),
        np.array([1], np.int32),
        np.array([1], dtype=">f4"),
        [np.float32(1)],
    ),
)
def test_wrong_input_type_or_dtype_is_not_silently_converted(raw):
    with pytest.raises(TypeError, match="float32"):
        normalize_contact_fp32(raw, expected_shape=(1,))


@pytest.mark.parametrize("expected", ((), (True,), (-1,), (np.int64(1),), [1], (2,), (1, 1)))
def test_invalid_shape_declaration_or_mismatch_is_rejected(expected):
    with pytest.raises((TypeError, ValueError)):
        normalize_contact_fp32(raw_bits((0x00000001,)), expected_shape=expected)


def test_scalar_array_is_outside_the_declared_load_layout():
    with pytest.raises(ValueError, match="shape"):
        normalize_contact_fp32(np.array(1, np.float32), expected_shape=())


def test_direct_receipt_construction_has_the_same_validation_and_ownership():
    raw = raw_bits((0x80000001, 0x00800000))
    result = ContactFP32Delivery(raw=raw, expected_shape=(2,))
    raw[:] = 0
    np.testing.assert_array_equal(result.raw.view(np.uint32), (0x80000001, 0x00800000))
    with pytest.raises(ValueError, match="finite"):
        ContactFP32Delivery(raw=raw_bits((0x7F800000,)), expected_shape=(1,))
