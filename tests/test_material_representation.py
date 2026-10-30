"""Independent dyadic input/publication controls; no native runtime required."""

import math
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.material_representation import (
    CODEC_BITS,
    NativeCompositeRepresentation,
    RepresentationBoundary,
    decode,
    deserialize_evidence,
    encode_values,
    exact,
    round_binary,
    serialize_evidence,
)
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import fingerprint
from phymeo.validation.material_representation import projection_charge, verify_rounding


@pytest.mark.parametrize(
    "bits,source,expected",
    [
        (32, F(1) + F(1, 2**24), 1.0),
        (32, F(1) + F(3, 2**24), 1.0 + 2**-22),
        (64, F(1) + F(1, 2**53), 1.0),
        (64, F(1) + F(3, 2**53), 1.0 + 2**-51),
        (32, -F(1) - F(1, 2**24), -1.0),
        (64, -F(1) - F(3, 2**53), -1.0 - 2**-51),
    ],
)
def test_exact_tie_even_codec_and_independent_cells(bits, source, expected):
    assert round_binary(source, bits) == expected
    assert verify_rounding(source, expected, bits) == F(expected) - source
    wrong = (
        np.nextafter(np.float32(expected), np.float32(np.inf))
        if bits == 32
        else math.nextafter(expected, math.inf)
    )
    with pytest.raises(SolverFailure, match="rounding"):
        verify_rounding(source, float(wrong), bits)


@pytest.mark.parametrize("scale", [F(1, 2), F(1), F(2)])
@pytest.mark.parametrize("shift", [F(0), F(2**10)])
def test_reference_displacement_retains_literal_small_update_and_inverse(scale, shift):
    reference = np.array([[float(scale + shift), 0.0, 0.0]])
    delta = float(scale / 2**25)
    native = np.array([[delta, 0.0, 0.0]], np.float32)
    world, witnesses = decode(reference, native)
    assert F(world[0, 0]) == scale * (1 + F(1, 2**25)) + shift
    assert witnesses[0].residual == 0
    forward = projection_charge(reference[0, 0], 0, delta, delta, 1, delta, 1)
    reverse = projection_charge(reference[0, 0], delta, 0, -delta, 1, -delta, 1)
    assert forward[:2] == reverse[:2] == (F(0), F(0))
    np.testing.assert_array_equal(decode(reference, np.zeros_like(native))[0], reference)
    assert world[0, 0] != float(np.float32(world[0, 0]))


def test_large_reference_decoder_loss_is_explicit_and_noncancelling():
    reference = np.array([[2.0**100, 0.0, 0.0]])
    displacement = np.array([[1.0, 0.0, 0.0]], np.float32)
    world, witnesses = decode(reference, displacement)
    assert world[0, 0] == reference[0, 0]
    assert witnesses[0].residual == -1
    charge, direct, terms = projection_charge(2.0**100, 0, 1, 1, 1, 1, 1)
    assert (charge, direct, terms) == (F(1), F(1), (F(0), F(0), F(0), F(-1)))
    reverse, _, _ = projection_charge(2.0**100, 1, 0, -1, 1, -1, 1)
    assert charge + reverse == 2
    assert (charge + reverse) / F(1, 128) > F(2, 100000)


@pytest.mark.parametrize(
    "source", [F(1, 2**127), F(1, 2**150), F(2) ** 128, F(2) ** 128 - F(2) ** 103]
)
def test_native_codec_normal_range_failures(source):
    with pytest.raises(SolverFailure):
        round_binary(source, 32)


def test_finite_codec_resources_and_world_subnormals_are_distinct():
    with pytest.raises(SolverFailure, match="resource"):
        round_binary(F(1, 2**CODEC_BITS), 64)
    with pytest.raises(SolverFailure, match="resource"):
        exact(1 << CODEC_BITS)
    assert round_binary(F(1, 2**1074), 64) == 2.0**-1074
    assert round_binary(F(1, 2**1075), 64) == 0
    assert math.copysign(1, round_binary(-0.0, 64)) == 1


def test_owned_boundary_exact_serialization_and_policy_rejection():
    values, witnesses = encode_values((F(1, 3), F(0)), (2,))
    boundary = RepresentationBoundary("control", values, witnesses)
    values[0] = 99
    assert boundary.computed[0] != 99 and not boundary.computed.flags.writeable
    restored = deserialize_evidence(serialize_evidence(boundary))
    assert fingerprint(boundary) == fingerprint(restored)
    assert restored.computed.tobytes() == boundary.computed.tobytes()
    with pytest.raises(ValueError):
        replace(NativeCompositeRepresentation(), codec_bits=8192)
    with pytest.raises(ValueError):
        NativeCompositeRepresentation(kind="two-word-position-f32-v2")
    with pytest.raises(ValueError):
        deserialize_evidence(b'{"type":"untrusted","fields":{}}')


def test_generic_references_permute_disconnect_and_keep_original_signed_zero():
    reference = np.array([[1.0, -0.0, 0.0], [1025.0, 0.0, 0.0]])
    original = reference.tobytes()
    d = np.array([[2**-25, 0, 0], [2**-25, 0, 0]], np.float32)
    world, _ = decode(reference, d)
    permuted, _ = decode(reference[::-1].copy(), d[::-1].copy())
    assert permuted.tobytes() == world[::-1].copy().tobytes()
    assert reference.tobytes() == original
    assert not np.signbit(world[:, 1:]).any()


def test_default_factories_preserve_host_and_v1_signatures_without_runtime():
    from test_composite_cloth import case, program_for

    from phymeo.backends.tyx_composite_cloth import TyxCompositeClothProgram

    model, material, _, _ = case()
    assert fingerprint(program_for(model, material).problem_signature()) == fingerprint(
        program_for(model, material, representation=None).problem_signature()
    )
    default = program_for(model, material, "interpret")
    explicit = program_for(model, material, "interpret", representation=None)
    assert isinstance(default, TyxCompositeClothProgram)
    assert fingerprint(default.problem_signature()) == fingerprint(explicit.problem_signature())
    assert "representation" not in default.problem_signature()
    represented = program_for(
        model, material, "interpret", representation=NativeCompositeRepresentation()
    )
    assert (
        represented.problem_signature()["world_publication"]
        == "RN64(reference+native_displacement)"
    )
