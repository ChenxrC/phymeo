"""Hand-authored saved-transition controls; no solver/kernel oracle."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "standalone_deformable_impulse_audit",
    Path(__file__).parents[2] / "examples" / "deformable_impulse_audit.py",
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def absorbed():
    # w=1, h=1/8, n=(1,0,0), delta=2^-26.
    # At x=1 the half-ULP is 2^-24: this positive displacement is absorbed.
    # Such delta is also consistent with length1, rest1+2^-23, alpha7, lambda0.
    x = np.array([[1, 0, 0], [0, 0, 0], [0.5, 0, 0]], np.float32)
    zero = np.zeros_like(x)
    frame = np.array(
        [
            [
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                2**-26,
                1,
                0,
                0,
                2**-26,
            ]
        ],
        np.float32,
    )
    return dict(
        input_position=x,
        input_velocity=zero.copy(),
        force=zero.copy(),
        inverse_mass=np.array([1, 0, 1], np.float32),
        gravity=np.zeros(3, np.float32),
        h=0.125,
        predicted_position=x.copy(),
        predicted_velocity=zero.copy(),
        edges=np.array([[0, 1]], np.int64),
        trace=(frame,),
        sweeps=1,
        final_position=x.copy(),
        final_velocity=zero.copy(),
    )


def ordered():
    # Two exactly represented movements: 1 -> 9/8 -> 1. Node3 is never touched.
    x = np.array([[1, 0, 0], [0, 0, 0], [2, 0, 0], [0.5, 0, 0]], np.float32)
    zero = np.zeros_like(x)
    frame = np.array(
        [
            [1, 0, 0, 0, 0, 0, 1.125, 0, 0, 0, 0, 0, 0, 0.125, 1, 0, 0, 0.125],
            [1.125, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0.125, -1, 0, 0, 0.125],
        ],
        np.float32,
    )
    return dict(
        input_position=x,
        input_velocity=zero.copy(),
        force=zero.copy(),
        inverse_mass=np.array([1, 0, 0, 1], np.float32),
        gravity=np.zeros(3, np.float32),
        h=0.125,
        predicted_position=x.copy(),
        predicted_velocity=zero.copy(),
        edges=np.array([[0, 1], [0, 2]], np.int64),
        trace=(frame,),
        sweeps=1,
        final_position=x.copy(),
        final_velocity=zero.copy(),
    )


def test_nonzero_material_impulse_absorbed_by_position_retains_raw_imbalance():
    values = absorbed()
    before = {k: v.tobytes() for k, v in values.items() if isinstance(v, np.ndarray)}
    result = audit.audit_deformable_impulse(**values)
    assert result["schema"] == "deformable-fp32-impulse-audit-v1"
    assert result["commanded_material_impulse"][0, 0] == 2**-23
    assert result["actual_impulse"][0, 0] == 0
    assert result["raw_impulse_imbalance"][0, 0] == -(2**-23)
    assert result["position_rounding_impulse"][0, 0] == -(2**-23)
    assert result["predictor_rounding_impulse"][0, 0] == 0
    assert result["reconstruction_rounding_impulse"][0, 0] == 0
    assert result["position_bound"][0, 0] >= 2**-23
    assert result["fixed_reaction_impulse"][1, 0] == 2**-23
    np.testing.assert_array_equal(result["bounded_remainder"], np.zeros((3, 3)))
    assert result["bound_method"] == "a-priori-binary32-half-ulp/exact-rational-v1"
    assert result["physical_conservation_claim"] is False
    for k, raw in before.items():
        assert values[k].tobytes() == raw


def test_exact_ordered_projection_keeps_unvisited_nodes_and_full_reaction():
    result = audit.audit_deformable_impulse(**ordered())
    np.testing.assert_array_equal(result["actual_impulse"], np.zeros((4, 3)))
    np.testing.assert_array_equal(result["raw_impulse_imbalance"], np.zeros((4, 3)))
    assert result["fixed_reaction_impulse"][1, 0] == 1
    assert result["fixed_reaction_impulse"][2, 0] == -1
    np.testing.assert_array_equal(result["bounded_remainder"], np.zeros((4, 3)))


def test_exact_nonzero_predictor_gravity_and_fixed_external_reaction():
    values = absorbed()
    values.update(
        inverse_mass=np.array([2, 0, 1], np.float32),
        input_velocity=np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32),
        force=np.array([[4, 0, 0], [8, 0, 0], [0, 0, 0]], np.float32),
        gravity=np.array([2, 0, 0], np.float32),
        edges=np.empty((0, 2), np.int64),
        trace=(),
        sweeps=0,
        predicted_velocity=np.array([[1.75, 0, 0], [0, 0, 0], [0.25, 0, 0]], np.float32),
        predicted_position=np.array([[1.21875, 0, 0], [0, 0, 0], [0.53125, 0, 0]], np.float32),
    )
    values["final_position"] = values["predicted_position"].copy()
    values["final_velocity"] = values["predicted_velocity"].copy()
    result = audit.audit_deformable_impulse(**values)
    assert result["actual_impulse"][0, 0] == 0.625
    assert result["external_impulse"][0, 0] == 0.5
    assert result["gravity_impulse"][0, 0] == 0.125
    assert result["fixed_reaction_impulse"][1, 0] == -1
    np.testing.assert_array_equal(result["raw_impulse_imbalance"], np.zeros((3, 3)))


@pytest.mark.parametrize(
    "fault",
    (
        "wrong-post",
        "wrong-pre",
        "wrong-lambda",
        "wrong-predicted-x",
        "wrong-predicted-v",
        "wrong-final-x",
        "wrong-final-v",
        "unvisited-moved",
        "fixed-moved",
        "fixed-velocity",
        "missing-row",
        "reordered",
        "missing-sweep",
        "extra-sweep",
    ),
)
def test_forged_or_incomplete_saved_transitions_fail(fault):
    values = ordered()
    if fault == "wrong-post":
        values["trace"][0][0, 6] = 1.25
    elif fault == "wrong-pre":
        values["trace"][0][1, 0] = 1
    elif fault == "wrong-lambda":
        values["trace"][0][0, 13] = 0.25
    elif fault == "wrong-predicted-x":
        values["predicted_position"][0, 0] = 1.25
    elif fault == "wrong-predicted-v":
        values["predicted_velocity"][0, 0] = 0.25
    elif fault == "wrong-final-x":
        values["final_position"][0, 0] = 1.25
    elif fault == "wrong-final-v":
        values["final_velocity"][0, 0] = 0.25
    elif fault == "unvisited-moved":
        values["final_position"][3, 0] = 0.75
    elif fault == "fixed-moved":
        values["final_position"][1, 0] = 0.25
    elif fault == "fixed-velocity":
        values["final_velocity"][1, 0] = 0.25
    elif fault == "missing-row":
        values["trace"] = (values["trace"][0][:1],)
    elif fault == "reordered":
        values["trace"] = (values["trace"][0][::-1].copy(),)
    elif fault == "missing-sweep":
        values["sweeps"] = 2
    elif fault == "extra-sweep":
        values["trace"] = values["trace"] * 2
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        audit.audit_deformable_impulse(**values)


@pytest.mark.parametrize(
    "fault",
    ("nonfinite", "subnormal-input", "subnormal-product", "dtype", "h", "edges", "bool-sweeps"),
)
def test_unsupported_saved_arithmetic_rejected(fault):
    values = absorbed()
    if fault == "nonfinite":
        values["force"][0, 0] = np.nan
    elif fault == "subnormal-input":
        values["force"][0, 0] = np.float32(2**-149)
    elif fault == "subnormal-product":
        values["inverse_mass"][0] = np.float32(2**-100)
        values["force"][0, 0] = np.float32(2**-100)
    elif fault == "dtype":
        values["predicted_position"] = values["predicted_position"].astype(np.float64)
    elif fault == "h":
        values["h"] = 0.1  # Not the stored binary32 scalar required by this contract.
    elif fault == "edges":
        values["edges"] = np.array([[0, 0]], np.int64)
    else:
        values["sweeps"] = True
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        audit.audit_deformable_impulse(**values)
