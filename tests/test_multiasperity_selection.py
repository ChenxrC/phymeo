"""Manufactured selector mechanism tests, not physical reference evidence."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from experiments.contact_selection.multiasperity_selection import (
    GEOMETRY,
    KNOTS,
    POINTS,
    VALIDATION,
    ForcePlan,
    choose_plan,
    evaluate,
)


def model():
    # Independent exact F=2*delta in every interval, in scipy coefficient order.
    return {
        "schema": "multi-asperity-force-plan-v1",
        "knots_m": list(KNOTS),
        "capabilities": ["force_n"],
        "calibration_force_n": [2 * x for x in KNOTS],
        "geometry": GEOMETRY,
        "modulus_pa": 1e6,
        "qualified_points_m": list(POINTS),
        "coefficients": [[0, 0, 0], [0, 0, 0], [2, 2, 2], [2 * x for x in KNOTS[:-1]]],
        "reference_checks": [
            {"delta_m": d, "kkt_256": True, "kkt_512": True, "force_grid_relative": 0.01}
            for d in POINTS
        ],
        "validation": [
            {"delta_m": d, "force_n": 2 * d, "predicted_n": 2 * d, "relative_error": 0.0}
            for d in VALIDATION
        ],
    }


def test_saved_polynomial_and_owned_immutable_plan():
    data = model()
    plan = choose_plan(data)
    data["coefficients"][2][0] = 999
    for delta in POINTS:
        assert plan.predict(delta) == {"force_n": pytest.approx(2 * delta)}
    with pytest.raises(FrozenInstanceError):
        plan.polynomial = ()


def test_cubic_horner_order_and_interval_origin():
    # F=(delta-origin)^3+2*(delta-origin)^2+3*(delta-origin)+4.
    c = ((1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4))
    x = 10e-6
    assert evaluate(c, KNOTS[2] + x) == pytest.approx(x**3 + 2 * x * x + 3 * x + 4)


@pytest.mark.parametrize(
    "options",
    [
        {"outputs": ("pressure_pa",)},
        {"outputs": ("force_n", "area_m2")},
        {"outputs": ("wrench",)},
        {"outputs": ("energy_j",)},
        {"geometry": "hertz"},
        {"modulus_pa": 2e6},
        {"modulus_pa": True},
    ],
)
def test_capability_and_material_refusal(options):
    with pytest.raises(ValueError):
        choose_plan(model(), **options)


@pytest.mark.parametrize("delta", [55e-6, 0, -1, True, float("nan"), float("inf")])
def test_no_continuous_domain_or_invalid_input_claim(delta):
    with pytest.raises(ValueError):
        choose_plan(model()).predict(delta)


@pytest.mark.parametrize(
    "field,value", [("kkt_512", False), ("kkt_256", False), ("force_grid_relative", 0.02001)]
)
def test_bad_reference_prevents_selection(field, value):
    data = model()
    data["reference_checks"][0][field] = value
    with pytest.raises(ValueError):
        choose_plan(data)


def test_holdout_failure_prevents_selection_without_refitting():
    data = model()
    original = deepcopy(data["coefficients"])
    data["validation"][1]["relative_error"] = 0.05001
    with pytest.raises(ValueError):
        choose_plan(data)
    assert data["coefficients"] == original


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "nonfinite", "negative", "flag", "shape"]
)
def test_malformed_evidence_is_not_admitted(change):
    data = model()
    if change == "duplicate":
        data["reference_checks"][0] = data["reference_checks"][1].copy()
    elif change == "missing":
        data["validation"].pop()
    elif change == "nonfinite":
        data["validation"][0]["relative_error"] = float("nan")
    elif change == "negative":
        data["validation"][0]["relative_error"] = -1
    elif change == "flag":
        data["reference_checks"][0]["kkt_256"] = 1
    else:
        data["coefficients"] = [[1]]
    with pytest.raises(ValueError):
        choose_plan(data)


def test_nonpositive_prediction_is_rejected():
    plan = ForcePlan(((0, 0, 0), (0, 0, 0), (0, 0, 0), (-1, -1, -1)))
    with pytest.raises(ValueError):
        plan.predict(POINTS[0])


@pytest.mark.parametrize(
    "fault", ["coefficients", "knots", "schema", "capabilities", "prediction", "truth"]
)
def test_current_model_is_bound_to_its_qualification_evidence(fault):
    data = model()
    if fault == "coefficients":
        data["coefficients"] = [[2 * v for v in row] for row in data["coefficients"]]
    elif fault == "knots":
        data["knots_m"][1] += 1e-6
    elif fault == "schema":
        data["schema"] = "different"
    elif fault == "capabilities":
        data["capabilities"].append("wrench")
    elif fault == "prediction":
        data["validation"][0]["predicted_n"] *= 2
    else:
        data["validation"][0]["force_n"] *= 2
    with pytest.raises(ValueError):
        choose_plan(data)


@pytest.mark.parametrize("force", [None, 0.0, -1.0])
def test_unavailable_reference_is_saved_as_refusal_before_fitting(tmp_path, force):
    from experiments.contact_selection.multiasperity_reference import configuration
    from experiments.contact_selection.multiasperity_selection import run

    reference = tmp_path / "reference"
    folder = reference / "c-n512-d020um"
    folder.mkdir(parents=True)
    case = {
        "grid": 512,
        "delta_m": 20e-6,
        "mode": "coupled",
        "config": configuration(512, 20e-6),
        "metrics": None if force is None else {"force_n": force},
    }
    path = folder / "case.json"
    raw = json.dumps(case)
    path.write_text(raw, encoding="utf-8")
    result = run(reference, tmp_path / "output")
    assert result["model"]["admission"] == "REFUSED_UNAVAILABLE_REFERENCE"
    assert result["new_solves"] == 0
    assert not result["research_target_passed"]
    assert "unavailable numerical force" in result["reason"]
    assert path.read_text(encoding="utf-8") == raw
    assert (tmp_path / "output" / "summary.json").is_file()
    assert not (tmp_path / "output" / "calibration-before-validation.json").exists()
