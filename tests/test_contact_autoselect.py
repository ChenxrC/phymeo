"""HOST mechanism tests using manufactured qualification/cost evidence.

The registry fixtures are intentionally synthetic: their error numbers are not
claims about FFT, Hertz, or measured timing. Formula expectations below are
independent hand calculations in SI. No external solver is imported or run.
"""

import copy
import hashlib
import json
import math
from dataclasses import FrozenInstanceError

import pytest

from experiments.contact_selection import autoselect as module
from experiments.contact_selection import run_calibration as runner

ALGORITHMS = ("linear", "hertz_shaped", "quadratic", "hertz_analytic")
DELTAS = (0.00005, 0.000075, 0.0001, 0.00015, 0.0002)
MECHANICAL = ("force", "tangent", "energy")
ALL_SCALARS = (*MECHANICAL, "area", "peak")


def _timing(median):
    return {
        "min_s": median * 0.9,
        "median_s": median,
        "max_s": median * 1.1,
        "batches": 7,
        "calls_per_batch": 1000,
    }


def _evidence():
    """Manufactured mechanism data; not a physical qualification fixture."""
    calibration = module.Calibration(2.0, "a" * 64)
    validation, costs = {}, {}
    force_costs = {"linear": 1e-6, "hertz_shaped": 2e-6, "quadratic": 3e-6, "hertz_analytic": 4e-6}
    constitutive_costs = {
        "linear": 8e-6,
        "hertz_shaped": 6e-6,
        "quadratic": 7e-6,
        "hertz_analytic": 5e-6,
    }
    errors = {"linear": 0.0, "hertz_shaped": 0.02, "quadratic": 0.04, "hertz_analytic": 0.0}
    for algorithm in ALGORITHMS:
        metrics = ALL_SCALARS if algorithm == "hertz_analytic" else MECHANICAL
        validation[algorithm] = [
            {"delta_m": delta, "relative_errors": {metric: errors[algorithm] for metric in metrics}}
            for delta in DELTAS
        ]
        costs[algorithm] = {
            "force": _timing(force_costs[algorithm]),
            "constitutive": _timing(constitutive_costs[algorithm]),
        }
    # The fastest candidate is perfect at the requested/calibration point but
    # exceeds the default gate at another validated point.
    validation["linear"][-1]["relative_errors"] = {metric: 0.06 for metric in MECHANICAL}
    costs["hertz_analytic"]["scalars"] = _timing(9e-6)
    references = {"calibration": "a" * 64, "historical": "b" * 64, "prospective": "c" * 64}
    return calibration, validation, costs, references


def _registry():
    return module.seal_registry(*_evidence())


def _reseal(registry):
    """Independent checksum encoding, including deliberate structural faults."""
    result = copy.deepcopy(registry)
    result.pop("binding_sha256", None)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["binding_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return result


def _cards(result):
    return {item["algorithm"]: item for item in result["candidates"]}


@pytest.mark.parametrize(
    "algorithm, force, tangent, energy",
    (
        ("linear", 8.0, 20000.0, 0.0016),
        ("hertz_shaped", 16.0, 60000.0, 0.00256),
        ("quadratic", 32.0, 160000.0, 0.0128 / 3.0),
    ),
)
def test_each_power_has_its_own_si_coefficient_and_work_integral(algorithm, force, tangent, energy):
    # F(100 um)=2 N. At four times the indentation, factors are 4, 8, 16.
    # dF/ddelta is N/m; integral_0^delta F ds is J, with U(0)=0.
    calibration = module.Calibration(2.0, "a" * 64)
    prediction = module.predict(algorithm, 0.0004, calibration, MECHANICAL)
    assert prediction == pytest.approx({"force": force, "tangent": tangent, "energy": energy})
    assert module.predict(algorithm, 0.0001, calibration)["force"] == pytest.approx(2.0)


def test_hertz_independent_si_values_and_no_calibration_dependence():
    # At R=.02 m, E*=1e6 Pa and delta=50 um: a=1 mm,
    # F=1/15 N, k=2000 N/m, U=1/750000 J, area=pi*1e-6 m2.
    expected = {
        "force": 1 / 15,
        "tangent": 2000.0,
        "energy": 1 / 750000,
        "area": math.pi * 1e-6,
        "peak": 100000 / math.pi,
    }
    first = module.predict("hertz_analytic", 0.00005, module.Calibration(2, "a" * 64), ALL_SCALARS)
    other = module.predict(
        "hertz_analytic", 0.00005, module.Calibration(800, "f" * 64), ALL_SCALARS
    )
    assert first == pytest.approx(expected, rel=2e-14, abs=0)
    assert other == first
    assert first["tangent"] * 0.00005 == pytest.approx(1.5 * first["force"])
    assert first["energy"] == pytest.approx(0.4 * first["force"] * 0.00005)


@pytest.mark.parametrize(
    "force, sha, delta",
    (
        (0, "a" * 64, 0.0001),
        (-1, "a" * 64, 0.0001),
        (True, "a" * 64, 0.0001),
        (math.inf, "a" * 64, 0.0001),
        (2, "A" * 64, 0.0001),
        (2, "a" * 63, 0.0001),
        (2, "a" * 64, 0.000075),
    ),
)
def test_calibration_requires_one_bound_positive_si_measurement(force, sha, delta):
    with pytest.raises(ValueError):
        module.Calibration(force, sha, delta)


def test_calibration_is_frozen_and_predictions_are_owned():
    calibration = module.Calibration(2, "a" * 64)
    with pytest.raises(FrozenInstanceError):
        calibration.force_n = 3
    result = module.predict("linear", 0.0001, calibration, MECHANICAL)
    result["force"] = -1
    assert module.predict("linear", 0.0001, calibration)["force"] == pytest.approx(2)
    doubled = module.predict("linear", 0.0001, module.Calibration(4, "a" * 64), MECHANICAL)
    assert doubled == pytest.approx({"force": 4, "tangent": 40000, "energy": 0.0002})


@pytest.mark.parametrize(
    "algorithm, delta, outputs",
    (
        ("unknown", 0.0001, ("force",)),
        ("linear", 0, ("force",)),
        ("linear", True, ("force",)),
        ("linear", math.nan, ("force",)),
        ("linear", math.inf, ("force",)),
        ("linear", 0.0001, ()),
        ("linear", 0.0001, ("force", "force")),
        ("quadratic", 0.0001, ("area",)),
        ("hertz_analytic", 0.0001, ("field",)),
    ),
)
def test_formula_rejects_bad_or_unsupported_requests(algorithm, delta, outputs):
    with pytest.raises(ValueError):
        module.predict(algorithm, delta, module.Calibration(2, "a" * 64), outputs)


def test_offline_validation_and_reference_sources_do_not_recalibrate(monkeypatch):
    calibration, rows, costs, references = _evidence()
    original_predict = module.predict
    monkeypatch.setattr(
        module, "predict", lambda *a, **k: pytest.fail("sealing must not evaluate or fit a formula")
    )
    first = module.seal_registry(calibration, rows, costs, references)
    rows["hertz_shaped"][0]["relative_errors"]["force"] = 0.03
    references["prospective"] = "d" * 64
    second = module.seal_registry(calibration, rows, costs, references)
    assert first["binding_sha256"] != second["binding_sha256"]
    assert (
        first["calibration"]
        == second["calibration"]
        == {
            "force_n": 2.0,
            "source_sha256": "a" * 64,
            "delta_m": 0.0001,
        }
    )
    monkeypatch.setattr(module, "predict", original_predict)
    a = module.Selector(first).select(0.00015, allowed=("hertz_shaped",))
    b = module.Selector(second).select(0.00015, allowed=("hertz_shaped",))
    assert a["status"] == b["status"] == "SELECTED"
    assert a["prediction"] == b["prediction"]


def test_default_includes_analytic_and_uses_worst_of_all_five_points():
    selection = module.Selector(_registry()).select(0.0001)
    cards = _cards(selection)
    assert set(cards) == set(ALGORITHMS)
    assert cards["hertz_analytic"]["eligible"]
    assert selection["algorithm"] == "hertz_shaped"
    assert cards["linear"]["reason"] == "ACCURACY_GATE_FAILED"
    assert cards["linear"]["failures"]["force"] == {
        "relative_error": 0.06,
        "delta_m": 0.0002,
        "requested_limit": 0.05,
    }
    # The current point's manufactured linear error is zero; it must not win.
    assert selection["prediction"]["force"] == pytest.approx(2.0)


def test_queries_evaluate_only_the_selected_formula_once_and_never_requalify(monkeypatch):
    selector = module.Selector(_registry())
    original = module.predict
    calls = []

    def spy(algorithm, delta, calibration, outputs):
        calls.append((algorithm, delta, calibration, outputs))
        return original(algorithm, delta, calibration, outputs)

    monkeypatch.setattr(module, "predict", spy)
    for name in ("seal_registry", "domain", "digest"):
        monkeypatch.setattr(
            module, name, lambda *a, **k: pytest.fail("query accessed offline qualification path")
        )
    result = selector.select(0.000075, profile="constitutive")
    assert result["algorithm"] == "hertz_analytic"
    assert calls == [("hertz_analytic", 0.000075, selector.calibration, MECHANICAL)]
    assert set(result["prediction"]) == set(MECHANICAL)
    assert result["latency_is_empirical_not_realtime_guarantee"] is True


@pytest.mark.parametrize(
    "delta, kwargs, reason",
    (
        (0.000125, {}, "OUTSIDE_VALIDATED_POINTS"),
        (0.000025, {}, "OUTSIDE_VALIDATED_POINTS"),
        (0.0001, {"physics": "dynamic"}, "UNSUPPORTED_PHYSICS"),
        (0.0001, {"radius_m": 0.03}, "UNSUPPORTED_PHYSICS"),
        (0.0001, {"modulus_pa": 2e6}, "UNSUPPORTED_PHYSICS"),
        (0.0001, {"domain_m": 0.32}, "UNSUPPORTED_PHYSICS"),
        (0.0001, {"profile": "field"}, "MISSING_OUTPUT"),
        (0.0001, {"allowed": ()}, "NO_ALLOWED_CANDIDATE"),
        (0.0001, {"allowed": ("mystery",)}, "INVALID_REQUEST"),
        (0.0001, {"allowed": ("linear", "linear")}, "INVALID_REQUEST"),
        (0.0001, {"tolerances": {}}, "INVALID_REQUEST"),
        (0.0001, {"tolerances": {"force": -0.1}}, "INVALID_REQUEST"),
        (0.0001, {"tolerances": {"force": 1.0}}, "INVALID_REQUEST"),
        (0.0001, {"budget_s": -1}, "INVALID_REQUEST"),
        (math.nan, {}, "INVALID_REQUEST"),
        (True, {}, "INVALID_REQUEST"),
    ),
)
def test_refusals_are_explanatory_and_never_predict(monkeypatch, delta, kwargs, reason):
    selector = module.Selector(_registry())
    monkeypatch.setattr(
        module, "predict", lambda *a, **k: pytest.fail("refused request evaluated a formula")
    )
    result = selector.select(delta, **kwargs)
    assert result["status"] == "REFUSED"
    assert result["reason"] == reason
    assert result["algorithm"] is None and result["prediction"] is None


def test_capability_filter_and_costs_match_requested_output_group():
    selector = module.Selector(_registry())
    permitted = ("hertz_shaped", "hertz_analytic")
    force = selector.select(0.0001, profile="force", allowed=permitted)
    mechanical = selector.select(0.0001, profile="constitutive", allowed=permitted)
    scalar = selector.select(0.0001, profile="scalars")
    assert force["algorithm"] == "hertz_shaped" and force["cost"]["median_s"] == 2e-6
    assert mechanical["algorithm"] == "hertz_analytic" and mechanical["cost"]["median_s"] == 5e-6
    assert scalar["algorithm"] == "hertz_analytic" and scalar["cost"]["median_s"] == 9e-6
    assert set(scalar["prediction"]) == set(ALL_SCALARS)
    for name in ("linear", "hertz_shaped", "quadratic"):
        assert _cards(scalar)[name]["reason"] == "MISSING_OUTPUT"
    reduced_only = selector.select(0.0001, profile="scalars", allowed=("hertz_shaped",))
    assert reduced_only["reason"] == "NO_ALLOWED_CANDIDATE"
    assert _cards(reduced_only)["hertz_analytic"]["reason"] == "NOT_ALLOWED"
    assert not scalar["speed_advantage_resolved"]


def test_budget_is_explicit_soft_median_bound_without_silent_fallback():
    selector = module.Selector(_registry())
    equal = selector.select(0.0001, budget_s=2e-6)
    assert equal["algorithm"] == "hertz_shaped"
    refused = selector.select(0.0001, budget_s=1.99e-6)
    assert refused["reason"] == "NO_ALLOWED_CANDIDATE" and refused["prediction"] is None
    assert _cards(refused)["linear"]["reason"] == "ACCURACY_GATE_FAILED"
    assert _cards(refused)["hertz_shaped"]["reason"] == "COST_BUDGET_EXCEEDED"
    assert selector.select(0.0001, budget_s=0)["status"] == "REFUSED"


def test_tightening_every_required_gate_cannot_increase_qualification():
    selector = module.Selector(_registry())
    sets = []
    for gate in (0.05, 0.01, 0.001, 0.0):
        result = selector.select(
            0.0001, profile="constitutive", tolerances={m: gate for m in MECHANICAL}
        )
        sets.append({c["algorithm"] for c in result["candidates"] if c["eligible"]})
    assert sets[0] == {"hertz_shaped", "quadratic", "hertz_analytic"}
    assert sets[1] == sets[2] == sets[3] == {"hertz_analytic"}
    assert all(after <= before for before, after in zip(sets, sets[1:]))
    rejected = selector.select(0.0001, allowed=("hertz_shaped",), tolerances={"force": 0.001})
    assert rejected["reason"] == "NO_ALLOWED_CANDIDATE"


def test_external_registry_and_return_mutations_do_not_change_loaded_selector():
    registry = _registry()
    selector = module.Selector(registry)
    first = selector.select(0.0001)
    expected_binding = first["binding_sha256"]
    registry["calibration"]["force_n"] = 900
    registry["cards"].clear()
    registry["references"].clear()
    first["cost"]["median_s"] = 123
    first["prediction"]["force"] = -999
    first["candidates"][1]["errors"]["force"]["relative_error"] = 0.99
    second = selector.select(0.0001)
    assert second["binding_sha256"] == expected_binding
    assert second["algorithm"] == "hertz_shaped"
    assert second["prediction"]["force"] == pytest.approx(2)
    assert second["cost"]["median_s"] == 2e-6
    assert _cards(second)["hertz_shaped"]["errors"]["force"]["relative_error"] == 0.02


def test_loaded_calibration_cannot_be_rebound_under_existing_qualification():
    selector = module.Selector(_registry())
    with pytest.raises((AttributeError, TypeError)):
        selector.calibration = module.Calibration(2000, "d" * 64)
    assert selector.select(0.0001)["prediction"]["force"] == pytest.approx(2)


def test_integrity_checksum_detects_change_without_resealing():
    registry = _registry()
    registry["calibration"]["force_n"] = 3
    with pytest.raises(ValueError, match="UNQUALIFIED_EVIDENCE"):
        module.Selector(registry)


@pytest.mark.parametrize(
    "fault",
    (
        "schema",
        "formula",
        "domain",
        "source_missing",
        "source_invalid",
        "source_mismatch",
        "candidate_missing",
        "capability",
        "point_missing",
        "point_order",
        "metric_missing",
        "metric_extra",
        "error_bool",
        "error_negative",
        "worst_missing",
        "worst_inconsistent",
        "cost_missing",
        "cost_extra",
        "timing_bounds",
        "timing_zero",
        "batches_wrong",
        "calls_wrong",
        "batches_float",
        "calls_float",
    ),
)
def test_recomputed_checksum_does_not_authorize_structurally_invalid_evidence(fault):
    registry = _registry()
    card = registry["cards"]["hertz_shaped"]
    timing = card["costs"]["force"]
    if fault == "schema":
        registry["schema"] = "other-v1"
    elif fault == "formula":
        registry["formula_version"] = "power-laws-and-hertz-v2"
    elif fault == "domain":
        registry["domain"]["radius_m"] = 0.03
    elif fault == "source_missing":
        registry["references"].pop("prospective")
    elif fault == "source_invalid":
        registry["references"]["prospective"] = "not-a-source-sha"
    elif fault == "source_mismatch":
        registry["references"]["calibration"] = "d" * 64
    elif fault == "candidate_missing":
        registry["cards"].pop("quadratic")
    elif fault == "capability":
        card["outputs"].append("area")
    elif fault == "point_missing":
        card["rows"].pop()
    elif fault == "point_order":
        card["rows"].reverse()
    elif fault == "metric_missing":
        card["rows"][0]["relative_errors"].pop("energy")
    elif fault == "metric_extra":
        card["rows"][0]["relative_errors"]["area"] = 0
    elif fault == "error_bool":
        card["rows"][0]["relative_errors"]["force"] = True
    elif fault == "error_negative":
        card["rows"][0]["relative_errors"]["force"] = -0.001
    elif fault == "worst_missing":
        card["worst"].pop("tangent")
    elif fault == "worst_inconsistent":
        card["worst"]["force"]["relative_error"] = 0
    elif fault == "cost_missing":
        card["costs"].pop("constitutive")
    elif fault == "cost_extra":
        card["costs"]["scalars"] = _timing(1e-6)
    elif fault == "timing_bounds":
        timing["min_s"] = timing["max_s"] * 2
    elif fault == "timing_zero":
        timing["min_s"] = 0
    elif fault == "batches_wrong":
        timing["batches"] = 6
    elif fault == "calls_wrong":
        timing["calls_per_batch"] = 999
    elif fault == "batches_float":
        timing["batches"] = 7.0
    else:
        timing["calls_per_batch"] = 1000.0
    with pytest.raises(ValueError, match="UNQUALIFIED_EVIDENCE"):
        module.Selector(_reseal(registry))


def test_measured_range_overlap_does_not_claim_a_resolved_speed_advantage():
    calibration, rows, costs, references = _evidence()
    separated = module.Selector(module.seal_registry(calibration, rows, costs, references)).select(
        0.0001
    )
    assert separated["speed_advantage_resolved"] is True
    costs["hertz_shaped"]["force"] = {**_timing(2e-6), "min_s": 1.8e-6, "max_s": 2.3e-6}
    costs["hertz_analytic"]["force"] = {**_timing(2.1e-6), "min_s": 1.9e-6, "max_s": 2.5e-6}
    overlapping = module.Selector(
        module.seal_registry(calibration, rows, costs, references)
    ).select(0.0001)
    assert overlapping["algorithm"] == "hertz_shaped"
    assert overlapping["speed_advantage_resolved"] is False
    assert "Overlapping" in overlapping["speed_note"]
    assert overlapping["cost"]["median_s"] == 2e-6
    assert overlapping["latency_is_empirical_not_realtime_guarantee"] is True


def _reference_case(delta, identifier, *, prospective=False):
    metrics = {"force_n": 2.0, "area_m2": 3e-6, "peak_pressure_pa": 50000.0}
    metrics.update({"kkt_pass": True} if prospective else {"kkt": {"passed": True}})
    return {
        "id": identifier,
        "grid": 512,
        "delta_m" if prospective else "delta": delta,
        "status": "FAIL_ACCURACY",  # A research-error failure is distinct from failed KKT.
        "config": {
            "radius_m": 0.02,
            "contact_modulus_pa": 1e6,
            "physical_sizes_m": [0.16, 0.16],
            "periodic": False,
        },
        "metrics": metrics,
        "timing": {"solver_s": 0.25},
    }


def _reference_inputs():
    historical = {
        "cases": [
            _reference_case(0.00005, "n512-d50um"),
            _reference_case(0.0001, "n512-d100um"),
            _reference_case(0.0002, "n512-d200um"),
        ]
    }
    prospective = {
        "cases": [
            _reference_case(0.000075, "n512-d75um", prospective=True),
            _reference_case(0.00015, "n512-d150um", prospective=True),
        ]
    }
    return historical, prospective


def test_fit_only_reads_the_fixed_kkt_true_one_point_and_owns_its_parameters():
    original = _reference_case(0.0001, "n512-d100um")
    fit = runner.fit_case(original, "a" * 64)
    assert fit == module.Calibration(2.0, "a" * 64)
    original["metrics"]["force_n"] = 900
    assert fit.force_n == 2.0
    changes = (
        ("delta", 0.000075),
        ("id", "n512-d75um"),
        ("grid", 256),
    )
    for key, value in changes:
        case = _reference_case(0.0001, "n512-d100um")
        case[key] = value
        with pytest.raises(ValueError):
            runner.fit_case(case, "a" * 64)
    for passed in (False, "false", 1):
        case = _reference_case(0.0001, "n512-d100um")
        case["metrics"]["kkt"]["passed"] = passed
        with pytest.raises(ValueError):
            runner.fit_case(case, "a" * 64)
    case = _reference_case(0.0001, "n512-d100um")
    case["config"]["periodic"] = True
    with pytest.raises(ValueError):
        runner.fit_case(case, "a" * 64)


def test_reference_rows_keep_roles_and_do_not_invent_fft_derivatives_or_energy():
    historical, prospective = _reference_inputs()
    # Other old grids exist in the historical nine-case experiment and are
    # intentionally irrelevant to the selected 512-grid qualification.
    unused = _reference_case(0.0001, "n128-d100um")
    unused["grid"] = 128
    unused["metrics"]["kkt"]["passed"] = False
    historical["cases"].append(unused)
    rows = runner.load_reference_rows(historical, prospective)
    assert [row["delta_m"] for row in rows] == list(DELTAS)
    assert [row["role"] for row in rows] == [
        "retrospective",
        "prospective-qualification",
        "calibration",
        "prospective-qualification",
        "retrospective",
    ]
    assert all(row["source_status"] == "FAIL_ACCURACY" for row in rows)
    assert all(row["fft"]["tangent"] is None and row["fft"]["energy"] is None for row in rows)
    assert all(set(row["oracle"]) == set(ALL_SCALARS) for row in rows)
    prospective["cases"][0]["metrics"]["force_n"] = 99
    assert rows[1]["fft"]["force"] == 2.0


@pytest.mark.parametrize(
    "fault",
    (
        "missing_prospective",
        "failed_prospective",
        "truthy_prospective",
        "failed_historical",
        "duplicate_prospective",
        "duplicate_historical",
        "wrong_grid",
        "wrong_material",
    ),
)
def test_incomplete_or_unusable_reference_evidence_cannot_qualify(fault):
    historical, prospective = _reference_inputs()
    if fault == "missing_prospective":
        prospective["cases"].pop()
    elif fault == "failed_prospective":
        prospective["cases"][0]["metrics"]["kkt_pass"] = False
    elif fault == "truthy_prospective":
        prospective["cases"][0]["metrics"]["kkt_pass"] = "false"
    elif fault == "failed_historical":
        historical["cases"][0]["metrics"]["kkt"]["passed"] = False
    elif fault == "duplicate_prospective":
        # A later good duplicate must not hide an earlier failed solve.
        prospective["cases"].append(copy.deepcopy(prospective["cases"][0]))
        prospective["cases"][0]["metrics"]["kkt_pass"] = False
    elif fault == "duplicate_historical":
        historical["cases"].append(copy.deepcopy(historical["cases"][0]))
        historical["cases"][0]["metrics"]["kkt"]["passed"] = False
    elif fault == "wrong_grid":
        prospective["cases"][0]["grid"] = 256
    else:
        prospective["cases"][0]["config"]["radius_m"] = 0.03
    with pytest.raises(ValueError):
        runner.load_reference_rows(historical, prospective)


def test_runner_oracle_has_independent_dimensional_values(monkeypatch):
    monkeypatch.setattr(
        runner, "predict", lambda *a, **k: pytest.fail("oracle used candidate producer")
    )
    assert runner.oracle(0.00005) == pytest.approx(
        {
            "force": 1 / 15,
            "tangent": 2000.0,
            "energy": 1 / 750000,
            "area": math.pi * 1e-6,
            "peak": 100000 / math.pi,
        },
        rel=2e-14,
        abs=0,
    )


def test_qualification_does_not_refit_and_keeps_fft_disagreement_separate():
    historical, prospective = _reference_inputs()
    calibration = runner.fit_case(historical["cases"][1], "a" * 64)
    references = runner.load_reference_rows(historical, prospective)
    first = runner.qualify(calibration, references)
    # Analytic baseline is qualified against physics, even though the supplied
    # manufactured FFT force is deliberately far from the Hertz value.
    analytic = first["hertz_analytic"][0]
    assert analytic["relative_errors"] == pytest.approx({m: 0.0 for m in ALL_SCALARS}, abs=2e-14)
    assert analytic["relative_to_fft"]["force"] == pytest.approx(29 / 30)
    assert analytic["relative_to_fft"]["tangent"] is None
    assert analytic["relative_to_fft"]["energy"] is None
    assert first["linear"][2]["prediction"]["force"] == pytest.approx(2.0)
    changed = copy.deepcopy(references)
    for row in changed:
        row["fft"]["force"] *= 100
        row["oracle"]["force"] *= 7
    second = runner.qualify(calibration, changed)
    assert calibration == module.Calibration(2.0, "a" * 64)
    for algorithm in ALGORITHMS:
        assert [r["prediction"] for r in first[algorithm]] == [
            r["prediction"] for r in second[algorithm]
        ]
    assert first["linear"][0]["relative_errors"] != second["linear"][0]["relative_errors"]
    assert first["linear"][0]["relative_to_fft"] != second["linear"][0]["relative_to_fft"]
