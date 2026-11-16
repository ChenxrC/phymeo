"""New-point schema/units/export tests only: no external FFT or physics execution."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.contact_selection import calibration_reference as reference


@pytest.mark.parametrize(
    "delta,force,area,tangent",
    [
        (75e-6, np.sqrt(6) / 20, 1.5e-6 * np.pi, 1000 * np.sqrt(6)),
        (150e-6, np.sqrt(3) / 5, 3e-6 * np.pi, 2000 * np.sqrt(3)),
    ],
)
def test_new_hertz_units_and_analytic_tangent(delta, force, area, tangent):
    result = reference.analytic_reference(delta)
    assert result["force_n"] == pytest.approx(force, rel=1e-14)
    assert result["area_m2"] == pytest.approx(area, rel=1e-14)
    assert result["tangent_n_m"] == pytest.approx(tangent, rel=1e-14)
    assert result["peak_pressure_pa"] == pytest.approx(50 * tangent / np.pi)
    assert result["tangent_n_m"] == pytest.approx(1.5 * force / delta)


@pytest.mark.parametrize("bad", [True, 50e-6, 100e-6, 200e-6, float("nan"), float("inf")])
def test_only_the_two_new_points_are_admitted(bad):
    with pytest.raises(ValueError):
        reference.configuration(bad)


def test_frozen_grid_and_solver_configuration_are_owned_and_unchanged():
    config = reference.configuration(75e-6)
    assert config["grid"] == 512
    assert config["computational_grid"] == [1024, 1024]
    assert config["physical_sizes_m"] == [0.16, 0.16]
    assert (config["radius_m"], config["contact_modulus_pa"]) == (0.02, 1e6)
    assert config["solver"] == {
        "pentol_m": 1e-10,
        "forcetol_n": 1e-10,
        "thermotol": 1e-8,
        "maxiter": 10000,
    }
    assert config["periodic"] is False
    config["solver"]["maxiter"] = 1
    assert reference.configuration(150e-6)["solver"]["maxiter"] == 10000
    inputs = reference.input_arrays(75e-6)
    assert inputs["x"][256] == 0
    assert inputs["x"][0] == -0.08
    assert inputs["heights"][0, 0] == pytest.approx(-0.32)
    assert inputs["heights"].shape == (512, 512)
    assert inputs["delta_m"].shape == ()


def test_nested_optimizer_arrays_preserve_data_dtype_shape_and_ownership():
    scalar = np.array(2.5, dtype=np.float64)
    vector = np.array([1, 2], dtype=np.int32)
    raw = {
        "maxcv": {"max_pres": scalar, "trace": [vector]},
        "nonfinite": np.array([np.nan, np.inf]),
    }
    encoded = reference.json_value(raw)
    scalar[...] = 99
    vector[:] = 7
    loaded = json.loads(json.dumps(encoded, allow_nan=False))
    assert loaded["maxcv"]["max_pres"] == {"ndarray_data": 2.5, "dtype": "float64", "shape": []}
    assert loaded["maxcv"]["trace"] == [{"ndarray_data": [1, 2], "dtype": "int32", "shape": [2]}]
    assert loaded["nonfinite"]["ndarray_data"] == [{"nonfinite": "nan"}, {"nonfinite": "inf"}]


def test_raw_arrays_are_captured_before_a_later_metadata_error():
    # Reproduce the problematic dictionary order with a deliberately unsupported
    # diagnostic; this is a serializer failure, not a numerical solver failure.
    raw = {"maxcv": object(), "x": np.array([[1.0]]), "jac": np.array([[2.0]])}
    arrays = {}
    with pytest.raises(TypeError):
        reference.optimizer_metadata(raw, arrays)
    np.testing.assert_array_equal(arrays["result_x"], [[1.0]])
    np.testing.assert_array_equal(arrays["result_jac"], [[2.0]])
    raw["x"][:] = 10
    assert arrays["result_x"][0, 0] == 1


def _manufactured_result(delta):
    """Export fixture with a hand-specified pixel load, not elastic physics."""
    x = (np.arange(512) - 256) * (0.16 / 512)
    heights = -(x[:, None] ** 2 + x[None, :] ** 2) / 0.04
    gap = np.maximum(-heights - delta, 0)
    u = np.zeros((1024, 1024))
    u[:512, :512] = heights + delta + gap
    force = np.zeros((512, 512))
    force[256, 256] = 2 * (0.16 / 512) ** 2
    result = SimpleNamespace(x=u, jac=force, offset=delta)
    substrate = SimpleNamespace(
        area_per_pt=(0.16 / 512) ** 2,
        local_topography_subdomain_slices=(slice(0, 512), slice(0, 512)),
    )
    system = SimpleNamespace(
        compute_gap=lambda value, offset: value[:512, :512] - (heights + offset)
    )
    reaction = np.zeros_like(u)
    reaction[:512, :512] = force
    return result, substrate, system, reaction, heights


def test_new_point_export_force_pressure_gap_and_kkt_are_separate_from_hertz():
    result, substrate, system, reaction, _ = _manufactured_result(75e-6)
    fields = reference.extract_fields(result, substrate, system, 75e-6)
    metrics = reference.assess_fields(fields, reaction, solver_success=True)
    assert fields["p"][256, 256] == 2
    assert fields["gap"][256, 256] == 0
    assert metrics["force_n"] == pytest.approx(2 * (0.16 / 512) ** 2)
    assert metrics["area_m2"] == pytest.approx((0.16 / 512) ** 2)
    assert metrics["peak_pressure_pa"] == 2
    assert metrics["kkt_pass"]
    assert not metrics["hertz_target_pass"]
    assert set(metrics["relative_errors"]) == {"force", "area", "peak"}
    reaction[700, 700] = 2e-9
    assert not reference.assess_fields(fields, reaction, solver_success=True)["kkt_pass"]
    result.offset = 100e-6
    with pytest.raises(ValueError, match="displacement"):
        reference.extract_fields(result, substrate, system, 75e-6)


def _fake_external(first_failure):
    """Dependency injection for orchestration only; never an FFT/Hertz oracle."""
    calls = []
    elastic_calls = []

    class Result(dict):
        __getattr__ = dict.__getitem__

    class Surface:
        def __init__(self, heights, sizes, **kwargs):
            assert sizes == (0.16, 0.16)
            assert kwargs == {"periodic": False, "unit": "m"}
            self.heights = heights

    class HalfSpace:
        def __init__(self, shape, **kwargs):
            assert shape == (512, 512)
            assert kwargs == {
                "young": 1e6,
                "physical_sizes": (0.16, 0.16),
                "fft": "serial",
                "check_boundaries": True,
            }
            self.area_per_pt = (0.16 / 512) ** 2
            self.local_topography_subdomain_slices = (slice(0, 512), slice(0, 512))

        def evaluate_force(self, value):
            assert value.shape == (1024, 1024)
            elastic_calls.append(self.delta)
            return -self.reaction

    class System:
        def __init__(self, substrate, surface):
            self.substrate, self.surface = substrate, surface

        def minimize_proxy(self, *, solver, **kwargs):
            result = solver(self.substrate, self.surface, **kwargs)
            if kwargs["offset"] == 75e-6 and first_failure == "boundary":
                raise RuntimeError("synthetic late boundary error")
            return result

        def compute_gap(self, value, offset):
            return value[:512, :512] - (self.surface.heights + offset)

    def solver(substrate, surface, **kwargs):
        delta = kwargs["offset"]
        assert kwargs == {
            "offset": delta,
            "external_force": None,
            "hardness": None,
            "pentol": 1e-10,
            "forcetol": 1e-10,
            "thermotol": 1e-8,
            "maxiter": 10000,
        }
        calls.append(delta)
        raw, _, _, reaction, _ = _manufactured_result(delta)
        substrate.reaction, substrate.delta = reaction, delta
        return Result(
            maxcv={"max_pres": np.array(0.0)},
            success=not (delta == 75e-6 and first_failure == "solver"),
            nit=3,
            nfev=7,
            x=raw.x,
            jac=raw.jac,
            offset=delta,
            active_set=raw.jac > 0,
            fun=0.0,
            message="explicit HOST fake; no physics claim",
        )

    return (Surface, HalfSpace, System, solver), calls, elastic_calls


@pytest.mark.parametrize("first_failure", ["boundary", "solver"])
def test_two_case_run_preserves_first_failure_and_does_not_repeat_or_skip_points(
    tmp_path, monkeypatch, first_failure
):
    api, calls, elastic_calls = _fake_external(first_failure)
    # Patch only this new driver's dependency seam, never the original runner.
    monkeypatch.setattr(reference, "_load_external", lambda: (api, {"test_fake": True}))
    monkeypatch.setattr(reference, "_process_memory", lambda: {"test_fake": True})
    output = tmp_path / "two-new-points"
    summary = reference.run(output)
    assert summary["schema"] == "phymeo-calibration-reference-v1"
    assert calls == [75e-6, 150e-6]
    assert elastic_calls == ([150e-6] if first_failure == "boundary" else [75e-6, 150e-6])
    assert [case["id"] for case in summary["cases"]] == ["n512-d075um", "n512-d150um"]
    first, second = summary["cases"]
    assert first["status"] == ("ERROR" if first_failure == "boundary" else "SOLVER_FAIL")
    assert first["usable_discrete_solution"] is False
    assert second["status"] == "HERTZ_TARGET_FAIL"
    assert second["metrics"]["kkt_pass"]
    assert summary["return_code"] == 1
    for case in summary["cases"]:
        data = json.loads((output / case["files"]["case"]).read_text(encoding="utf-8"))
        assert data["optimizer"]["maxcv"]["max_pres"]["shape"] == []
        assert data["timing"]["solver_s"] >= 0
        assert data["timing"]["total_case_s"] >= data["timing"]["solver_s"]
        with np.load(output / case["files"]["arrays"], allow_pickle=False) as saved:
            assert saved["result_x"].shape == (1024, 1024)
            assert saved["result_jac"].shape == (512, 512)
            assert float(saved["delta_m"]) == case["delta_m"]
            np.testing.assert_array_equal(saved["p"], saved["f"] / (0.16 / 512) ** 2)
    assert json.loads((output / "summary.json").read_text())["return_code"] == 1
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        reference.run(output)
    assert (output / "summary.json").read_bytes() == before
    assert calls == [75e-6, 150e-6]


def test_dependency_failure_still_saves_both_inputs_and_no_relative_output(tmp_path, monkeypatch):
    def unavailable():
        raise ImportError("synthetic missing external reference")

    monkeypatch.setattr(reference, "_load_external", unavailable)
    with pytest.raises(ValueError, match="absolute"):
        reference.run("relative-not-created")
    output = tmp_path / "unavailable"
    summary = reference.run(output)
    assert len(summary["cases"]) == 2
    for case in summary["cases"]:
        assert case["status"] == "UNSUPPORTED"
        assert case["metrics"] is None
        with np.load(output / case["files"]["arrays"], allow_pickle=False) as saved:
            assert set(saved.files) == {"x", "heights", "delta_m"}
        assert (output / case["files"]["case"]).is_file()
