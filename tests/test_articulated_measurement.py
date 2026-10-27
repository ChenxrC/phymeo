"""Performance records must retain physical checks and rejected-call evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phymeo.experiments.articulated import run_articulated_scene
from phymeo.experiments.rigid_cloth import SceneRunFailure


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scene", ["S2", "S3", "S4"])
def test_measurement_uses_real_step_and_retains_each_substep_quality(scene):
    run = run_articulated_scene(scene=scene, frames=2)
    result = script("measure_articulated").measure(run, frame=0, warmups=2, repeats=1, calls=2)
    assert result["status"] == "complete"
    assert result["acceptance_protocol"] is False  # Small test is not the full benchmark.
    assert result["accepted_measured_steps"] == 2
    assert result["failed_calls"] == 0
    assert result["warmup_calls"] == 2
    assert len(result["samples"]) == 2
    assert len(result["repeats"]) == 1
    for sample in result["samples"]:
        assert sample["step_ms"] > 0
        assert len(sample["substeps"]) == 2
        for step in sample["substeps"]:
            assert step["position_iterations"] >= 1
            assert step["velocity_iterations"] >= 1
            assert step["max_residual"] <= 1e-8
            assert step["quality"]["fixed_position_error"] == 0
            assert step["quality"]["robot_balance_error"] < 1e-10
            assert set(step["contacts"]) == set(run.config["node_domains"])
            assert "position" in step["timings_ms"]
    assert result["exact_saved_state"] is True


def test_measurement_rejects_altered_physics_metadata_before_timing():
    module = script("measure_articulated")
    run = run_articulated_scene(scene="S2", frames=2)
    run.metrics["max_robot_balance_error"] = 100.0
    with pytest.raises(ValueError, match="summary"):
        module.measure(run, frame=0, warmups=2, repeats=1, calls=1)


def test_failed_physical_audit_is_recorded_not_counted_as_valid_latency(monkeypatch):
    module = script("measure_articulated")
    run = run_articulated_scene(scene="S2", frames=2)
    create = module.make_articulated_runtime
    closed = []

    class BadCandidate:
        def __init__(self, *args, **kwargs):
            self.inner = create(*args, **kwargs)

        def step(self, *args, **kwargs):
            result = self.inner.step(*args, **kwargs)
            result.substep_states[0].domains["robot"].v[0] += 1.0
            return result

        def close(self):
            self.inner.close()
            closed.append(True)

    monkeypatch.setattr(module, "make_articulated_runtime", BadCandidate)
    record = {}
    with pytest.raises(SceneRunFailure):
        module.measure(run, frame=0, warmups=2, repeats=1, calls=1, record=record)
    assert closed == [True]
    assert record["status"] == "failed"
    assert record["failed_calls"] == 1
    assert record["accepted_measured_steps"] == 0
    assert record["samples"] == []
    assert record["failure"]["phase"] == "first"


@pytest.mark.parametrize("options", [{"frame": True}, {"frame": 2}, {"calls": 0}])
def test_invalid_benchmark_workload_is_not_silently_changed(options):
    module = script("measure_articulated")
    run = run_articulated_scene(scene="S2", frames=2)
    config = dict(frame=0, warmups=2, repeats=1, calls=1)
    config.update(options)
    with pytest.raises((TypeError, ValueError)):
        module.measure(run, **config)


def test_installed_articulated_pair_smoke_checks_an_independent_mass_solution():
    # The same callable is executed by the wheel entry point outside the checkout.
    result = script("smoke_installed").check_articulated_pair()
    np.testing.assert_allclose(result["robot_q"], [-0.01 / 3], atol=1e-12)
    np.testing.assert_allclose(result["robot_v"], [-1 / 3], atol=1e-12)
    np.testing.assert_allclose(result["node_v"], [[0.0, 2 / 3, 0.0]], atol=1e-12)


@pytest.mark.parametrize("phase", ["construction", "close"])
def test_resource_failure_cannot_leave_completed_measurement(monkeypatch, phase):
    module = script("measure_articulated")
    run = run_articulated_scene(scene="S2", frames=2)
    create = module.make_articulated_runtime

    def broken(*args, **kwargs):
        if phase == "construction":
            raise RuntimeError("injected construction failure")
        runtime = create(*args, **kwargs)
        close = runtime.close

        def bad_close():
            close()
            raise RuntimeError("injected close failure")

        runtime.close = bad_close
        return runtime

    monkeypatch.setattr(module, "make_articulated_runtime", broken)
    record = {}
    with pytest.raises(RuntimeError, match="injected"):
        module.measure(run, frame=0, warmups=2, repeats=1, calls=1, record=record)
    assert record["status"] == "failed"
    assert record["failure"]["phase"] == phase
