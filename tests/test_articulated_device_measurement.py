"""Device timings must retain actual execution, audit and failure evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.tyx


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/measure_articulated_device.py"
    assert path.exists(), "P6 performance recorder missing"
    spec = importlib.util.spec_from_file_location("measure_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace(tyx_mode):
    from phymeo.experiments.articulated_device import run_articulated_device

    return run_articulated_device(scene="S2", mode=tyx_mode, frames=1)


def test_actual_device_step_measurement_keeps_all_boundaries(trace):
    report = script().measure(trace, frame=0, warmups=2, repeats=1, calls=1)
    assert report["status"] == "complete"
    assert report["acceptance_protocol"] is False
    assert report["accepted_measured_steps"] == 1
    assert report["warmup_calls"] == 2
    assert report["failed_calls"] == 0
    assert report["exact_saved_state"] is True
    sample = report["samples"][0]
    assert sample["step_ms"] > 0 and sample["audit_ms"] > 0
    for substep in sample["substeps"]:
        execution = substep["execution"]
        assert execution["launches"] == execution["completed_launches"] > 0
        assert substep["quality"]["robot_balance_error"] <= 5e-5
        assert substep["max_residual"] <= 2e-6


@pytest.mark.parametrize("phase", ["construction", "close", "audit"])
def test_failed_device_call_never_leaves_complete_record(trace, monkeypatch, phase):
    module = script()
    create = module.make_tyx_articulated_runtime
    closed = []

    def broken(*args, **kwargs):
        if phase == "construction":
            raise RuntimeError("injected construction failure")
        runtime = create(*args, **kwargs)
        close = runtime.close
        step = runtime.step

        def bad_step(*args, **kwargs):
            result = step(*args, **kwargs)
            result.substep_states[0].domains["robot"].v[0] += 1
            return result

        def final_close():
            close()
            closed.append(True)
            if phase == "close":
                raise RuntimeError("injected close failure")

        runtime.close = final_close
        if phase == "audit":
            runtime.step = bad_step
        return runtime

    monkeypatch.setattr(module, "make_tyx_articulated_runtime", broken)
    record = {}
    with pytest.raises(RuntimeError):
        module.measure(trace, frame=0, warmups=2, repeats=1, calls=1, record=record)
    assert record["status"] == "failed"
    if phase != "construction":
        assert closed == [True]
    if phase == "audit":
        assert record["accepted_measured_steps"] == 0
        assert record["samples"] == []


def test_reference_cannot_count_as_device_timing():
    from phymeo.experiments.articulated_device import run_articulated_reference

    with pytest.raises(ValueError, match="device|mode"):
        script().measure(run_articulated_reference(frames=1), frame=0, repeats=1, calls=1)


def test_wrong_installed_dependency_revision_cannot_count_as_frozen_measurement(trace, monkeypatch):
    module = script()
    monkeypatch.setattr(
        module,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda path: "{}"),
        raising=False,
    )
    record = {}
    with pytest.raises(ValueError, match="tyx|revision"):
        module.measure(trace, frame=0, repeats=1, calls=1, record=record)
    assert record["status"] == "failed"
    assert record["failure"]["phase"] == "construction"
