"""C1's new cost harness must preserve physical outputs, keys and complete path receipts."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scene_scale_cases import (
    BALL,
    BOX,
    STEPS,
    H,
    builder,
    certify_frame,
    matrix,
    oracle,
    seeds,
)

import phymeo as ph

ROOT = Path(__file__).resolve().parents[2]


def driver():
    path = ROOT / "scripts" / "benchmark_scene_scale_costs.py"
    spec = importlib.util.spec_from_file_location("_scene_scale_cost_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cost_harness_preserves_a_real_loaded_step_and_freezes_nine_cases(tmp_path):
    # A real existing public positive runs BEFORE checking the missing measurement tool.
    make = builder(ph, "loaded", 1)
    with make.build() as scene:
        frame = scene.step(H)
        certify_frame(frame, scene.rigid_rows, "loaded", 1, 0)
        assert frame.loads.contacts[0].force_n == 4.0
        assert scene.geometry().time_s == H
    tool = driver()
    assert tuple(tool.MATRIX) == matrix()
    assert tool.WINDOWS == ("cold", "warm")
    assert tool.STEPS == 32 and tool.H == 1 / 128


@pytest.mark.parametrize("case", ("free", "loaded", "birth"))
def test_independent_scalar_oracles_distinguish_occupancy_and_first_force(case):
    expected = oracle(seeds(case, 1)[0])
    magnitudes = np.linalg.norm(expected["force_on_ball_n"], axis=1)
    assert expected["position_m"].shape == (STEPS + 1, 3)
    assert min(expected["entry_gap_m"]) > 0
    if case == "free":
        assert not magnitudes.any()
        assert expected["position_m"][-1, 2] == 223 / 256
    elif case == "loaded":
        np.testing.assert_array_equal(magnitudes, 4)
        assert expected["position_m"][-1, 0] == 1 / 128
    else:
        # Step 9 enters with exactly zero layer compression; step 10 first applies force.
        assert tuple(np.flatnonzero(magnitudes > 1e-10)[:1]) == (9,)
        assert magnitudes[-1] > 0


@pytest.mark.parametrize(
    "feature, transformed", (("face", False), ("edge", False), ("corner", True))
)
def test_real_birth_trajectory_has_bound_material_points_and_accepted_endpoints(
    feature, transformed
):
    make = builder(ph, "birth", 1, feature=feature, transformed=transformed)
    with make.build() as scene:
        assert scene.sample().loads.contacts == ()
        for step in range(STEPS):
            frame = scene.step(H)
            certify_frame(
                frame,
                scene.rigid_rows,
                "birth",
                1,
                step,
                feature=feature,
                transformed=transformed,
            )
            assert scene.geometry().time_s == (step + 1) * H


def test_model_capacity_is_a_real_public_limit_not_a_benchmark_knob():
    make = builder(ph, "free", 128)
    make.add_sphere(100, 1 / 8, mass=1, position=(0, 0, 20), world=9999)
    with pytest.raises(ValueError, match="capacity"):
        make.build()


def test_capacity_rejection_preserves_state_time_frame_then_the_same_scene_recovers():
    with builder(ph, "loaded", 1).build() as scene:
        scene.step(H)
        old, time, frame = scene.state, scene.geometry().time_s, scene.last_frame
        with pytest.raises(ph.SolverFailure, match="capacity"):
            scene.step(H, capacity=0)
        np.testing.assert_array_equal(scene.state.rigid.position, old.rigid.position)
        assert scene.geometry().time_s == time
        np.testing.assert_array_equal(
            scene.last_frame.state.rigid.position, frame.state.rigid.position
        )
        accepted = scene.step(H)
        certify_frame(accepted, scene.rigid_rows, "loaded", 1, 1)


def test_streamed_window_round_trips_all_public_fields_and_rejects_changed_receipts(tmp_path):
    tool = driver()
    destination = tmp_path / "window.json.gz"
    with builder(ph, "loaded", 1).build() as scene:
        first = scene.step(H)
        first_value = tool.public_value(dict(frame=first, geometry=scene.geometry()))
        second = scene.step(H)
        second_value = tool.public_value(dict(frame=second, geometry=scene.geometry()))
    with tool.WindowWriter(destination, dict(worlds=1, case="loaded")) as writer:
        writer.append(first_value)
        writer.append(second_value)
    decoded = list(tool.read_window(destination))
    assert decoded == [first_value, second_value]
    # Equal sums do not substitute for the actual per-world fields.
    changed = tool.public_value(first_value)
    changed["intentional_extra_field"] = True
    assert tool.logical_digest(changed) != tool.logical_digest(first_value)
    with gzip.open(destination, "rt", encoding="utf8") as stream:
        lines = [json.loads(line) for line in stream]
    lines[1]["delta"]["set"]["changed_public_field"] = 123
    corrupted = tmp_path / "corrupted.json.gz"
    with gzip.open(corrupted, "wt", encoding="utf8") as stream:
        for row in lines:
            stream.write(json.dumps(row) + "\n")
    with pytest.raises(AssertionError, match="digest mismatch"):
        list(tool.read_window(corrupted))


def test_w32_local_ids_and_mirrors_are_distinct_original_material_addresses():
    frozen = seeds("birth", 32)
    assert len({row.world for row in frozen}) == 32
    assert {row.mirror for row in frozen} == {-1, 1}
    assert (BOX, BALL) == (2, 9)
    assert frozen[0].normal == (0.6, 0.8, 0)


@pytest.mark.tyx
def test_w1_device_window_publishes_the_same_full_physical_contract(tyx_mode, tmp_path):
    with builder(ph, "loaded", 1).build(backend=tyx_mode, cache_root=tmp_path) as scene:
        for step in range(STEPS):
            frame = scene.step(H)
            certify_frame(frame, scene.rigid_rows, "loaded", 1, step)
        assert frame.rigid_report.backend == f"tyx:{tyx_mode}"
        assert scene.stats()["geometry"]["completed_launches"] > 0


def test_diagnostic_wall_partition_does_not_add_nested_device_api_times():
    path = ROOT / "scripts" / "profile_scene_scale_costs.py"
    spec = importlib.util.spec_from_file_location("_scene_scale_profile_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    timeline = module.CallTimeline()
    timeline.records = [
        (10, 90, 1, "response_law_wrapper", 0),
        (20, 30, 2, "h2d_api", 1),
        (35, 45, 3, "kernel_call_api", 1),
        (45, 60, 4, "synchronization_api", 1),
        (60, 70, 5, "d2h_api", 1),
    ]
    result = timeline.partition(0, 100)
    assert result["inclusive_ns"]["response_law_wrapper"] == 80
    assert result["exclusive_ns"]["response_law_wrapper"] == 35
    assert result["exclusive_ns"]["other_public_caller"] == 20
    assert sum(result["exclusive_ns"].values()) == 100
    assert result["pure_gpu_device_ns"] is None
    assert result["calls"]["kernel_call_api"] == 1
