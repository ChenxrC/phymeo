"""C3 measures the same public scene with a scoped retained-dispatch control."""

import copy
import importlib.util
import sys
from pathlib import Path
from time import perf_counter_ns

import pytest
from scene_scale_cases import H, builder, certify_frame

import phymeo as ph
from phymeo.geometry import collision_pipeline

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full_dispatch", "filtered_dispatch")


def _tool():
    path = ROOT / "scripts" / "profile_scene_loaded_narrow.py"
    assert path.is_file(), "C3 same-session narrow driver is missing"
    spec = importlib.util.spec_from_file_location("_c3_loaded_narrow_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loaded_narrow_pairs_real_public_output_before_accepting_the_driver():
    # Qualify the unchanged public scene before the original missing-driver RED.
    with builder(ph, "loaded", 1).build() as scene:
        frame = scene.step(H)
        certify_frame(frame, scene.rigid_rows, "loaded", 1, 0)
        assert frame.loads.contacts[0].force_n == 4.0
        assert scene.geometry().time_s == H
    tool = _tool()
    assert tool.MATRIX == (("loaded", 1), ("birth", 1), ("loaded", 32), ("birth", 32))
    assert tool.FRAMES == tuple(range(1, 17))
    assert tool.VARIANTS == VARIANTS
    assert tool.PUBLIC_CLOCK_FIELDS == ()
    assert tool.H == 1 / 128


def test_interleaving_is_fixed_balanced_and_does_not_select_faster_observations():
    tool = _tool()
    orders = [tool.pair_order(frame) for frame in tool.FRAMES]
    assert orders == [VARIANTS, VARIANTS[::-1]] * 8
    assert sum(order[0] == VARIANTS[0] for order in orders) == 8
    for bad in (0, 17, -1, 1.5, True):
        with pytest.raises(ValueError, match="declared frame"):
            tool.pair_order(bad)


def test_dispatch_control_restores_the_product_even_when_a_public_call_fails():
    tool = _tool()
    original = collision_pipeline._complete_pairs
    with pytest.raises(RuntimeError, match="injected caller failure"):
        with tool.dispatch_variant("full_dispatch"):
            assert collision_pipeline._complete_pairs is collision_pipeline._detect_pairs
            with pytest.raises(RuntimeError, match="active"):
                with tool.dispatch_variant("filtered_dispatch"):
                    pass
            raise RuntimeError("injected caller failure")
    assert collision_pipeline._complete_pairs is original
    with tool.dispatch_variant("filtered_dispatch"):
        assert collision_pipeline._complete_pairs is original
    assert collision_pipeline._complete_pairs is original
    with pytest.raises(ValueError, match="variant"):
        with tool.dispatch_variant("unknown"):
            pass
    assert collision_pipeline._complete_pairs is original


def test_full_public_comparison_never_strips_unknown_fields_or_physical_time():
    tool = _tool()
    value = {"frame": {"time_s": H, "normal": [0.0, 0.0, 1.0]}, "history": [1, 2]}
    tool.assert_full_equal(value, copy.deepcopy(value))
    changed = copy.deepcopy(value)
    changed["frame"]["time_s"] = 2 * H
    with pytest.raises(AssertionError, match="time_s"):
        tool.assert_full_equal(value, changed)
    changed = copy.deepcopy(value)
    changed["new_public_field"] = 0
    with pytest.raises(AssertionError, match="fields"):
        tool.assert_full_equal(value, changed)
    changed = copy.deepcopy(value)
    changed["history"][1] = 3
    with pytest.raises(AssertionError, match="history"):
        tool.assert_full_equal(value, changed)


def test_counter_comparison_only_normalizes_four_declared_geometry_clocks():
    tool = _tool()
    clocks = (
        "geometry.envelope_ns",
        "geometry.broad_ns",
        "geometry.midphase_ns",
        "geometry.narrow_ns",
    )
    assert tool.COUNTER_CLOCK_FIELDS == clocks
    left = {key: 10 for key in clocks}
    left.update({"geometry.completed_launches": 2, "new_time_like_count": 9})
    right = {key: 20 for key in clocks}
    right.update({"geometry.completed_launches": 2, "new_time_like_count": 9})
    tool.assert_counters_equal(left, right)
    assert left[clocks[0]] == 10 and right[clocks[0]] == 20
    right["new_time_like_count"] = 10
    with pytest.raises(AssertionError, match="new_time_like_count"):
        tool.assert_counters_equal(left, right)
    with pytest.raises(AssertionError, match="missing"):
        tool.assert_counters_equal({}, {})


def test_mutually_exclusive_spans_cover_the_caller_without_nested_double_counting():
    tool = _tool()
    timeline = tool.NarrowTimeline()
    timeline.records = [
        (10, 90, 1, "geometry_narrow", 0),
        (20, 40, 2, "hull_reconstruction", 1),
        (40, 65, 3, "projection_feature_search", 1),
        (65, 80, 4, "projection_distance_material_finish", 1),
        (45, 50, 5, "material_chart_weights", 2),
    ]
    measured = timeline.partition(0, 100)
    assert measured["exclusive_ns"] == {
        "other_public_caller": 20,
        "geometry_narrow": 20,
        "hull_reconstruction": 20,
        "projection_feature_search": 20,
        "material_chart_weights": 5,
        "projection_distance_material_finish": 15,
    }
    assert sum(measured["exclusive_ns"].values()) == measured["caller_wall_ns"] == 100
    assert measured["inclusive_ns"]["geometry_narrow"] == 80
    assert measured["calls"]["hull_reconstruction"] == 1
    assert measured["pure_gpu_device_ns"] is None


def test_real_trace_separates_geometry_features_and_path_proof_with_identical_payload():
    tool = _tool()
    observations = []
    timeline = tool.NarrowTimeline()
    for traced in (False, True):
        with builder(ph, "loaded", 1).build() as scene:
            if traced:
                with timeline.active(scene):
                    begin = perf_counter_ns()
                    frame = scene.step(H)
                    geometry = scene.geometry()
                    end = perf_counter_ns()
            else:
                frame = scene.step(H)
                geometry = scene.geometry()
            certify_frame(frame, scene.rigid_rows, "loaded", 1, 0)
            observations.append(tool.canonical_payload(scene, frame, geometry))
    tool.assert_full_equal(*observations)
    measured = timeline.partition(begin, end)
    for label in (
        "hull_reconstruction",
        "projection_feature_search",
        "projection_distance_material_finish",
        "path_interval_verify",
    ):
        assert measured["calls"].get(label, 0) > 0, label
    assert sum(measured["exclusive_ns"].values()) == end - begin
    assert timeline.source_binding["sha256"]
    assert set(timeline.source_binding["anchors"]) == {
        "projection_feature_search", "projection_distance_material_finish"
    }
    assert sys.getprofile() is None and sys.gettrace() is None


def test_independent_birth_pair_archives_every_declared_frame_and_first_force(tmp_path):
    tool = _tool()
    report = tool.run_pair("birth", 1, "untraced", "host", tmp_path / "cache", tmp_path / "pair")
    assert report["success"]
    assert [row["frame"] for row in report["rows"]] == list(tool.FRAMES)
    assert report["excluded_public_fields"] == []
    for row in report["rows"]:
        left, right = (row["observations"][variant] for variant in VARIANTS)
        assert left["frame_logical_sha256"] == right["frame_logical_sha256"]
        assert bool(left["force_worlds"]) == (row["frame"] >= 10)
        assert left["force_worlds"] == right["force_worlds"]
        assert row["order"] == list(tool.pair_order(row["frame"]))
    for variant in VARIANTS:
        output = report["outputs"][variant]
        frames = list(tool.driver.read_window(Path(output["path"])))
        assert len(frames) == 16
        assert output["decode_verified_frames"] == 16
        assert "contact_events" in frames[0]


@pytest.mark.parametrize("changed", ("diagnostic", "protocol"))
def test_new_diagnostic_and_protocol_sources_are_bound_and_checked(tmp_path, monkeypatch, changed):
    tool = _tool()
    copied = {}
    for name, source in tool.EXTRA_TOOL_PATHS.items():
        path = tmp_path / source.name
        path.write_bytes(source.read_bytes())
        copied[name] = path
    monkeypatch.setattr(tool, "EXTRA_TOOL_PATHS", copied)
    before = tool.provenance()
    copied[changed].write_bytes(copied[changed].read_bytes() + b"\n# binding negative control\n")
    after = tool.provenance()
    assert before["extra_tools"][changed] != after["extra_tools"][changed]
    with pytest.raises(AssertionError, match="diagnostic tool changed"):
        tool.verify_provenance(before, after)


def test_public_capacity_failure_restores_dispatch_and_keeps_the_next_step_valid():
    tool = _tool()
    original = collision_pipeline._complete_pairs
    with builder(ph, "loaded", 1).build() as scene:
        state_before = tool.driver.public_value(scene.state)
        with pytest.raises(ph.SolverFailure, match="capacity"):
            with tool.dispatch_variant("full_dispatch"):
                scene.step(H, capacity=0)
        assert collision_pipeline._complete_pairs is original
        assert scene.geometry().time_s == 0
        assert scene.last_frame is None
        tool.assert_full_equal(state_before, tool.driver.public_value(scene.state))
        with tool.dispatch_variant("filtered_dispatch"):
            frame = scene.step(H)
        certify_frame(frame, scene.rigid_rows, "loaded", 1, 0)
        assert frame.loads.contacts[0].force_n == 4.0
    assert collision_pipeline._complete_pairs is original
