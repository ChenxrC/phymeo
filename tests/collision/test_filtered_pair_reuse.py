"""Complete ordinary detect records with reusable immutable filter rows."""

import copy
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
import phymeo.geometry.collision_pipeline as pipeline


def _filter_scene():
    """One real touching pair, an AABB miss, and all four filter priorities."""
    keys = tuple(
        ph.CollisionKey(*key)
        for key in ((0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0), (1, 1, 0))
    )
    a, same, mask, excluded, _, _, other_world = keys
    shapes = [
        ph.CollisionShape(key, "sphere", (7,), 0.125, group=1, mask=1) for key in keys
    ]
    shapes[1] = ph.CollisionShape(same, "sphere", (7,), 0.125, group=1, mask=0)
    shapes[2] = ph.CollisionShape(mask, "sphere", (7,), 0.125, group=2, mask=2)
    model = ph.CollisionModel(
        tuple(reversed(shapes)),
        ((a, same), (a, mask), (a, excluded), (a, other_world)),
    )
    centers = (0, 4, 2, 6, 0.25, 8, 0)
    paths = {key: ph.CollisionMotion([[x, 0, 0]]) for key, x in zip(keys, centers, strict=True)}
    return model, paths, keys


def _by_pair(result):
    return {(row.first, row.second): row for row in result.pairs}


def _assert_filter_scene(result, keys):
    a, same, mask, excluded, touching, far, other_world = keys
    assert result.status == "ok" and result.required_candidates == 1
    assert [(row.first, row.second) for row in result.pairs] == list(combinations(keys, 2))
    rows = _by_pair(result)
    for second, reason in (
        (other_world, "different_world"),
        (same, "same_rigid_body"),
        (mask, "mask"),
        (excluded, "excluded"),
    ):
        row = rows[a, second]
        assert (row.candidate, row.status, row.reason, row.family) == (
            False, "filtered", reason, "RR"
        )
        assert row.witness is None and row.sweep is None and not row.witness_shapes
    miss = rows[a, far]
    assert (miss.candidate, miss.status, miss.reason) == (False, "separated", "aabb")
    contact = rows[a, touching]
    assert contact.candidate and contact.status == "touching"
    assert contact.witness.gap_m == 0
    assert contact.witness.first_weights == ((7, 1.0),)
    assert contact.witness.second_weights == ((7, 1.0),)
    np.testing.assert_array_equal(contact.witness.first_point, (0.125, 0, 0))
    np.testing.assert_array_equal(contact.witness.second_point, (0.125, 0, 0))
    np.testing.assert_array_equal(contact.witness.normal, (1, 0, 0))


def test_repeat_public_detect_does_not_dispatch_invariant_filtered_rows(monkeypatch):
    """RED on the old path: its second public query redispatches codes 1--4."""
    model, paths, keys = _filter_scene()
    original_dispatch, original_pair = pipeline._detect_pairs, pipeline._pair
    dispatched, constructed = [], []

    def dispatch(*args, **kwargs):
        dispatched.extend(int(code) for code in args[2])
        return original_dispatch(*args, **kwargs)

    def pair(*args, **kwargs):
        constructed.append(int(args[4]))
        return original_pair(*args, **kwargs)

    with ph.CollisionWorkspace(model) as work:
        before = ph.detect(model, paths, ph.CollisionPolicy(), work)
        _assert_filter_scene(before, keys)
        saved_pairs = before.pairs
        monkeypatch.setattr(pipeline, "_detect_pairs", dispatch)
        monkeypatch.setattr(pipeline, "_pair", pair)
        after = ph.detect(model, paths, ph.CollisionPolicy(), work)
        _assert_filter_scene(after, keys)
        assert after == before and before.pairs is saved_pairs
        assert work.last_result is after
        assert 0 in dispatched and 5 in dispatched
        repeated = [code for code in dispatched if 1 <= code <= 4]
        assert not repeated, f"invariant filter rows still reached narrow dispatch: {repeated}"
        assert not [code for code in constructed if 1 <= code <= 4]
        for old, new in zip(before.pairs, after.pairs, strict=True):
            if old.status == "filtered":
                assert old is new
                with pytest.raises(FrozenInstanceError):
                    new.reason = "changed"


@pytest.mark.parametrize("broad_phase", ["all_pairs", "sap", "bvh"])
def test_runtime_aabb_transitions_and_previous_complete_publication(broad_phase):
    model, paths, keys = _filter_scene()
    a, _, _, _, touching, _, _ = keys
    with ph.CollisionWorkspace(model, broad_phase=broad_phase) as work:
        initial = ph.detect(model, paths, ph.CollisionPolicy(), work)
        saved_pairs, saved_bounds = initial.pairs, initial.bounds
        distant = {**paths, touching: ph.CollisionMotion([[20, 0, 0]])}
        separated = ph.detect(model, distant, ph.CollisionPolicy(), work)
        assert separated.required_candidates == 0
        row = _by_pair(separated)[a, touching]
        assert (row.candidate, row.status, row.reason) == (False, "separated", "aabb")
        assert separated.bounds != saved_bounds
        restored = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert restored == initial
        assert initial.pairs is saved_pairs and initial.bounds is saved_bounds
        for old, clear, again in zip(saved_pairs, separated.pairs, restored.pairs, strict=True):
            if old.status == "filtered":
                assert old is clear is again
        _assert_filter_scene(initial, keys)


def test_filtered_inputs_still_validate_envelopes_before_backend(monkeypatch):
    model, paths, keys = _filter_scene()
    calls = []
    with ph.CollisionWorkspace(model) as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        classify = work._classify_world_envelopes

        def record(*args, **kwargs):
            calls.append(True)
            return classify(*args, **kwargs)

        monkeypatch.setattr(work, "_classify_world_envelopes", record)
        for invalid, message in (
            (ph.CollisionMotion([[1e100, 0, 0]]), "FP32"),
            (ph.CollisionMotion([[0, 0, 0], [1, 0, 0]]), "registered vertices"),
        ):
            with pytest.raises(ValueError, match=message):
                ph.detect(
                    model, {**paths, keys[-1]: invalid}, ph.CollisionPolicy(), work
                )
            assert work.last_result is previous
        assert not calls
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == previous
        assert len(calls) == 1


def test_only_filtered_rr_rd_dd_rows_still_use_real_classification(monkeypatch):
    keys = tuple(ph.CollisionKey(world, 7, 3) for world in range(4))
    kinds = ("sphere", "triangle", "triangle", "sphere")
    vertices = {"sphere": (9,), "triangle": (2, 9, 17)}
    model = ph.CollisionModel(
        [ph.CollisionShape(k, kind, vertices[kind], 0.125 if kind == "sphere" else 0)
         for k, kind in zip(keys, kinds, strict=True)]
    )
    paths = {
        k: ph.CollisionMotion([[0, 0, 0]] if kind == "sphere" else [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        for k, kind in zip(keys, kinds, strict=True)
    }
    calls = []
    with ph.CollisionWorkspace(model) as work:
        classify = work._classify_world_envelopes

        def record(*args, **kwargs):
            calls.append(True)
            return classify(*args, **kwargs)

        monkeypatch.setattr(work, "_classify_world_envelopes", record)
        before = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        after = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert len(calls) == 2 and before == after
        assert before.required_candidates == 0 and len(before.pairs) == 6
        assert [row.family for row in before.pairs] == ["RD", "RD", "RR", "DD", "RD", "RD"]
        assert all(row.reason == "different_world" for row in before.pairs)
        assert all(a is b for a, b in zip(before.pairs, after.pairs, strict=True))


def test_actual_backend_code_is_not_overridden_by_a_model_template(monkeypatch):
    model, paths, keys = _filter_scene()
    with ph.CollisionWorkspace(model) as work:
        before = ph.detect(model, paths, ph.CollisionPolicy(), work)
        classify = work._classify_world_envelopes
        target = [(row.first, row.second) for row in before.pairs].index((keys[0], keys[-1]))

        def changed_code(*args, **kwargs):
            bounds, flags = classify(*args, **kwargs)
            flags = flags.copy()
            flags[target] = 3
            return bounds, flags

        monkeypatch.setattr(work, "_classify_world_envelopes", changed_code)
        after = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert after.pairs[target].reason == "mask"
        assert before.pairs[target].reason == "different_world"
        assert after.pairs[target] is not before.pairs[target]


@pytest.mark.parametrize("kind", ["hull", "mesh", "plate", "tetrahedron"])
def test_aabb_filtered_geometry_still_reports_invalid_material(kind):
    first, second = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    tetra = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    flat = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.25, 0.25, 0]]
    if kind == "hull":
        a = ph.CollisionShape(first, kind, (0, 1, 2, 3))
        b = ph.CollisionShape(second, kind, (0, 1, 2, 3))
        pa, pb = flat, np.asarray(tetra) + [8, 0, 0]
    elif kind == "mesh":
        a = ph.CollisionShape(
            first, kind, (0, 1, 2, 3),
            triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        )
        b = ph.CollisionShape(second, "sphere", (0,), 0.125)
        pa, pb = flat, [[8, 0, 0]]
    elif kind == "plate":
        a = ph.CollisionShape(first, kind, (0, 1, 2, 3), 0.125)
        b = ph.CollisionShape(second, "triangle", (0, 1, 2))
        pa = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.25]]
        pb = [[8, 0, 0], [9, 0, 0], [8, 1, 0]]
    else:
        a = ph.CollisionShape(first, kind, (0, 1, 2, 3))
        b = ph.CollisionShape(second, kind, (0, 1, 2, 3))
        pa, pb = flat, np.asarray(tetra) + [8, 0, 0]
    other_world = ph.CollisionKey(1, 1, 0)
    remote = ph.CollisionShape(other_world, "sphere", (0,), 0.125)
    model = ph.CollisionModel((a, b, remote))
    paths = {
        first: ph.CollisionMotion(pa),
        second: ph.CollisionMotion(pb),
        other_world: ph.CollisionMotion([[0, 0, 0]]),
    }
    with ph.CollisionWorkspace(model) as work:
        for _ in range(2):
            result = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
            assert result.status == "ok" and result.required_candidates == 0
            row = _by_pair(result)[first, second]
            assert not row.candidate
            assert row.status == "invalid_geometry" and row.reason != "aabb"
            assert row.witness is None


def test_capacity_and_transaction_failure_keep_previous_tuple_and_retry():
    model, paths, keys = _filter_scene()
    with ph.CollisionWorkspace(model) as work:
        before = ph.detect(model, paths, ph.CollisionPolicy(), work)
        saved_pairs = before.pairs
        failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert failed.status == "capacity_exceeded" and failed.required_candidates == 1
        assert not failed.pairs and not failed.bounds and work.last_result is before
        distant = {**paths, keys[4]: ph.CollisionMotion([[20, 0, 0]])}
        with pytest.raises(RuntimeError, match="abort"):
            with work.transaction():
                staged = ph.detect(model, distant, ph.CollisionPolicy(), work)
                assert staged.required_candidates == 0 and work.last_result is before
                raise RuntimeError("abort")
        assert work.last_result is before and before.pairs is saved_pairs
        assert ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=1), work) == before
        _assert_filter_scene(before, keys)


@pytest.mark.tyx
def test_device_filters_keep_real_classify_capacity_and_failure(tyx_mode, tmp_path, monkeypatch):
    model, paths, keys = _filter_scene()
    with ph.CollisionWorkspace(model) as host:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), host)

    def forbidden(*args, **kwargs):
        raise AssertionError("selected device fell back to HOST broad phase")

    monkeypatch.setattr(pipeline, "_host_query", forbidden)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        first = ph.detect(model, paths, ph.CollisionPolicy(), work)
        counts = work.stats()
        failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert failed.status == "capacity_exceeded" and work.last_result is first
        restored = ph.detect(model, paths, ph.CollisionPolicy(), work)
        _assert_filter_scene(restored, keys)
        assert first.pairs == restored.pairs == expected.pairs
        assert first.bounds == expected.bounds
        assert all(
            a is b for a, b in zip(first.pairs, restored.pairs, strict=True)
            if a.status == "filtered"
        )
        later = work.stats()
        assert counts["completed_launches"] == 2 and later["completed_launches"] == 6
        assert counts["allocations"] == later["allocations"]
        if tyx_mode == "cuda":
            assert later["dynamic_uploads"] == 3 and later["downloads"] == 6
            assert later["synchronizations"] == 3

        def fault(*args, **kwargs):
            raise RuntimeError("injected classification fault")

        monkeypatch.setattr(work._engine, "query", fault)
        with pytest.raises(RuntimeError, match="injected"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is restored
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)


@pytest.mark.tyx
def test_filtered_slots_preserve_global_feature_batches(tyx_mode, tmp_path, monkeypatch):
    """Nine moving sphere pairs straddle the old 128-row global batch boundary."""
    shapes, paths = [], {}
    for world in range(9):
        for body in range(2):
            key = ph.CollisionKey(world, body, 0)
            shapes.append(ph.CollisionShape(key, "sphere", (13,), 0.125))
            paths[key] = (
                ph.CollisionMotion([[-0.5, 0, 0]], [[0.5, 0, 0]])
                if body == 0 else ph.CollisionMotion([[0, 0, 0]])
            )
    model = ph.CollisionModel(shapes)
    complete = pipeline._complete_pairs
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path / "cache") as work:
        entry = work._engine.entry_times
        submissions = []

        def record(features):
            if features:
                submissions.append(len(features))
            return entry(features)

        monkeypatch.setattr(work._engine, "entry_times", record)
        monkeypatch.setattr(pipeline, "_complete_pairs", pipeline._detect_pairs)
        original = ph.detect(model, paths, ph.CollisionPolicy(), work)
        original_batches = tuple(submissions)
        before = work.stats()
        submissions.clear()
        monkeypatch.setattr(pipeline, "_complete_pairs", complete)
        reused = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert reused == original and len(reused.pairs) == 153
        assert len(original_batches) == 2 and tuple(submissions) == original_batches
        assert work.stats()["completed_launches"] == 2 * before["completed_launches"]
        contacts = [row for row in reused.pairs if row.candidate]
        assert len(contacts) == 9 and all(row.status == "hit" for row in contacts)
        for row in contacts:
            lo, hi = row.sweep.first_t
            assert lo <= 0.25 <= hi
            assert row.witness.first_weights == row.witness.second_weights == ((13, 1.0),)


def _comparison_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_scene_scale_frames.py"
    spec = importlib.util.spec_from_file_location("_c2_comparison_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol_row():
    return dict(
        step=1, entry_s=0.0, end_s=1 / 128,
        step_ns=12, geometry_snapshot_ns=2, public_to_end_ns=14,
        oracle_ns=3, archive_ns=4,
        actual_contacts=1, actual_sites=2, actual_required_candidates=1,
        published_pairs=21, different_world_rows=6, force_worlds=[0],
        counters={
            "geometry.envelope_ns": 2, "geometry.broad_ns": 2,
            "geometry.midphase_ns": 0, "geometry.narrow_ns": 3,
            "geometry.completed_launches": 4, "geometry.downloaded_bytes": 96,
        },
        native_counters={"cuda_kernel_launches": 5, "cuda_h2d_bytes": 128},
        frame_logical_sha256="same-full-frame",
        future_diagnostic={"physical_sample_time_s": 0.0, "unlisted_cost_ns": 9},
    )


def test_comparison_allows_only_the_enumerated_wall_clock_values():
    compare = _comparison_module()
    before, after = _protocol_row(), _protocol_row()
    for key in compare.ROW_TIMINGS:
        after[key] += 10
    for key in compare.COUNTER_TIMINGS:
        after["counters"][key] += 20
    compare.compare_step_rows(before, after)
    assert compare.normalized_step(before).keys() == before.keys()
    assert compare.normalized_step(before)["counters"].keys() == before["counters"].keys()
    assert before["step_ns"] == 12 and after["step_ns"] == 22


@pytest.mark.parametrize(
    "path",
    [
        ("actual_sites",),
        ("actual_required_candidates",),
        ("published_pairs",),
        ("different_world_rows",),
        ("entry_s",),
        ("end_s",),
        ("counters", "geometry.completed_launches"),
        ("counters", "geometry.downloaded_bytes"),
        ("native_counters", "cuda_kernel_launches"),
        ("future_diagnostic", "physical_sample_time_s"),
        ("future_diagnostic", "unlisted_cost_ns"),
    ],
)
def test_comparison_rejects_changed_counts_physical_time_and_unlisted_fields(path):
    compare = _comparison_module()
    before, after = _protocol_row(), _protocol_row()
    target = after
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] += 1
    with pytest.raises(AssertionError, match="changed"):
        compare.compare_step_rows(before, after)


@pytest.mark.parametrize("operation", ["delete", "add", "type"])
def test_comparison_rejects_missing_added_or_retyped_output(operation):
    compare = _comparison_module()
    before = {"frame": {"pair": {"normal": [0.0, 0.0, 1.0], "candidate": True}}}
    after = copy.deepcopy(before)
    if operation == "delete":
        del after["frame"]["pair"]["normal"]
    elif operation == "add":
        after["frame"]["pair"]["new_unrecorded_quantity"] = 1
    else:
        after["frame"]["pair"]["candidate"] = 1
    with pytest.raises(AssertionError, match="changed"):
        compare.compare_values(before, after)


def _source_receipt(compare, head):
    return dict(
        source_head=head, root="C:/frozen", module="C:/frozen/src/phymeo/__init__.py",
        product_dirty=False,
        imported_files={
            compare.PRODUCT_CHANGE: head,
            "src/phymeo/core/result.py": "unchanged-product",
        },
        tools={
            name: {"path": "C:/frozen/" + suffix, "sha256": "fixed-" + name}
            for name, suffix in compare.TOOL_SUFFIXES.items()
        },
        driver_sha256="fixed-benchmark",
    )


@pytest.mark.parametrize("changed", ["cases", "profile", "product"])
def test_comparison_rejects_changed_original_tools_or_other_product_files(changed):
    compare = _comparison_module()
    before_source = _source_receipt(compare, "old")
    after_source = _source_receipt(compare, "new")
    before = dict(source=before_source, source_after=copy.deepcopy(before_source))
    after = dict(source=after_source, source_after=copy.deepcopy(after_source))
    assert compare._check_sources(before, after) == [compare.PRODUCT_CHANGE]
    for stage in ("source", "source_after"):
        if changed == "product":
            after[stage]["imported_files"]["src/phymeo/core/result.py"] = "changed"
        else:
            after[stage]["tools"][changed]["sha256"] = "changed"
    with pytest.raises(AssertionError, match="changed|changes"):
        compare._check_sources(before, after)


def _protocol_run(directory, compare, head, *, mutate=False):
    """Tiny pure codec receipt, not a substitute for the actual nine physics runs."""
    directory.mkdir()
    driver = compare.driver
    windows = []
    for phase in driver.WINDOWS:
        rows = []
        with driver.WindowWriter(directory / (phase + ".json.gz"), {"phase": phase}) as writer:
            for step in range(driver.STEPS):
                frame = {
                    "frame": {"sample": step, "normal": [0, 0, 1], "future_private_field": 7},
                    "geometry": {"time_s": (step + 1) * driver.H},
                    "contact_history": [],
                }
                if mutate and step == 7:
                    frame["frame"]["future_private_field"] = 8
                row = _protocol_row()
                row.update(
                    step=step + 1, entry_s=step * driver.H, end_s=(step + 1) * driver.H,
                    frame_logical_sha256=writer.append(frame),
                )
                rows.append(row)
        windows.append(dict(
            phase=phase, successful_steps=driver.STEPS, rows=rows, build_ns=7,
            output=writer.stats(),
            public_statistics=driver._distribution([r["public_to_end_ns"] for r in rows]),
            step_statistics=driver._distribution([r["step_ns"] for r in rows]),
            native_build_counters={"cuda_kernel_launches": 0},
            native_window_counters={"cuda_kernel_launches": 160},
            max_physical_errors={"position_m": 0.0},
        ))
    source = _source_receipt(compare, head)
    manifest = dict(
        schema=driver.SCHEMA, success=True, backend="cuda",
        source=source, source_after=copy.deepcopy(source),
        case="free", worlds=1, steps_per_window=driver.STEPS, h_s=driver.H,
        matrix=[list(pair) for pair in driver.MATRIX],
        started_utc="2026-11-05T00:00:00+00:00", ended_utc="2026-11-05T00:00:01+00:00",
        declaration_ns=1, windows=windows,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")


def test_full_archive_comparison_reuses_acceptance_and_locates_unknown_field(
    tmp_path, monkeypatch
):
    compare = _comparison_module()
    old, same, changed = (tmp_path / name for name in ("old", "same", "changed"))
    _protocol_run(old, compare, "old")
    _protocol_run(same, compare, "new")
    _protocol_run(changed, compare, "new", mutate=True)
    original_reader = compare.driver.read_window

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted old frames were unnecessarily decoded again")

    monkeypatch.setattr(compare.driver, "read_window", forbidden)
    result = compare.compare_case(old, same)
    assert result["success"]
    assert all(row["extra_full_frames_decoded"] == 0 for row in result["windows"])
    monkeypatch.setattr(compare.driver, "read_window", original_reader)
    assert [row["compared_frames"] for row in result["windows"]] == [32, 32]
    assert all(row["excluded_public_fields"] == [] for row in result["windows"])
    with pytest.raises(AssertionError, match="future_private_field"):
        compare.compare_case(old, changed)
