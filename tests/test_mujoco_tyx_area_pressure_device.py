"""Accepted area-device copy contracts; short real steps, no old trajectory rerun."""

from collections.abc import Mapping
from contextlib import ExitStack
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import tyx as tx
from test_mujoco_tyx_area_pressure_lifecycle import H, _advance, _load, _readonly, _two_worlds

from experiments.area_pressure.loaded_support import DUAL, build_support
from phymeo.backends.mujoco_area_pressure_state import MuJoCoTyxAreaSceneCheckpoint
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
FIELDS = {
    "entry_q",
    "entry_v",
    "entry_clock",
    "entry_meta",
    "area_meta",
    "area_ids",
    "area_data",
    "area_measure_words",
    "endpoint_q",
    "endpoint_v",
    "endpoint_clock",
}


def _array(stack, mode, values):
    values = np.array(values, copy=True)
    if mode == "interpret":
        return values
    result = tx.CudaArray(values)
    stack.callback(result.free)
    return result


def _outputs(stack, runtime, mode, sentinel=0):
    return {
        name: _array(stack, mode, np.full(shape, sentinel, dtype))
        for name, (shape, dtype) in runtime.area_pressure_device_layout.items()
    }


def _host(outputs, mode):
    return {
        name: value.copy() if mode == "interpret" else value.to_numpy()
        for name, value in outputs.items()
    }


def _copy(runtime, outputs):
    return runtime.copy_area_pressure(outputs, expected_revision=runtime.checkpoint_revision)


def _capture(stack, runtime, mode):
    outputs = _outputs(stack, runtime, mode)
    binding = _copy(runtime, outputs)
    return binding, _host(outputs, mode)


def _bytes_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].tobytes() == expected[name].tobytes(), name


def _rows_equal(actual, expected, rows):
    _bytes_equal(actual, {name: value[list(rows)] for name, value in expected.items()})


def _time(words):
    return words[..., 0].astype(np.float64) - words[..., 1].astype(np.float64)


def _check_values(runtime, binding, arrays, observations):
    assert binding["schema"] == "mujoco-tyx-area-device-copy-v1"
    assert binding["revision"] == runtime.checkpoint_revision
    assert binding["world_keys"] == runtime.world_keys
    assert binding["pair_names"] == tuple(
        (pair["identity"].compliant_body, pair["identity"].rigid_body)
        for pair in observations[runtime.world_keys[0]]["pairs"]
    )
    _readonly(binding)
    words = np.ascontiguousarray(arrays["area_measure_words"], dtype="<i4")
    area = words.view("<f8").reshape(words.shape[:2])
    for row, key in enumerate(runtime.world_keys):
        frame = observations[key]
        nq, nv = len(frame["endpoint_q"]), len(frame["endpoint_v"])
        np.testing.assert_array_equal(
            arrays["endpoint_q"][row, :nq], frame["endpoint_q"].astype(np.float32)
        )
        np.testing.assert_array_equal(
            arrays["endpoint_v"][row, :nv], frame["endpoint_v"].astype(np.float32)
        )
        np.testing.assert_array_equal(
            arrays["entry_q"][row, :nq], frame["input_q"].astype(np.float32)
        )
        np.testing.assert_array_equal(
            arrays["entry_v"][row, :nv], frame["input_v"].astype(np.float32)
        )
        for name, width in (
            ("entry_q", nq),
            ("entry_v", nv),
            ("endpoint_q", nq),
            ("endpoint_v", nv),
        ):
            np.testing.assert_array_equal(arrays[name][row, width:], 0)
        assert _time(arrays["entry_clock"][row, :2]) == frame["input_time"]
        assert _time(arrays["entry_clock"][row, 2:]) == frame["output_time"]
        assert _time(arrays["endpoint_clock"][row]) == frame["endpoint_time"]
        np.testing.assert_array_equal(
            arrays["entry_meta"][row], [1, frame["input_tick"], frame["output_tick"], 0, 0, 0]
        )
        cursor, positive = 0, 0
        for ordinal, pair in enumerate(frame["pairs"]):
            quad = pair["source_quadrature"]
            count = len(quad.dA)
            sl = slice(cursor, cursor + count)
            ids = arrays["area_ids"][row, sl]
            np.testing.assert_array_equal(ids[:, 0], 1)
            np.testing.assert_array_equal(ids[:, 1], ordinal)
            np.testing.assert_array_equal(ids[:, 2], np.arange(cursor, cursor + count))
            np.testing.assert_array_equal(ids[:, 3:5], np.tile(pair["body_ids"], (count, 1)))
            np.testing.assert_array_equal(ids[:, 5], 1)
            # Binary64 measure is preserved independently of FP32 point/force storage.
            assert area[row, sl].tobytes() == np.asarray(quad.dA, dtype="<f8").tobytes()
            expected = np.column_stack(
                (
                    pair["points"],
                    pair["normals"],
                    pair["stiffness"],
                    pair["surrogate_gap"],
                    pair["normal_force"],
                    pair["force_on_a"],
                    pair["torque_on_a_world_origin"],
                )
            )
            assert expected.shape == (count, 15)
            np.testing.assert_array_equal(
                arrays["area_data"][row, sl, :9], expected[:, :9].astype(np.float32)
            )
            np.testing.assert_allclose(
                arrays["area_data"][row, sl, 9:], expected[:, 9:], rtol=3e-5, atol=2e-6
            )
            catalog = binding["catalogs"][key][ordinal]
            assert set(catalog) == {"identity", "stamp", "sample_ids", "source_ids"}
            for name in catalog:
                assert catalog[name] == pair[name]
            positive += int(np.count_nonzero(pair["normal_force"] > 0))
            cursor += count
        assert len(binding["catalogs"][key]) == len(frame["pairs"])
        np.testing.assert_array_equal(arrays["area_meta"][row], [1, cursor, positive, 0])
        for name in ("area_data", "area_ids", "area_measure_words"):
            np.testing.assert_array_equal(arrays[name][row, cursor:], 0)


@pytest.mark.parametrize("mode", MODES)
def test_full_batch_device_fields_and_no_host_numeric_producer(mode, monkeypatch):
    runtime, scene = _two_worlds(mode)
    with runtime, ExitStack() as stack:
        _load(runtime, -3.0)
        _advance(scene, 2)
        expected = runtime.area_pressure_snapshot()
        layout = runtime.area_pressure_device_layout
        assert isinstance(layout, MappingProxyType) and set(layout) == FIELDS
        with pytest.raises(TypeError):
            layout["invented"] = ((1,), np.dtype(np.float32))
        for shape, dtype in layout.values():
            assert type(shape) is tuple and isinstance(dtype, np.dtype)
            assert shape[0] == 2
        assert layout["area_ids"][0][-1] == 6
        assert layout["area_data"][0][-1] == 15
        assert layout["area_measure_words"][0][-1] == 2
        assert layout["area_measure_words"][1] == np.dtype(np.int32)
        before = runtime.checkpoint().to_bytes()
        outputs = _outputs(stack, runtime, mode, sentinel=71)
        counts = dict(runtime.counters)
        # Only identity metadata remains accessible to the copy path. Numeric
        # receipt fields cannot serve as a hidden replacement for device bytes.
        metadata_only = {
            key: SimpleNamespace(
                raw_contact_fields=MappingProxyType(
                    {
                        "area_pressure": tuple(
                            MappingProxyType(
                                {
                                    name: pair[name]
                                    for name in ("identity", "stamp", "sample_ids", "source_ids")
                                }
                            )
                            for pair in expected[key]["pairs"]
                        )
                    }
                )
            )
            for key in runtime.world_keys
        }

        def forbidden(*args, **kwargs):
            raise AssertionError("device copy attempted a HOST numeric producer/readback")

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_receipts", metadata_only)
            for name in ("snapshot", "integration_snapshot", "area_pressure_snapshot"):
                patch.setattr(runtime, name, forbidden)
            patch.setattr(runtime._device, "_get", forbidden)
            patch.setattr(runtime._device, "download", forbidden)
            for name in ("mj_forward", "mj_step", "mj_kinematics", "mj_comPos", "mj_comVel"):
                patch.setattr(runtime._mj, name, forbidden)
            if mode == "cuda":
                patch.setattr(tx.CudaArray, "to_numpy", forbidden)
            binding = _copy(runtime, outputs)
        for name in ("completed_kernels", "uploaded_bytes", "downloaded_bytes"):
            assert runtime.counters[name] == counts[name]
        arrays = _host(outputs, mode)
        _check_values(runtime, binding, arrays, expected)
        assert runtime.checkpoint().to_bytes() == before
        # A caller may overwrite its successful copies without touching accepted banks.
        for name, value in outputs.items():
            if mode == "interpret":
                value.fill(19)
            else:
                shape, dtype = layout[name]
                value.copy_from(np.full(shape, 19, dtype))
        _, copied_again = _capture(stack, runtime, mode)
        _bytes_equal(copied_again, arrays)


@pytest.mark.parametrize("mode", MODES)
def test_initial_invalid_successful_empty_and_copy_storage_ownership(mode):
    runtime, scene = build_support(mode, H, DUAL, initial_q=0.0)
    with runtime, ExitStack() as stack:
        outputs = _outputs(stack, runtime, mode, sentinel=23)
        initial_binding = _copy(runtime, outputs)
        initial = _host(outputs, mode)
        assert initial_binding["catalogs"][13] == ()
        for name in (
            "entry_q",
            "entry_v",
            "entry_clock",
            "entry_meta",
            "area_meta",
            "area_ids",
            "area_data",
            "area_measure_words",
        ):
            np.testing.assert_array_equal(initial[name], 0)
        _advance(scene, 1)
        _bytes_equal(_host(outputs, mode), initial)
        empty_binding, empty = _capture(stack, runtime, mode)
        np.testing.assert_array_equal(empty["entry_meta"][0], [1, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(empty["area_meta"][0], [1, 0, 0, 0])
        assert len(empty_binding["catalogs"][13]) == 2
        assert all(catalog["sample_ids"] == () for catalog in empty_binding["catalogs"][13])
        for name in ("area_ids", "area_data", "area_measure_words"):
            np.testing.assert_array_equal(empty[name], 0)
        held = _outputs(stack, runtime, mode)
        _copy(runtime, held)
        _advance(scene, 1)
        scene.reset()
        _bytes_equal(_host(held, mode), empty)
        _, reset = _capture(stack, runtime, mode)
        _bytes_equal(reset, initial)
        _readonly(initial_binding)
        _readonly(empty_binding)


@pytest.mark.parametrize("mode", MODES)
def test_device_banks_roundtrip_reordered_subset_clone_and_continuation(mode):
    import phymeo.backends.mujoco_checkpoint as checkpoints

    runtime, scene = _two_worlds(mode)
    with runtime, ExitStack() as stack:
        _load(runtime, -3.0)
        _advance(scene, 2)
        checkpoint = MuJoCoTyxAreaSceneCheckpoint.from_bytes(scene.checkpoint().to_bytes())
        original_binding, original = _capture(stack, runtime, mode)
        clones = []
        for selected, rows in (((41, 13), (1, 0)), ((41,), (1,))):
            cloned = scene.clone(world_keys=selected)
            stack.callback(cloned.runtime.close)
            clone_binding, clone_values = _capture(stack, cloned.runtime, mode)
            assert clone_binding["world_keys"] == selected
            _rows_equal(clone_values, original, rows)
            for key in selected:
                assert clone_binding["catalogs"][key] == original_binding["catalogs"][key]
            clones.append((cloned, rows))
        _advance(scene, 1)
        _, continued = _capture(stack, runtime, mode)
        for cloned, rows in clones:
            _advance(cloned, 1)
            _, clone_values = _capture(stack, cloned.runtime, mode)
            _rows_equal(clone_values, continued, rows)
        scene.restore(checkpoint, expected_revision=runtime.checkpoint_revision)
        restored_binding, restored = _capture(stack, runtime, mode)
        _bytes_equal(restored, original)
        assert restored_binding["catalogs"] == original_binding["catalogs"]
        _advance(scene, 1)
        _, replay = _capture(stack, runtime, mode)
        _bytes_equal(replay, continued)
        # A valid checksum cannot authorize an area bank inconsistent with its
        # original quadrature, typed force rows and historical receipt.
        runtime_checkpoint = checkpoint.runtime_checkpoint
        forged_state = dict(runtime_checkpoint._read(runtime_checkpoint.binding))
        forged_entry = dict(forged_state["accepted_entry"])
        data = forged_entry["area_data"].copy()
        assert data[0, 0, 8] > 0
        data[0, 0, 8] += 0.25
        forged_entry["area_data"] = data
        forged_state["accepted_entry"] = forged_entry
        forged = checkpoints._make_checkpoint(
            runtime_checkpoint.binding, runtime_checkpoint.source_revision, forged_state
        )
        envelope = MuJoCoTyxAreaSceneCheckpoint.make(scene.assets, forged)
        revision, capsule = runtime.checkpoint_revision, runtime.checkpoint().to_bytes()
        with pytest.raises(ValueError):
            scene.restore(envelope, expected_revision=revision)
        _, retained = _capture(stack, runtime, mode)
        _bytes_equal(retained, continued)
        assert runtime.checkpoint_revision == revision
        assert runtime.checkpoint().to_bytes() == capsule
        # Pre-device I3a capsules lack both the layout binding and resident
        # fields. They must be refused, never backfilled from HOST receipts.
        old_state = dict(runtime_checkpoint._read(runtime_checkpoint.binding))
        old_entry = dict(old_state["accepted_entry"])
        old_binding = dict(runtime_checkpoint.binding)
        old_entry_binding = dict(old_binding["accepted_entry"])
        old_layout = dict(old_entry_binding["layout"])
        for name in ("area_meta", "area_ids", "area_data", "area_measure_words"):
            del old_entry[name]
            del old_layout[name]
        del old_entry_binding["area_schema"]
        del old_entry_binding["area_measure"]
        old_entry_binding["layout"] = old_layout
        old_binding["accepted_entry"] = old_entry_binding
        old_state["accepted_entry"] = old_entry
        legacy = checkpoints._make_checkpoint(
            old_binding, runtime_checkpoint.source_revision, old_state
        )
        with pytest.raises(ValueError):
            runtime.restore(legacy, expected_revision=revision)
        _, retained = _capture(stack, runtime, mode)
        _bytes_equal(retained, continued)
        assert runtime.checkpoint_revision == revision
        assert runtime.checkpoint().to_bytes() == capsule


@pytest.mark.parametrize("mode", MODES)
def test_output_preflight_rejects_every_destination_before_any_write(mode, monkeypatch):
    runtime, scene = _two_worlds(mode)
    with runtime, ExitStack() as stack:
        _advance(scene, 1)
        outputs = _outputs(stack, runtime, mode, sentinel=37)
        untouched = _host(outputs, mode)
        revision, before = runtime.checkpoint_revision, runtime.checkpoint().to_bytes()

        def reject(candidate, *, expected=revision):
            with pytest.raises((ValueError, TypeError)):
                runtime.copy_area_pressure(candidate, expected_revision=expected)
            _bytes_equal(_host(outputs, mode), untouched)
            assert runtime.checkpoint_revision == revision
            assert runtime.checkpoint().to_bytes() == before

        reject(outputs, expected=revision - 1)
        reject(outputs, expected=False)
        reject(outputs, expected=float(revision))
        missing = dict(outputs)
        del missing["endpoint_clock"]
        reject(missing)
        reject(dict(outputs, extra=outputs["endpoint_clock"]))
        shape, dtype = runtime.area_pressure_device_layout["endpoint_clock"]
        reject(dict(outputs, endpoint_clock=_array(stack, mode, np.zeros(shape, np.float64))))
        reject(
            dict(
                outputs,
                endpoint_clock=_array(stack, mode, np.zeros((*shape[:-1], shape[-1] + 1), dtype)),
            )
        )
        if mode == "interpret":
            readonly = np.zeros(shape, dtype)
            readonly.setflags(write=False)
            reject(dict(outputs, endpoint_clock=readonly))
        else:
            with monkeypatch.context() as patch:
                patch.setattr(outputs["endpoint_clock"], "device_ordinal", 1)
                with pytest.raises(ValueError):
                    _copy(runtime, outputs)
            _bytes_equal(_host(outputs, mode), untouched)
        # Distinct caller views of the same allocation must be detected.
        qshape, qdtype = runtime.area_pressure_device_layout["entry_q"]
        if mode == "interpret":
            overlap = outputs["endpoint_q"].view()
        else:
            overlap = tx.CudaArray.from_external(
                int(outputs["endpoint_q"].device_ptr),
                qshape,
                qdtype,
                owner=outputs["endpoint_q"],
                device=outputs["endpoint_q"].device_ordinal,
            )
            stack.callback(overlap.free)
        reject(dict(outputs, entry_q=overlap))
        reject(dict(outputs, endpoint_q=runtime._device.q[runtime._device.active]))
        items_called = []

        class SwappedItems(Mapping):
            def __iter__(self):
                return iter(outputs)

            def __len__(self):
                return len(outputs)

            def __getitem__(self, name):
                return outputs[name]

            def items(self):
                items_called.append(True)
                return dict(outputs, endpoint_q=runtime._device.q[runtime._device.active]).items()

        # The preflight-approved mapping must be captured once; a caller's
        # alternate items() cannot replace destinations after validation.
        runtime.copy_area_pressure(SwappedItems(), expected_revision=revision)
        assert items_called == []
        np.testing.assert_array_equal(_host(outputs, mode)["entry_meta"][:, 2], 1)
        assert runtime.checkpoint().to_bytes() == before


@pytest.mark.parametrize("mode", MODES)
def test_late_download_and_capacity_failures_preserve_accepted_device_banks(mode, monkeypatch):
    runtime, scene = _two_worlds(mode)
    with runtime, ExitStack() as stack:
        _advance(scene, 1)
        binding, before = _capture(stack, runtime, mode)
        revision, capsule = runtime.checkpoint_revision, runtime.checkpoint().to_bytes()
        geometry = scene.last_geometry
        download, seen = runtime._device.download, []

        def fail_after_download():
            download()
            seen.append(True)
            raise RuntimeError("deliberate post-download area fault")

        with monkeypatch.context() as patch:
            patch.setattr(runtime._device, "download", fail_after_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                _advance(scene, 1)
        assert seen == [True]
        assert caught.value.acceptance_known and not caught.value.physics_accepted
        assert caught.value.accepted_worlds == {13: False, 41: False}
        assert runtime.checkpoint_revision == revision
        assert scene.last_geometry is geometry
        after_binding, after = _capture(stack, runtime, mode)
        _bytes_equal(after, before)
        assert after_binding == binding
        assert runtime.checkpoint().to_bytes() == capsule
        _advance(scene, 1)
        _, accepted = _capture(stack, runtime, mode)
        np.testing.assert_array_equal(accepted["entry_meta"][:, 2], 2)
    limited, limited_scene = build_support(mode, H, DUAL, initial_q=-0.0025, capacity=2)
    with limited, ExitStack() as stack:
        binding, before = _capture(stack, limited, mode)
        revision, capsule = limited.checkpoint_revision, limited.checkpoint().to_bytes()
        with pytest.raises(ValueError, match="capacity"):
            _advance(limited_scene, 1)
        after_binding, after = _capture(stack, limited, mode)
        _bytes_equal(after, before)
        assert after_binding == binding and limited.checkpoint_revision == revision
        assert limited.checkpoint().to_bytes() == capsule


@pytest.mark.parametrize("mode", MODES)
def test_stage_bounds_and_unavailable_world_moment_are_not_physics_failure(mode):
    from phymeo.backends.mujoco_tyx._area_entry_kernels import stage

    runtime, _ = build_support(mode, H, DUAL)
    with runtime, ExitStack() as stack:
        device = runtime._device
        samples = np.zeros((1, 2, 8), np.float32)
        samples[0, 0] = [1e30, 0, 0, 0, 1, 0, 1, -1]
        row_ids = np.zeros((1, 2, 5), np.int32)
        row_ids[0, 0] = [7, 0, -1, -1, -1]
        row_data = np.zeros((1, 2, 14), np.float32)
        row_data[0, 0, 9] = 1e20
        initial = [
            np.array([1], np.int32),
            np.zeros((1, 2), np.int32),
            np.array([[1, 0]], np.int32),
            samples,
            np.array([0.1, 0], dtype="<f8").view("<i4").reshape(1, 2, 2),
            np.array([[0, 0, 1]], np.int32),
            row_ids,
            row_data,
            np.array([[1, 0, 1, 0, 0, 0]], np.int32),
            np.zeros((1, 4), np.int32),
            np.zeros((1, 2, 6), np.int32),
            np.zeros((1, 2, 15), np.float32),
            np.zeros((1, 2, 2), np.int32),
            np.zeros(1, np.int32),
        ]
        args = [_array(stack, mode, value) for value in initial]

        def run():
            with np.errstate(over="ignore", invalid="ignore"), tx.use_runtime(device.runtime):
                stage(1, 2, 1, *args, mode=mode)
                device.runtime.sync()
            return [v.copy() if mode == "interpret" else v.to_numpy() for v in args]

        got = run()
        np.testing.assert_array_equal(got[-1], 0)
        np.testing.assert_array_equal(got[9], [[1, 1, 1, 0]])
        assert got[10][0, 0, 5] == 0
        np.testing.assert_array_equal(got[11][0, 0, 12:15], 0)
        assert got[11][0, 0, 8] == np.float32(1e20)
        # INT_MIN must be rejected before subtracting size/indexing rows.
        bad_counts = np.array([[0, 0, np.iinfo(np.int32).min]], np.int32)
        if mode == "interpret":
            args[5][:] = bad_counts
        else:
            args[5].copy_from(bad_counts)
        got = run()
        np.testing.assert_array_equal(got[-1], 1)
        np.testing.assert_array_equal(got[9], 0)
