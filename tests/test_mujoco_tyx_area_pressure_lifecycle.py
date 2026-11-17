"""Actual area-contact continuation and immutable HOST observation lifecycle.

This new file exercises the existing loaded-support fixture; it does not rerun
the previous complete support qualification. Authoring performs no execution.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure import (
    H as HINGE_H,
)
from test_mujoco_tyx_area_pressure import (
    _hinge_slider_xml,
    _mapping,
    _runtime,
)

from experiments.area_pressure.loaded_support import DUAL, build_support
from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    MuJoCoTyxAreaPressureScene,
)
from phymeo.backends.mujoco_area_pressure_state import MuJoCoTyxAreaSceneCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
KEYS = (13, 41)


def _same(actual, expected):
    """All scientific bytes match; only live revision/configuration tokens differ."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            if field.name not in ("configuration_id", "source_revision"):
                _same(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and set(actual) == set(expected)
        for key in expected:
            if key not in ("configuration_id", "source_revision"):
                _same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _same(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def _readonly(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    elif is_dataclass(value):
        for field in fields(value):
            _readonly(getattr(value, field.name))
    elif isinstance(value, Mapping):
        assert isinstance(value, MappingProxyType)
        with pytest.raises(TypeError):
            value["forbidden-write"] = 1
        for item in value.values():
            _readonly(item)
    elif isinstance(value, (tuple, list)):
        assert isinstance(value, tuple)
        for item in value:
            _readonly(item)


def _load(runtime, force):
    runtime.set_controls(
        {key: MuJoCoControl([], qfrc_applied=[force]) for key in runtime.world_keys}
    )


def _advance(scene, count):
    for _ in range(count):
        assert scene.step(H, expected_revision=scene.runtime.checkpoint_revision) == dict.fromkeys(
            scene.runtime.world_keys, True
        )


def _state(runtime):
    return runtime.integration_snapshot(), runtime.area_pressure_snapshot()


def _two_worlds(mode):
    template, scene = build_support(mode, H, DUAL, initial_q=-0.0025)
    try:
        compilation, assets = template.compilation, scene.assets
    finally:
        template.close()
    runtime = MuJoCoTyxRuntime(
        compilation,
        mode=mode,
        world_keys=KEYS,
        initial_states={
            13: MuJoCoInitialState([-0.0025], [0.0]),
            41: MuJoCoInitialState([-0.0027], [0.01]),
        },
        area_pressure_pairs=tuple(
            (item.identity.compliant_body, item.identity.rigid_body) for item in assets
        ),
        constraint_capacity=64,
    )
    return runtime, MuJoCoTyxAreaPressureScene(runtime, assets)


def _retained(scene, before, observation, raw, revision, geometry):
    runtime = scene.runtime
    _same(runtime.integration_snapshot(), before)
    _same(runtime.area_pressure_snapshot(), observation)
    assert runtime.checkpoint_revision == revision
    assert runtime.checkpoint().to_bytes() == raw
    assert scene.last_geometry is geometry
    for key in runtime.world_keys:
        assert runtime.integration_snapshot()[key].last_step is before[key].last_step


@pytest.mark.parametrize("mode", MODES)
def test_loaded_checkpoint_bytes_restore_and_fifty_step_bitwise_continuation(mode, tmp_path):
    runtime, scene = build_support(mode, H, DUAL, initial_q=-0.0025)
    with runtime:
        _load(runtime, -3.0)
        _advance(scene, 30)
        at_checkpoint = _state(runtime)
        assert at_checkpoint[0][13].tick == 30
        assert all(np.sum(pair["normal_force"]) > 0 for pair in at_checkpoint[1][13]["pairs"])
        checkpoint = scene.checkpoint()
        path = tmp_path / "loaded-step-30.area-checkpoint"
        path.write_bytes(checkpoint.to_bytes())
        restored = MuJoCoTyxAreaSceneCheckpoint.from_bytes(path.read_bytes())
        assert restored.to_bytes() == checkpoint.to_bytes()
        assert restored.asset_sha256 == checkpoint.asset_sha256 and len(restored.asset_sha256) == 64
        assert restored.runtime_checkpoint.to_bytes() == checkpoint.runtime_checkpoint.to_bytes()
        expected = []
        for _ in range(50):
            _advance(scene, 1)
            expected.append(_state(runtime))
        assert expected[-1][0][13].tick == 80
        np.savez_compressed(
            tmp_path / "expected-steps-31-through-80.npz",
            q=np.stack([item[0][13].q for item in expected]),
            v=np.stack([item[0][13].v for item in expected]),
            generalized_contact_force=np.stack(
                [item[0][13].last_step.qfrc_constraint for item in expected]
            ),
        )
        scene.restore(restored, expected_revision=runtime.checkpoint_revision)
        assert dict(scene.last_geometry) == {}
        _same(_state(runtime), at_checkpoint)
        # Restoring historical records must retain their original provenance.
        original_stamp = at_checkpoint[1][13]["pairs"][0]["stamp"]
        assert runtime.area_pressure_snapshot()[13]["pairs"][0]["stamp"] == original_stamp
        assert runtime.area_pressure_configuration_id(13) != original_stamp.configuration_id
        for wanted in expected:
            _advance(scene, 1)
            _same(_state(runtime), wanted)
        assert runtime.integration_snapshot()[13].tick == 80


@pytest.mark.parametrize("mode", MODES)
def test_clone_subset_preserves_history_and_owns_independent_same_mode_continuation(mode):
    runtime, scene = _two_worlds(mode)
    with runtime:
        _load(runtime, -3.0)
        _advance(scene, 3)
        reordered = scene.clone(world_keys=(41, 13))
        with reordered.runtime:
            assert reordered.runtime.world_keys == (41, 13)
            _same(reordered.runtime.area_pressure_snapshot(), runtime.area_pressure_snapshot())
        cloned = scene.clone(world_keys=(41,))
        with cloned.runtime:
            assert cloned.runtime is not runtime and cloned.runtime.world_keys == (41,)
            assert cloned.assets is not scene.assets
            assert cloned.checkpoint().asset_sha256 == scene.checkpoint().asset_sha256
            _same(cloned.runtime.integration_snapshot(), {41: runtime.integration_snapshot()[41]})
            _same(
                cloned.runtime.area_pressure_snapshot(), {41: runtime.area_pressure_snapshot()[41]}
            )
            assert dict(cloned.last_geometry) == {}
            for _ in range(4):
                _advance(scene, 1)
                _advance(cloned, 1)
                _same(
                    cloned.runtime.integration_snapshot(), {41: runtime.integration_snapshot()[41]}
                )
                _same(
                    cloned.runtime.area_pressure_snapshot(),
                    {41: runtime.area_pressure_snapshot()[41]},
                )
            old = cloned.runtime.area_pressure_snapshot()
            tick = cloned.runtime.integration_snapshot()[41].tick
            runtime.close()
            _advance(cloned, 1)
            assert cloned.runtime.integration_snapshot()[41].tick == tick + 1
            _readonly(old)
            assert old[41]["endpoint_tick"] == tick


@pytest.mark.parametrize("mode", MODES)
def test_host_observation_history_controls_and_selected_reset_do_not_launch(mode):
    runtime, scene = _two_worlds(mode)
    with runtime:
        initial = runtime.area_pressure_snapshot()
        assert set(initial) == set(KEYS)
        assert all(not frame["valid"] and frame["pairs"] == () for frame in initial.values())
        assert all("input_q" not in frame for frame in initial.values())
        _load(runtime, -3.0)
        _advance(scene, 2)
        counters = dict(runtime.counters)
        history = runtime.area_pressure_snapshot()
        selected = runtime.area_pressure_snapshot(keys=(41,))
        _same(selected, {41: history[41]})
        _readonly(history)
        for name in ("completed_kernels", "synchronizations", "uploaded_bytes", "downloaded_bytes"):
            assert runtime.counters[name] == counters[name]
        for frame in history.values():
            assert frame["schema"] == "mujoco-tyx-area-observation-v1"
            assert frame["storage"] == "host-owned-accepted-receipt" and frame["valid"]
            assert frame["input_tick"] + 1 == frame["output_tick"] == frame["endpoint_tick"]
            assert frame["output_time"] == frame["endpoint_time"]
            np.testing.assert_array_equal(frame["applied_control"].qfrc_applied, [-3.0])
            np.testing.assert_array_equal(frame["current_control"].qfrc_applied, [-3.0])
        _load(runtime, 9.0)
        after_control = runtime.area_pressure_snapshot()
        for key in KEYS:
            expected = dict(history[key], current_control=after_control[key]["current_control"])
            _same(after_control[key], expected)
            np.testing.assert_array_equal(
                after_control[key]["applied_control"].qfrc_applied, [-3.0]
            )
            np.testing.assert_array_equal(after_control[key]["current_control"].qfrc_applied, [9.0])
        assert after_control[13]["source_revision"] > history[13]["source_revision"]
        assert dict(scene.last_geometry) == {}
        np.testing.assert_array_equal(
            runtime.integration_snapshot()[13].held_control.qfrc_applied, [9.0]
        )
        scene.reset(keys=(13,))
        after_reset = runtime.area_pressure_snapshot()
        assert not after_reset[13]["valid"] and after_reset[13]["pairs"] == ()
        assert after_reset[13]["endpoint_tick"] == 0 and "applied_control" not in after_reset[13]
        _same(after_reset[41], after_control[41])
        assert history[13]["valid"] and history[13]["endpoint_tick"] == 2
        _readonly(history)


@pytest.mark.parametrize("mode", MODES)
def test_changed_assets_and_late_restore_failure_leave_accepted_scene_untouched(mode, monkeypatch):
    import phymeo.backends.mujoco_checkpoint as checkpoints

    runtime, scene = build_support(mode, H, DUAL, initial_q=-0.0025)
    with runtime:
        _load(runtime, -3.0)
        _advance(scene, 2)
        saved = scene.checkpoint()
        saved_state = _state(runtime)
        _advance(scene, 3)
        before, observation = _state(runtime)
        raw, revision, geometry = (
            runtime.checkpoint().to_bytes(),
            runtime.checkpoint_revision,
            scene.last_geometry,
        )
        scene.reset(keys=())
        _retained(scene, before, observation, raw, revision, geometry)
        asset = scene.assets[0]
        changed_pressure = replace(
            asset.pressure_local, nodal_pressure=asset.pressure_local.nodal_pressure * 2
        )
        changed = BodyAreaPressureAsset(asset.identity, changed_pressure, asset.rigid_local)
        incompatible = MuJoCoTyxAreaPressureScene(runtime, (changed, scene.assets[1]))
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                pytest.fail("asset mismatch reached replacement verification")

            patch.setattr(checkpoints, "_verify_bank", forbidden)
            with pytest.raises(ValueError):
                incompatible.restore(saved, expected_revision=revision)
        _retained(scene, before, observation, raw, revision, geometry)
        verified = []
        original = checkpoints._verify_bank

        def late_failure(*args, **kwargs):
            original(*args, **kwargs)
            verified.append(True)
            raise RuntimeError("deliberate verified area replacement failure")

        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, "_verify_bank", late_failure)
            with pytest.raises(RuntimeError, match="verified area replacement"):
                scene.restore(saved, expected_revision=revision)
        assert verified == [True]
        _retained(scene, before, observation, raw, revision, geometry)
        scene.restore(saved, expected_revision=revision)
        _same(_state(runtime), saved_state)
        assert dict(scene.last_geometry) == {}
        _advance(scene, 1)
        assert runtime.integration_snapshot()[13].tick == 3


@pytest.mark.parametrize("mode", MODES)
def test_rechecksummed_area_metadata_cannot_bypass_restore_validation(mode, monkeypatch):
    import phymeo.backends.mujoco_checkpoint as checkpoints

    runtime, scene = build_support(mode, H, DUAL, initial_q=-0.0025)
    with runtime:
        _load(runtime, -3.0)
        _advance(scene, 2)
        original = runtime.checkpoint()
        before, observation = _state(runtime)
        raw, revision, geometry = (
            original.to_bytes(),
            runtime.checkpoint_revision,
            scene.last_geometry,
        )
        for fault in (
            "normal-force",
            "source-area",
            "historical-stamp",
            "raw-efc-force",
            "input-phase",
            "missing-entry",
        ):
            state = dict(original._read(original.binding))
            worlds = list(state["worlds"])
            world = dict(worlds[0])
            receipt = world["receipt"]
            fields_raw = dict(receipt.raw_contact_fields)
            pairs = list(fields_raw["area_pressure"])
            pair = dict(pairs[1])
            assert len(pair["normal_force"]) > 0
            if fault == "normal-force":
                force = pair["normal_force"].copy()
                force[0] += 0.125
                pair["normal_force"] = force
            elif fault == "source-area":
                quad = pair["source_quadrature"]
                pair["source_quadrature"] = replace(
                    quad, dA=quad.dA * 2, total_area_m2=quad.total_area_m2 * 2
                )
            elif fault == "historical-stamp":
                pair["stamp"] = replace(
                    pair["stamp"], sample_time_s=pair["stamp"].sample_time_s + H
                )
            elif fault == "raw-efc-force":
                raw_rows = fields_raw["efc_data"].copy()
                start = int(fields_raw["counts"][2]) - sum(
                    len(item["normal_force"]) for item in pairs
                )
                raw_rows[start + len(pairs[0]["normal_force"]), 9] += 0.125
                fields_raw["efc_data"] = raw_rows
            elif fault == "input-phase":
                receipt = replace(receipt, contact_time=receipt.output_time)
            else:
                del state["accepted_entry"]
            pairs[1] = MappingProxyType(pair)
            fields_raw["area_pressure"] = tuple(pairs)
            world["receipt"] = replace(receipt, raw_contact_fields=MappingProxyType(fields_raw))
            worlds[0] = world
            state["worlds"] = tuple(worlds)
            binding = dict(original.binding)
            if fault == "missing-entry":
                del binding["accepted_entry"]
            forged = checkpoints._make_checkpoint(binding, original.source_revision, state)
            envelope = MuJoCoTyxAreaSceneCheckpoint.make(scene.assets, forged)
            with monkeypatch.context() as patch:

                def forbidden(*args, **kwargs):
                    pytest.fail("malformed history reached replacement verification")

                patch.setattr(checkpoints, "_verify_bank", forbidden)
                with pytest.raises(ValueError):
                    scene.restore(envelope, expected_revision=revision)
            _retained(scene, before, observation, raw, revision, geometry)
        _advance(scene, 1)
        assert runtime.integration_snapshot()[13].tick == 3


@pytest.mark.parametrize("mode", MODES)
def test_actual_offset_com_hinge_observation_preserves_three_moment_references(mode):
    _, _, runtime = _runtime(_hinge_slider_xml(), mode)
    with runtime:
        before = runtime.integration_snapshot()[11]
        packet = _mapping(runtime)
        runtime.step_area_pressure(HINGE_H, packet, expected_revision=runtime.checkpoint_revision)
        state = runtime.integration_snapshot()[11]
        counters = dict(runtime.counters)
        frame = runtime.area_pressure_snapshot()[11]
        _readonly(frame)
        assert runtime.counters == counters
        np.testing.assert_array_equal(frame["input_q"], before.q)
        np.testing.assert_array_equal(frame["input_v"], before.v)
        np.testing.assert_array_equal(frame["endpoint_q"], state.q)
        np.testing.assert_array_equal(frame["endpoint_v"], state.v)
        assert frame["input_time"] == before.time and frame["output_time"] == state.time
        pair = frame["pairs"][0]
        quad = pair["source_quadrature"]
        assert quad.identity == packet[11][0].identity and quad.stamp == packet[11][0].stamp
        np.testing.assert_array_equal(pair["dA"], quad.dA)
        assert np.sum(pair["normal_force"]) > 0
        dynamics = state.last_step.dynamics
        for side, body in zip(("a", "b"), pair["body_ids"], strict=True):
            force = pair["force_on_" + side]
            summed = np.sum(force, axis=0)
            np.testing.assert_array_equal(pair["force_sum_" + side], summed)
            for label, origin in (
                ("world", np.zeros(3)),
                ("body", dynamics.body_position[body]),
                ("com", dynamics.body_com[body]),
            ):
                wanted = np.sum(np.cross(pair["points"] - origin, force), axis=0)
                np.testing.assert_allclose(
                    pair["moment_" + label + "_" + side], wanted, rtol=0, atol=1e-12
                )
            np.testing.assert_allclose(
                pair["moment_world_" + side],
                pair["moment_com_" + side] + np.cross(dynamics.body_com[body], summed),
                rtol=0,
                atol=1e-12,
            )
        assert not np.array_equal(pair["moment_world_a"], pair["moment_body_a"])
        assert not np.array_equal(pair["moment_body_a"], pair["moment_com_a"])
        np.testing.assert_array_equal(pair["force_sum_a"] + pair["force_sum_b"], 0)
        np.testing.assert_allclose(
            pair["moment_world_a"] + pair["moment_world_b"], 0, rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("mode", MODES)
def test_same_model_with_reordered_area_pairs_rejects_runtime_capsule(mode, monkeypatch):
    runtime, scene = build_support(mode, H, DUAL, initial_q=-0.0025)
    with runtime:
        _advance(scene, 1)
        checkpoint = runtime.checkpoint()
        names = tuple(reversed(runtime._area_pressure_pairs))
        with MuJoCoTyxRuntime(
            runtime.compilation,
            mode=mode,
            world_keys=runtime.world_keys,
            area_pressure_pairs=names,
            constraint_capacity=64,
        ) as other:
            before = other.checkpoint().to_bytes()
            revision = other.checkpoint_revision

            def forbidden():
                pytest.fail("wrong area pair order reached device allocation")

            monkeypatch.setattr(other, "_build_device", forbidden)
            with pytest.raises(ValueError, match="binding"):
                other.restore(checkpoint, expected_revision=revision)
            assert other.checkpoint_revision == revision
            assert other.checkpoint().to_bytes() == before
