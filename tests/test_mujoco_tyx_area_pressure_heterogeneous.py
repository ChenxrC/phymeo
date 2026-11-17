"""Short real-step heterogeneous area routing and copy-owned observation tests."""

from contextlib import ExitStack
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure import PAIR, H, _hinge_slider_xml, _surface
from test_mujoco_tyx_area_pressure_lifecycle import _readonly, _same

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_observation import ManipulationSuccessReader
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (91, 7, 41, 11)
AREA_KEYS = (91, 41, 11)
AREA_FIELDS = ("area_meta", "area_ids", "area_data", "area_measure_words")
FREE_XML = f'''<mujoco model="heterogeneous-area-free">
  <compiler angle="radian" fusestatic="false"/>
  <option timestep="{H}" gravity="0 0 0" integrator="Euler"
    solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal"/>
  <worldbody>
    <body name="compliant" pos="0 0 .3">
      <joint name="floating" type="free" damping=".4"/>
      <inertial pos=".02 0 -.01" mass=".5" diaginertia=".02 .03 .04"/>
      <geom type="sphere" size=".06" contype="0" conaffinity="0"/>
    </body>
    <body name="rigid" pos="0 0 .12">
      <geom type="sphere" size=".06" contype="0" conaffinity="0"/>
    </body>
  </worldbody></mujoco>'''
NATIVE_XML = f'''<mujoco model="heterogeneous-native-contact">
  <option timestep="{H}" gravity="0 0 0" integrator="Euler"
    solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal"/>
  <default><geom condim="1" friction="0 0 0" solref=".02 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="ball" pos="0 0 .099"><joint type="slide" axis="0 0 1"/>
      <geom name="ball-geom" type="sphere" size=".1" mass=".7"/>
    </body>
  </worldbody></mujoco>'''


def _declarations():
    sources = {"hinge": _hinge_slider_xml(), "native": NATIVE_XML, "free": FREE_XML}
    capacities = {"hinge": 8, "native": 32, "free": 16}
    models = tuple(
        MuJoCoTyxModelSpec(
            name,
            compile_mujoco_xml(xml, require_asset=False),
            contact_capacity=4,
            constraint_capacity=capacities[name],
            area_pressure_pairs=() if name == "native" else (PAIR,),
        )
        for name, xml in sources.items()
    )
    initial = {
        91: MuJoCoInitialState(
            [0, 0, 0.3, 1, 0, 0, 0], [-0.03, 0.01, 0, 0.1, 0.02, -0.01], time=0.125
        ),
        7: MuJoCoInitialState([0], [-0.01], time=0.125),
        41: MuJoCoInitialState([0, 0], [-0.2, 0.1], time=0.125),
        11: MuJoCoInitialState([0, 0], [-0.1, 0.05], time=0.125),
    }
    names = {91: "free", 7: "native", 41: "hinge", 11: "hinge"}
    worlds = tuple(MuJoCoTyxWorldSpec(key, names[key], initial[key]) for key in KEYS)
    return models, worlds


def _make(stack, mode):
    models, worlds = _declarations()
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(models, worlds, mode=mode))
    workspace = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
    return batch, workspace, models, worlds


def _packet(owner, keys=AREA_KEYS, *, empty=(11,)):
    return {key: (_surface(owner, key, empty=key in empty),) for key in keys}


def _step(batch, *, empty=(11,)):
    result = batch.step_area_pressure(
        H, _packet(batch, empty=empty), expected_revision=batch.checkpoint_revision
    )
    assert result == dict.fromkeys(batch.world_keys, True)


def _refresh(batch, workspace):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    return frame, workspace.readback(frame)


def _exact_arrays(actual, expected):
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype and actual[name].shape == value.shape
        assert actual[name].tobytes() == value.tobytes(), name


def _assert_area(batch, frame, arrays):
    states = batch.integration_snapshot()
    _readonly(frame.area_catalogs)
    assert arrays["area_ids"].shape == (4, 16, 6)
    assert arrays["area_data"].shape == (4, 16, 15)
    assert arrays["area_measure_words"].shape == (4, 16, 2)
    assert arrays["area_meta"].dtype == arrays["area_ids"].dtype == np.int32
    assert arrays["area_data"].dtype == np.float32
    words = np.ascontiguousarray(arrays["area_measure_words"], dtype="<i4")
    measures = words.view("<f8").reshape(4, 16)
    for row, key in enumerate(KEYS):
        state, layout = states[key], batch.layout[key]
        for name, width in (("q", layout.nq), ("v", layout.nv)):
            np.testing.assert_array_equal(arrays[name][row, :width], getattr(state, name))
            np.testing.assert_array_equal(arrays[name][row, width:], 0)
            np.testing.assert_array_equal(
                arrays[name + "_mask"][row], np.arange(arrays[name].shape[1]) < width
            )
        np.testing.assert_array_equal(arrays["area_declared_mask"][row], [int(key in AREA_KEYS)])
        if key not in AREA_KEYS:
            assert frame.area_catalogs[key] == ()
            for name in AREA_FIELDS:
                np.testing.assert_array_equal(arrays[name][row], 0)
            assert arrays["entry_meta"][row, 3] > 0
            assert np.any(arrays["entry_contact_ids"][row, :, 0] == 1)
            continue
        receipt = state.last_step
        assert arrays["entry_meta"][row, 1] == receipt.tick - 1
        assert arrays["entry_meta"][row, 2] == state.tick
        for name, width in (("q", layout.nq), ("v", layout.nv)):
            np.testing.assert_array_equal(
                arrays["entry_" + name][row, :width], getattr(receipt, "input_" + name)
            )
            np.testing.assert_array_equal(arrays["entry_" + name][row, width:], 0)
        pair = receipt.raw_contact_fields["area_pressure"][0]
        n = len(pair["sample_ids"])
        positive = int(np.count_nonzero(pair["normal_force"] > 0))
        np.testing.assert_array_equal(arrays["area_meta"][row], [1, n, positive, 0])
        ids, data = arrays["area_ids"][row, :n], arrays["area_data"][row, :n]
        np.testing.assert_array_equal(ids[:, 0], 1)
        np.testing.assert_array_equal(ids[:, 1], 0)
        np.testing.assert_array_equal(ids[:, 2], np.arange(n))
        np.testing.assert_array_equal(ids[:, 3:5], np.tile(pair["body_ids"], (n, 1)))
        np.testing.assert_array_equal(ids[:, 5], 1)
        expected = np.column_stack(
            (
                pair["points"],
                pair["normals"],
                pair["stiffness"],
                pair["surrogate_gap"],
                pair["normal_force"],
            )
        )
        np.testing.assert_array_equal(data[:, :9], expected.astype(np.float32))
        np.testing.assert_allclose(data[:, 9:12], pair["force_on_a"], rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(
            data[:, 12:15], pair["torque_on_a_world_origin"], rtol=3e-5, atol=2e-6
        )
        assert measures[row, :n].tobytes() == np.asarray(pair["dA"], dtype="<f8").tobytes()
        assert len(frame.area_catalogs[key]) == 1
        for name in ("identity", "stamp", "sample_ids", "source_ids"):
            assert frame.area_catalogs[key][0][name] == pair[name]
        for name in ("area_ids", "area_data", "area_measure_words"):
            np.testing.assert_array_equal(arrays[name][row, n:], 0)


@pytest.mark.parametrize("mode", MODES)
def test_mixed_models_shared_bucket_padding_and_independent_leaf_steps(mode):
    with ExitStack() as stack:
        batch, ws, specs, worlds = _make(stack, mode)
        assert batch.world_keys == KEYS
        assert batch.layout[41].bucket == batch.layout[11].bucket
        assert len({row.bucket for row in batch.layout.values()}) == 3
        assert (batch.layout[91].nq, batch.layout[91].nv) == (7, 6)
        assert (batch.layout[41].nq, batch.layout[41].nv) == (2, 2)
        assert ws.binding["contact_tokens"]
        assert ws.binding["area_capacity"] == 16
        assert ws.binding["area_pair_names"] == {"hinge": (PAIR,), "free": (PAIR,), "native": ()}
        initial, arrays = _refresh(batch, ws)
        assert all(initial.area_catalogs[key] == () for key in KEYS)
        np.testing.assert_array_equal(arrays["area_meta"], 0)
        leaves = {}
        for world in worlds:
            spec = next(item for item in specs if item.key == world.model_key)
            leaves[world.world] = stack.enter_context(
                MuJoCoTyxRuntime(
                    spec.compilation,
                    mode=mode,
                    world_keys=(world.world,),
                    initial_states={world.world: world.initial_state},
                    contact_capacity=spec.contact_capacity,
                    constraint_capacity=spec.constraint_capacity,
                    area_pressure_pairs=spec.area_pressure_pairs,
                )
            )
        for empty in ((11,), ()):
            _step(batch, empty=empty)
            for key, leaf in leaves.items():
                if key in AREA_KEYS:
                    leaf.step_area_pressure(
                        H,
                        _packet(leaf, (key,), empty=empty),
                        expected_revision=leaf.checkpoint_revision,
                    )
                else:
                    leaf.step(H)
                _same(batch.integration_snapshot()[key], leaf.integration_snapshot()[key])
            frame, arrays = _refresh(batch, ws)
            _assert_area(batch, frame, arrays)


@pytest.mark.parametrize("mode", MODES)
def test_area_preflight_and_last_bucket_failure_are_atomic_and_retry(mode, monkeypatch):
    with ExitStack() as stack:
        batch, ws, _, _ = _make(stack, mode)
        _step(batch)
        frame, arrays = _refresh(batch, ws)
        capsule, revision, last = (
            batch.checkpoint().to_bytes(),
            batch.checkpoint_revision,
            batch.last_step,
        )
        good = _packet(batch)
        stale = replace(good[91][0], stamp=replace(good[91][0].stamp, step_index=0))
        triangle = good[41][0].triangles[0]
        excessive = replace(
            good[41][0],
            triangles=tuple(replace(triangle, triangle_id=f"capacity-{i}") for i in range(3)),
        )
        invalid = [
            {key: value for key, value in good.items() if key != 91},
            dict(good),
            {**good, 91: (stale,)},
            {**good, 41: (excessive,)},
        ]
        invalid[1][7] = ()  # A native world must never be silently accepted as an area mapping key.
        for mapping in invalid:
            with pytest.raises(ValueError):
                batch.step_area_pressure(H, mapping, expected_revision=revision)
            assert (
                batch.checkpoint().to_bytes() == capsule and batch.checkpoint_revision == revision
            )
            assert batch.last_step is last
            _exact_arrays(ws.readback(frame), arrays)
        with pytest.raises((ValueError, PhysicsStepFailure)):
            batch.step(H)
        with pytest.raises(ValueError):
            batch.step_area_pressure(H, good, expected_revision=revision - 1)
        assert batch.checkpoint().to_bytes() == capsule and batch.checkpoint_revision == revision
        _exact_arrays(ws.readback(frame), arrays)
        target = batch._buckets[-1]
        download, seen = target._device.download, []

        def fail():
            download()
            seen.append(True)
            raise RuntimeError("deliberate last heterogeneous area download")

        with monkeypatch.context() as patch:
            patch.setattr(target._device, "download", fail)
            with pytest.raises(PhysicsStepFailure) as caught:
                batch.step_area_pressure(H, good, expected_revision=revision)
        assert seen == [True] and caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
        assert batch.checkpoint().to_bytes() == capsule and batch.checkpoint_revision == revision
        assert batch.last_step is last
        _exact_arrays(ws.readback(frame), arrays)
        assert batch.step_area_pressure(H, good, expected_revision=revision) == dict.fromkeys(
            KEYS, True
        )
        frame, arrays = _refresh(batch, ws)
        _assert_area(batch, frame, arrays)


@pytest.mark.parametrize("mode", MODES)
def test_area_workspace_reset_restore_new_owners_and_stale_frames(mode):
    with ExitStack() as stack:
        batch, ws, _, _ = _make(stack, mode)
        _step(batch, empty=())
        saved_frame, saved = _refresh(batch, ws)
        checkpoint = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(batch.checkpoint().to_bytes())
        owners = tuple(batch._buckets)
        batch.reset((41,))
        assert any(new is not old for new, old in zip(batch._buckets, owners, strict=True))
        with pytest.raises(ValueError):
            ws.readback(saved_frame)
        with pytest.raises(ValueError):
            ws.refresh(expected_revision=saved_frame.revision)
        reset_frame, reset = _refresh(batch, ws)
        row = KEYS.index(41)
        np.testing.assert_array_equal(reset["area_meta"][row], 0)
        assert reset_frame.area_catalogs[41] == ()
        np.testing.assert_array_equal(reset["area_declared_mask"][row], [1])
        for key in (91, 11):
            slot = KEYS.index(key)
            for name in AREA_FIELDS:
                assert reset[name][slot].tobytes() == saved[name][slot].tobytes()
            assert reset_frame.area_catalogs[key] == saved_frame.area_catalogs[key]
        owners = tuple(batch._buckets)
        batch.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        assert all(new is not old for new, old in zip(batch._buckets, owners, strict=True))
        with pytest.raises(ValueError):
            ws.readback(reset_frame)
        restored_frame, restored = _refresh(batch, ws)
        _exact_arrays(restored, saved)
        assert restored_frame.area_catalogs == saved_frame.area_catalogs
        _step(batch, empty=())
        with pytest.raises(ValueError):
            ws.readback(restored_frame)
        frame, arrays = _refresh(batch, ws)
        _assert_area(batch, frame, arrays)


@pytest.mark.parametrize("mode", MODES)
def test_refresh_is_device_sourced_and_last_bucket_failure_keeps_frame(mode, monkeypatch):
    with ExitStack() as stack:
        batch, ws, _, _ = _make(stack, mode)
        _step(batch, empty=())
        frame, expected = _refresh(batch, ws)

        def forbidden(*args, **kwargs):
            raise AssertionError("area refresh used a numeric HOST source")

        with monkeypatch.context() as patch:
            for name in ("snapshot", "integration_snapshot"):
                patch.setattr(batch, name, forbidden)
            for owner in batch._buckets:
                metadata_only = {}
                for key, receipt in owner._receipts.items():
                    pairs = receipt.raw_contact_fields.get("area_pressure", ())
                    metadata_only[key] = SimpleNamespace(
                        raw_contact_fields=MappingProxyType(
                            {
                                "area_pressure": tuple(
                                    MappingProxyType(
                                        {
                                            name: pair[name]
                                            for name in (
                                                "identity",
                                                "stamp",
                                                "sample_ids",
                                                "source_ids",
                                            )
                                        }
                                    )
                                    for pair in pairs
                                )
                            }
                        )
                    )
                patch.setattr(owner, "_receipts", metadata_only)
                for name in ("snapshot", "integration_snapshot", "area_pressure_snapshot"):
                    patch.setattr(owner, name, forbidden)
                patch.setattr(owner._device, "_get", forbidden)
                patch.setattr(owner._device, "download", forbidden)
                for name in ("mj_forward", "mj_step", "mj_kinematics", "mj_comPos", "mj_comVel"):
                    patch.setattr(owner._mj, name, forbidden)
                if mode == "cuda":
                    for source in (
                        *owner._device.q,
                        *owner._device.v,
                        *owner._device.time,
                        *(value for banks in owner._device.entry.values() for value in banks),
                    ):
                        patch.setattr(source, "to_numpy", forbidden)
            fresh = ws.refresh(expected_revision=batch.checkpoint_revision)
        _exact_arrays(ws.readback(fresh), expected)
        assert fresh.area_catalogs == frame.area_catalogs
        original, calls = ws._area_observer.refresh_bucket, []

        def fail(index, owner, destination, options):
            result = original(index, owner, destination, options)
            calls.append(index)
            if index == len(batch._buckets) - 1:
                raise RuntimeError("deliberate final area refresh")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(ws._area_observer, "refresh_bucket", fail)
            with pytest.raises(RuntimeError, match="final area refresh"):
                ws.refresh(expected_revision=batch.checkpoint_revision)
        assert calls == list(range(len(batch._buckets)))
        _exact_arrays(ws.readback(fresh), expected)
        retry, data = _refresh(batch, ws)
        _exact_arrays(data, expected)
        assert retry.area_catalogs == fresh.area_catalogs


@pytest.mark.parametrize("mode", MODES)
def test_same_source_different_area_declarations_do_not_merge_or_restore(mode):
    source = compile_mujoco_xml(_hinge_slider_xml(), require_asset=False)
    worlds = (MuJoCoTyxWorldSpec(41, "first"), MuJoCoTyxWorldSpec(11, "second"))
    first = MuJoCoTyxModelSpec("first", source, constraint_capacity=8, area_pressure_pairs=(PAIR,))
    reverse = MuJoCoTyxModelSpec(
        "second", source, constraint_capacity=8, area_pressure_pairs=(PAIR[::-1],)
    )
    same = replace(reverse, area_pressure_pairs=(PAIR,))
    with (
        MuJoCoTyxHeterogeneousBatch((first, reverse), worlds, mode=mode) as distinct,
        MuJoCoTyxHeterogeneousBatch((first, same), worlds, mode=mode) as shared,
    ):
        assert distinct.layout[41].bucket != distinct.layout[11].bucket
        assert shared.layout[41].bucket == shared.layout[11].bucket
        with distinct.prepare_device_workspace(
            include_contacts=True, include_area_pressure=False
        ) as native_only:
            with pytest.raises(
                ValueError, match="area contact needs an explicit body-pair success adapter"
            ):
                ManipulationSuccessReader(native_only, {})
        a, b = distinct.checkpoint(), shared.checkpoint()
        for destination, foreign, original in ((distinct, b, a), (shared, a, b)):
            revision = destination.checkpoint_revision
            with pytest.raises(ValueError):
                destination.restore(foreign, expected_revision=revision)
            assert destination.checkpoint_revision == revision
            assert destination.checkpoint().to_bytes() == original.to_bytes()


@pytest.mark.cuda
def test_cuda_area_torch_copies_survive_refresh_reset_and_caller_mutation():
    import torch

    assert torch.cuda.is_available()
    with ExitStack() as stack:
        batch, ws, _, _ = _make(stack, "cuda")
        _step(batch, empty=())
        frame, reference = _refresh(batch, ws)
        first, second = ws.to_torch(frame), ws.to_torch(frame)
        wanted = {name: value.clone() for name, value in second.items()}
        for name, value in second.items():
            assert value.device.type == "cuda"
            assert value.data_ptr() != first[name].data_ptr()
            np.testing.assert_array_equal(value.cpu().numpy(), reference[name])
        assert second["area_data"].dtype == torch.float32
        assert second["area_measure_words"].dtype == second["area_ids"].dtype == torch.int32
        first["area_data"].fill_(123)
        _step(batch, empty=())
        _refresh(batch, ws)
        batch.reset((41,))
        _refresh(batch, ws)
        torch.cuda.synchronize()
        for name, value in second.items():
            assert torch.equal(value, wanted[name])


@pytest.mark.parametrize("mode", MODES)
def test_optional_area_on_native_only_and_corrupt_body_identity(mode, monkeypatch):
    from test_mujoco_tyx_area_pressure_device import _array

    source = compile_mujoco_xml(NATIVE_XML, require_asset=False)
    with (
        MuJoCoTyxHeterogeneousBatch(
            (MuJoCoTyxModelSpec("native", source),),
            (MuJoCoTyxWorldSpec(7, "native"),),
            mode=mode,
        ) as native,
        native.prepare_device_workspace(include_area_pressure=True) as native_ws,
    ):
        native.step(H)
        frame, values = _refresh(native, native_ws)
        assert native_ws.binding["area_capacity"] == 1
        assert frame.area_catalogs[7] == ()
        for name in (*AREA_FIELDS, "area_declared_mask"):
            assert not np.any(values[name])

    with ExitStack() as stack:
        batch, ws, _, _ = _make(stack, mode)
        _step(batch, empty=())
        frame, expected = _refresh(batch, ws)
        owner = batch._buckets[batch.layout[91].bucket]
        device = owner._device
        original = device.entry["area_ids"][device.active]
        ids = original.copy() if mode == "interpret" else original.to_numpy()
        ids[0, 0, 4] = 0  # Valid body index, wrong declared body for this pair.
        replacement = dict(device.entry)
        banks = list(replacement["area_ids"])
        banks[device.active] = _array(stack, mode, ids)
        replacement["area_ids"] = tuple(banks)
        with monkeypatch.context() as patch:
            patch.setattr(device, "entry", replacement)
            with pytest.raises(ValueError, match="invalid accepted-entry"):
                ws.refresh(expected_revision=batch.checkpoint_revision)
        _exact_arrays(ws.readback(frame), expected)
        retry, values = _refresh(batch, ws)
        _exact_arrays(values, expected)
        assert retry.area_catalogs == frame.area_catalogs
