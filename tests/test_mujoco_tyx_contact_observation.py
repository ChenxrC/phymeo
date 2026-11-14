"""B2c public accepted-entry observations; author-only, UNEXECUTED.

Native FK and stored receipts are independent diagnostic oracles, never output
producers. These tests do not claim manipulation success or expanded physics.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest
import test_mujoco_tyx_body_observation as bodies
import test_mujoco_tyx_entry_record as records
import test_mujoco_tyx_heterogeneous_io as io
from test_mujoco_tyx_activation import save

from phymeo.backends import mujoco_entry_checkpoint as entry_checkpoint
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

MODES, H = records.MODES, records.H
ENTRY_BODIES = tuple("entry_" + name for name in bodies.BODY_FIELDS)
ENTRY_FIELDS = (
    records.FIELDS
    | set(ENTRY_BODIES)
    | {
        "entry_body_mask",
        "entry_contact_wrench_world",
        "entry_contact_body_wrench",
    }
)


def forbidden(*args, **kwargs):
    raise AssertionError("contact observation used HOST receipts/state/readback as a producer")


class NoHostState(Mapping):
    __getitem__ = forbidden
    __iter__ = forbidden
    __len__ = forbidden
    __getattr__ = forbidden


def refresh(mj, batch, ws, monkeypatch):
    with monkeypatch.context() as guard:
        guard.setattr(entry_checkpoint, "_accepted_entry", forbidden)
        guard.setattr(batch, "_last_step", NoHostState())
        for owner in batch._buckets:
            guard.setattr(owner, "_data", NoHostState())
            guard.setattr(owner, "_receipts", NoHostState())
            guard.setattr(owner._device, "_get", forbidden)
        return bodies.refresh_without_host_physics(mj, batch, ws, monkeypatch)


def expectations(mj, batch, models):
    raw = [records.record(owner) for owner in batch._buckets]
    expected = {}
    for key in batch.world_keys:
        layout = batch.layout[key]
        values, index = raw[layout.bucket], layout.row
        if not values["entry_meta"][index, 0]:
            expected[key] = None
            continue
        stamp = values["entry_clock"][index]
        state = SimpleNamespace(
            q=values["entry_q"][index, : layout.nq],
            v=values["entry_v"][index, : layout.nv],
            act=values["entry_act"][index, : layout.na],
            time=float(stamp[0]) - float(stamp[1]),
        )
        expected[key] = bodies.native_bodies(mj, models[layout.model_key], state)
    return raw, expected


def wrenches(ids, data, position):
    """Reference forces act on geom2; both moments are about BODY ORIGINS."""
    frame, point = data[4:13].reshape(3, 3), data[1:4]
    force = frame.T @ data[13:16]
    couple = frame.T @ data[16:19]
    b1, b2 = map(int, ids[4:6])
    result = np.empty((2, 6))
    result[0, :3], result[1, :3] = -force, force
    result[0, 3:] = -couple + np.cross(point - position[b1], -force)
    result[1, 3:] = couple + np.cross(point - position[b2], force)
    return np.concatenate((force, couple)), result


def check_values(batch, ws, frame, values, raw, expected):
    io.compare_frame(batch, ws, frame)
    assert ws.binding["contact_tokens"] is True
    assert ws.binding["entry_phase"] == "accepted-step-input"
    assert ws.binding["phase"] == "accepted-integration-endpoint"
    assert ws.binding["body_velocity"] == "world-linear-and-angular-at-body-origin"
    assert ENTRY_FIELDS <= set(values)
    n = len(batch.world_keys)
    nb = max(row.nbody for row in batch.layout.values())
    nc = max(1, *(spec.contact_capacity for spec in batch._models))
    assert ws.binding["entry_contact_capacity"] == nc
    assert ws.binding["entry_body_kinematics"] is True
    shapes = {
        "entry_q": (n, max(1, max(row.nq for row in batch.layout.values()))),
        "entry_v": (n, max(1, max(row.nv for row in batch.layout.values()))),
        "entry_act": (n, max(1, max(row.na for row in batch.layout.values()))),
        "entry_clock": (n, 4),
        "entry_meta": (n, 6),
        "entry_contact_ids": (n, nc, 8),
        "entry_contact_data": (n, nc, 20),
        "entry_body_mask": (n, nb),
        "entry_contact_wrench_world": (n, nc, 6),
        "entry_contact_body_wrench": (n, nc, 2, 6),
    }
    for name in ENTRY_BODIES:
        shapes[name] = (n, nb, 3, 3) if name.endswith("rotation") else (n, nb, 3)
    for name, shape in shapes.items():
        array = values[name]
        integer = name in ("entry_meta", "entry_contact_ids", "entry_body_mask")
        assert array.shape == shape and array.dtype == np.dtype(np.int32 if integer else np.float32)
        assert np.isfinite(array).all() and not array.flags.writeable
    for index, key in enumerate(batch.world_keys):
        layout = batch.layout[key]
        source, row = raw[layout.bucket], layout.row
        for name, width in (("q", layout.nq), ("v", layout.nv), ("act", layout.na)):
            field = "entry_" + name
            assert io.exact(values[field][index, :width]) == io.exact(source[field][row, :width])
            np.testing.assert_array_equal(values[field][index, width:], 0)
        for name in ("entry_clock", "entry_meta"):
            assert io.exact(values[name][index]) == io.exact(source[name][row])
        for name in ("entry_contact_ids", "entry_contact_data"):
            width = source[name].shape[1]
            assert io.exact(values[name][index, :width]) == io.exact(source[name][row])
            np.testing.assert_array_equal(values[name][index, width:], 0)
        valid, _, output_tick, count, _, overflow = map(int, values["entry_meta"][index])
        assert output_tick == frame.ticks[index] and overflow == 0
        mask = np.arange(nb) < (layout.nbody if valid else 0)
        np.testing.assert_array_equal(values["entry_body_mask"][index], mask)
        for name in ENTRY_BODIES:
            if valid:
                field = name.removeprefix("entry_")
                np.testing.assert_allclose(
                    values[name][index, : layout.nbody],
                    expected[key][field],
                    **(bodies.TWIST_TOL if "velocity" in name else bodies.POSE_TOL),
                    err_msg=f"{key}:{name}",
                )
            np.testing.assert_array_equal(values[name][index, layout.nbody if valid else 0 :], 0)
        for ci in range(count):
            ids, data = source["entry_contact_ids"][row, ci], source["entry_contact_data"][row, ci]
            world, pair = wrenches(ids, data.astype(np.float64), expected[key]["body_position"])
            np.testing.assert_allclose(
                values["entry_contact_wrench_world"][index, ci],
                world,
                **records.normal.FORCE_TOL,
            )
            np.testing.assert_allclose(
                values["entry_contact_body_wrench"][index, ci],
                pair,
                **records.normal.FORCE_TOL,
            )
            # Tight algebra check against exported origins isolates transform/sign
            # errors from the separately checked native FK accuracy.
            world, pair = wrenches(
                ids, data.astype(np.float64), values["entry_body_position"][index]
            )
            np.testing.assert_allclose(
                values["entry_contact_wrench_world"][index, ci],
                world,
                rtol=3e-5,
                atol=4e-5,
            )
            np.testing.assert_allclose(
                values["entry_contact_body_wrench"][index, ci],
                pair,
                rtol=3e-5,
                atol=4e-5,
            )
            np.testing.assert_allclose(pair[0, :3] + pair[1, :3], 0, rtol=0, atol=1e-6)
        for name in ("entry_contact_wrench_world", "entry_contact_body_wrench"):
            np.testing.assert_array_equal(values[name][index, count:], 0)


def observe(mj, batch, ws, models, monkeypatch, history, path, label):
    raw, expected = expectations(mj, batch, models)
    states = batch.integration_snapshot()
    physics = bodies.physical_bytes(batch)
    source = tuple(spec.compilation.source.primary_xml for spec in batch._models)
    payload = dict(
        label=label,
        source_hex=tuple(s.hex() for s in source),
        input_entry=raw,
        native_entry=expected,
        states=states,
    )
    history.append(payload)
    save(path, history)  # Preserve complete inputs even if the first refresh fails.
    try:
        frame = refresh(mj, batch, ws, monkeypatch)
    except BaseException as error:
        payload["refresh_error"] = repr(error)
        save(path, history)
        raise
    values = ws.readback(frame)
    payload.update(frame=frame, values=values, receipt=batch.last_step)
    save(path, history)
    assert bodies.physical_bytes(batch) == physics
    assert io.exact(batch.integration_snapshot()) == io.exact(states)
    assert tuple(spec.compilation.source.primary_xml for spec in batch._models) == source
    check_values(batch, ws, frame, values, raw, expected)
    if ws.binding["endpoint_body_kinematics"]:
        endpoint = {
            key: bodies.native_bodies(mj, models[batch.layout[key].model_key], states[key])
            for key in batch.world_keys
        }
        bodies.check_bodies(batch, ws, frame, endpoint)
    else:
        assert not set(bodies.BODY_FIELDS).intersection(values)
    return frame, values


def entry_only(values):
    return {name: values[name] for name in ENTRY_FIELDS}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reverse", (False, True))
def test_w8_public_entry_routes_w1_order_and_input_phase(mode, reverse, tmp_path, monkeypatch):
    mj = records.normal._deps()
    keys = records.KEYS[::-1] if reverse else records.KEYS
    # Keep model/bucket order fixed while independently permuting public worlds.
    specs, declared, models, _ = records.declarations(mj, records.KEYS)
    by_key = {world.world: world for world in declared}
    worlds = tuple(by_key[key] for key in keys)
    history, path = [], tmp_path / "contact-w8.json.gz"
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        ws = stack.enter_context(
            batch.prepare_device_workspace(include_contacts=True, include_bodies=reverse)
        )
        assert ws.binding["endpoint_body_kinematics"] is reverse
        if reverse:
            assert any(batch.layout[key].bucket != index for index, key in enumerate(keys))
        singles = {}
        for world in worlds:
            spec = next(spec for spec in specs if spec.key == world.model_key)
            single = stack.enter_context(MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode=mode))
            single_ws = stack.enter_context(single.prepare_device_workspace(include_contacts=True))
            singles[world.world] = single, single_ws
        model = models["floating"]
        free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        assert len(free) == 2 and np.all(model.jnt_qposadr[free] > 0)
        for tick in range(3):
            frame, values = observe(mj, batch, ws, models, monkeypatch, history, path, f"w8-{tick}")
            assert frame.ticks == (tick,) * 8
            for index, key in enumerate(keys):
                single, single_ws = singles[key]
                _, one = observe(
                    mj, single, single_ws, models, monkeypatch, history, path, f"w1-{key}-{tick}"
                )
                row = batch.layout[key]
                for name in ENTRY_FIELDS:
                    width = {"entry_q": row.nq, "entry_v": row.nv, "entry_act": max(1, row.na)}.get(
                        name
                    )
                    if name in ENTRY_BODIES or name == "entry_body_mask":
                        width = row.nbody
                    if name.startswith("entry_contact_"):
                        width = one[name].shape[1]
                    actual = values[name][index] if width is None else values[name][index, :width]
                    assert io.exact(actual) == io.exact(one[name][0])
            if tick:
                index = keys.index(91)
                count = batch.layout[91].nbody
                endpoint = bodies.native_bodies(mj, model, batch.integration_snapshot()[91])
                assert (
                    np.max(
                        np.abs(
                            values["entry_body_position"][index, :count] - endpoint["body_position"]
                        )
                    )
                    > 1e-6
                )
                marker = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "fixed-marker")
                np.testing.assert_allclose(
                    values["entry_body_position"][index, marker], [-2, 1, 0.4], **bodies.POSE_TOL
                )
                np.testing.assert_array_equal(
                    values["entry_body_linear_velocity"][index, marker], 0
                )
                np.testing.assert_array_equal(
                    values["entry_body_angular_velocity"][index, marker], 0
                )
                index = keys.index(2)
                ids = values["entry_contact_ids"][index, 0]
                second = int(ids[5])
                data = values["entry_contact_data"][index, 0].astype(np.float64)
                position = values["entry_body_position"][index]
                world, pair = wrenches(ids, data, position)
                rotation = values["entry_body_rotation"][index, second]
                com = position[second] + rotation @ models["multi-free"].body_ipos[second]
                wrong_com_moment = world[3:] + np.cross(data[1:4] - com, world[:3])
                assert np.linalg.norm(pair[1, 3:] - wrong_com_moment) > 1e-5
            if tick < 2:
                packets = {key: records.command(models[records.KINDS[key]], tick) for key in keys}
                batch.set_controls(packets)
                assert batch.step(H) == dict.fromkeys(keys, True)
                for key, (single, _) in singles.items():
                    single.set_controls({key: packets[key]})
                    assert single.step(H) == {key: True}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dimension", (4, 6))
def test_public_pure_couple_slot_mapping_and_body_origin_reactions(
    mode, dimension, tmp_path, monkeypatch
):
    mj = records.normal._deps()
    xml = records.mixed.free_xml(dim=dimension, offset=False, two=True)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    model = compiled.fresh_native_model()
    q = model.qpos0.copy()
    q[3:7] = (np.cos(0.01), 0, np.sin(0.01), 0)
    # The accepted B2a fixture: v=-omega_world x r cancels tangential
    # translation at the actual sphere-plane midpoint, retaining pure couple.
    sine, cosine = np.sin(0.02), np.cos(0.02)
    if dimension == 4:
        velocity = [0, -0.28 * cosine, -0.005, -2 * sine, 0, 2 * cosine]
    else:
        rz = -(0.199 + 0.14 * sine) / 2
        velocity = [0, 1.5 * rz, -0.005, 1.5 * cosine, 0, 1.5 * sine]
    spec = MuJoCoTyxModelSpec("couple", compiled)
    world = MuJoCoTyxWorldSpec(41, "couple", MuJoCoInitialState(q, velocity, time=0.125))
    history, path = [], tmp_path / "contact-couple.json.gz"
    with (
        MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode=mode) as batch,
        batch.prepare_device_workspace(include_contacts=True) as ws,
    ):
        for tick in range(2):
            before, packet = batch.integration_snapshot()[41], records.command(model, tick)
            oracle = records.native_contacts(mj, model, before, packet)
            batch.set_controls({41: packet})
            assert batch.step(H) == {41: True}
            _, values = observe(
                mj, batch, ws, {"couple": model}, monkeypatch, history, path, f"couple-{tick}"
            )
            receipt = batch.last_step.receipts[41]
            history[-1]["native_contacts"] = oracle
            save(path, history)
            records.assert_native(receipt, oracle)
            assert len(receipt.contacts) == 1
            assert int(receipt.raw_contact_fields["contact_map"][0, 0]) == 1
            ids = values["entry_contact_ids"][0, 0]
            assert ids[7] == dimension and ids[3] == mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "b")
            pair = values["entry_contact_body_wrench"][0, 0]
            position = values["entry_body_position"][0]
            for side in (0, 1):
                assert np.linalg.norm(pair[side, :3]) > 1e-3
                assert np.linalg.norm(pair[side, 3:]) > 1e-6
            moment_at_world_origin = sum(
                (
                    pair[side, 3:] + np.cross(position[int(ids[4 + side])], pair[side, :3])
                    for side in (0, 1)
                ),
                np.zeros(3),
            )
            np.testing.assert_allclose(moment_at_world_origin, 0, rtol=0, atol=4e-5)
            if tick == 0:
                components = slice(3, 4) if dimension == 4 else slice(4, 6)
                assert np.linalg.norm(receipt.contacts[0].wrench_contact_frame[components]) > 1e-6
                assert np.linalg.norm(oracle[0]["wrench_contact_frame"][components]) > 1e-6
                assert np.linalg.norm(values["entry_contact_wrench_world"][0, 0, 3:]) > 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_invalid_empty_inactive_activation_controls_and_owner_rebinding(
    mode, tmp_path, monkeypatch
):
    from phymeo.backends.mujoco_checkpoint import _make_checkpoint
    from phymeo.backends.mujoco_heterogeneous_checkpoint import _capture

    mj = records.normal._deps()
    normal_source = records.normal._slider_pair(z=".11171875", inactive=True)
    specs = (
        MuJoCoTyxModelSpec("normal", compile_mujoco_xml(normal_source, require_asset=False)),
        MuJoCoTyxModelSpec(
            "activation",
            compile_mujoco_xml(
                records.composition.constrained_source(),
                require_asset=False,
            ),
            contact_capacity=0,
        ),
    )
    models = {spec.key: spec.compilation.fresh_native_model() for spec in specs}
    start = records.forest.initial(mj, models["activation"])
    q = start.q.copy()
    q[0] = 0.0195
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, "normal", MuJoCoInitialState([offset], [0], time=0.125))
        for key, offset in zip((41, 7, 305), (-0.02, 0, 0.1), strict=True)
    ) + (
        MuJoCoTyxWorldSpec(
            91, "activation", MuJoCoInitialState(q, start.v, act=[0.07], time=0.125)
        ),
    )
    history, path = [], tmp_path / "contact-lifecycle.json.gz"
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        ws = stack.enter_context(batch.prepare_device_workspace(include_contacts=True))
        plain = stack.enter_context(batch.prepare_device_workspace())
        assert plain.binding["contact_tokens"] is False
        baseline_physics = bodies.physical_bytes(batch)
        plain_values = plain.readback(
            bodies.refresh_without_host_physics(mj, batch, plain, monkeypatch)
        )
        assert not any(name.startswith("entry_") for name in plain_values)
        assert bodies.physical_bytes(batch) == baseline_physics
        initial_frame, initial = observe(
            mj, batch, ws, models, monkeypatch, history, path, "initial-invalid"
        )
        assert len(initial_frame.world_boundaries) == len(batch.world_keys)
        assert all(token[2] == "initial" for token in initial_frame.world_boundaries)
        assert len({token[0] for token in initial_frame.world_boundaries}) == 1
        for name in ENTRY_FIELDS:
            np.testing.assert_array_equal(initial[name], 0)
        assert ws.binding["endpoint_body_kinematics"] is False
        packets = {world.world: records.command(models[world.model_key]) for world in worlds}
        batch.set_controls(packets)
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        old, accepted = observe(mj, batch, ws, models, monkeypatch, history, path, "accepted")
        assert old.world_boundaries == initial_frame.world_boundaries
        np.testing.assert_array_equal(
            accepted["entry_meta"][:, [0, 3, 4]],
            [
                [1, 2, 2],
                [1, 2, 0],
                [1, 0, 0],
                [1, 0, 0],
            ],
        )
        for name in ("entry_contact_wrench_world", "entry_contact_body_wrench"):
            np.testing.assert_array_equal(accepted[name][1:], 0)
        assert accepted["entry_body_mask"][
            1
        ].any()  # Detected inactive contacts still have a real entry pose.
        assert accepted["entry_body_mask"][
            2
        ].any()  # A successful empty set is not an invalid observation.
        np.testing.assert_array_equal(accepted["entry_contact_ids"][3], 0)
        np.testing.assert_array_equal(accepted["entry_contact_data"][3], 0)
        packets = {world.world: records.command(models[world.model_key], 1) for world in worlds}
        batch.set_controls(packets)
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.readback(old)
        held, held_values = observe(
            mj, batch, ws, models, monkeypatch, history, path, "held-controls"
        )
        assert held.revision > old.revision and held.epoch == old.epoch + 1
        assert held.ticks == old.ticks
        assert held.world_boundaries == old.world_boundaries
        assert io.exact(entry_only(held_values)) == io.exact(entry_only(accepted))
        capsule = batch.checkpoint()
        (tmp_path / "entry-observation.checkpoint").write_bytes(capsule.to_bytes())
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        successor_frame, successor = observe(
            mj, batch, ws, models, monkeypatch, history, path, "successor"
        )
        previous_owners = tuple(batch._buckets)
        batch.reset_states(
            {7: MuJoCoInitialState([0.1], [0])}, expected_revision=batch.checkpoint_revision
        )
        assert any(a is not b for a, b in zip(previous_owners, batch._buckets, strict=True))
        reset_frame, reset = observe(
            mj, batch, ws, models, monkeypatch, history, path, "partial-reset"
        )
        for index, key in enumerate(batch.world_keys):
            previous, current = (
                successor_frame.world_boundaries[index],
                reset_frame.world_boundaries[index],
            )
            if key == 7:
                assert current[0] == previous[0] and current[1] > previous[1]
                assert current[2] == "reset-states"
            else:
                assert current == previous
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.readback(successor_frame)
        for name in ENTRY_FIELDS:
            np.testing.assert_array_equal(reset[name][1], 0)
            for index in (0, 2, 3):
                assert io.exact(reset[name][index]) == io.exact(successor[name][index])
        batch.restore(capsule, expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.readback(reset_frame)
        restored_frame, restored = observe(
            mj, batch, ws, models, monkeypatch, history, path, "restored-owner"
        )
        for previous, current in zip(
            reset_frame.world_boundaries, restored_frame.world_boundaries, strict=True
        ):
            assert current[0] == previous[0] and current[1] > previous[1]
            assert current[2] == "restore"
        assert io.exact(restored) == io.exact(held_values)
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        replay_frame, replay = observe(
            mj, batch, ws, models, monkeypatch, history, path, "replayed"
        )
        assert replay_frame.world_boundaries == restored_frame.world_boundaries
        assert io.exact(replay) == io.exact(successor)
        # Synthetic old-format capsule, consumed through the PUBLIC restore.
        # It retains endpoint/ticks but has no historical entry record.
        cp = batch.checkpoint()
        leaves, publication = cp._read(cp.binding)
        legacy_leaves = []
        for leaf in leaves:
            binding, state = dict(leaf.binding), dict(leaf._read(leaf.binding))
            binding.pop("accepted_entry")
            state.pop("accepted_entry")
            legacy_leaves.append(_make_checkpoint(binding, leaf.source_revision, state))
        legacy = _capture(cp.binding, cp.source_revision, tuple(legacy_leaves), publication)
        (tmp_path / "entry-observation-legacy.checkpoint").write_bytes(legacy.to_bytes())
        batch.restore(legacy, expected_revision=batch.checkpoint_revision)
        legacy_frame, invalid = observe(
            mj, batch, ws, models, monkeypatch, history, path, "legacy-invalid"
        )
        assert all(tick > 0 for tick in legacy_frame.ticks)
        for previous, current in zip(
            replay_frame.world_boundaries, legacy_frame.world_boundaries, strict=True
        ):
            assert current[0] == previous[0] and current[1] > previous[1]
            assert current[2] == "restore"
        for name in ENTRY_FIELDS:
            expected = np.zeros_like(invalid[name])
            if name == "entry_meta":
                expected[:, 2] = legacy_frame.ticks
            assert io.exact(invalid[name]) == io.exact(expected)
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        _, valid = observe(mj, batch, ws, models, monkeypatch, history, path, "legacy-next-step")
        np.testing.assert_array_equal(valid["entry_meta"][:, 0], 1)
        np.testing.assert_array_equal(valid["entry_meta"][:, 1], legacy_frame.ticks)
        # The no-contact activation bucket alone has the required C=1 sentinel.
        zero = stack.enter_context(
            MuJoCoTyxHeterogeneousBatch((specs[1],), (worlds[-1],), mode=mode)
        )
        zero_ws = stack.enter_context(zero.prepare_device_workspace(include_contacts=True))
        zero.set_controls({91: records.command(models["activation"])})
        assert zero.step(H) == {91: True}
        _, zero_values = observe(
            mj, zero, zero_ws, models, monkeypatch, history, path, "zero-capacity-sentinel"
        )
        assert zero_values["entry_contact_ids"].shape == (1, 1, 8)
        assert zero_values["entry_contact_data"].shape == (1, 1, 20)
        assert zero_values["entry_meta"][0, 0] == 1 and zero_values["entry_meta"][0, 3] == 0


def write_device(device, target, values):
    with device.tx.use_runtime(device.runtime):
        device._put(target, values)
        device.runtime.sync()


@pytest.mark.parametrize("mode", MODES)
def test_late_bucket_and_nonfinite_entry_rejection_keep_frame_then_retry(
    mode, tmp_path, monkeypatch
):
    mj = records.normal._deps()
    specs, worlds, models, _ = records.declarations(mj, (41, 11, 2))
    history, path = [], tmp_path / "contact-rejection.json.gz"
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch,
        batch.prepare_device_workspace(include_contacts=True, include_bodies=True) as ws,
    ):
        batch.set_controls(
            {world.world: records.command(models[world.model_key]) for world in worlds}
        )
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        old, values = observe(mj, batch, ws, models, monkeypatch, history, path, "accepted-frame")
        revision, receipt = batch.checkpoint_revision, batch.last_step
        states, physical = io.exact(batch.integration_snapshot()), bodies.physical_bytes(batch)
        original, visited = type(ws._entry_observer).refresh_bucket, []

        def fail_last(observer, index, owner, destination, options):
            original(observer, index, owner, destination, options)
            visited.append(index)
            if index == len(batch._buckets) - 1:
                raise RuntimeError("authored final entry observation rejection")

        with monkeypatch.context() as fault:
            fault.setattr(type(ws._entry_observer), "refresh_bucket", fail_last)
            with pytest.raises(RuntimeError, match="final entry observation"):
                refresh(mj, batch, ws, monkeypatch)
        history.append(
            dict(label="late-bucket-rejected", visited=tuple(visited), retained=ws.readback(old))
        )
        save(path, history)
        assert visited == list(range(len(batch._buckets)))
        assert io.exact(ws.readback(old)) == io.exact(values)
        assert bodies.physical_bytes(batch) == physical
        assert batch.checkpoint_revision == revision and batch.last_step is receipt
        assert io.exact(batch.integration_snapshot()) == states
        target = batch._buckets[-1]._device
        accepted_record = records.record(batch._buckets[-1])
        assert accepted_record["entry_meta"][0, 0] == 1
        assert accepted_record["entry_meta"][0, 3] > 0
        assert accepted_record["entry_contact_ids"][0, 0, 1] == 1
        saved = accepted_record["entry_contact_data"]
        bad = saved.copy()
        bad[0, 0, 13] = np.nan  # Explicit fault in a valid accepted normal-force token.
        accepted_buffer = target.entry["entry_contact_data"][target.active]
        write_device(target, accepted_buffer, bad)
        injected = bodies.physical_bytes(batch)
        history.append(
            dict(label="injected-valid-force-NaN", input_entry=bad, retained_values=values)
        )
        save(path, history)
        try:
            with pytest.raises(ValueError, match="entry|contact|nonfinite"):
                refresh(mj, batch, ws, monkeypatch)
            assert io.exact(ws.readback(old)) == io.exact(values)
            assert bodies.physical_bytes(batch) == injected
            assert batch.checkpoint_revision == revision and batch.last_step is receipt
            assert io.exact(batch.integration_snapshot()) == states
        finally:
            write_device(target, accepted_buffer, saved)
        assert bodies.physical_bytes(batch) == physical
        retry, actual = observe(mj, batch, ws, models, monkeypatch, history, path, "retry")
        assert retry.epoch == old.epoch + 1 and retry.revision == old.revision
        assert retry.world_boundaries == old.world_boundaries
        assert io.exact(actual) == io.exact(values)
        with pytest.raises(ValueError, match="superseded"):
            ws.readback(old)


@pytest.mark.cuda
def test_cuda_public_contact_exports_are_independent_d2d_copies_after_close(tmp_path, monkeypatch):
    import torch

    mj = records.normal._deps()
    specs, worlds, models, _ = records.declarations(mj, (41, 2))
    history, path = [], tmp_path / "contact-torch.json.gz"
    with (
        MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch,
        batch.prepare_device_workspace(include_contacts=True) as ws,
    ):
        batch.set_controls(
            {world.world: records.command(models[world.model_key]) for world in worlds}
        )
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        old, values = observe(mj, batch, ws, models, monkeypatch, history, path, "torch-source")
        physical = bodies.physical_bytes(batch)
        with monkeypatch.context() as guard:
            guard.setattr(type(ws._banks[ws._active]["entry_q"]), "to_numpy", forbidden)
            first = ws.to_torch(old)
            second = ws.to_torch(old)
        for name in ENTRY_FIELDS:
            assert first[name].is_cuda and second[name].is_cuda
            assert first[name].data_ptr() != second[name].data_ptr()
            assert io.exact(first[name].cpu().numpy()) == io.exact(values[name])
            assert io.exact(second[name].cpu().numpy()) == io.exact(values[name])
            first[name].fill_(37)
        assert io.exact(ws.readback(old)) == io.exact(values)
        assert bodies.physical_bytes(batch) == physical
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        current, updated = observe(
            mj, batch, ws, models, monkeypatch, history, path, "torch-next-step"
        )
        new = ws.to_torch(current)
        for name in ENTRY_FIELDS:
            assert new[name].data_ptr() not in (first[name].data_ptr(), second[name].data_ptr())
            assert io.exact(new[name].cpu().numpy()) == io.exact(updated[name])
            assert torch.all(first[name] == 37).item()
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.to_torch(old)
        ws.close()
        with pytest.raises(RuntimeError, match="closed"):
            ws.to_torch(current)
        with pytest.raises(RuntimeError, match="closed"):
            ws.refresh(expected_revision=batch.checkpoint_revision)
    for name in ENTRY_FIELDS:
        assert torch.all(first[name] == 37).item()
        assert io.exact(second[name].cpu().numpy()) == io.exact(values[name])
        assert io.exact(new[name].cpu().numpy()) == io.exact(updated[name])
