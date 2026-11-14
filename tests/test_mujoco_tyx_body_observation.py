"""B1 endpoint body observations; author-only, execution belongs to the parent.

Independent native FK/body-origin Jacobians are diagnostic oracles only.
This suite does not authorize public device contact tokens or task success.
"""

from contextlib import ExitStack

import numpy as np
import pytest
import test_mujoco_tyx_ball as ball
import test_mujoco_tyx_forest as forest
import test_mujoco_tyx_free_ball as free_ball
import test_mujoco_tyx_heterogeneous_io as io
import test_mujoco_tyx_runtime as legacy
import test_mujoco_tyx_tree as tree
from test_mujoco_tyx_activation import save

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes


MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (41, 7, 305, 11, 203, 59, 91, 2)
FAMILY = dict(zip(KEYS, ("t0", "scalar", "ball", "free-ball", "two-free", "tree", "free-ball", "two-free"), strict=True))
BODY_FIELDS = ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity")
# Existing floating pose gates and fixed/floating Jacobian component gates.
POSE_TOL = dict(rtol=3e-5, atol=4e-5)
TWIST_TOL = dict(rtol=2e-4, atol=5e-5)


def declarations(mj, keys=KEYS):
    sources = {
        "t0": legacy.source(free=True, plane=False),
        "scalar": legacy.SLIDER,
        "tree": tree.MIXED_XML,
        "ball": ball.HINGE_BALL_SLIDE,
        "free-ball": free_ball.source("mixed"),
        "two-free": forest.source("two-free", object_first=False),
    }
    names = tuple(dict.fromkeys(FAMILY[key] for key in keys))
    specs = tuple(MuJoCoTyxModelSpec(name, compile_mujoco_xml(sources[name], require_asset=False)) for name in names)
    models = {spec.key: spec.compilation.fresh_native_model() for spec in specs}
    worlds = []
    for key in keys:
        name, slot = FAMILY[key], KEYS.index(key)
        model = models[name]
        if name == "ball":
            initial = ball.initial(mj, model, slot)
        elif name == "free-ball":
            initial = free_ball.initial(mj, model, slot)
        elif name == "two-free":
            initial = forest.initial(mj, model, slot=slot)
        else:
            q = model.qpos0.copy()
            v = np.linspace(.2, .7, model.nv)
            if name == "t0":
                quat = np.array([.9, .17, -.23, .11])
                q[3:7] = quat / np.linalg.norm(quat)
            else:
                q += .11 + .01*slot
            initial = MuJoCoInitialState(q, v, time=.125)
        worlds.append(MuJoCoTyxWorldSpec(key, name, initial))
    return specs, tuple(worlds), models


def controls(batch):
    return {key: MuJoCoControl(np.full(row.nu, .025, np.float32)) for key, row in batch.layout.items()}


def native_bodies(mj, model, state):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = state.q, state.v, state.act, state.time
    mj.mj_forward(model, data)
    linear, angular, com_linear = (np.zeros((model.nbody, 3)) for _ in range(3))
    for body in range(model.nbody):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jp, jr, body)
        linear[body], angular[body] = jp @ state.v, jr @ state.v
        mj.mj_jacBodyCom(model, data, jp, jr, body)
        com_linear[body] = jp @ state.v
    return {
        "body_position": data.xpos.copy(),
        "body_rotation": data.xmat.reshape(model.nbody, 3, 3).copy(),
        "body_linear_velocity": linear,
        "body_angular_velocity": angular,
        "com_linear_velocity": com_linear,
    }


def check_bodies(batch, ws, frame, expected):
    values = io.compare_frame(batch, ws, frame)
    assert ws.binding["endpoint_body_kinematics"] is True
    assert ws.binding["body_velocity"] == "world-linear-and-angular-at-body-origin"
    assert ws.binding["phase"] == "accepted-integration-endpoint"
    assert values["body_mask"].dtype == np.int32
    width = max(row.nbody for row in batch.layout.values())
    for name in BODY_FIELDS:
        tail = (3, 3) if name == "body_rotation" else (3,)
        assert values[name].shape == (len(batch.world_keys), width, *tail)
        assert values[name].dtype == np.float32 and np.isfinite(values[name]).all()
        assert not values[name].flags.writeable
    for index, key in enumerate(batch.world_keys):
        count = batch.layout[key].nbody
        np.testing.assert_array_equal(values["body_mask"][index], np.arange(width) < count)
        for name in BODY_FIELDS:
            np.testing.assert_allclose(values[name][index, :count], expected[key][name],
                                       **(TWIST_TOL if "velocity" in name else POSE_TOL),
                                       err_msg=f"{key}:{name}")
            np.testing.assert_array_equal(values[name][index, count:], 0)
    return values


def physical_bytes(batch):
    """Diagnostic bytes of every real solver allocation, including disposable scratch."""
    result = []
    for owner in batch._buckets:
        device = owner._device
        with device.tx.use_runtime(device.runtime):
            device.runtime.sync()
            row = {}
            for entry in device.allocation_manifest.entries:
                name, dot, bank = entry.name.partition(".")
                buffer = getattr(device, name)
                if dot:
                    buffer = buffer[int(bank)]
                row[entry.name] = device._get(buffer)
        result.append(io.exact(row))
    return tuple(result)


def refresh_without_host_physics(mj, batch, ws, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("body refresh used HOST physics/state reconstruction")

    with monkeypatch.context() as guard:
        for name in ("mj_step", "mj_forward", "mj_kinematics", "mj_comPos", "mj_comVel",
                     "mj_jacBody", "mj_jacBodyCom", "mj_fullM", "mj_rne", "mj_jac"):
            guard.setattr(mj, name, forbidden)
        for name in ("solve", "inv", "cholesky", "lstsq"):
            guard.setattr(np.linalg, name, forbidden)
        guard.setattr(batch, "integration_snapshot", forbidden)
        for owner in batch._buckets:
            guard.setattr(owner, "snapshot", forbidden)
            guard.setattr(owner, "integration_snapshot", forbidden)
        return ws.refresh(expected_revision=batch.checkpoint_revision)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reverse", (False, True))
def test_w8_endpoint_bodies_match_native_and_w1_without_changing_physics(mode, reverse, tmp_path, monkeypatch):
    mj = legacy.deps(mode)
    keys = KEYS[::-1] if reverse else KEYS
    specs, worlds, models = declarations(mj, keys)
    history = []
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        ws = stack.enter_context(batch.prepare_device_workspace(include_bodies=True))
        singles = {}
        for world in worlds:
            spec = next(spec for spec in specs if spec.key == world.model_key)
            single = stack.enter_context(MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode=mode))
            singles[world.world] = (single, stack.enter_context(single.prepare_device_workspace(include_bodies=True)))
        source = tuple(spec.compilation.source.primary_xml for spec in specs)
        assert len({row.nbody for row in batch.layout.values()}) > 1
        multi = models["two-free"]
        free_ids = np.flatnonzero(multi.jnt_type == mj.mjtJoint.mjJNT_FREE)
        assert len(free_ids) == 2 and all(multi.jnt_qposadr[free_ids] > 0)
        for tick in range(3):
            states = batch.integration_snapshot()
            expected = {key: native_bodies(mj, models[FAMILY[key]], states[key]) for key in keys}
            before, allocations = io.exact(states), physical_bytes(batch)
            frame = refresh_without_host_physics(mj, batch, ws, monkeypatch)
            values = check_bodies(batch, ws, frame, expected)
            history.append(dict(frame=frame, values=values, native=expected, step=batch.last_step))
            assert io.exact(batch.integration_snapshot()) == before
            assert physical_bytes(batch) == allocations
            assert tuple(spec.compilation.source.primary_xml for spec in specs) == source
            assert frame.ticks == (tick,)*8
            for index, key in enumerate(keys):
                single, single_ws = singles[key]
                assert io.exact(single.integration_snapshot()[key]) == io.exact(states[key])
                single_frame = refresh_without_host_physics(mj, single, single_ws, monkeypatch)
                single_values = check_bodies(single, single_ws, single_frame, {key: expected[key]})
                for name in BODY_FIELDS:
                    assert io.exact(values[name][index, :batch.layout[key].nbody]) == io.exact(single_values[name][0])
                if tick and states[key].last_step.dynamics is not None:
                    entry = states[key].last_step.dynamics.body_position
                    assert np.max(np.abs(values["body_position"][index, :len(entry)] - entry)) > 1e-6
            # Eccentric FREE root: origin velocity differs from COM, angular v is local in qvel.
            root = int(models["free-ball"].body("root").id)
            item = expected[11]
            assert np.linalg.norm(item["body_linear_velocity"][root] - item["com_linear_velocity"][root]) > 1e-3
            qa = int(models["free-ball"].jnt_dofadr[models["free-ball"].joint("free").id])
            local = states[11].v[qa + 3:qa + 6]
            np.testing.assert_allclose(item["body_angular_velocity"][root], item["body_rotation"][root] @ local, rtol=0, atol=1e-12)
            assert np.linalg.norm(item["body_angular_velocity"][root] - local) > 1e-3
            if tick < 2:
                packet = controls(batch)
                batch.set_controls(packet)
                assert batch.step(H) == dict.fromkeys(keys, True)
                for key, (single, _) in singles.items():
                    single.set_controls({key: packet[key]})
                    assert single.step(H) == {key: True}
    save(tmp_path / "endpoint-bodies.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_body_observation_is_opt_in_and_keeps_existing_native_fields(mode):
    mj = legacy.deps(mode)
    specs, worlds, _ = declarations(mj, (7, 11))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch, batch.prepare_device_workspace() as ws:
        data = io.compare_frame(batch, ws, ws.refresh(expected_revision=0))
        assert ws.binding["endpoint_body_kinematics"] is False
        assert set(data) == {"q", "v", "act", "clock", "q_mask", "v_mask", "act_mask", "ctrl_mask", "body_mask"}
        assert not set(BODY_FIELDS).intersection(data)


@pytest.mark.parametrize("mode", MODES)
def test_last_body_observation_failure_keeps_frame_and_solver_scratch(mode, monkeypatch):
    mj = legacy.deps(mode)
    specs, worlds, models = declarations(mj)
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch, batch.prepare_device_workspace(include_bodies=True) as ws:
        batch.set_controls(controls(batch))
        batch.step(H)
        old = ws.refresh(expected_revision=batch.checkpoint_revision)
        previous = io.exact(ws.readback(old))
        states, scratch = io.exact(batch.integration_snapshot()), physical_bytes(batch)
        revision, receipt = batch.checkpoint_revision, batch.last_step
        original = type(ws._body_observer).refresh_bucket
        visited = []

        def fail_last(observer, index, owner, destination, options):
            original(observer, index, owner, destination, options)
            visited.append(index)
            if index == len(batch._buckets) - 1:
                raise RuntimeError("authored final body observation failure")

        with monkeypatch.context() as patch:
            patch.setattr(type(ws._body_observer), "refresh_bucket", fail_last)
            with pytest.raises(RuntimeError, match="final body observation"):
                ws.refresh(expected_revision=revision)
        assert visited == list(range(len(batch._buckets)))
        assert io.exact(ws.readback(old)) == previous
        assert batch.checkpoint_revision == revision and batch.last_step is receipt
        assert io.exact(batch.integration_snapshot()) == states
        assert physical_bytes(batch) == scratch
        current = batch.integration_snapshot()
        expected = {key: native_bodies(mj, models[FAMILY[key]], current[key]) for key in KEYS}
        retry = refresh_without_host_physics(mj, batch, ws, monkeypatch)
        assert io.exact(check_bodies(batch, ws, retry, expected)) == previous


@pytest.mark.parametrize("mode", MODES)
def test_body_workspace_follows_partial_reset_restore_and_replay(mode, tmp_path, monkeypatch):
    mj = legacy.deps(mode)
    specs, worlds, models = declarations(mj)
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch, batch.prepare_device_workspace(include_bodies=True) as ws:
        batch.set_controls(controls(batch))
        batch.step(H)
        saved_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        saved = ws.readback(saved_frame)
        capsule = batch.checkpoint()
        batch.step(H)
        successor_states = io.exact(batch.integration_snapshot())
        successor = ws.readback(ws.refresh(expected_revision=batch.checkpoint_revision))
        selection = (91, 2)
        reset = {world.world: MuJoCoInitialState(world.initial_state.q, -world.initial_state.v)
                 for world in worlds if world.world in selection}
        owners = tuple(batch._buckets)
        before_reset = batch.integration_snapshot()
        batch.reset_states(reset, expected_revision=batch.checkpoint_revision)
        assert any(old is not new for old, new in zip(owners, batch._buckets, strict=True))
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.readback(saved_frame)
        states = batch.integration_snapshot()
        expected = {key: native_bodies(mj, models[FAMILY[key]], states[key]) for key in KEYS}
        reset_frame = refresh_without_host_physics(mj, batch, ws, monkeypatch)
        reset_values = check_bodies(batch, ws, reset_frame, expected)
        for i, key in enumerate(KEYS):
            if key not in selection:
                assert io.exact(states[key]) == io.exact(before_reset[key])
                for name in BODY_FIELDS:
                    assert io.exact(reset_values[name][i]) == io.exact(successor[name][i])
            else:
                assert states[key].tick == 0
        batch.restore(capsule, expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.readback(reset_frame)
        restored = ws.readback(refresh_without_host_physics(mj, batch, ws, monkeypatch))
        assert io.exact(restored) == io.exact(saved)
        batch.step(H)
        replay = ws.readback(refresh_without_host_physics(mj, batch, ws, monkeypatch))
        assert io.exact(replay) == io.exact(successor)
        assert io.exact(batch.integration_snapshot()) == successor_states
        save(tmp_path / "body-boundaries.json.gz", dict(saved=saved, reset=reset_values, restored=restored, successor=successor, replay=replay))


@pytest.mark.cuda
def test_cuda_body_exports_are_torch_owned_copies(tmp_path):
    import torch

    mj = legacy.deps("cuda")
    specs, worlds, models = declarations(mj, (11, 2))
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch, batch.prepare_device_workspace(include_bodies=True) as ws:
        frame = ws.refresh(expected_revision=0)
        raw = ws.readback(frame)
        exported = ws.to_torch(frame)
        for name in BODY_FIELDS:
            assert exported[name].is_cuda
            assert io.exact(exported[name].cpu().numpy()) == io.exact(raw[name])
            exported[name].fill_(42)
        assert io.exact(ws.readback(frame)) == io.exact(raw)
        batch.set_controls(controls(batch))
        batch.step(H)
        new_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        new = ws.to_torch(new_frame)
        for name in BODY_FIELDS:
            assert new[name].is_cuda and new[name].data_ptr() != exported[name].data_ptr()
            assert torch.all(exported[name] == 42).item()
        with pytest.raises(ValueError, match="stale|superseded"):
            ws.to_torch(frame)
        states = batch.integration_snapshot()
        expected = {key: native_bodies(mj, models[FAMILY[key]], states[key]) for key in batch.world_keys}
        save(tmp_path / "body-torch.json.gz", check_bodies(batch, ws, new_frame, expected))


@pytest.mark.cuda
def test_eight_original_manipulation_scenes_export_native_endpoint_bodies(tmp_path, monkeypatch):
    mj = legacy.deps("cuda")
    scenes = manipulation_scenes()
    specs = tuple(MuJoCoTyxModelSpec(scene.key, compile_mujoco_xml(scene.mjcf, require_asset=False)) for scene in scenes)
    worlds = tuple(MuJoCoTyxWorldSpec(key, scene.key) for key, scene in zip(KEYS, scenes, strict=True))
    models = {key: spec.compilation.fresh_native_model() for key, spec in zip(KEYS, specs, strict=True)}
    history = []
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch, batch.prepare_device_workspace(include_bodies=True) as ws:
        for tick in range(3):
            states = batch.integration_snapshot()
            expected = {key: native_bodies(mj, models[key], states[key]) for key in KEYS}
            frame = refresh_without_host_physics(mj, batch, ws, monkeypatch)
            values = check_bodies(batch, ws, frame, expected)
            tensors = ws.to_torch(frame)
            for name in BODY_FIELDS:
                assert tensors[name].is_cuda
                assert io.exact(tensors[name].cpu().numpy()) == io.exact(values[name])
            history.append(dict(frame=frame, values=values, native=expected, step=batch.last_step))
            assert frame.ticks == (tick,)*8
            if tick < 2:
                batch.set_controls(controls(batch))
                assert batch.step(H) == dict.fromkeys(KEYS, True)
    save(tmp_path / "manipulation-body-endpoints.json.gz", history)
