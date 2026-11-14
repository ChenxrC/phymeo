"""Eight authored manipulation worlds on one GPU through public native-state IO.

This is a 32-step physics/IO integration contract, not manipulation success,
policy inference, or training. Only the parent executes this authored test.
"""

from contextlib import ExitStack

import numpy as np
import pytest
import test_mujoco_tyx_activation as activation
import test_mujoco_tyx_heterogeneous_io as io

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.manipulation_tasks import H, manipulation_scenes


@pytest.mark.cuda
def test_eight_manipulation_scenes_public_gpu_io_match_w1_for_32_steps(tmp_path, monkeypatch):
    import mujoco as mj
    import torch
    from tyx.torch import from_torch, stream_from_torch

    scenes = manipulation_scenes()
    keys = (41, 7, 305, 11, 203, 59, 91, 2)
    assert len(scenes) == len(keys) == 8
    specs = tuple(
        MuJoCoTyxModelSpec(scene.key, compile_mujoco_xml(scene.mjcf, require_asset=False))
        for scene in scenes
    )
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, scene.key) for key, scene in zip(keys, scenes, strict=True)
    )
    # Static native model metadata only: no endpoint CPU forward or CPU stepping.
    geometry = {}
    for key, scene, spec in zip(keys, scenes, specs, strict=True):
        model = spec.compilation.fresh_native_model()
        joint_id = int(model.joint("object-free").id)
        geometry[key] = (
            int(model.geom(scene.support_geom_name).id),
            frozenset(int(model.geom(name).id) for name in scene.object_geom_names),
            int(model.jnt_qposadr[joint_id]),
        )

    history = []
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
        ws = stack.enter_context(batch.prepare_device_workspace())
        single = {
            world.world: stack.enter_context(
                MuJoCoRuntime(
                    spec.compilation,
                    world_keys=(world.world,),
                    initial_states={world.world: world.initial_state},
                    mode="cuda",
                    implementation="tyx",
                )
            )
            for world, spec in zip(worlds, specs, strict=True)
        }
        assert batch.world_keys == keys and batch.timestep == H == 0.001
        assert batch.capabilities["bucket_count"] == batch.capabilities["world_count"] == 8
        assert all(owner.capabilities["gpu"] for owner in single.values())
        initial = batch.integration_snapshot()
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert frame.ticks == (0,) * 8
        assert ws.binding["phase"] == "accepted-integration-endpoint"
        arrays = ws.to_torch(frame)
        history.append(dict(frame=frame, workspace=io.compare_frame(batch, ws, frame)))
        device = arrays["q"].device
        assert device.type == "cuda" and device.index == ws.binding["device"]
        producer = torch.cuda.Stream(device=device)
        stream = stream_from_torch(producer, backend="cuda")
        stack.callback(stream.close)
        supported = set()

        def forbidden(*args, **kwargs):
            raise AssertionError("manipulation GPU step attempted CPU dynamics")

        try:
            with monkeypatch.context() as guard:
                for name in (
                    "mj_step",
                    "mj_forward",
                    "mj_fullM",
                    "mj_rne",
                    "mj_jac",
                    "mj_collision",
                    "mj_fwdConstraint",
                ):
                    guard.setattr(mj, name, forbidden)
                for name in ("solve", "inv", "cholesky", "lstsq"):
                    guard.setattr(np.linalg, name, forbidden)
                for tick in range(1, 33):
                    packets = {
                        key: MuJoCoControl(
                            np.full(batch.layout[key].nu, 0.025, np.float32),
                            qfrc_applied=np.zeros(batch.layout[key].nv, np.float32),
                            xfrc_applied=np.zeros((batch.layout[key].nbody, 6), np.float32),
                        )
                        for key in keys
                    }
                    with torch.cuda.stream(producer), ExitStack() as inputs:
                        mask = arrays["ctrl_mask"].bool()
                        assert (~mask).any().item(), "heterogeneous actuator padding is exercised"
                        ctrl = torch.full(mask.shape, 0.025, device=device, dtype=torch.float32)
                        ctrl[~mask] = float("nan")
                        qfrc = torch.zeros_like(arrays["v"])
                        xfrc = torch.zeros(
                            (8, arrays["body_mask"].shape[1], 6), device=device, dtype=torch.float32
                        )
                        views = tuple(
                            from_torch(value, backend="cuda") for value in (ctrl, qfrc, xfrc)
                        )
                        for view in views:
                            inputs.callback(view.free)
                        control_receipt = batch.set_controls_from_device(
                            views[0],
                            qfrc_applied=views[1],
                            xfrc_applied=views[2],
                            world_keys=keys,
                            expected_revision=batch.checkpoint_revision,
                            producer_stream=stream,
                        )
                    for key, owner in single.items():
                        owner.set_controls({key: packets[key]})
                        assert owner.step(H) == {key: True}
                    assert batch.step(H) == dict.fromkeys(keys, True)
                    states = batch.integration_snapshot()
                    references = {
                        key: owner.integration_snapshot()[key] for key, owner in single.items()
                    }
                    evidence = dict(
                        batch_step=batch.last_step, controls=control_receipt, w1=references
                    )
                    history.append(evidence)
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    diagnostic = ws.readback(frame)
                    evidence.update(frame=frame, workspace=diagnostic)
                    arrays = ws.to_torch(frame)
                    io.compare_frame(batch, ws, frame)
                    assert frame.ticks == (tick,) * 8
                    assert batch.last_step.world_keys == keys
                    assert all(batch.last_step.accepted.values())
                    for name, tensor in arrays.items():
                        assert tensor.device == device and torch.isfinite(tensor).all().item()
                        assert io.exact(tensor.cpu().numpy()) == io.exact(diagnostic[name])
                    for key in keys:
                        state, reference = states[key], references[key]
                        assert state.tick == tick and np.isfinite(state.time)
                        for name in ("q", "v", "act", "ctrl", "time", "tick"):
                            assert io.exact(getattr(state, name)) == io.exact(
                                getattr(reference, name)
                            ), (key, tick, name)
                        for name in ("q", "v", "act", "ctrl", "qfrc_applied", "xfrc_applied"):
                            assert np.isfinite(getattr(state, name)).all(), (key, tick, name)
                        assert not state.qfrc_applied.any() and not state.xfrc_applied.any()
                        receipt = batch.last_step.receipts[key]
                        assert io.exact(receipt.control) == io.exact(packets[key])
                        assert receipt.backend == "mujoco-tyx-cuda"
                        assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
                        support_id, object_ids, qa = geometry[key]
                        for contact in receipt.contacts:
                            first, second = contact.geometry_ids
                            if first == support_id and second in object_ids:
                                object_force = contact.force_on_geometry2_world
                            elif second == support_id and first in object_ids:
                                object_force = contact.force_on_geometry1_world
                            else:
                                continue
                            assert tick > 1, "the source object begins 1 mm above support"
                            assert contact.dimension == 3
                            if contact.active and contact.normal_force > 0 and object_force[2] > 0:
                                assert receipt.input_q[qa + 2] < initial[key].q[qa + 2]
                                supported.add(key)
            assert supported == set(keys), "all eight FREE objects must reach physical support"
            assert len(history) == 33
            assert all(state.tick == 32 for state in batch.integration_snapshot().values())
        finally:
            # Preserve complete receipts and explicit diagnostic copies, including failure prefixes.
            activation.save(tmp_path / "manipulation-w8-gpu-32.json.gz", history)
