"""Send accepted TYX body poses through the Windows CUDA/Vulkan compute probe.

The two explicit CUDA-to-host copies per exposure are verification oracles.
The shader operates on the original 48 FP32 bit patterns in the shared SSBO.
"""
import argparse
import ctypes as ct
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from verify_cuda_vulkan_exchange import Result, cu, imported_slots, vk


KEYS = (41, 7)
CONTROLS = (4., -4.)
H = .001
WORDS = 48
POSE_BYTES = WORDS * 4
MARKER = 0x3f800000  # FP32 1: the audit counts words distinct from this marker.
SLIDER = """<mujoco model="opposing-sliders"><option timestep=".001" gravity="0 0 0"/>
<worldbody><body name="mass"><joint name="slide" type="slide" axis="0 0 1"
 armature=".04" damping=".2"/><geom name="sphere" type="sphere" size=".1" mass=".5"/>
</body></worldbody><actuator><motor name="drive" joint="slide" gear="2"
 ctrllimited="true" ctrlrange="-5 5"/></actuator></mujoco>"""


def _physics_bytes(batch):
    return {k: int(v) for k, v in batch.counters["observed_device_counters"].items()
            if k.endswith("_bytes") and isinstance(v, int)}


def _verify_poses(mj, model, references, words, tick):
    rows = words.view(np.float32).reshape(4, 3, 4)
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite source body transform")
    checks = []
    for i, (world, state) in enumerate(zip(KEYS, references, strict=True)):
        mj.mj_forward(model, state)
        native = rows[2*i:2*i+2]
        expected_r = state.xmat.reshape(2, 3, 3)
        np.testing.assert_allclose(native[:, :, :3], expected_r, rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(native[:, :, 3], state.xpos, rtol=3e-5, atol=4e-5)
        if tick and native[1, 2, 3] * CONTROLS[i] <= 0:
            raise ValueError(f"world {world} did not move along its commanded direction")
        checks.append(dict(world=world, control=CONTROLS[i], body_count=2,
                           position_max_abs_m=float(np.max(np.abs(native[:, :, 3] - state.xpos))),
                           rotation_max_abs=float(np.max(np.abs(native[:, :, :3] - expected_r))),
                           slider_z_m=float(native[1, 2, 3]), reference_z_m=float(state.xpos[1, 2])))
    return checks


def _exchange(driver, dll, owner, slots, slot, first_use, tensor):
    address, writable, ready, stream = slots[slot]
    wait = driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
    signal = driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
    result = Result()
    try:
        if first_use:
            cu(driver.cuWaitExternalSemaphoresAsync, [writable], [wait], 1, stream)
        # On the next generation, this CUDA stream already consumed writable
        # when it audited the preceding Vulkan result for this exact slot.
        cu(driver.cuMemcpyDtoDAsync, address, tensor.data_ptr(), POSE_BYTES, stream)
        cu(driver.cuSignalExternalSemaphoresAsync, [ready], [signal], 1, stream)
        vk(dll, "submit", owner, slot, MARKER)
        vk(dll, "finish", owner, slot, 10_000_000_000, ct.byref(result))
        cu(driver.cuWaitExternalSemaphoresAsync, [writable], [wait], 1, stream)
    finally:
        # Keep the Torch allocation alive while the imported CUDA stream uses it.
        cu(driver.cuStreamSynchronize, stream)

    original, returned = np.empty(WORDS, np.uint32), np.empty(WORDS, np.uint32)
    cu(driver.cuMemcpyDtoH, original.ctypes.data, tensor.data_ptr(), POSE_BYTES)
    cu(driver.cuMemcpyDtoH, returned.ctypes.data, address, POSE_BYTES)
    expected = original ^ np.uint32(0xa5a55a5a) ^ np.arange(WORDS, dtype=np.uint32)
    np.testing.assert_array_equal(returned, expected)
    if (result.checked, result.mismatches, result.first_input, result.last_input) != (
        WORDS, int((original != MARKER).sum()), int(original[0]), int(original[-1])
    ):
        raise RuntimeError("Vulkan shader input audit disagrees with the source bit patterns")
    return original, dict(checked_words=result.checked, words_different_from_marker=result.mismatches,
                          first_input_word=result.first_input, last_input_word=result.last_input,
                          source_sha256=sha256(original.tobytes()).hexdigest(),
                          result_sha256=sha256(returned.tobytes()).hexdigest(),
                          verified_returned_words=WORDS)


def run(library, shader, device, validation):
    import mujoco as mj
    import torch
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    spec = MuJoCoTyxModelSpec("slider", compile_mujoco_xml(SLIDER, require_asset=False))
    model = spec.compilation.fresh_native_model()
    worlds = tuple(MuJoCoTyxWorldSpec(k, spec.key) for k in KEYS)
    references = tuple(mj.MjData(model) for _ in KEYS)
    records = []
    with (
        torch.cuda.device(device),
        imported_slots(library, shader, device, WORDS, validation) as (driver, dll, owner, slots, uuid),
        MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch,
        batch.prepare_device_workspace(include_bodies=True) as workspace,
    ):
        if workspace.binding["device"] != device or len(slots) != 2 or model.nbody != 2:
            raise ValueError("physics device, slot count or native body layout differs")
        batch.set_controls({k: MuJoCoControl([u]) for k, u in zip(KEYS, CONTROLS, strict=True)})
        initial_physics = _physics_bytes(batch)
        previous_physics = initial_physics
        previous_tick = 0
        for sample, tick in enumerate((0, 12, 24)):
            for _ in range(previous_tick, tick):
                if batch.step(H) != {k: True for k in KEYS}:
                    raise RuntimeError("physics batch rejected a slider step")
                for state, control in zip(references, CONTROLS, strict=True):
                    state.ctrl[:] = control
                    mj.mj_step(model, state)
            slot = sample % len(slots)
            before = workspace.counters
            frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
            handle = int(slots[slot][3])
            if not handle:
                raise RuntimeError("expected a nondefault CUDA stream")
            stream = torch.cuda.ExternalStream(handle, device=device)
            with torch.cuda.stream(stream):
                snapshot = workspace.body_pose_snapshot(frame)
            tensor = snapshot.transforms
            if (snapshot.frame is not frame or snapshot.world_keys != KEYS
                    or snapshot.body_offsets != (0, 2, 4) or frame.ticks != (tick, tick)
                    or frame.revision != batch.checkpoint_revision
                    or (tick and frame.revision != batch.last_step.revision_after)
                    or tensor.device != torch.device("cuda", device) or not tensor.is_contiguous()
                    or tensor.dtype != torch.float32 or tuple(tensor.shape) != (4, 3, 4)):
                raise RuntimeError("snapshot disagrees with its accepted frame, device or body layout")
            original, audit = _exchange(driver, dll, owner, slots, slot, sample < len(slots), tensor)
            after = workspace.counters
            pose_writes = after["output_d2d_bytes"] - before["output_d2d_bytes"]
            if pose_writes != POSE_BYTES or after["explicit_readback_bytes"] != before["explicit_readback_bytes"]:
                raise RuntimeError("unexpected body snapshot transfer counters")
            current = _physics_bytes(batch)
            records.append(dict(
                slot=slot, slot_generation=sample // len(slots), tick=tick, revision=frame.revision,
                workspace_epoch=frame.epoch, world_keys=snapshot.world_keys, world_ticks=frame.ticks,
                world_boundaries=frame.world_boundaries, body_offsets=snapshot.body_offsets,
                native_comparison=_verify_poses(mj, model, references, original, tick), vulkan=audit,
                bytes=dict(physics_solver={k: v - previous_physics.get(k, 0) for k,v in current.items()},
                           observer_status_d2h=after["status_downloaded_bytes"]-before["status_downloaded_bytes"],
                           snapshot_device_write=pose_writes, snapshot_to_ssbo_d2d=POSE_BYTES,
                           pose_h2d=0, pose_d2h_for_transport=0,
                           source_oracle_d2h=POSE_BYTES, result_oracle_d2h=POSE_BYTES,
                           vulkan_audit_to_host=ct.sizeof(Result))))
            previous_physics, previous_tick = current, tick
        counters = batch.counters
        if (counters["completed_steps"], counters["accepted_world_steps"]) != (24, 48):
            raise RuntimeError("unexpected physical step count")
        layout_bytes = workspace.counters["body_pose_layout_upload_bytes"]
        device_name = torch.cuda.get_device_name(device)
        validation_errors = dll.pv0_validation_errors(owner)
    return dict(
        scope="Live accepted body-pose transport through a Vulkan compute SSBO",
        device_uuid=uuid.hex(), device_name=device_name, cuda_ordinal=device, slots=2,
        phase=workspace.binding["phase"], compiled_timestep_s=H, completed_steps=24, accepted_world_steps=48,
        pose_format="contiguous FP32 body-to-world [R|p], 3x4 per body, native body ID order",
        physics_bytes_after_initial_controls=initial_physics, layout_upload_bytes=layout_bytes,
        audit="Both 192-byte host reads occur after Vulkan completion and are used only for verification.",
        validation_enabled=validation, validation_errors=validation_errors,
        native_live_after=dll.pv0_live(), exchanges=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--shader", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "win32" or args.device < 0 or args.report.exists():
        parser.error("requires Win32, a nonnegative CUDA ordinal and a new report path")
    report = run(args.library, args.shader, args.device, args.validation)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"{report['device_name']}: verified 3 accepted pose batches, 144 Vulkan words")


if __name__ == "__main__":
    main()
