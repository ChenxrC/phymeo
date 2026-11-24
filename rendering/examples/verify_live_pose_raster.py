"""Measure resident raster captures from live TYX slider poses and a host pose oracle."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _panda_scene import box, look_at
from _r4_archive import _pose
from verify_live_pose_exchange import CONTROLS, H, KEYS, _physics_bytes
from phymeo_rendering.assets import Material
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_pose_packing import DeviceVisualBinding
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, Instance, NodePose, Pose, SceneFrame, VisualScene


SIZE = (320, 240)
VIEWS = {"front": (.46, -.52, .31), "reverse": (-.48, .43, .38)}
XML = """<mujoco model="visual-slider"><option timestep=".001" gravity="0 0 0"/>
<default><geom type="box" contype="0" conaffinity="0"/></default>
<worldbody>
  <geom name="receiver" size=".42 .33 .012" pos="0 0 -.17" rgba=".46 .53 .53 1"/>
  <body name="carriage" pos="0 0 .015" euler="0 0 90">
    <inertial pos="0 0 0" mass=".5" diaginertia=".001 .001 .001"/>
    <joint name="slide" type="slide" axis="0 0 1" armature=".04" damping=".2"/>
    <geom name="block" size=".073 .045 .034" pos=".012 -.014 .016"
          euler="0 0 18" rgba=".83 .29 .07 1"/>
    <geom name="cap" size=".033 .052 .015" pos="-.014 -.005 .065"
          euler="12 0 -17" rgba=".13 .37 .61 1"/>
  </body>
</worldbody>
<actuator><motor name="drive" joint="slide" gear="2" ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>"""


def _scene(model, directory):
    meshes, materials, instances = [], [], []
    body_nodes = {model.body(i).name: i for i in range(model.nbody)}
    for i in range(model.ngeom):
        name = model.geom(i).name
        mesh = box(directory, name, 2 * model.geom_size[i])
        material = Material(name, tuple(map(float, model.geom_rgba[i, :3])), 0., .7)
        w, x, y, z = map(float, model.geom_quat[i])
        local = Pose(tuple(map(float, model.geom_pos[i])), (x, y, z, w))
        meshes.append(mesh)
        materials.append(material)
        instances.append(Instance(i + 1, mesh.id, material.id, model.body(int(model.geom_bodyid[i])).name, local))
    scene = VisualScene("native-visual-slider-" + sha256(XML.encode()).hexdigest()[:12],
                        tuple(meshes), tuple(materials), tuple(instances),
                        (DirectionalLight((.6, 0., -.8), (1., 1., 1.)),))
    return scene, body_nodes


def _planes(capture, key, camera, time_s):
    if (capture.frame_key, capture.camera, capture.time_s) != (key, camera, time_s):
        raise ValueError("captured planes belong to another world, frame, time or camera")
    return {p.channel: p.data for p in capture.planes}


def _compare(host, shared, block_id):
    required = set(Channel)
    if host.keys() != required or shared.keys() != required:
        raise ValueError("expected RGB, forward depth, instance ID and valid planes")
    if host[Channel.INSTANCE_ID] != shared[Channel.INSTANCE_ID] or host[Channel.VALID] != shared[Channel.VALID]:
        raise ValueError("instance ID or visibility differs between host and shared transforms")
    w, h = SIZE
    ids = np.frombuffer(shared[Channel.INSTANCE_ID], "<u4").reshape(h, w)
    depth = np.frombuffer(shared[Channel.FORWARD_DEPTH], "<f4").reshape(h, w)
    ref_depth = np.frombuffer(host[Channel.FORWARD_DEPTH], "<f4").reshape(h, w)
    hit = ids != 0
    if (not hit.any() or not np.isfinite(depth[hit]).all() or not np.isfinite(ref_depth[hit]).all()
            or not np.isposinf(depth[~hit]).all() or not np.isposinf(ref_depth[~hit]).all()):
        raise ValueError("invalid hit or miss depth")
    depth_error = float(np.abs(depth[hit] - ref_depth[hit]).max())
    rgb = np.frombuffer(shared[Channel.RGB], "u1").reshape(h, w, 3)
    ref_rgb = np.frombuffer(host[Channel.RGB], "u1").reshape(h, w, 3)
    error = np.abs(rgb.astype(np.int16) - ref_rgb)
    if depth_error > 2e-5 or error.max() > 1:
        raise ValueError(f"host/shared raster differs: depth {depth_error} m; RGB {error.max()} codes")
    y, x = np.nonzero(ids == block_id)
    if len(x) < 20:
        raise ValueError("moving slider is absent or too small in the exposure")
    metrics = dict(exact_instance_id=True, exact_valid=True, rgb_max_codes=int(error.max()),
                   rgb_changed_pixels=int(np.any(error != 0, axis=2).sum()), depth_max_abs_m=depth_error,
                   jointly_valid_pixels=int(hit.sum()), moving_pixels=len(x),
                   centroid_xy=[float(x.mean()), float(y.mean())],
                   plane_sha256={ch.value: sha256(data).hexdigest() for ch, data in shared.items()})
    return metrics, (rgb, depth, ids)


def _shadow_change(before, after, receiver):
    rgb0, z0, id0 = before
    rgb1, z1, id1 = after
    fixed = (id0 == receiver) & (id1 == receiver) & (z0 == z1)
    # Direct directional shading, one fixed receiver material, fixed camera and
    # rigid receiver. Restrict the test to its unchanged primary hits.
    delta = rgb1.mean(axis=2) - rgb0.mean(axis=2)
    flat = int(np.abs(np.where(fixed, delta, 0.)).argmax())
    y, x = np.unravel_index(flat, fixed.shape)
    return dict(fixed_receiver_pixels=int(fixed.sum()), brightened_18=int((fixed & (delta >= 18)).sum()),
                darkened_18=int((fixed & (delta <= -18)).sum()), witness_xy=[int(x), int(y)],
                witness_before_rgb=rgb0[y, x].tolist(), witness_after_rgb=rgb1[y, x].tolist())


def _sheet(path, images, labels):
    w, h = SIZE
    pad, caption = 8, 27
    canvas = Image.new("RGB", (len(labels)*(w+pad)+pad, 4*(h+caption+pad)+pad), "#1c272b")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for row, (key, view) in enumerate((k, v) for k in KEYS for v in VIEWS):
        y = pad + row*(h+caption+pad)
        for col, label in enumerate(labels):
            x = pad + col*(w+pad)
            draw.text((x, y+2), f"World {key}   {view}   |   {label}", font=font, fill="white")
            canvas.paste(Image.fromarray(images[label, key, view][0]), (x, y+caption))
    canvas.save(path)


def run(args):
    import mujoco as mj
    import torch
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch, MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("choose an empty raster proof directory")
    mesh_dir = args.output / "meshes"
    mesh_dir.mkdir()
    spec = MuJoCoTyxModelSpec("slider", compile_mujoco_xml(XML, require_asset=False))
    model = spec.compilation.fresh_native_model()
    scene, nodes = _scene(model, mesh_dir)
    receiver, block = (int(model.geom(name).id) + 1 for name in ("receiver", "block"))
    common = dict(library_path=args.library, shader_directory=args.shader_directory or args.library.parent,
                  device_index=args.device_index, validation=args.validation, resident=True)
    cameras = {v: look_at(eye, (0., 0., -.025), (0., 0., 1.)) for v, eye in VIEWS.items()}
    worlds = tuple(MuJoCoTyxWorldSpec(k, spec.key) for k in KEYS)
    reference = {k: mj.MjData(model) for k in KEYS}
    host, shared, bindings = {}, {}, {}
    exposures, witnesses, images, centroids, rows = [], {}, {}, {}, []
    stale_after_reset = False
    labels = ("Start", "Step 40", "Step 80", "Reset 41")
    audit_pose_bytes = host_image_bytes = shared_image_bytes = 0
    with (
        torch.cuda.device(args.device),
        MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch,
        batch.prepare_device_workspace(include_bodies=True) as ws,
        ExitStack() as stack,
    ):
        if ws.binding["device"] != args.device:
            raise ValueError("TYX workspace inhabits another CUDA adapter")
        stream = torch.cuda.Stream(device=args.device)
        for key in KEYS:
            host[key] = NativeBackend(**common).open_scene(scene)
            stack.callback(host[key].close)
            shared[key] = NativeBackend(**common, shared_cuda_device=args.device).open_scene(scene)
            stack.callback(shared[key].close)
            bindings[key] = DeviceVisualBinding(scene, ws, key, nodes)
        batch.set_controls({k: MuJoCoControl([u]) for k, u in zip(KEYS, CONTROLS, strict=True)})
        initial_bytes = _physics_bytes(batch)
        for stage, label in enumerate(labels):
            if stage in (1, 2):
                for _ in range(40):
                    if batch.step(H) != dict.fromkeys(KEYS, True):
                        raise RuntimeError("physical slider step rejected")
                    for k, u in zip(KEYS, CONTROLS, strict=True):
                        reference[k].ctrl[:] = u
                        mj.mj_step(model, reference[k])
            elif stage == 3:
                batch.reset_states({KEYS[0]: MuJoCoInitialState([-.04], [0.], time=0.)},
                                   expected_revision=batch.checkpoint_revision)
                reference[KEYS[0]] = mj.MjData(model)
                reference[KEYS[0]].qpos[0] = -.04
                try:
                    with shared[KEYS[0]].publish_device_poses(
                            bindings[KEYS[0]], ws, snapshot, producer_stream=stream):
                        raise AssertionError("published an exposure from before reset")
                except ValueError as error:
                    if "stale heterogeneous frame" not in str(error):
                        raise
                    stale_after_reset = True
            frame = ws.refresh(expected_revision=batch.checkpoint_revision)
            expected_ticks = ((0, 80) if stage == 3 else (40*stage, 40*stage))
            if frame.ticks != expected_ticks:
                raise ValueError("accepted world ticks disagree with the requested step or reset")
            with torch.cuda.stream(stream):
                snapshot = ws.body_pose_snapshot(frame)
            # Explicit independent validation frontend. Only the host baseline
            # and native MuJoCo oracle below ever consume this bulk CPU copy.
            host_poses = snapshot.transforms.cpu().numpy()
            audit_pose_bytes += host_poses.nbytes
            for world_row, k in enumerate(KEYS):
                state = reference[k]
                mj.mj_forward(model, state)
                start, stop = snapshot.body_offsets[world_row:world_row+2]
                poses = host_poses[start:stop]
                np.testing.assert_allclose(poses[:, :, :3], state.xmat.reshape(-1, 3, 3), rtol=3e-5, atol=4e-5)
                np.testing.assert_allclose(poses[:, :, 3], state.xpos, rtol=3e-5, atol=4e-5)
                binding = bindings[k]
                time_s = ws.accepted_frame_time(frame, k)
                if abs(time_s - state.time) > 2e-6:
                    raise ValueError("physics endpoint time differs from native integration")
                host_frame = SceneFrame(binding.frame_key(snapshot), time_s, tuple(
                    NodePose(node, _pose(poses[i, :, 3], poses[i, :, :3].astype(np.float64)))
                    for node, i in nodes.items()))
                if host_frame.key.epoch != int(stage == 3 and k == KEYS[0]):
                    raise ValueError("wrong world boundary generation")
                with shared[k].publish_device_poses(binding, ws, snapshot, producer_stream=stream) as exposure:
                    if exposure.key != host_frame.key or exposure.time_s != time_s:
                        raise ValueError("different shared and baseline source exposures")
                    for view, pose in cameras.items():
                        camera = Camera(view, exposure.key, pose, *SIZE, 340., 340., 160., 120., .025, 4.)
                        request = CaptureRequest(camera, RenderPath.RASTER, tuple(Channel))
                        sh = _planes(shared[k].capture_device_poses(exposure, request), exposure.key, camera, time_s)
                        ref = _planes(host[k].capture(host_frame, request), host_frame.key, camera, time_s)
                        metrics, image = _compare(ref, sh, block)
                        images[label, k, view] = image
                        centroids[label, k, view] = metrics["centroid_xy"]
                        exposures.append(dict(stage=label, world=k, camera=view, tick=exposure.key.tick,
                                              epoch=exposure.key.epoch, revision=exposure.key.revision,
                                              time_s=time_s, **metrics))
                        shared_image_bytes += sum(map(len, sh.values()))
                        host_image_bytes += sum(map(len, ref.values()))
            rows.append(dict(stage=label, ticks=frame.ticks, revision=frame.revision,
                             world_boundaries=frame.world_boundaries, native_worlds_checked=len(KEYS)))

        for world_i, k in enumerate(KEYS):
            for view in VIEWS:
                a = centroids[labels[0], k, view]
                b = centroids[labels[2], k, view]
                dy = b[1] - a[1]
                if dy * CONTROLS[world_i] >= 0 or abs(dy) < 3:
                    raise ValueError(f"missing opposite vertical image motion in world {k}, camera {view}")
                proof = _shadow_change(images[labels[0], k, view], images[labels[2], k, view], receiver)
                proof["slider_vertical_motion_px"] = dy
                witnesses[f"{k}/{view}"] = proof
                end = images[labels[2], k, view]
                reset = images[labels[3], k, view]
                identical = all(np.array_equal(x, y) for x, y in zip(end, reset, strict=True))
                if identical != (k == KEYS[1]):
                    raise ValueError("partial reset did not remain isolated to its selected world")
            if not any(witnesses[f"{k}/{v}"]["brightened_18"] for v in VIEWS):
                raise ValueError(f"no fixed receiver shadow cleared in world {k}")
        if any(bindings[k].packs != len(labels) for k in KEYS):
            raise ValueError("two-camera capture repacked published transforms")
        stats = batch.counters
        if stats["completed_steps"] != 80 or stats["accepted_world_steps"] != 160:
            raise RuntimeError("unexpected number of real physics steps")
        final_bytes = _physics_bytes(batch)
        counters = {}
        for k in KEYS:
            baseline, accelerated = (o[k].diagnostics()["device"] for o in (host, shared))
            ext = accelerated["resident_shared_matrices"]
            if (ext["publications"] != len(labels) or ext["captures"] != len(labels)*len(VIEWS)
                    or ext["matrix_d2d_bytes"] != bindings[k].packed_output_bytes
                    or accelerated["resident_host_transform_input_bytes"] != 0
                    or baseline["resident_host_transform_input_bytes"] <= 0):
                raise ValueError(f"unexpected resident pose-transfer counters in world {k}")
            counters[k] = dict(shared=ext, host_transform_input_bytes=baseline["resident_host_transform_input_bytes"],
                               shared_host_transform_input_bytes=accelerated["resident_host_transform_input_bytes"],
                               static_visual_binding_h2d=bindings[k].layout_upload_bytes,
                               gpu_matrix_pack_output_bytes=bindings[k].packed_output_bytes,
                               pack_status_d2h=bindings[k].status_readback_bytes)
        workspace_counts = dict(ws.counters)
    # The first session opened is the last closed by the ExitStack.
    final_device = host[KEYS[0]].diagnostics()["device"]
    cleanup = {name: final_device[name] for name in ("live_resources", "live_contexts", "live_frames", "validation_errors")}
    if any(cleanup.values()):
        raise RuntimeError(f"unclean Vulkan teardown: {cleanup}")
    for k in KEYS:
        final = shared[k].diagnostics()["device"]["resident_shared_matrices"]
        if not final["closed"] or final["imported_cuda_resources"] or final["exports_open"]:
            raise RuntimeError(f"shared CUDA resources survived closing world {k}")
        counters[k]["shared_after_close"] = final
    _sheet(args.output / "live-pose-raster.png", images, labels)
    report = dict(scene=scene.key, render_path="resident software raster/shadows", device=final_device["device_name"],
                  validation_enabled=args.validation, cleanup=cleanup, physics_steps=80, accepted_world_steps=160,
                  world_keys=KEYS, labels=labels, frames=rows, exposures=exposures, motion_and_shadow=witnesses,
                  pre_reset_snapshot_rejected=stale_after_reset,
                  physics_bytes_initial=initial_bytes, physics_bytes_final=final_bytes, workspace=workspace_counts,
                  world_counters=counters, audit_pose_d2h_bytes=audit_pose_bytes,
                  baseline_image_bytes_read_back=host_image_bytes, shared_image_bytes_read_back=shared_image_bytes)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified {len(exposures)} host/shared camera pairs; {args.output}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--library", type=Path, required=True)
    cli.add_argument("--shader-directory", type=Path)
    cli.add_argument("--device-index", type=int, default=0)
    cli.add_argument("--device", type=int, default=0)
    cli.add_argument("--validation", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
