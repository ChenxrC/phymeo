"""Reproduce resident Vulkan ownership, moving shadows, sensors and full Panda.

Explicit device execution only, for example from the checkout root:
  python -B rendering/examples/native_resident_smoke.py --library BUILD/phymeo_native.dll \
      --device-index 1 --output NEW_DIRECTORY --full-panda

Set PYTHONPATH=rendering/src and, when needed, VK_LAYER_PATH to the installed
Khronos validation-layer manifest directory. Uses real GPU raster attachments;
there is no simulation, fallback renderer, image editing or timing threshold.
The optional complete Panda case reuses _panda_scene and therefore needs NumPy.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends.native import NativeBackend, _instance_matrices, _parameters
from phymeo_rendering.backends._native_resident import ResidentDevice, pack_scene
from phymeo_rendering.cameras import Camera
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import (
    DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene,
)

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 640, 480


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _frame(x=0.0, tick=0, key="resident-smoke"):
    nodes = (
        NodePose("fixed", Pose()),
        NodePose("moving", Pose((x, 0.0, 0.0))),
    )
    return SceneFrame(
        FrameKey(key, "resident-smoke-world", 0, tick, tick, "authored"),
        tick * 0.01,
        nodes,
    )

def _request(frame):
    return CaptureRequest(
        Camera("overhead", frame.key, Pose((0.0, 0.0, 1.0625)),
               WIDTH, HEIGHT, 500.0, 500.0, 320.0, 240.0, 0.1, 5.0),
        RenderPath.RASTER, tuple(Channel),
    )


def _sample(data, x, y, fmt):
    return struct.unpack_from(fmt, data, struct.calcsize(fmt) * (y * WIDTH + x))[0]


def _close_clean(owner):
    owner.close()
    info = owner.diagnostics()["device"] if hasattr(owner, "diagnostics") else owner.info()
    keys = ("live_contexts", "live_frames", "live_resources", "validation_errors")
    require(tuple(info[key] for key in keys) == (0, 0, 0, 0), f"resource cleanup: {info}")
    telemetry = info.get("resident_telemetry")
    if telemetry is not None and telemetry.get("final_after_destroy"):
        require(telemetry["device_local_bytes_current"] == 0, "device-local allocation survived close")
        require(telemetry["host_visible_bytes_current"] == 0, "host-visible allocation survived close")
        require(telemetry["allocation_count"] == telemetry["free_count"], "allocation/free imbalance")
    return info


def _scene():
    fixtures = ROOT / "rendering/tests/fixtures/g1"
    return VisualScene(
        "resident-smoke",
        (Mesh("ground", (fixtures / "table.gltf").as_uri()),
         Mesh("block", (fixtures / "block.gltf").as_uri())),
        (Material("matte", (0.45, 0.45, 0.45), 0.0, 0.7),),
        (Instance(1, "ground", "matte", "fixed", Pose((0.0, 0.0, -0.0625))),
         Instance(21, "block", "matte", "moving", Pose((0.0, 0.0, 0.125)))),
        (DirectionalLight((0.6, 0.0, -0.8), (1.0, 1.0, 1.0)),),
    )


def moving_shadow(backend, scene):
    """A fixed receiver must brighten after the same occluder moves to either side.

    An independent pinhole/box calculation gives receiver z=1.125 m and blocker
    top z=0.875 m. The fixed receiver lies outside the blocker primary silhouette,
    while the initial toward-light ray intersects its actual box. The two later
    poses move that box clear of the same ray. No accepted pixel is synthesized.
    """
    session = backend.open_scene(scene)
    result = {"frames": []}
    try:
        for tick, x in enumerate((0.0, 0.35, -0.35)):
            frame = _frame(x, tick)
            captured = session.capture(frame, _request(frame))
            data = {plane.channel: plane.data for plane in captured.planes}
            require(_sample(data[Channel.INSTANCE_ID], 400, 240, "<I") == 1, "receiver ID")
            require(abs(_sample(data[Channel.FORWARD_DEPTH], 400, 240, "<f") - 1.125) < 2e-6,
                    "receiver depth")
            block_x = round(320 + x * 500 / 0.875)
            require(_sample(data[Channel.INSTANCE_ID], block_x, 240, "<I") == 21, "blocker ID")
            require(abs(_sample(data[Channel.FORWARD_DEPTH], block_x, 240, "<f") - 0.875) < 2e-6,
                    "blocker depth")
            require(data[Channel.VALID][240 * WIDTH + 400] == 1, "receiver valid")
            offset = (240 * WIDTH + 400) * 3
            info = session.diagnostics()["device"]
            require(info["resident_loaded"] == 1 and info["live_resources"] > 0,
                    "resident geometry was released before drawing")
            result["frames"].append({
                "occluder_x_m": x, "receiver_rgb": list(data[Channel.RGB][offset:offset + 3]),
                "planes_sha256": {key.value: hashlib.sha256(value).hexdigest() for key, value in data.items()},
                "device": info,
            })
        brightness = [sum(row["receiver_rgb"]) for row in result["frames"]]
        require(brightness[0] == 0 and brightness[1] > brightness[0] + 100
                and brightness[2] > brightness[0] + 100, f"moving shadow did not clear: {brightness}")
        infos = [row["device"] for row in result["frames"]]
        require(all(info["resident_upload_calls"] == 1 for info in infos), "static geometry reuploaded")
        require(len({info["resident_geometry_upload_bytes"] for info in infos}) == 1,
                "geometry bytes changed across motion")
        require(infos[-1]["resident_target_rebuilds"] == 1, "same-size targets were not retained")
        require(infos[-1]["resident_transform_upload_bytes"] == 3 * 2 * 128, "dynamic upload bytes")
        require(infos[-1]["resident_host_transform_input_bytes"] == 3 * 2 * 64, "host matrix bytes")
    finally:
        result["closed"] = _close_clean(session)
    return result


def empty_capture(backend):
    scene = VisualScene("resident-empty", (), (), ())
    session = backend.open_scene(scene)
    result = {}
    try:
        frame = _frame(key=scene.key)
        data = {plane.channel: plane.data for plane in session.capture(frame, _request(frame)).planes}
        for channel in (Channel.RGB, Channel.INSTANCE_ID, Channel.VALID):
            require(not any(data[channel]), f"nonzero empty {channel.value}")
        require(all(math.isinf(value[0]) and value[0] > 0
                    for value in struct.iter_unpack("<f", data[Channel.FORWARD_DEPTH])),
                "empty depth must be positive infinity")
        result["all_empty_channels_checked"] = True
    finally:
        result["closed"] = _close_clean(session)
    return result


def borrowed_host_lifetime(library, device_index, scene):
    """Destroy every caller upload array before the first actual GPU draw."""
    decoded = {mesh.id: load_mesh(mesh) for mesh in scene.meshes}
    packed = pack_scene(scene, decoded)
    device = ResidentDevice(library, library.parent, device_index, True)
    result = {}
    try:
        device.upload(*packed)
        for array in packed:
            ct.memset(ct.addressof(array), 0, ct.sizeof(array))
        del packed
        gc.collect()
        frame = _frame()
        triangles = sum(len(decoded[item.mesh_id].triangles) for item in scene.instances)
        params = _parameters(scene, _request(frame), triangles, 1.0)
        params.counts[3] = len(scene.instances)
        raw = device.render(_instance_matrices(scene, frame), params, WIDTH, HEIGHT)
        pixels = WIDTH * HEIGHT
        ids, depth = raw[pixels * 8:pixels * 12], raw[pixels * 4:pixels * 8]
        require(_sample(ids, 400, 240, "<I") == 1 and _sample(ids, 320, 240, "<I") == 21,
                "caller array mutation corrupted owned geometry")
        require(abs(_sample(depth, 400, 240, "<f") - 1.125) < 2e-6, "owned geometry depth")
        result["host_arrays_zeroed_freed_and_collected_before_draw"] = True
    finally:
        result["closed"] = _close_clean(device)
    return result


def full_panda(backend, directory):
    from _panda_scene import build_scene, cameras
    from phymeo_rendering.adapters.panda_replay import PandaReplaySource

    fixtures = ROOT / "rendering/tests/fixtures/panda_replay"
    assets = ROOT / "rendering/assets/panda"
    initial = PandaReplaySource(VisualScene("input", (), (), ()),
                                fixtures / "panda.urdf", fixtures / "trace.json")
    geometry = initial.provenance(next(initial.frames()))["geometry"]
    directory.mkdir()
    scene = build_scene(assets, directory, geometry["cube_full_size_m"], geometry["table_full_size_m"])
    source = PandaReplaySource(scene, fixtures / "panda.urdf", fixtures / "trace.json")
    before = time.perf_counter()
    session = backend.open_scene(scene)
    result = {"open_wall_s": time.perf_counter() - before, "frames": []}
    try:
        allowed_ids = {0} | {instance.id for instance in scene.instances}
        for frame in source.frames():
            original = cameras(frame)[0]
            camera = replace(
                original, width=320, height=240, fx_px=original.fx_px / 3,
                fy_px=original.fy_px / 3, cx_px=original.cx_px / 3, cy_px=original.cy_px / 3,
            )
            before = time.perf_counter()
            captured = session.capture(frame, CaptureRequest(camera, RenderPath.RASTER, tuple(Channel)))
            elapsed = time.perf_counter() - before
            data = {plane.channel: plane.data for plane in captured.planes}
            observed = {value[0] for value in struct.iter_unpack("<I", data[Channel.INSTANCE_ID])}
            require(observed <= allowed_ids, "full Panda emitted an undeclared ID")
            require(any(100 <= identity <= 158 for identity in observed), "full Panda is not visible")
            result["frames"].append({
                "source_time_s": frame.time_s, "capture_wall_s": elapsed,
                "visible_ids": sorted(observed),
                "planes_sha256": {key.value: hashlib.sha256(value).hexdigest() for key, value in data.items()},
            })
        info = session.diagnostics()["device"]
        require(info["resident_instanced_triangles"] == 134960, "complete Panda triangle count")
        require(info["resident_upload_calls"] == 1, "complete Panda was reuploaded")
        result["live"] = info
    finally:
        result["closed"] = _close_clean(session)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-panda", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = args.library.resolve()
    report = {"schema": "phymeo-native-resident-smoke-v1", "status": "running",
              "library_path": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "shader_sha256": {name: hashlib.sha256((library.parent / name).read_bytes()).hexdigest()
                                for name in ("resident.vert.spv", "resident.frag.spv")},
              "device_index": args.device_index}
    backend = NativeBackend(library_path=library, shader_directory=library.parent,
                            device_index=args.device_index, validation=True, resident=True)
    scene = _scene()
    try:
        report["moving_shadow"] = moving_shadow(backend, scene)
        report["empty"] = empty_capture(backend)
        report["host_pointer_lifetime"] = borrowed_host_lifetime(library, args.device_index, scene)
        if args.full_panda:
            report["full_panda"] = full_panda(backend, output / "geometry")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "receipt.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8",
        )
    print(json.dumps({"status": report["status"], "receipt": str(output / "receipt.json")}, indent=2))


if __name__ == "__main__":
    main()
