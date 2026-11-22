"""Render and encode the archived Panda grasp with a paired lighting control."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from _panda_motion import PandaMotionSource
from _panda_scene import build_scene, cameras, look_at
from phymeo_rendering.assets import Material
from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import VisualScene
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/panda_replay"
MOTION = ROOT / "tests/fixtures/panda_motion/trajectory.json"
WIDTH, HEIGHT = 768, 576
PASSES = {
    "single": ("single",),
    "studio": ("studio",),
    "materials": ("materials-wide", "materials-detail"),
    "copper": ("copper-detail",),
}


def movie_camera(frame, name, phase):
    if name in ("single", "studio"):
        fixed = cameras(frame)[0]
        scale = WIDTH / fixed.width
        return replace(
            fixed,
            width=WIDTH,
            height=HEIGHT,
            fx_px=scale * fixed.fx_px,
            fy_px=scale * fixed.fy_px,
            cx_px=scale * fixed.cx_px,
            cy_px=scale * fixed.cy_px,
        )
    # The second film presents two views of the same archive. Smooth camera
    # travel reveals view-dependent highlights; it changes no object pose.
    eased = phase * phase * (3 - 2 * phase)
    delta = math.radians(-7 + 17 * eased)
    axis = (0.12, 0.0)
    dx, dy = 1.27 - axis[0], -0.86
    eye = (
        axis[0] + dx * math.cos(delta) - dy * math.sin(delta),
        dx * math.sin(delta) + dy * math.cos(delta),
        0.79,
    )
    if name == "materials-wide":
        target, focal = (0.12, 0, 0.415), 895
    else:
        cube = next(n.world_pose.position_m for n in frame.nodes if n.node_id == "cube")
        target = (cube[0], cube[1], cube[2] + 0.034)
        focal = 2250
    camera_id = "materials-detail" if name == "copper-detail" else name
    return Camera(
        camera_id,
        frame.key,
        look_at(eye, target, (0, 0, 1)),
        WIDTH,
        HEIGHT,
        focal,
        focal,
        WIDTH / 2,
        HEIGHT / 2,
        0.005,
        5.0,
    )


def write_json(path, value):
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    )


def render_pass(args):
    stage = args.pass_name
    if args.motion.resolve() == MOTION:
        expected = MOTION.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    else:
        expected = args.subset_sha256

    def source_for(scene):
        return PandaMotionSource(
            scene,
            FIXTURE / "panda.urdf",
            args.motion,
            sample_hz=48,
            playback_fps=24,
            expected_subset_sha256=expected,
        )

    source_input = source_for(VisualScene("panda-movie-input", (), (), ()))
    geometry = source_input.geometry
    root = args.work
    root.mkdir(parents=True, exist_ok=True)
    destinations = {name: root / name for name in PASSES[stage]}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError(f"Use a fresh {stage} output directory")
    assets = ROOT / "assets/panda"
    meshes = root / ("geometry-" + stage)
    meshes.mkdir()
    scene = build_scene(assets, meshes, geometry["cube_full_size_m"], geometry["table_full_size_m"])
    if stage in ("materials", "copper"):
        from _panda_lookdev import lookdev_scene

        scene = lookdev_scene(scene, meshes)
    if stage == "copper":
        finish = Material("lookdev-polished-copper", (0.93, 0.61, 0.47), 1.0, 0.18)
        scene = replace(
            scene,
            key=scene.key + ":copper",
            materials=(*scene.materials, finish),
            instances=tuple(
                replace(part, material_id=finish.id) if part.node_id == "cube" else part
                for part in scene.instances
            ),
        )
    for dest in destinations.values():
        dest.mkdir()
    source = source_for(scene)
    frames = tuple(source.frames())
    studio = None if stage == "single" else SapienStudio(center_m=(0.18, 0, 0.42), scale_m=0.85)
    path = RenderPath.PATH_TRACE
    backend = SapienBackend(
        SapienConfig(device=args.device, path=path, studio=studio, samples_per_pixel=args.spp)
    )
    session = backend.open_scene(scene)
    rows = []
    try:
        for i, frame in enumerate(frames):
            row = {"record": frame.key.revision, "time_s": frame.time_s, "captures": {}}
            for name, dest in destinations.items():
                cam = movie_camera(frame, name, i / (len(frames) - 1))
                capture = session.capture(frame, CaptureRequest(cam, path, (Channel.RGB,)))
                raw = capture.planes[0].data
                filename = dest / f"{i:04d}.png"
                Image.frombytes("RGB", (WIDTH, HEIGHT), raw).save(filename, compress_level=2)
                row["captures"][name] = {
                    "camera": asdict(cam),
                    "rgb_sha256": hashlib.sha256(raw).hexdigest(),
                }
            rows.append(row)
            if (i + 1) % 18 == 0 or i + 1 == len(frames):
                print(f"{stage}: {i + 1}/{len(frames)} archive samples", flush=True)
    finally:
        session.close()
    diag = session.diagnostics()
    if diag["physics_steps"] != 0 or diag["lifecycle"]["completed_captures"] != len(rows) * len(
        destinations
    ):
        raise RuntimeError("Incomplete pure-render capture")
    scene_info = asdict(scene)
    for mesh in scene_info["meshes"]:
        mesh["uri"] = mesh["uri"].rsplit("/", 1)[-1]
    write_json(
        root / f"{stage}-capture.json",
        {
            "scene": scene_info,
            "source": source.timeline(),
            "device": diag,
            "frames": rows,
            "manufacturer_manifest_sha256": hashlib.sha256(
                (assets / "manifest.json").read_bytes()
            ).hexdigest(),
            "procedural_mesh_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in meshes.glob("*.gltf")
            },
        },
    )


def encode(args):
    from _panda_movie_encode import encode_movies

    captures = {
        stage: json.loads((args.work / f"{stage}-capture.json").read_bytes()) for stage in PASSES
    }
    single, studio, materials, copper = (captures[key] for key in PASSES)
    render_settings = (
        "samples_per_pixel",
        "max_bounces",
        "denoiser",
        "device_name",
        "effective_path",
    )
    if any(
        {k: c["device"].get(k) for k in render_settings}
        != {k: studio["device"].get(k) for k in render_settings}
        for c in captures.values()
    ):
        raise ValueError("Movies require matched renderer and sampling settings")
    if single["scene"] != studio["scene"] or single["source"] != studio["source"]:
        raise ValueError("Lighting comparison requires identical geometry, materials and motion")
    if any(len(c["frames"]) != 108 for c in captures.values()):
        raise ValueError("Expected 108 distinct captured poses in each pass")
    if materials["procedural_mesh_sha256"] != copper["procedural_mesh_sha256"]:
        raise ValueError("Payload finishes require identical geometry")
    red_scene, copper_scene = materials["scene"], copper["scene"]
    if (
        red_scene["meshes"] != copper_scene["meshes"]
        or red_scene["lights"] != copper_scene["lights"]
        or red_scene["materials"] != copper_scene["materials"][:-1]
        or materials["device"]["studio"] != copper["device"]["studio"]
    ):
        raise ValueError("A property outside the payload finish changed")
    changed = [
        (a, b)
        for a, b in zip(red_scene["instances"], copper_scene["instances"], strict=True)
        if a != b
    ]
    if (
        len(changed) != 1
        or changed[0][0]["node_id"] != "cube"
        or changed[0][1] != {**changed[0][0], "material_id": "lookdev-polished-copper"}
    ):
        raise ValueError("Only the archived payload should use the alternate finish")
    for a, b, c, d in zip(
        single["frames"], studio["frames"], materials["frames"], copper["frames"], strict=True
    ):
        if a["captures"]["single"]["camera"] != b["captures"]["studio"]["camera"]:
            raise ValueError("Paired lighting camera or frame key diverged")
        if len({(r["record"], r["time_s"]) for r in (a, b, c, d)}) != 1:
            raise ValueError("Material film diverged from archived timeline")
        red_camera = c["captures"]["materials-detail"]["camera"]
        metal_camera = d["captures"]["copper-detail"]["camera"]
        if {
            **red_camera,
            "frame_key": {**red_camera["frame_key"], "scene_key": copper_scene["key"]},
        } != metal_camera:
            raise ValueError("Payload finish comparison camera or archive frame diverged")
    for capture in captures.values():
        for index, exposure in enumerate(capture["frames"]):
            for view, measurement in exposure["captures"].items():
                with Image.open(args.work / view / f"{index:04d}.png") as image:
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest != measurement["rgb_sha256"]:
                    raise ValueError(f"Image differs from GPU readback at {view} frame {index}")
    receipt = encode_movies(args.work, args.output, 24, 108)
    write_json(
        args.output / "recording.json",
        {
            "source": single["source"],
            "encoded": receipt,
            "archive_world": single["frames"][0]["captures"]["single"]["camera"]["frame_key"][
                "world_id"
            ],
            "manufacturer_manifest_sha256": studio["manufacturer_manifest_sha256"],
            "renderer": {k: studio["device"].get(k) for k in render_settings},
            "measurements": {
                k: {
                    "geometry": c["device"]["geometry"],
                    "physics_steps": c["device"]["physics_steps"],
                    "exposures": c["device"]["lifecycle"]["completed_captures"],
                    "capture_manifest_sha256": hashlib.sha256(
                        (args.work / f"{k}-capture.json").read_bytes()
                    ).hexdigest(),
                }
                for k, c in captures.items()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest="command", required=True)
    capture = jobs.add_parser("render")
    capture.add_argument("--pass", choices=PASSES, dest="pass_name", required=True)
    capture.add_argument("--motion", type=Path, default=MOTION)
    capture.add_argument("--subset-sha256")
    capture.add_argument("--work", type=Path, required=True)
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--spp", type=int, default=256)
    video = jobs.add_parser("encode")
    video.add_argument("--work", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        render_pass(args)
    else:
        encode(args)


if __name__ == "__main__":
    main()
