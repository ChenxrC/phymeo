"""Render and audit two archived Panda grasp endpoints. See results/r1/README.md."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from _panda_anchors import freeze_anchors, measure, reference_links
from _panda_scene import build_scene, cameras, marker, pose_matrix, unit
from _panda_wireframe import draw_collision_overlay
from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.assets import Material
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene
from PIL import Image, ImageDraw


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def subset(scene, key, instances):
    return replace(
        scene,
        key=key,
        instances=tuple(instances),
        meshes=tuple(m for m in scene.meshes if any(i.mesh_id == m.id for i in instances)),
        materials=tuple(
            m for m in scene.materials if any(i.material_id == m.id for i in instances)
        ),
    )


def expect_failure(action, error):
    try:
        action()
    except error as exc:
        return str(exc)
    raise AssertionError("An invalid input unexpectedly succeeded")


def render_set(backend, source, path, angles, dst, prefix, sensors, retain_raw=True):
    """Save real public captures and enough metadata to reproduce every exposure."""
    captures, evidence = {}, []
    session = backend.open_scene(source.scene())
    try:
        for i, frame in enumerate(source.frames()):
            before = source.provenance(frame)
            for camera in cameras(frame):
                if camera.id not in angles:
                    continue
                channels = tuple(Channel) if sensors else (Channel.RGB,)
                request = CaptureRequest(camera, path, channels)
                if not session.supports(request):
                    raise UnsupportedRequestError(f"{prefix} cannot fulfill {camera.id}")
                # The consumer must reject a frame from another world/revision.
                failures = {}
                for field in ("world_id", "revision"):
                    value = "wrong-archive" if field == "world_id" else frame.key.revision + 1
                    bad = replace(frame, key=replace(frame.key, **{field: value}))
                    failures[field] = expect_failure(
                        lambda: session.capture(bad, request), ValueError
                    )
                cap = session.capture(frame, request)
                planes = {p.channel: p.data for p in cap.planes}
                name = f"{prefix}-{i + 1}-{camera.id}"
                image = Image.frombytes("RGB", (camera.width, camera.height), planes[Channel.RGB])
                image.save(dst / (name + ".png"))
                if retain_raw:
                    for channel, raw in planes.items():
                        if channel is not Channel.RGB:
                            (dst / (name + "-" + channel.value + ".raw")).write_bytes(raw)
                evidence.append(
                    dict(
                        name=name,
                        camera=asdict(camera),
                        source_record=before["record_index"],
                        time_s=frame.time_s,
                        negative_controls=failures,
                        channel_sha256={
                            k.value: hashlib.sha256(v).hexdigest() for k, v in planes.items()
                        },
                    )
                )
                captures[i, camera.id] = planes
            if source.provenance(frame) != before:
                raise AssertionError("Exposure changed its archived input")
    finally:
        session.close()
    save_json(
        dst / (prefix + "-capture.json"),
        dict(
            exposures=evidence,
            device=session.diagnostics(),
            lights=[asdict(light) for light in source.scene().lights],
        ),
    )
    return captures


def id_plane(data, camera):
    return np.frombuffer(data[Channel.INSTANCE_ID], "<u4").reshape(camera.height, camera.width)


def main(args):
    root = Path(__file__).resolve().parents[2]
    fixture = root / "rendering/tests/fixtures/panda_replay"
    urdf = fixture / "panda.urdf"
    excerpt = fixture / "trace.json"
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    geo = out / "geometry"
    geo.mkdir()

    def replay(scene):
        return PandaReplaySource(scene, urdf, excerpt)

    empty = replay(VisualScene("panda-r1-input", (), (), ()))
    inputs = empty.provenance(next(empty.frames()))["geometry"]
    scene = build_scene(args.assets, geo, inputs["cube_full_size_m"], inputs["table_full_size_m"])
    # Keep both the source manifest binding and the consumed render declaration.
    # Paths are converted to portable asset names; all generated metre-scale
    # meshes are also saved next to this evidence.
    manifest = (args.assets / "manifest.json").read_bytes()
    static = asdict(scene)
    for mesh in static["meshes"]:
        mesh["uri"] = mesh["uri"].rsplit("/", 1)[-1]
    static["panda_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    static["prepared_counts"] = json.loads(manifest)["totals"]
    static["authored_geometry_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in geo.glob("*.gltf")
    }
    save_json(out / "scene.json", static)
    source = replay(scene)
    frames = list(source.frames())
    fixed = [cameras(f)[0] for f in frames]
    refs = []
    for f in frames:
        record = source.provenance(f)["record"]
        links = reference_links(record["q"])
        links["table"] = pose_matrix(Pose(tuple(inputs["table_center_m"])))
        links["cube"] = pose_matrix(Pose(tuple(record["cube"]), tuple(record["xyzw"])))
        observed = {node.node_id: node.world_pose for node in f.nodes}
        for name, matrix in links.items():
            np.testing.assert_allclose(matrix, pose_matrix(observed[name]), atol=2e-12, rtol=0)
        refs.append(links)
    anchors = freeze_anchors(scene, fixed, refs)
    save_json(out / "source-anchors.json", anchors)
    print(
        "Froze visible source triangles:",
        [(a["name"], a["owner"], a["triangle"]) for a in anchors],
        flush=True,
    )

    def sapien(path, studio=None):
        return SapienBackend(
            SapienConfig(path=path, device=args.device, samples_per_pixel=args.spp, studio=studio)
        )

    raster, pt = RenderPath.RASTER, RenderPath.PATH_TRACE
    broken = subset(scene, "panda-r1-missing-asset", [scene.instances[0]])
    missing = replace(broken.meshes[0], uri=(geo / "deliberately-missing.gltf").resolve().as_uri())
    missing_error = expect_failure(
        lambda: sapien(raster).open_scene(replace(broken, meshes=(missing,))),
        (BackendUnavailableError, FileNotFoundError),
    )
    if "deliberately-missing.gltf" not in missing_error:
        raise AssertionError(
            "Missing-asset control failed for an unrelated reason: " + missing_error
        )
    studio = SapienStudio(center_m=(0.18, 0, 0.42), scale_m=0.85)
    for name, lighting in (("single", None), ("studio", studio)):
        angles = {"fixed"} if lighting is None else {"fixed", "detail"}
        render_set(sapien(pt, lighting), source, pt, angles, out, name, False)
        print("Captured both poses:", name, flush=True)
    fill = DirectionalLight(tuple(unit((-0.6, 0.8, -0.65))), (1.35, 1.35, 1.35))
    # This sensor rig uses one forward-facing key on both raster backends.
    sensor_scene = replace(scene, key="panda-r1-raster-fill-v1", lights=(fill,))
    sensors = render_set(
        sapien(raster), replay(sensor_scene), raster, {"fixed", "wrist"}, out, "raster", True
    )
    print("Captured both sensor cameras", flush=True)

    markers = marker(geo)
    diag_scene = VisualScene(
        "panda-r1-material-fiducials",
        (markers,),
        (Material("tag", (0.5, 0.5, 0.5), 0, 0.8),),
        tuple(
            Instance(10 + i, markers.id, "tag", a["node"], Pose(tuple(a["node_point"])))
            for i, a in enumerate(anchors)
        ),
        scene.lights,
    )
    tags = render_set(sapien(raster), replay(diag_scene), raster, {"fixed"}, out, "fiducials", True)
    reports = []
    wrist_observation = []
    lighting_observation = []
    for i, (frame, camera) in enumerate(zip(frames, fixed, strict=True)):
        data = sensors[i, "fixed"]
        z = np.frombuffer(data[Channel.FORWARD_DEPTH], "<f4").reshape(camera.height, camera.width)
        measured = measure(
            anchors, i, id_plane(data, camera), z, id_plane(tags[i, "fixed"], camera)
        )
        reports.append(measured)
        robot = id_plane(data, camera) >= 100
        luma = []
        for name in ("single", "studio"):
            with Image.open(out / f"{name}-{i + 1}-fixed.png") as image:
                samples = np.asarray(image)[robot] @ np.array([0.2126, 0.7152, 0.0722])
            luma.append(
                dict(
                    path=name,
                    median_srgb8=float(np.median(samples)),
                    near_black_percent=float(100 * np.mean(samples <= 16)),
                    near_white_percent=float(100 * np.mean(samples >= 250)),
                )
            )
        lighting_observation.append(dict(robot_pixels=int(robot.sum()), levels=luma))
        wrist = cameras(frame)[1]
        wrist_ids = id_plane(sensors[i, "wrist"], wrist)
        counts = {
            label: int(np.isin(wrist_ids, ids).sum())
            for label, ids in {
                "payload": [1],
                "left_finger": [j.id for j in scene.instances if j.node_id == "panda_leftfinger"],
                "right_finger": [j.id for j in scene.instances if j.node_id == "panda_rightfinger"],
            }.items()
        }
        if min(counts.values()) == 0:
            raise AssertionError(f"Missing task object in wrist image: {counts}")
        wrist_observation.append(counts)
        with Image.open(out / f"studio-{i + 1}-fixed.png") as im:
            overlay = draw_collision_overlay(im, urdf, frame, camera)
            overlay.save(out / f"collision-{i + 1}.png")
            annotated = im.copy()
        draw = ImageDraw.Draw(annotated)
        for m in measured:
            x, y = m["observed_centroid"]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 183, 79), width=2)
            draw.text((x + 7, y - 8), m["name"], fill="white", stroke_width=2, stroke_fill="black")
        annotated.save(out / f"anchors-{i + 1}.png")

    result = dict(
        provenance=[source.provenance(f) for f in frames],
        anchors=reports,
        studio=asdict(studio),
        wrist_pixels=wrist_observation,
        lighting=lighting_observation,
        missing_asset_error=missing_error,
        visual_declaration_sha256=hashlib.sha256((out / "scene.json").read_bytes()).hexdigest(),
        pt_samples_per_pixel=args.spp,
        zero_physics_steps=True,
    )
    if args.native_library:
        native = NativeBackend(
            library_path=args.native_library,
            shader_directory=args.native_library.parent,
            device_index=args.native_index,
            validation=True,
        )
        result["native_full_scene"] = expect_failure(
            lambda: native.open_scene(sensor_scene), UnsupportedRequestError
        )
        local = subset(
            sensor_scene,
            "panda-r1-native-fingers-and-payload",
            [
                i
                for i in sensor_scene.instances
                if i.id in (1, 2) or i.node_id in ("panda_leftfinger", "panda_rightfinger")
            ],
        )
        render_set(native, replay(local), raster, {"wrist"}, out, "native-detail", True)
    save_json(out / "summary.json", result)
    print(
        "Max observed anchor error (px):",
        max(m["error_px"] for group in reports for m in group),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spp", default=512, type=int)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-index", default=0, type=int)
    main(parser.parse_args())
