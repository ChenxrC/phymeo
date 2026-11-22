"""Shared, opt-in device acceptance on the fixed G1 table."""

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zlib

from phymeo_rendering.adapters import AuthoredSceneSource
from phymeo_rendering.backends import SessionClosedError, UnsupportedRequestError
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, Pose

from ._g1_table import FIXTURES, table_frames, table_scene


# Pixel-centre probes from the 500 px pinhole and known box heights (metres).
# (468, 231) sees the elevated block in front of the rubber sample.
PROBES = ((1, 1, 0, math.inf), (120, 160, 1, 1.125),
          (195, 240, 11, 1.0), (320, 240, 12, 1.0), (445, 240, 13, 1.0),
          (468, 231, 21, 0.875))
SHADOW = (368, 232, 380, 248)
CONTROL = (260, 232, 272, 248)


def write_png(path, width, height, rgb):
    """Lossless RGB8 preview. The device's original bytes are retained separately."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[i:i + stride] for i in range(0, len(rgb), stride))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def region(rgb, box):
    l, t, r, b = box
    return [rgb[(y * 640 + x)*3:(y * 640 + x)*3 + 3]
            for y in range(t, b) for x in range(l, r)]


def mean_brightness(values):
    return sum(sum(pixel) for pixel in values) / (3 * len(values))


class FrozenTableSource(AuthoredSceneSource):
    def __init__(self, scene, frames):
        super().__init__(scene, frames)
        self.steps = 0

    def step(self):
        self.steps += 1
        raise AssertionError("A rendering consumer advanced its source")


def snapshot(scene, frame):
    encoded = json.dumps([asdict(scene), asdict(frame)], sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def exercise_table(test, factory, label, path, *, sensors):
    folder = Path(os.environ["PHYMEO_R0_OUTPUT"]) / label
    folder.mkdir(parents=True, exist_ok=False)
    reference_scene = table_scene()
    frames = table_frames()
    channels = tuple(Channel) if sensors else (Channel.RGB,)
    report = {"path": path.value, "channels": [ch.value for ch in channels],
              "assets": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                         for file in (FIXTURES / "table.gltf", FIXTURES / "block.gltf")},
              "samples": [], "shadow_roi": SHADOW, "control_roi": CONTROL}
    rgb_pair = []
    sensor_pair = []
    for lighting, scene in (
        ("overhead", reference_scene),
        ("inclined", replace(reference_scene, key="g3-table-inclined",
                             lights=(DirectionalLight((0.6, 0.0, -0.8), (1., 1., 1.)),))),
    ):
        source = FrozenTableSource(scene, (replace(f, key=replace(f.key, scene_key=scene.key)) for f in frames))
        frame, foreign = source.frames()
        digest = snapshot(source.scene(), frame)
        camera = Camera("table-front", frame.key, Pose((0., 0., 1.0625)), 640, 480, 500., 500., 320., 240., 0.1, 5.)
        request = CaptureRequest(camera, path, channels)
        backend = factory()
        session = backend.open_scene(source.scene())
        try:
            test.assertTrue(session.supports(request))
            with test.assertRaises(ValueError):
                session.capture(foreign, request)
            other_path = RenderPath.PATH_TRACE if path is RenderPath.RASTER else RenderPath.RASTER
            wrong = replace(request, path=other_path)
            test.assertFalse(session.supports(wrong))
            with test.assertRaises(UnsupportedRequestError):
                session.capture(frame, wrong)
            capture = session.capture(frame, request)
        finally:
            session.close()
        with test.assertRaises(SessionClosedError):
            session.capture(frame, request)
        test.assertEqual(source.steps, 0)
        test.assertEqual(snapshot(source.scene(), frame), digest)
        sample = {"lighting": lighting, "snapshot_sha256": digest, "source_steps": source.steps,
                  "camera": asdict(camera), "light": asdict(scene.lights[0]), "time_s": frame.time_s,
                  "planes": {}, "probes": []}
        if label == "native-raster":
            sample["device"] = session.diagnostics()
        else:
            sample["device"] = backend.last_open_diagnostics
        for plane in capture.planes:
            name = f"{lighting}-{plane.channel.value}.raw"
            (folder / name).write_bytes(plane.data)
            sample["planes"][name] = hashlib.sha256(plane.data).hexdigest()
        planes = {p.channel: p.data for p in capture.planes}
        rgb = planes[Channel.RGB]
        write_png(folder / f"{lighting}.png", 640, 480, rgb)
        rgb_pair.append(rgb)
        if sensors:
            sensor_pair.append(tuple(planes[ch] for ch in (Channel.INSTANCE_ID, Channel.VALID, Channel.FORWARD_DEPTH)))
            for u, v, ident, z in PROBES:
                i = v * 640 + u
                observed_id = struct.unpack_from("<I", planes[Channel.INSTANCE_ID], 4*i)[0]
                depth = struct.unpack_from("<f", planes[Channel.FORWARD_DEPTH], 4*i)[0]
                valid = planes[Channel.VALID][i]
                sample["probes"].append([u, v, observed_id, depth if math.isfinite(depth) else "inf", valid])
        report["samples"].append(sample)
        (folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    before, after = (region(rgb, SHADOW) for rgb in rgb_pair)
    control0, control1 = (region(rgb, CONTROL) for rgb in rgb_pair)
    changed = sum(any(abs(a-b) >= 2 for a, b in zip(p, q)) for p, q in zip(before, after))
    shadow_drop = mean_brightness(before) - mean_brightness(after)
    control_drop = mean_brightness(control0) - mean_brightness(control1)
    report.update(changed_shadow_pixels=changed, shadow_pixels=len(before),
                  shadow_drop_srgb8=shadow_drop, control_drop_srgb8=control_drop)
    (folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test.assertGreaterEqual(changed, 32, label)
    test.assertGreater(shadow_drop, max(0., control_drop), label)
    if sensors:
        test.assertEqual(sensor_pair[0], sensor_pair[1], "Lighting changed geometric visibility")
        for sample in report["samples"]:
            for actual, (_, _, ident, depth) in zip(sample["probes"], PROBES, strict=True):
                test.assertEqual((actual[2], actual[4]), (ident, int(ident != 0)), actual)
                if ident:
                    test.assertAlmostEqual(actual[3], depth, delta=0.002)
                else:
                    test.assertEqual(actual[3], "inf")
