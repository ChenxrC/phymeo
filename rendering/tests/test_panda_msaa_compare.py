"""HOST contract tests for one frozen Panda AA comparison; no device claims."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))


class FilmSampleTests(unittest.TestCase):
    def setUp(self):
        self.film = importlib.import_module("panda_realtime_film")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def arguments(self, **changes):
        values = dict(mode="native", output=self.work / "capture", excerpt=self.work / "trace.json",
            surface_roughness=False, photographic=False, reference_seed=None, area_light_grid=2,
            daylight=False, native_studio=False, tiles=1, workshop=False, resolution="2p",
            display_transform="clip", shadow_mode="software", library=None, device_index=1,
            validation=False, device="cuda:0", samples=64, camera="fixed", stride=1, frame=None,
            light_strength=.5, light_yaw=25., cube_size=256)
        values.update(changes)
        values["excerpt"].write_text("{}", encoding="utf-8")
        return SimpleNamespace(**values)

    def capture_through_boundary(self, args, *, old_constructor=False):
        # A CPU stand-in is only used at the unavailable hardware boundary.
        # It proves film argument forwarding and metadata, not rendering quality.
        from phymeo_rendering.cameras import Camera
        from phymeo_rendering.scene import FrameKey, Pose, SceneFrame, VisualScene
        scene = VisualScene("host-film", (), (), ())
        frame = SceneFrame(FrameKey(scene.key, "archive", 0, 0, 0, "authored"), 1 / 30, ())
        camera = Camera("host", frame.key, Pose(), 3, 2, 2., 2., 1.5, 1., .1, 10.)
        seen = {}
        class Session:
            def capture(self, actual, request):
                return SimpleNamespace(frame_key=actual.key, time_s=actual.time_s,
                    camera=request.camera, planes=(SimpleNamespace(data=bytes(18)),))
            def close(self):
                pass
            def diagnostics(self):
                return {"physics_steps": 0, "device": {"resident_upload_calls": 1,
                    "resident_captures": 1}}
        class Backend:
            def open_scene(self, actual):
                return Session()
        def old(*, library_path, resident, device_index, validation, exposure):
            seen.update(library_path=library_path, resident=resident, device_index=device_index,
                        validation=validation, exposure=exposure)
            return Backend()
        def current(**kwargs):
            seen.update(kwargs)
            return Backend()
        with patch.object(self.film, "NativeBackend", old if old_constructor else current), \
             patch.object(self.film, "load_episode", return_value=(scene, (frame,))), \
             patch.object(self.film, "fixed", return_value=camera):
            self.film.capture(args)
        return seen, json.loads((args.output / "capture.json").read_text())

    def test_old_namespace_still_calls_old_one_sample_constructor(self):
        seen, receipt = self.capture_through_boundary(self.arguments(), old_constructor=True)
        self.assertNotIn("msaa_samples", seen)
        self.assertEqual(receipt["msaa_samples"], 1)

    def test_explicit_one_sample_does_not_require_new_backend_keyword(self):
        seen, receipt = self.capture_through_boundary(self.arguments(msaa_samples=1), old_constructor=True)
        self.assertNotIn("msaa_samples", seen)
        self.assertEqual(receipt["msaa_samples"], 1)

    def test_four_samples_reaches_real_backend_constructor_argument(self):
        seen, receipt = self.capture_through_boundary(self.arguments(msaa_samples=4))
        self.assertEqual(seen["msaa_samples"], 4)
        self.assertEqual(receipt["msaa_samples"], 4)

    def test_invalid_msaa_rejects_before_creating_output_or_loading_assets(self):
        for value in (True, False, 0, 2, 8, 4., "4", None):
            with self.subTest(value=value):
                args = self.arguments(msaa_samples=value)
                with self.assertRaisesRegex((TypeError, ValueError), "msaa_samples"):
                    self.film.capture(args)
                self.assertFalse(args.output.exists())

    def test_four_samples_rejects_reference_instead_of_silently_ignoring(self):
        args = self.arguments(mode="studio", msaa_samples=4)
        with self.assertRaisesRegex(ValueError, "native|MSAA|msaa_samples"):
            self.film.capture(args)
        self.assertFalse(args.output.exists())


class PairedFilmTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("panda_msaa_compare")

    def protocol_and_receipts(self):
        protocol = {"schema": "phymeo-panda-msaa-protocol-v1",
            "profile": self.module.frozen_profile(), "library_sha256": "a" * 64,
            "input_binding": {"scene_sha256": "b" * 64, "source_excerpt_sha256": "c" * 64,
                "frames": [{"index": i, "time_s": (i + 1) / 30,
                    "state_sha256": f"{i:064x}", "camera": {"intrinsics": [900., 900., 640., 360.],
                    "pose": [0., 0., 1.], "frame_revision": i}} for i in range(300)]}}
        receipts = []
        for samples in (1, 4):
            receipts.append({"msaa_samples": samples, "input_binding": copy.deepcopy(protocol["input_binding"]),
                "frames": 300, "size": [1280, 720], "fps": 30, "warmup": 10,
                "capture_ms": [10.] * 300, "consumer_work_ms": [1000.] * 300,
                "p50_ms": 10., "p95_ms": 10., "max_ms": 10.,
                "frame_sha256": [f"{i:064x}" for i in range(300)],
                "profile": copy.deepcopy(protocol["profile"]), "backend": {
                    "library_sha256": "a" * 64,
                    "msaa": {"requested_samples": samples, "actual_color_samples": samples, "sensor_samples": 1, "sensor_sampling": "deterministic pixel center", "resolve_space": "display-linear then IEC sRGB once" if samples == 4 else "none (single sample)"},
                    "device": {"closed": True, "resident_upload_calls": 1,
                        "resident_captures": 310, "live_contexts": 0, "live_frames": 0,
                        "live_resources": 0, "validation_errors": 0}}})
        return protocol, receipts[0], receipts[1]

    def test_frozen_neutral_roughness_daylight_profile_has_one_variable(self):
        p = self.module.frozen_profile()
        for key, expected in {"samples": [1, 4], "frames": 300, "fps": 30, "size": [1280, 720],
                "warmup": 10, "camera": "sequence", "display_transform": "pbr-neutral",
                "photographic": True, "surface_roughness": True, "cube_size": 256,
                "daylight": True, "light_strength": .5, "light_yaw": 25.,
                "exposure": 8., "shadow_mode": "ray_query", "area_light_grid": 2}.items():
            self.assertEqual(p[key], expected, key)
        p["size"][0] = 1
        self.assertEqual(self.module.frozen_profile()["size"], [1280, 720])

    def test_compare_uses_public_readback_times_and_keeps_performance_separate(self):
        protocol, one, four = self.protocol_and_receipts()
        result = self.module.compare_receipts(protocol, one, four)
        self.assertTrue(result["input_match"])
        self.assertEqual(result["modes"]["4"]["p95_ms"], 10.)
        self.assertTrue(result["fast_profile_passed"])
        four["capture_ms"] = [40.] * 300
        four.update(p50_ms=40., p95_ms=40., max_ms=40.)
        result = self.module.compare_receipts(protocol, one, four)
        self.assertTrue(result["input_match"])
        self.assertFalse(result["fast_profile_passed"])
        self.assertEqual(result["modes"]["4"]["p95_ms"], 40.)

    def test_camera_clock_pose_and_source_mismatch_each_reject(self):
        for target in ("camera", "clock", "state", "source", "scene"):
            with self.subTest(target=target):
                protocol, one, four = self.protocol_and_receipts()
                binding = four["input_binding"]
                if target == "camera": binding["frames"][27]["camera"]["intrinsics"][0] += .1
                elif target == "clock": binding["frames"][27]["time_s"] += .001
                elif target == "state": binding["frames"][27]["state_sha256"] = "d" * 64
                elif target == "source": binding["source_excerpt_sha256"] = "d" * 64
                else: binding["scene_sha256"] = "d" * 64
                with self.assertRaisesRegex(ValueError, "binding|input"):
                    self.module.compare_receipts(protocol, one, four)

    def test_both_receipts_cannot_collude_against_frozen_protocol(self):
        protocol, one, four = self.protocol_and_receipts()
        for receipt in (one, four):
            receipt["profile"]["display_transform"] = "clip"
        with self.assertRaisesRegex(ValueError, "profile|protocol"):
            self.module.compare_receipts(protocol, one, four)

    def test_unwarmed_incomplete_fallback_or_foreign_dll_reject(self):
        for target in ("warmup", "count", "samples", "dll", "unclosed", "upload"):
            with self.subTest(target=target):
                protocol, one, four = self.protocol_and_receipts()
                if target == "warmup": four["warmup"] = 9
                elif target == "count": four["capture_ms"].pop()
                elif target == "samples": four["backend"]["msaa"]["actual_color_samples"] = 1
                elif target == "dll": four["backend"]["library_sha256"] = "d" * 64
                elif target == "unclosed": four["backend"]["device"]["closed"] = False
                else: four["backend"]["device"]["resident_upload_calls"] = 2
                with self.assertRaises(ValueError):
                    self.module.compare_receipts(protocol, one, four)

    def test_false_percentile_and_nonfinite_timing_reject(self):
        for target in ("claimed", "nan", "negative"):
            with self.subTest(target=target):
                protocol, one, four = self.protocol_and_receipts()
                if target == "claimed": four["p95_ms"] = 1.
                else: four["capture_ms"][10] = float("nan") if target == "nan" else -1.
                with self.assertRaisesRegex(ValueError, "timing|p95|capture"):
                    self.module.compare_receipts(protocol, one, four)

    def test_codec_frame_error_reports_full_mean_and_worst_without_hiding_frame(self):
        report = self.module.summarize_codec_errors([
            {"index": 0, "mae_rgb8": 1., "max_rgb8": 3},
            {"index": 1, "mae_rgb8": 2., "max_rgb8": 9},
            {"index": 2, "mae_rgb8": 3., "max_rgb8": 5}])
        self.assertEqual(report["mean_mae_rgb8"], 2.)
        self.assertEqual(report["worst_mae_frame"], 2)
        self.assertEqual(report["max_rgb8"], 9)
        self.assertEqual(report["worst_max_frame"], 1)
        with self.assertRaisesRegex(ValueError, "contiguous|index"):
            self.module.summarize_codec_errors([{"index": 1, "mae_rgb8": 0., "max_rgb8": 0}])


if __name__ == "__main__":
    unittest.main()