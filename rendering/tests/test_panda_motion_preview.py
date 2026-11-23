"""HOST contracts for the public Panda motion-preview runner; no device evidence."""
from copy import deepcopy
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


class MotionPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = importlib.import_module("panda_motion_preview")

    def test_presets_share_optics_and_only_change_bounces_or_extent(self):
        a = self.study.preset("preview720")
        b = self.study.preset("quality720")
        c = self.study.preset("fast540")
        self.assertEqual(a["extent"], [1280, 720])
        self.assertEqual(c["extent"], [960, 540])
        self.assertEqual(a["settings"]["indirect_bounces"], 1)
        self.assertEqual(b["settings"]["indirect_bounces"], 2)
        b["settings"]["indirect_bounces"] = 1
        self.assertEqual(a["settings"], b["settings"])
        self.assertEqual(a["settings"], c["settings"])
        self.assertEqual(a["settings"]["spatial_filter"], "guided_settling")
        self.assertEqual(a["settings"]["reflection_mode"], "single_bounce")
        self.assertEqual(a["settings"]["area_light_grid"], 2)
        self.assertEqual(a["settings"]["msaa_samples"], 1)
        self.assertEqual(a["exposure"], 8.)
        self.assertFalse(c["eligible_for_720p_claim"])

    def test_direct_keeps_mirror_and_360_only_changes_pixel_grid(self):
        a = self.study.preset("preview720")
        d = self.study.preset("direct720")
        self.assertEqual(d["settings"]["reflection_mode"], "single_bounce")
        expected = dict(a["settings"], indirect_mode="none", progressive=False, spatial_filter="none")
        self.assertEqual(d["settings"], expected)
        low = self.study.preset("fast360")
        self.assertEqual(low["extent"], [640, 360])
        self.assertEqual(low["settings"], a["settings"])
        self.assertFalse(low["eligible_for_720p_claim"])

    def test_close_checks_the_actual_nested_allocation_telemetry(self):
        d = dict(closed=True, live_contexts=0, live_frames=0, live_resources=0, validation_errors=0,
                 resident_telemetry=dict(final_after_destroy=True, device_local_bytes_current=0,
                                         host_visible_bytes_current=0))
        self.study.verify_close(d)
        d["resident_telemetry"]["device_local_bytes_current"] = 16
        with self.assertRaises(ValueError):
            self.study.verify_close(d)

    def test_presets_are_independent_and_unknown_names_rejected(self):
        a = self.study.preset("preview720")
        a["settings"]["indirect_bounces"] = 99
        self.assertEqual(self.study.preset("preview720")["settings"]["indirect_bounces"], 1)
        with self.assertRaises(ValueError):
            self.study.preset("automatic")

    def test_full_and_diagnostic_registration_are_explicit(self):
        self.assertEqual(self.study.pose_indices(False), tuple(range(300)))
        self.assertEqual(self.study.pose_indices(True), (0, 59, 60, 149, 239, 299))
        self.assertEqual(self.study.WARMUP, 10)
        self.assertEqual(self.study.QUICK_WARMUP, 2)
        self.assertEqual(self.study.STOP_BATCHES, (1, 2, 4, 8, 16))

    def test_completed_frame_percentiles_do_not_use_nominal_movie_fps(self):
        value = self.study.timing_summary(list(range(1, 301)), full=True)
        self.assertEqual(value["p50_ms"], 150)
        self.assertEqual(value["p95_ms"], 285)
        self.assertEqual(value["max_ms"], 300)
        self.assertAlmostEqual(value["completed_fps_from_mean"], 1000 / 150.5)
        self.assertEqual(value["within_33_3_ms"], 33)
        self.assertFalse(value["latency_gate_passed"])
        short = self.study.timing_summary([1.] * 6, full=False)
        self.assertIsNone(short["latency_gate_passed"])
        self.assertEqual(short["scope"], "six-pose diagnostic; not 300-frame acceptance")

    def test_nonfinite_or_incomplete_timing_is_not_reported_as_pass(self):
        for times, full in (([], False), ([0.], False), ([math.nan], False),
                            ([math.inf], False), ([2.] * 299, True)):
            with self.subTest(times=len(times), full=full), self.assertRaises(ValueError):
                self.study.timing_summary(times, full=full)

    def test_capture_checks_full_frame_time_camera_and_channel_order(self):
        from phymeo_rendering.outputs import Channel
        f = SimpleNamespace(key="frame-a", time_s=2.)
        req = SimpleNamespace(camera="camera-a", channels=(Channel.RGB,))
        shot = SimpleNamespace(frame_key=f.key, time_s=f.time_s, camera=req.camera,
                               planes=(SimpleNamespace(channel=Channel.RGB, data=b"rgb"),))
        self.study.verify_capture(shot, f, req)
        for field, value in (("frame_key", "old"), ("time_s", 1.), ("camera", "other"),
                             ("planes", (SimpleNamespace(channel=Channel.FORWARD_DEPTH),))):
            bad = deepcopy(shot)
            setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.verify_capture(bad, f, req)

    @staticmethod
    def row(pose=60, rgb="same", sensor="geom"):
        return dict(pose_index=pose, frame_key={"tick": pose}, time_s=pose / 30,
                    state_sha256="state", camera_sha256="camera",
                    rgb_sha256=rgb, sensors={"depth": sensor})

    def test_rgb_and_all_are_matched_by_state_not_just_loop_index(self):
        a = dict(preset="preview720", channels="rgb", extent=[1280, 720],
                 frames=[self.row()], status="complete")
        b = deepcopy(a)
        b["channels"] = "all"
        self.assertTrue(self.study.compare_runs([a, b])["rgb_vs_all_exact"])
        for field, value in (("rgb_sha256", "different"), ("time_s", -1.),
                             ("frame_key", {"tick": 99}), ("state_sha256", "new"),
                             ("camera_sha256", "new")):
            bad = deepcopy(b)
            bad["frames"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.compare_runs([a, bad])

    def test_two_bounces_cannot_change_sensor_planes(self):
        a = dict(preset="preview720", channels="all", extent=[1280, 720],
                 frames=[self.row()], status="complete")
        b = deepcopy(a)
        b["preset"] = "quality720"
        b["frames"][0]["rgb_sha256"] = "allowed-radiometric-change"
        self.assertTrue(self.study.compare_runs([a, b])["same_extent_sensors_exact"])
        b["frames"][0]["sensors"]["depth"] = "different"
        with self.assertRaises(ValueError):
            self.study.compare_runs([a, b])

    def test_low_resolution_is_not_compared_as_equal_sensor_grid(self):
        a = dict(preset="preview720", channels="all", extent=[1280, 720],
                 frames=[self.row()], status="complete")
        b = deepcopy(a)
        b.update(preset="fast540", extent=[960, 540])
        b["frames"][0].update(rgb_sha256="low", sensors={"depth": "low"},
                              camera_sha256="scaled")
        result = self.study.compare_runs([a, b])
        self.assertIsNone(result["same_extent_sensors_exact"])
        self.assertIn("different resolutions", result["limitations"])

    def test_no_missing_or_duplicate_pair_can_silently_pass(self):
        a = dict(preset="preview720", channels="rgb", extent=[1280, 720],
                 frames=[self.row()], status="complete")
        with self.assertRaises(ValueError):
            self.study.compare_runs([a, deepcopy(a)])
        b = deepcopy(a)
        b["channels"] = "all"
        b["frames"] = []
        with self.assertRaises(ValueError):
            self.study.compare_runs([a, b])

    def test_profile_deltas_require_exact_capture_and_timestamp_counts(self):
        before = dict(captures=10, timestamp_samples=10, gpu_render_ns=20,
                      gpu_copy_ns=30, gpu_total_ns=50, host_wait_ns=60)
        after = {k: v + (6 if k in ("captures", "timestamp_samples") else 6000000)
                 for k, v in before.items()}
        result = self.study.profile_delta(before, after, 6)
        self.assertEqual(result["mean_ms"]["gpu_total_ns"], 1.)
        after["timestamp_samples"] -= 1
        with self.assertRaises(ValueError):
            self.study.profile_delta(before, after, 6)

    def test_backend_binding_rejects_wrong_bounce_or_unfiltered_selection(self):
        p = self.study.preset("preview720")
        d = dict(shadow_mode="ray_query", area_light_grid=2,
                 msaa={"actual_color_samples": 1}, reflection={"mode": "single_bounce"},
                 indirect={"mode": "diffuse_bounce", "bounces": 1, "samples_per_pixel": 16},
                 progressive={"enabled": True, "batches": 1, "primary_samples_per_pixel": 16},
                 spatial_filter={"mode": "guided_settling"}, normal_maps={"count": 1})
        self.study.verify_backend(d, p)
        for group, key, value in (("indirect", "bounces", 2),
                                  ("spatial_filter", "mode", "none"),
                                  ("indirect", "samples_per_pixel", 4)):
            bad = deepcopy(d)
            bad[group][key] = value
            with self.subTest(group=group), self.assertRaises(ValueError):
                self.study.verify_backend(bad, p)


if __name__ == "__main__":
    unittest.main()
