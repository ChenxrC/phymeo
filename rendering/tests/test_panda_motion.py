"""Motion sampling checks; real-archive cases skip when its source is unavailable."""

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from phymeo_rendering.scene import VisualScene

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SPEC = importlib.util.spec_from_file_location("_panda_motion", EXAMPLES / "_panda_motion.py")
motion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motion)
FIXTURE = Path(__file__).parent / "fixtures" / "panda_replay"


class SamplingTests(unittest.TestCase):
    def test_checked_in_subset_preserves_real_motion_without_external_archive(self):
        directory = Path(__file__).parent / "fixtures" / "panda_motion"
        path = directory / "trajectory.json"
        expected = "2e3687f3e0c62a0acfbbd1893154874a45eadc15feb10b4d8fbd6ff5ea7b91cc"
        self.assertEqual((directory / "trajectory.sha256").read_text().split()[0], expected)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        source = motion.PandaMotionSource(
            VisualScene("portable-motion", (), (), ()),
            FIXTURE / "panda.urdf",
            path,
            expected_subset_sha256=expected,
        )
        frames = tuple(source.frames())
        self.assertEqual(len(frames), 108)
        self.assertEqual((frames[0].key.revision, frames[-1].key.revision), (32, 3599))
        self.assertEqual((frames[0].time_s, frames[-1].time_s), (0.020625, 2.25))
        first, last = source.provenance(frames[0]), source.provenance(frames[-1])
        for evidence in (first, last):
            self.assertEqual(
                evidence["archive_sha256"],
                "d74e42dd2d3f7735e197b0a845e602217b243a8513c73b7ab09c02995ccb666b",
            )
            self.assertIsNone(evidence["live_physics_identity"])
        self.assertGreater(abs(last["record"]["q"][1] - first["record"]["q"][1]), 0.2)
        self.assertGreater(last["record"]["cube"][0] - first["record"]["cube"][0], 0.07)
        self.assertEqual(source.timeline()["playback_duration_s"], 4.5)

    def test_48_hz_right_edges_use_108_original_records(self):
        selected = motion.sample_indices(record_count=3600, dt_s=0.000625, sample_hz=48)
        self.assertEqual(len(selected), 108)
        self.assertEqual(selected[:6], (32, 66, 99, 132, 166, 199))
        self.assertEqual(selected[-1], 3599)
        self.assertEqual(len(set(selected)), 108)
        for k, index in enumerate(selected):
            self.assertLessEqual(abs((index + 1) * 0.000625 - (k + 1) / 48), 0.0003125)

    def test_tie_uses_earlier_record_and_partial_last_slot_is_kept(self):
        self.assertEqual(motion.sample_indices(record_count=3, dt_s=0.125, sample_hz=4), (1, 2))
        self.assertEqual(motion.sample_indices(record_count=5, dt_s=0.1, sample_hz=4), (1, 4))

    def test_invalid_sampling_is_rejected_without_duplicates(self):
        for rate in (0, -1, True, float("nan"), float("inf"), 1601):
            with self.subTest(rate=rate), self.assertRaises((ValueError, TypeError)):
                motion.sample_indices(record_count=3600, dt_s=0.000625, sample_hz=rate)
        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises((ValueError, TypeError)):
                motion.sample_indices(record_count=count, dt_s=0.000625, sample_hz=48)

    def test_arbitrary_input_cannot_claim_original_archive(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.json"
            path.write_text('{"trace": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                motion.PandaMotionSource(
                    VisualScene("test", (), (), ()), FIXTURE / "panda.urdf", path
                )


class RealArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("PHYMEO_PANDA_ARCHIVE")
        if not configured:
            raise unittest.SkipTest("original archive comparison requires PHYMEO_PANDA_ARCHIVE")
        cls.archive = Path(configured)
        if not cls.archive.is_file():
            raise FileNotFoundError(f"configured Panda archive unavailable: {cls.archive}")

    def test_full_archive_and_trusted_subset_are_identical(self):
        scene = VisualScene("motion-test", (), (), ())
        direct = motion.PandaMotionSource(scene, FIXTURE / "panda.urdf", self.archive)
        frames = tuple(direct.frames())
        self.assertEqual(len(frames), 108)
        self.assertEqual(frames[0].key.revision, 32)
        self.assertEqual(frames[-1].key.revision, 3599)
        self.assertEqual(frames[0].time_s, 0.020625)
        self.assertEqual(frames[-1].time_s, 2.25)
        timeline = direct.timeline()
        self.assertEqual(timeline["playback_duration_s"], 4.5)
        self.assertEqual(timeline["nominal_speed"], 0.5)
        self.assertEqual(timeline["source_window_s"], [0.0, 2.25])
        self.assertFalse(timeline["interpolation"])
        self.assertEqual(direct.geometry["cube_full_size_m"], [0.046, 0.058, 0.04])
        archive = json.loads(self.archive.read_text(encoding="utf-8"))
        for frame in frames:
            evidence = direct.provenance(frame)
            self.assertEqual(evidence["record"], archive["trace"][frame.key.revision])
            self.assertIsNone(evidence["live_physics_identity"])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.json"
            receipt = motion.write_motion_subset(self.archive, path)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "trusted.*SHA|expected_subset"):
                motion.PandaMotionSource(scene, FIXTURE / "panda.urdf", path)
            loaded = motion.PandaMotionSource(
                scene, FIXTURE / "panda.urdf", path, expected_subset_sha256=receipt["sha256"]
            )
            self.assertEqual(tuple(loaded.frames()), frames)
            self.assertEqual(loaded.timeline(), direct.timeline())
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                motion.PandaMotionSource(
                    scene, FIXTURE / "panda.urdf", path, expected_subset_sha256=receipt["sha256"]
                )

    def test_reordered_joint_names_and_foreign_model_rejected(self):
        scene = VisualScene("motion-test", (), (), ())
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "motion.json"
            motion.write_motion_subset(self.archive, path)
            original = path.read_bytes()
            subset = json.loads(original)
            inputs = json.loads(subset["inputs_json"])
            inputs["joint_names"][0:2] = reversed(inputs["joint_names"][0:2])
            subset["inputs_json"] = json.dumps(inputs)
            path.write_text(json.dumps(subset), encoding="utf-8")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "joint"):
                motion.PandaMotionSource(
                    scene, FIXTURE / "panda.urdf", path, expected_subset_sha256=checksum
                )
            path.write_bytes(original)
            wrong_model = root / "panda.urdf"
            wrong_model.write_bytes((FIXTURE / "panda.urdf").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "model|URDF"):
                motion.PandaMotionSource(
                    scene,
                    wrong_model,
                    path,
                    expected_subset_sha256=hashlib.sha256(original).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
