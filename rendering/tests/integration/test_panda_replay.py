"""Opt-in G4/R1 archive-to-camera acceptance on public GPU render sessions."""

import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("PHYMEO_R1_DEVICE") == "1", "Panda R1 requires explicit GPU capture"
)
class PandaReplayCapture(unittest.TestCase):
    def test_archived_grasp_camera_observations(self):
        examples = Path(__file__).resolve().parents[2] / "examples"
        output = Path(os.environ["PHYMEO_R1_OUTPUT"])
        library = os.environ.get("PHYMEO_R0_LIBRARY")
        args = SimpleNamespace(
            assets=Path(os.environ["PHYMEO_R1_ASSETS"]),
            output=output,
            device=os.environ["PHYMEO_R0_SAPIEN_DEVICE"],
            spp=512,
            native_library=Path(library) if library else None,
            native_index=int(os.environ.get("PHYMEO_R0_DEVICE_INDEX", "0")),
        )
        with patch.object(sys, "path", [str(examples), *sys.path]):
            runpy.run_path(str(examples / "panda_replay.py"))["main"](args)
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual([p["record_index"] for p in report["provenance"]], [479, 2159])
        for pose in report["anchors"]:
            self.assertEqual({p["name"] for p in pose}, {"base", "wrist", "finger"})
            self.assertLessEqual(max(p["error_px"] for p in pose), 1)
        for view in ("single", "studio"):
            meta = json.loads((output / (view + "-capture.json")).read_text(encoding="utf-8"))
            self.assertEqual(meta["device"]["effective_path"], "path_trace")
            self.assertEqual(meta["device"]["physics_steps"], 0)
        self.assertEqual(len(report["wrist_pixels"]), 2)
        if library:
            self.assertIn("4096 triangles", report["native_full_scene"])
            self.assertTrue((output / "native-detail-2-wrist.png").is_file())
