"""Native-only input/camera seams; device execution is separately selected."""

import math
import tempfile
import unittest
from pathlib import Path

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.scene import Instance, VisualScene


class NativeConfiguration(unittest.TestCase):
    def test_invalid_native_configuration_is_rejected(self):
        for fields, error in (
            ({"device_index": True}, TypeError),
            ({"device_index": -1}, ValueError),
            ({"validation": 1}, TypeError),
            ({"exposure": math.nan}, ValueError),
            ({"exposure": -1}, ValueError),
            ({"exposure": True}, TypeError),
        ):
            with self.subTest(fields=fields), self.assertRaises(error):
                NativeBackend(**fields)

    def test_missing_explicit_library_is_not_a_fallback(self):
        scene = VisualScene(
            "native", (Mesh("mesh", "absent.obj"),),
            (Material("paint", (0.2, 0.3, 0.4), 0.0, 0.5),),
            (Instance(1, "mesh", "paint", "node"),),
        )
        with tempfile.TemporaryDirectory() as root:
            backend = NativeBackend(library_path=Path(root) / "absent.dll")
            with self.assertRaisesRegex(BackendUnavailableError, "library"):
                backend.open_scene(scene)


if __name__ == "__main__":
    unittest.main()
