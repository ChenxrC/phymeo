import os
import unittest
from pathlib import Path

from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.pipeline import RenderPath

from ._r0_table import exercise_lifetime, exercise_table


@unittest.skipUnless(os.environ.get("PHYMEO_R0_NATIVE") == "1", "explicit native device test")
class NativeTableTest(unittest.TestCase):
    @staticmethod
    def backend():
        library = Path(os.environ["PHYMEO_R0_LIBRARY"])
        return NativeBackend(
            library_path=library,
            shader_directory=library.parent,
            device_index=int(os.environ["PHYMEO_R0_DEVICE_INDEX"]),
            validation=True,
        )

    def test_fixed_table_on_vulkan(self):
        exercise_table(self, self.backend, "native-raster", RenderPath.RASTER, sensors=True)

    def test_owned_resources_after_twenty_sessions(self):
        def check(test, session):
            info = session.diagnostics()["device"]
            fields = ("live_contexts", "live_frames", "live_resources", "validation_errors")
            counts = {field: info[field] for field in fields}
            test.assertEqual(tuple(counts.values()), (0, 0, 0, 0))
            test.assertEqual(info["captures"], 1)
            return counts

        exercise_lifetime(self, self.backend, "native-raster", RenderPath.RASTER, check)
