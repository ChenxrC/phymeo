import os
import unittest

from phymeo_rendering.pipeline import RenderPath

from ._r0_table import exercise_table


@unittest.skipUnless(os.environ.get("PHYMEO_R0_REFERENCE") == "1", "explicit reference device test")
class ReferenceTableTest(unittest.TestCase):
    def capture_table(self, path, sensors):
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig

        def factory():
            return SapienBackend(SapienConfig(path=path, device=os.environ["PHYMEO_R0_SAPIEN_DEVICE"]))
        exercise_table(self, factory, "sapien-" + path.value, path, sensors=sensors)

    def test_reference_raster(self):
        self.capture_table(RenderPath.RASTER, True)

    def test_reference_path_tracing(self):
        self.capture_table(RenderPath.PATH_TRACE, False)
