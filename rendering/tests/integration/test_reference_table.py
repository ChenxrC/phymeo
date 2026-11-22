import os
import unittest

from phymeo_rendering.pipeline import RenderPath

from ._r0_table import exercise_lifetime, exercise_table


@unittest.skipUnless(os.environ.get("PHYMEO_R0_REFERENCE") == "1", "explicit reference device test")
class ReferenceTableTest(unittest.TestCase):
    @staticmethod
    def factory(path):
        from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig

        def make():
            return SapienBackend(
                SapienConfig(path=path, device=os.environ["PHYMEO_R0_SAPIEN_DEVICE"])
            )

        return make

    def capture_table(self, path, sensors):
        exercise_table(self, self.factory(path), "sapien-" + path.value, path, sensors=sensors)

    def test_reference_raster(self):
        self.capture_table(RenderPath.RASTER, True)

    def test_reference_path_tracing(self):
        self.capture_table(RenderPath.PATH_TRACE, False)

    def test_owned_entities_after_twenty_sessions_per_path(self):
        def check(test, session):
            info = session.diagnostics()["lifecycle"]
            test.assertTrue(info["closed"])
            test.assertEqual(info["attached_entities_after_clear"], 0)
            test.assertEqual(info["attached_cameras_after_clear"], 0)
            test.assertEqual(info["attached_render_bodies_after_clear"], 0)
            test.assertEqual(info["completed_captures"], 1)
            return info

        for path in (RenderPath.RASTER, RenderPath.PATH_TRACE):
            with self.subTest(path=path):
                exercise_lifetime(self, self.factory(path), "sapien-" + path.value, path, check)
