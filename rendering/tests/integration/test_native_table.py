import os
from pathlib import Path
import unittest

from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.pipeline import RenderPath

from ._r0_table import exercise_table


@unittest.skipUnless(os.environ.get("PHYMEO_R0_NATIVE") == "1", "explicit native device test")
class NativeTableTest(unittest.TestCase):
    def test_fixed_table_on_vulkan(self):
        library = Path(os.environ["PHYMEO_R0_LIBRARY"])
        def factory():
            return NativeBackend(library_path=library, shader_directory=library.parent,
                                 device_index=int(os.environ["PHYMEO_R0_DEVICE_INDEX"]), validation=True)
        exercise_table(self, factory, "native-raster", RenderPath.RASTER, sensors=True)
