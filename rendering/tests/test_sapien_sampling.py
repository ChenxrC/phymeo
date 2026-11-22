"""Explicitly seeded reference sampling uses an isolated shader pack."""
import tempfile
import unittest
from pathlib import Path

from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends.sapien import SapienConfig
from phymeo_rendering.pipeline import RenderPath


class ReferenceStreamTest(unittest.TestCase):
    def test_seeded_mode_requires_an_explicit_uint32_and_path_tracing(self):
        self.assertEqual(SapienConfig(path=RenderPath.PATH_TRACE, sampling_mode="shifted-halton",
                                      seed=0).seed, 0)
        for seed in (None, -1, 2**32, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                SapienConfig(path=RenderPath.PATH_TRACE, sampling_mode="shifted-halton", seed=seed)
        with self.assertRaises(UnsupportedRequestError):
            SapienConfig(sampling_mode="shifted-halton", seed=1)
        with self.assertRaises(ValueError):
            SapienConfig(path=RenderPath.PATH_TRACE, sampling_mode="typo", seed=1)

    def test_variant_preserves_official_pack_and_changes_both_random_sequences(self):
        from phymeo_rendering.backends._sapien_sampling import make_seeded_pack

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = root / "official"
            original.mkdir()
            shader = ("uint magic = 0x9e3779b9;  // use fixed seed for consistency\n"
                      "ray.seed = seed;\nvec2 r = halton2d(frameCount * spp + i, 2, 3);\n")
            (original / "camera.rgen").write_text(shader)
            (original / "random.glsl").write_bytes(b"unchanged")
            one = make_seeded_pack(original, root / "one", 11)
            two = make_seeded_pack(original, root / "two", 31)
            self.assertEqual((original / "camera.rgen").read_text(), shader)
            self.assertNotEqual((one / "camera.rgen").read_bytes(), (two / "camera.rgen").read_bytes())
            self.assertEqual((one / "random.glsl").read_bytes(), b"unchanged")
            result = (one / "camera.rgen").read_text()
            self.assertIn("0x0000000bu", result)
            self.assertIn("fract(halton2d(", result)
            (original / "camera.rgen").write_text(shader + shader)
            with self.assertRaisesRegex(ValueError, "sampling"):
                make_seeded_pack(original, root / "bad", 11)
