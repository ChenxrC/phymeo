"""Portable SAPIEN adapter contracts; no graphics runtime is required."""
import builtins
from dataclasses import FrozenInstanceError
import math
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.pipeline import RenderPath
from phymeo_rendering.scene import VisualScene


class SapienConfigurationTest(unittest.TestCase):
    def test_explicit_path_and_unsupported_seed_are_not_silently_ignored(self):
        from phymeo_rendering.backends.sapien import SapienConfig
        config = SapienConfig()
        self.assertIs(config.path, RenderPath.RASTER)
        self.assertEqual((config.samples_per_pixel, config.max_bounces), (64, 4))
        self.assertIsNone(config.seed)
        self.assertEqual(config.denoiser, "none")
        with self.assertRaises(FrozenInstanceError):
            config.seed = 7
        with self.assertRaisesRegex(UnsupportedRequestError, "seed"):
            SapienConfig(path=RenderPath.PATH_TRACE, seed=7)
        with self.assertRaisesRegex(UnsupportedRequestError, "denoiser"):
            SapienConfig(denoiser="optix")

    def test_configuration_rejects_invalid_fields(self):
        from phymeo_rendering.backends.sapien import SapienConfig
        for kwargs, error in (
            ({"path": "raster"}, TypeError),
            ({"device": 1}, TypeError), ({"device": " "}, ValueError),
            ({"samples_per_pixel": True}, TypeError),
            ({"samples_per_pixel": 0}, ValueError),
            ({"samples_per_pixel": 2**32}, ValueError),
            ({"max_bounces": 1.5}, TypeError),
            ({"max_bounces": -1}, ValueError),
            ({"seed": True}, TypeError),
            ({"denoiser": None}, TypeError),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                SapienConfig(**kwargs)

    def test_dependency_failure_is_chained_only_on_explicit_open(self):
        from phymeo_rendering.backends.sapien import SapienBackend
        original = builtins.__import__
        calls = []
        missing = ModuleNotFoundError("deliberately unavailable sapien")
        def no_sapien(name, *args, **kwargs):
            if name == "sapien" or name.startswith("sapien."):
                calls.append(name)
                raise missing
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=no_sapien):
            backend = SapienBackend()
            self.assertEqual(calls, [])
            with self.assertRaises(BackendUnavailableError) as caught:
                backend.open_scene(VisualScene("empty", (), (), ()))
        self.assertIs(caught.exception.__cause__, missing)
        self.assertEqual(calls, ["sapien"])

    def test_linear_hdr_has_exactly_one_standard_srgb_encoding(self):
        from phymeo_rendering.backends._sapien_pixels import encode_linear_rgb
        # Independent sRGB reference anchors. Gamma 2.2 would produce 14, not 10,
        # for the breakpoint and 31, not 25, for linear 0.01.
        result = encode_linear_rgb((0.0, 0.0031308, 0.01, 0.18, 0.5, 1.0, 2.0, -0.1, 0.0))
        self.assertEqual(result, bytes((0, 10, 25, 118, 188, 255, 255, 0, 0)))
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "finite"):
                encode_linear_rgb((bad, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
