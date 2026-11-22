"""Independent Khronos PBR Neutral display checks; no native renderer changes.

Oracle equations: KhronosGroup/ToneMapping b5a2eed5ddf6c2227090449399de9c7affb9e4c9,
PBR_Neutral/README.md specification. The implementation uses Decimal scalar
arithmetic and never calls the product display or tone-map helpers.
"""
from dataclasses import fields
from decimal import Decimal, localcontext
import importlib.util
import math
import os
from pathlib import Path
import unittest

import numpy as np

from phymeo_rendering.backends._sapien_pixels import encode_linear_image
from phymeo_rendering.backends.sapien import SapienConfig


ROOT = Path(__file__).resolve().parents[1]


def neutral_reference(rgb, exposure=1.):
    with localcontext() as context:
        context.prec = 70
        values = [max(Decimal(0), Decimal.from_float(float(value))*Decimal.from_float(float(exposure)))
                  for value in rgb]
        minimum = min(values)
        fresnel = Decimal("0.04")
        offset = minimum-minimum*minimum/(4*fresnel) if minimum <= 2*fresnel else fresnel
        reduced = [value-offset for value in values]
        peak = max(reduced)
        compression = Decimal("0.76")
        if peak <= compression:
            return tuple(float(value) for value in reduced)
        shoulder = 1-(1-compression)**2/(peak+1-2*compression)
        chroma_weight = 1/(Decimal("0.15")*(peak-shoulder)+1)
        return tuple(float(value*shoulder/peak*chroma_weight+shoulder*(1-chroma_weight))
                     for value in reduced)


def code(linear):
    value = min(1., max(0., float(linear)))
    display = 12.92*value if value <= .0031308 else 1.055*value**(1/2.4)-.055
    return min(255, int(math.floor(display*255+.5)))


def reference_bytes(image, *, exposure=1., transform="pbr-neutral"):
    output = bytearray()
    for pixel in np.asarray(image).reshape(-1, 3):
        linear = neutral_reference(pixel, exposure) if transform == "pbr-neutral" else tuple(float(v)*exposure for v in pixel)
        output.extend(code(value) for value in linear)
    return bytes(output)


class ReferenceNeutralTests(unittest.TestCase):
    def test_config_adds_only_an_explicit_reference_display_choice(self):
        original = ("path", "device", "samples_per_pixel", "max_bounces", "seed", "denoiser",
                    "studio", "sampling_mode", "exposure")
        self.assertEqual(tuple(field.name for field in fields(SapienConfig)), original+("display_transform",))
        self.assertEqual(SapienConfig().display_transform, "clip")
        self.assertEqual(SapienConfig(display_transform="pbr-neutral").display_transform, "pbr-neutral")
        for invalid in (None, True, 1, "aces", "PBR-neutral", ""):
            with self.subTest(invalid=invalid), self.assertRaisesRegex((TypeError, ValueError), "display_transform"):
                SapienConfig(display_transform=invalid)

    def test_explicit_clip_preserves_every_previous_code_boundary(self):
        midway = (np.arange(255, dtype=np.float64)+.5)/255
        linear = np.where(midway <= .04045, midway/12.92, ((midway+.055)/1.055)**2.4).astype(np.float32)
        image = np.stack((np.nextafter(linear, -np.float32(np.inf)), linear,
                          np.nextafter(linear, np.float32(np.inf))), axis=-1)[None]
        existing = encode_linear_image(image)
        self.assertEqual(encode_linear_image(image, display_transform="clip"), existing)
        self.assertEqual(existing, reference_bytes(image, transform="clip"))
        mixed = np.array([[[-.1, .0, .3], [.7, 1., 10.]]], dtype=np.float32)
        for exposure in (0., 1., 2., 8.):
            self.assertEqual(encode_linear_image(mixed, exposure=exposure, display_transform="clip"),
                             encode_linear_image(mixed, exposure=exposure))

    def test_black_gray_white_saturated_and_overexposed_match_independent_scalar(self):
        pixels = [(0.,)*3, (.02,)*3, (.08,)*3, (.18,)*3, (.5,)*3, (.8,)*3, (1.,)*3,
                  (2.,)*3, (4.,)*3, (8.,)*3, (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                  (8., .4, .1), (.1, 4., 1.5), (1.5, .1, 4.), (.04, .2, .7)]
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                image = np.asarray([pixels], dtype=dtype)
                self.assertEqual(encode_linear_image(image, display_transform="pbr-neutral"), reference_bytes(image))

    def test_toe_and_compression_neighborhoods_do_not_jump_or_cross_channels(self):
        for dtype in (np.float32, np.float64):
            pixels = []
            for level in (.08, .8):
                value = dtype(level)
                for neighbor in (np.nextafter(value, dtype(-np.inf)), value,
                                 np.nextafter(value, dtype(np.inf))):
                    pixels.extend(((neighbor, neighbor, neighbor),
                                   (neighbor, .11, .23), (.23, neighbor, .11), (.11, .23, neighbor)))
            image = np.asarray([pixels], dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assertEqual(encode_linear_image(image, display_transform="pbr-neutral"), reference_bytes(image))

    def test_exposure_precedes_tone_mapping_and_srgb_exactly_once(self):
        image = np.array([[[.23, .04, .37], [.16, .22, .07], [1.5, .7, .21]]], dtype=np.float32)
        expected = reference_bytes(image, exposure=3.5)
        actual = encode_linear_image(image, exposure=3.5, display_transform="pbr-neutral")
        self.assertEqual(actual, expected)
        premature = np.array([[neutral_reference(pixel) for pixel in image[0]]], dtype=np.float64)
        self.assertNotEqual(actual, reference_bytes(premature, exposure=3.5))
        self.assertEqual(encode_linear_image(image, exposure=0., display_transform="pbr-neutral"), bytes(9))

    def test_strided_readback_remains_owned_unmodified_and_handles_finite_hdr(self):
        rgba = np.linspace(-.2, 12., 17*23*4, dtype=np.float32).reshape(17, 23, 4)
        view = rgba[::-1, ::2, :3]
        original = rgba.copy()
        encoded = encode_linear_image(view, exposure=2., display_transform="pbr-neutral")
        self.assertEqual(encoded, reference_bytes(view, exposure=2.))
        np.testing.assert_array_equal(rgba, original)
        rgba.fill(0.)
        self.assertNotEqual(encoded, bytes(len(encoded)))
        largest = np.full((1, 2, 3), np.finfo(np.float32).max, dtype=np.float32)
        self.assertEqual(encode_linear_image(largest, exposure=8., display_transform="pbr-neutral"), bytes([255])*6)

    def test_nonfinite_readback_is_rejected_before_display_compression(self):
        for invalid in (np.nan, np.inf, -np.inf):
            image = np.ones((1, 1, 3), dtype=np.float32)
            image[0, 0, 1] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                encode_linear_image(image, display_transform="pbr-neutral")
        with self.assertRaisesRegex(ValueError, "display_transform"):
            encode_linear_image(np.zeros((1, 1, 3), dtype=np.float32), display_transform="unknown")

    def test_preregistered_hdr_grays_remain_distinct_in_display_space(self):
        levels = (1., 2., 4., 8.)
        image = np.array([[(v,)*3 for v in levels]], dtype=np.float32)
        clipped = encode_linear_image(image, display_transform="clip")
        neutral = encode_linear_image(image, display_transform="pbr-neutral")
        self.assertEqual(clipped, bytes([255])*12)
        values = tuple(neutral[::3])
        self.assertEqual(tuple(sorted(set(values))), values)
        self.assertLess(max(values), 255)
        self.assertGreaterEqual(values[-1]-values[0], 10)


@unittest.skipUnless(os.environ.get("PHYMEO_SAPIEN_NEUTRAL_DEVICE") == "1", "explicit reference display device slot")
class ReferenceNeutralDeviceTest(unittest.TestCase):
    def test_real_radiance_highlights_recover_without_sensor_changes(self):
        output = os.environ.get("PHYMEO_SAPIEN_NEUTRAL_OUTPUT")
        if output is None:
            self.fail("PHYMEO_SAPIEN_NEUTRAL_OUTPUT must be a NEW persistent evidence directory")
        spec = importlib.util.spec_from_file_location("reference_neutral_calibration",
                                                     ROOT/"examples/reference_neutral_calibration.py")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        result = probe.run_calibration(Path(output), device=os.environ.get("PHYMEO_SAPIEN_DEVICE_NAME", "cuda:0"))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["raster_sensors_unchanged"])
        self.assertGreaterEqual(result["pt_foreground_recovered_pixels"], 64)
        for shot in result["captures"]:
            raw = np.fromfile(Path(output)/shot["linear_file"], dtype="<f4").reshape(shot["shape"])
            actual = (Path(output)/shot["rgb_file"]).read_bytes()
            self.assertEqual(actual, reference_bytes(raw, exposure=shot["exposure"], transform=shot["display_transform"]))
            self.assertEqual(shot["diagnostics"]["display_transform"], shot["display_transform"])


if __name__ == "__main__":
    unittest.main()
