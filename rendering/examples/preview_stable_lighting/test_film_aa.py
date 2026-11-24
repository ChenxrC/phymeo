from dataclasses import dataclass
import unittest

import numpy as np

from .film_aa import block_mean, scaled_camera


@dataclass(frozen=True)
class Lens:
    width: int = 13
    height: int = 9
    fx_px: float = 11.
    fy_px: float = 8.
    cx_px: float = 6.3
    cy_px: float = 4.1


class FilmSampling(unittest.TestCase):
    def test_true_subpixel_camera_coordinates(self):
        lens = Lens()
        scaled = scaled_camera(lens, 2)
        self.assertEqual((scaled.width, scaled.height), (26, 18))
        for x in (0, 1, 7, 12):
            for fine_x in (2*x, 2*x+1):
                ray_x = (fine_x+.5-scaled.cx_px) / scaled.fx_px
                self.assertAlmostEqual(ray_x, ((fine_x+.5)/2-lens.cx_px)/lens.fx_px)
        crop = scaled_camera(lens, 8, (2, 1, 8, 6))
        self.assertAlmostEqual((4.5-crop.cy_px)/crop.fy_px, (1+4.5/8-lens.cy_px)/lens.fy_px)
        self.assertEqual((crop.width, crop.height), (48, 40))

    def test_independent_hdr_pixel_areas(self):
        source = np.zeros((2, 4, 3), dtype=np.float32)
        source[0, 0] = (12., 1., 0.)
        source[:, 2:] = (2., 1., 4.)
        color = block_mean(source, 2)
        np.testing.assert_array_equal(color, [[[3., .25, 0.], [2., 1., 4.]]])
        source[:] = 0
        self.assertEqual(color[0, 1, 2], 4.)

    def test_incomplete_groups_and_bad_crop_are_rejected(self):
        with self.assertRaises(ValueError):
            block_mean(np.zeros((2, 3, 3)), 2)
        with self.assertRaises(ValueError):
            scaled_camera(Lens(), 2, (0, 0, 14, 9))
