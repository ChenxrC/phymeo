"""The fast float32 camera readback must match the public scalar display transform."""
import unittest

import numpy as np

from phymeo_rendering.backends._sapien_pixels import encode_linear_rgb


class DisplayConversion(unittest.TestCase):
    def test_exposure_is_applied_to_linear_radiance_before_srgb(self):
        from phymeo_rendering.backends._sapien_pixels import encode_linear_image
        from phymeo_rendering.backends.sapien import SapienConfig

        film = SapienConfig(exposure=8.)
        image = np.array([[[.01, .06, .19], [.001, 1., 0.]]], dtype=np.float32)
        self.assertEqual(encode_linear_image(image, exposure=film.exposure),
                         encode_linear_rgb((8. * image).flat))
        self.assertEqual(encode_linear_image(image, exposure=0.), bytes(6))
        for invalid in (-1, np.inf, np.nan, True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                SapienConfig(exposure=invalid)

    def test_every_srgb_code_boundary_and_neighbors_match_scalar(self):
        from phymeo_rendering.backends._sapien_pixels import encode_linear_image

        mid=(np.arange(255,dtype=np.float64)+.5)/255
        linear=np.where(mid<=.04045,mid/12.92,((mid+.055)/1.055)**2.4).astype(np.float32)
        pixels=np.stack((np.nextafter(linear,-np.float32(np.inf)),linear,
                         np.nextafter(linear,np.float32(np.inf))),axis=-1)[None]
        self.assertEqual(encode_linear_image(pixels),encode_linear_rgb(pixels.flat))

    def test_strided_rgba_input_keeps_row_channel_order_and_clips_hdr(self):
        from phymeo_rendering.backends._sapien_pixels import encode_linear_image

        rgba=np.linspace(-.2,2.,23*31*4,dtype=np.float32).reshape(23,31,4)
        pixels=rgba[::-1,::2,:3]
        self.assertEqual(encode_linear_image(pixels),encode_linear_rgb(pixels.flat))
        self.assertEqual(encode_linear_image(pixels.astype(np.float64)),encode_linear_rgb(pixels.flat))
        for bad in (np.nan,np.inf,-np.inf):
            data=pixels.copy()
            data[0,0,0]=bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError,"finite"):
                encode_linear_image(data)
