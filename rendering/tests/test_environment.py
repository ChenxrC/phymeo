"""Orientation, linear radiance, file identity and resource bounds for environment lighting."""
from hashlib import sha256
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "light.hdr"

    def declaration(self, raw, **options):
        from phymeo_rendering.environment import RadianceEnvironment

        self.path.write_bytes(raw)
        return RadianceEnvironment(self.path.as_uri(), sha256(raw).hexdigest(), **options)

    def test_literal_and_run_packets_preserve_hdr_and_zero(self):
        from phymeo_rendering.environment import load_environment

        row = bytes([2, 2, 0, 8, 8, 0, 10, 20, 30, 40, 50, 60, 70, 136, 50, 136, 255,
                     8, 0, 127, 128, 129, 130, 131, 132, 133])
        raw = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n" + row * 4
        source = load_environment(self.declaration(raw))
        self.assertEqual(source.shape, (4, 8, 3))
        np.testing.assert_array_equal(source[0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(source[0, 7], [70./8, 50./8, 255./8])

    def test_lookup_longitude_seam_poles_and_positive_yaw(self):
        from phymeo_rendering.environment import sample_environment

        panorama = np.zeros((4, 8, 3))
        panorama[..., 0] = np.arange(8)
        panorama[..., 1] = np.arange(4)[:, None]
        directions = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        sampled = sample_environment(panorama, directions, yaw_degrees=0)
        np.testing.assert_allclose(sampled[:4, 0], [3.5, 5.5, 3.5, 1.5])
        np.testing.assert_allclose(sampled[:, 1], [1.5, 1.5, 1.5, 1.5, 0., 3.])
        np.testing.assert_allclose(sample_environment(panorama, [[0, 1, 0]], yaw_degrees=90)[0],
                                   sampled[0], atol=1e-12)
        seam = sample_environment(panorama, [[-1, -.000001, 0], [-1, .000001, 0]], yaw_degrees=0)
        self.assertLess(abs(seam[0, 0]-seam[1, 0]), .0001)

    def test_sapien_face_centres_and_off_axis(self):
        from phymeo_rendering.environment import sapien_face_directions

        expected = np.array([[1,0,0],[-1,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0]])
        for index, centre in enumerate(expected):
            directions = sapien_face_directions(index, 3)
            np.testing.assert_allclose(directions[1,1], centre)
            np.testing.assert_allclose(np.linalg.norm(directions,axis=2), 1)
        # At the high/right of sampler +X: GPU y>0,z<0; public z>0,y<0.
        point = sapien_face_directions(0,3)[0,2]
        self.assertGreater(point[2],0)
        self.assertLess(point[1],0)

    def test_pin_and_extent_rejected_before_decode(self):
        from phymeo_rendering.environment import load_environment

        env = self.declaration(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 8192 +X 16384\n")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_environment(env)
        self.path.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_environment(env)

    def test_bad_packets_axis_and_transfer_are_rejected(self):
        from phymeo_rendering.environment import load_environment

        header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n"
        for data in [header+b"\x02\x02\x00\x08\x00", header+b"\x02\x02\x00\x08\x90\xff",
                     header[:-1], header.replace(b"-Y",b"+Y"), header.replace(b"rgbe",b"xyze"),
                     header.replace(b"FORMAT",b"EXPOSURE=2\nFORMAT")]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                load_environment(self.declaration(data))

    def test_six_owned_hdr_faces_and_immutable_inputs(self):
        from phymeo_rendering.environment import RadianceEnvironment, write_sapien_faces

        raw = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n"+bytes([128,64,32,129])*32
        env = self.declaration(raw, strength=2, yaw_degrees=-45, face_size=16)
        generated = self.root / "six"
        generated.mkdir()
        paths, report = write_sapien_faces(env, generated)
        self.assertEqual(len(set(paths)),6)
        for path in paths:
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"#?RADIANCE"))
            header, scanlines = data.split(b"-Y 16 +X 16\n")
            self.assertIn(b"32-bit_rle_rgbe", header)
            row = bytes((2,2,0,16)) + b"".join(bytes([16])+bytes([c])*16 for c in (128,64,32,130))
            self.assertEqual(scanlines, row*16)
        self.assertEqual(report["source_sha256"], env.sha256)
        self.assertEqual(report["peak_linear_radiance"], 2.)
        self.assertEqual(self.path.read_bytes(), raw)
        for args in [dict(strength=-1),dict(strength=math.inf),dict(yaw_degrees=True),dict(face_size=4000)]:
            with self.assertRaises((TypeError,ValueError)):
                RadianceEnvironment(env.uri, env.sha256, **args)
