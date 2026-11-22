"""Independent energy and axis checks for native diffuse and gloss prefiltering."""
from hashlib import sha256
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


def square_cells(path, rgb):
    height,width=rgb.shape[:2]
    row=np.ones((height,width,4),dtype=np.uint8)*129
    row[...,:3]=rgb
    raw=f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()+row.tobytes()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


class DiffuseAndGloss(unittest.TestCase):
    def test_uniform_radiance_all_mips_and_lambert_coefficients(self):
        from phymeo_rendering.environment import RadianceEnvironment,prepare_native_environment

        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"white.hdr"
            rgb=np.broadcast_to([96,32,64],(8,16,3))
            data=prepare_native_environment(RadianceEnvironment(p.as_uri(),square_cells(p,rgb),face_size=16))
            self.assertEqual((data.side,data.mip_levels),(16,5))
            all_texels=sum(6*(16>>level)**2 for level in range(5))
            tex=np.frombuffer(data.rgba_f32le,"<f4").reshape(all_texels,4)
            np.testing.assert_array_equal(tex,np.broadcast_to([.75,.25,.5,1],tex.shape))
            sh=np.frombuffer(data.diffuse_sh_f32le,"<f4").reshape(9,4)
            np.testing.assert_allclose(sh[0,:3]/math.sqrt(4*math.pi),[.75,.25,.5],atol=1e-7)
            np.testing.assert_allclose(sh[1:],0,atol=1e-7)
            np.testing.assert_array_equal(sh[:,3],0)
            data.diagnostics["source_sha256"]="changed"
            self.assertNotEqual(data.diagnostics["source_sha256"],"changed")

    def test_half_sky_has_exact_cosine_response_and_rotates(self):
        from phymeo_rendering.environment import RadianceEnvironment,prepare_native_environment

        # Red radiance +X hemisphere, blue -X hemisphere. Analytically
        # the normalized Lambert integral is (1 + N.x)/2 for red.
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"hemisphere.hdr"
            rgb=np.full((16,32,3),[0,0,128],dtype=np.uint8)
            rgb[:,8:24,:]=[128,0,0]
            digest=square_cells(p,rgb)
            for yaw,expected_axis in [(0,3),(90,1)]:
                data=prepare_native_environment(RadianceEnvironment(p.as_uri(),digest,yaw_degrees=yaw,face_size=16))
                coeff=np.frombuffer(data.diffuse_sh_f32le,"<f4").reshape(9,4)[:,:3]
                np.testing.assert_allclose(coeff[0]/math.sqrt(4*math.pi),[.5,0,.5],atol=1e-6)
                np.testing.assert_allclose(coeff[expected_axis]*math.sqrt(3/(4*math.pi)),[.5,0,-.5],atol=1e-6)
                np.testing.assert_allclose(np.delete(coeff,[0,expected_axis],axis=0),0,atol=1e-6)
                glossy=np.frombuffer(data.rgba_f32le,"<f4")[:6*16*16*4].reshape(6,16,16,4)
                face = 0 if yaw==0 else 4
                np.testing.assert_array_equal(glossy[face,8,8],[1,0,0,1])
                coarse=np.frombuffer(data.rgba_f32le,"<f4")[-24:].reshape(6,4)
                self.assertGreater(coarse[face,0],coarse[face,2])
                self.assertGreater(coarse[face,0],.85)
                np.testing.assert_allclose(coarse[:,0]+coarse[:,2],1,atol=1e-6)
