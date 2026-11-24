"""Preservation of registered geometry and link-local R1 material boundaries."""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from _panda_scene import write_mesh
from galaxea_r1.finish_details import CAMERA, CAMERA_RIM, WHEEL, _camera_images, _camera_uv, apply_details
from phymeo_rendering.assets import Material
from phymeo_rendering.mesh_data import _local_path, load_mesh
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene
from phymeo_rendering.texture_data import load_texture


def _buffer(mesh):
    doc = json.loads(_local_path(mesh.uri).read_bytes())
    return base64.b64decode(doc["buffers"][0]["uri"].split(",")[1])


class R1DetailFinishTests(unittest.TestCase):
    def test_split_wheel_has_one_tread_and_identical_geometry(self):
        # A narrow ring in XZ shares positions across both glTF chunks. Neither
        # half spans the wheel on its own. A disconnected hub spans the origin.
        directions = np.array([(1,0,0), (0,0,1), (-1,0,0), (0,0,-1)], dtype=float)
        positions = np.concatenate((.07*directions, .062*directions))
        def strip(i):
            j = (i+1)%4
            return [(i,j,4+j), (i,4+j,4+i)]

        artifacts = Path(__file__).resolve().parents[3] / "artifacts"
        artifacts.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=artifacts) as folder:
            root = Path(folder)
            source = []
            for part in range(2):
                triangles = np.array(strip(part*2) + strip(part*2+1))
                ids, inverse = np.unique(triangles, return_inverse=True)
                p = list(positions[ids])
                t = list(inverse.reshape(-1, 3))
                if part == 1:
                    hub = len(p)
                    p.extend([(0,0,0), (.025,0,0), (0,0,.025)])
                    t.append((hub,hub+1,hub+2))
                source.append(write_mesh(root, f"wheel-{part}", p, [(0,1,0)]*len(p), t))
            material = Material("wheel", (.4,.4,.4), 0., .5)
            instances = tuple(Instance(i+1, m.id, "wheel", "wheel_motor_link2", Pose((0,1,0)))
                              for i,m in enumerate(source))
            scene = VisualScene("wheel-test", tuple(source), (material,), instances,
                                (DirectionalLight((0,0,-1),(1,1,1)),))
            result = apply_details(scene, root/"finishes")
            self.assertEqual(result.instances, scene.instances)
            self.assertEqual(result.lights, scene.lights)
            self.assertNotEqual(result.key, scene.key)
            self.assertEqual(result, apply_details(scene, root/"finishes"))
            for original, derived in zip(source, result.meshes):
                a,b = load_mesh(original), load_mesh(derived)
                self.assertEqual(original.id, derived.id)
                self.assertEqual((a.positions, a.normals, a.triangles), (b.positions, b.normals, b.triangles))
                self.assertTrue(_buffer(derived).startswith(_buffer(original)))
                for (x,_,z), (u,v) in zip(b.positions, b.texcoords):
                    self.assertEqual(u, .875 if np.hypot(x,z) > .06 else .125)
                    self.assertEqual(v, .5)
            mat, = result.materials
            color = load_texture(mat.base_color_map)
            rough = load_texture(mat.roughness_map.image)
            for x, swatch in zip((8,56), WHEEL):
                self.assertEqual(tuple(color.rgb8[3*x:3*x+3]), swatch[0])
                self.assertEqual(rough.rgb8[3*x], round(255*swatch[1]))

    def test_camera_charts_keep_windows_separate_inside_large_triangle(self):
        # The first triangle spans both lenses. Testing its midpoint catches
        # interpolation of vertex categories in place of true planar mapping.
        points = np.array([(0,0,0), (.120,0,0), (0,.0096,.002), (.120,.0096,.002),
                           (0,.0107,.002), (0,.016,.002), (0,0,-.020), (0,0,0),
                           (.120,0,.002)])
        groups = np.array([1]*7 + [0,1])
        islands = np.array([7]*6 + [8,9,8])
        components = [dict(group=0,faces=44), dict(group=1,faces=8064)]
        patches = [dict(group=7,low=[0,0,0],high=[.120,.016,.002]),
                   dict(group=8,low=[0,0,-.020],high=[.120,0,.002]),
                   dict(group=9,low=[0,0,0],high=[0,0,.002])]
        uv, _ = _camera_uv(points, groups, components, islands, patches)
        color, rough = _camera_images()
        self.assertEqual(color.shape, rough.shape)

        def check_texture(coordinates, swatch):
            # Require all four hardware bilinear samples to agree, not only
            # the nearest texel, for interiors of these broad finishes.
            h,w = color.shape[:2]
            x,y = np.floor(coordinates * [w,h] - .5).astype(int)
            for dy in (0,1):
                for dx in (0,1):
                    self.assertEqual(tuple(color[y+dy,x+dx]), swatch[0])
                    self.assertEqual(rough[y+dy,x+dx,0], round(255*swatch[1]))

        check_texture(uv[0], CAMERA[1])
        check_texture(uv[1], CAMERA[1])
        check_texture((uv[0] + uv[1]) / 2, CAMERA[0])
        check_texture(np.mean(uv[[0,1,2]], axis=0), CAMERA[0])
        check_texture(uv[4], CAMERA_RIM)
        check_texture(uv[5], CAMERA[0])
        for coords in uv[6:]:
            check_texture(coords, CAMERA[0])
        np.testing.assert_equal(uv[6], uv[8])
        with self.assertRaisesRegex(ValueError, "missing ZED aperture"):
            _camera_uv(points[:3], groups[:3], components, islands[:3], patches)


if __name__ == "__main__":
    unittest.main()
