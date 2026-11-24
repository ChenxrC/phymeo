from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from phymeo_rendering.scene import Pose
from galaxea_r1.model import Model, _joints, _origin, _read_pinned


class ModelContract(unittest.TestCase):
    def test_parent_rotated_slider_and_unwrapped_wheel(self):
        tree = ET.fromstring('''<robot>
        <joint name="base_rot" type="continuous"><parent link="base_link"/><child link="arm"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="finger" type="prismatic"><parent link="arm"/><child link="tip"/>
        <origin xyz="1 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper=".1"/></joint>
        </robot>''')
        model = Model(None, ('base_link', 'arm', 'tip'), _joints(tree), 0., .1, {})
        from math import pi
        nodes = model.pose_nodes({'base_rot': 2*pi+pi/2, 'finger': .05}, Pose((1., 0., 0.)))
        self.assertAlmostEqual(nodes[-1].world_pose.position_m[0], 2.)
        self.assertAlmostEqual(nodes[-1].world_pose.position_m[1], 1.05)
        with self.assertRaises(ValueError):
            model.pose_nodes({'finger': .2})
        with self.assertRaises(ValueError):
            model.pose_nodes({'wrong_joint': .1})

    def test_urdf_origin_uses_fixed_xyz_roll_pitch_yaw(self):
        pose = _origin(ET.fromstring('<origin xyz="1 2 3" rpy="1.5707963267948966 0 1.5707963267948966"/>'))
        self.assertEqual(pose.position_m, (1., 2., 3.))
        for value in pose.orientation_xyzw:
            self.assertAlmostEqual(value, .5)

    def test_official_blob_identity_fails_on_changed_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'visual.STL').write_bytes(b'changed STL')
            with self.assertRaisesRegex(ValueError, 'pinned Git blob'):
                _read_pinned(root, 'visual.STL', {'visual.STL': '0'*40})
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                _read_pinned(root, 'other.STL', {})


if __name__ == '__main__':
    unittest.main()
