import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from galaxea_r1.model import Joint, Model
from galaxea_r1.workshop_motion import ARM, FINGERS, LAYOUT, OPEN, _Arm, _calibrate, _path
from phymeo_rendering.scene import Pose, compose_pose


class WorkshopMotion(unittest.TestCase):
    def test_only_closed_gripper_crosses_between_fixtures(self):
        resting_z = LAYOUT.support_z + LAYOUT.payload_size[2]/2
        pick, place = np.r_[LAYOUT.pick_xy, resting_z], np.r_[LAYOUT.place_xy, resting_z]
        for t, pose, width in ((.29,pick,0.), (.37,pick,1.), (.47,pick+[0,0,.145],1.),
                               (.66,place+[0,0,.145],1.), (.76,place,1.), (.84,place,0.)):
            p, closed, _ = _path(t,LAYOUT)
            np.testing.assert_allclose(p,pose,atol=1e-12)
            self.assertAlmostEqual(closed,width)
        for t in (.48,.52,.56,.60,.65):
            p, closed, phase = _path(t,LAYOUT)
            self.assertEqual((closed,phase),(1.,"transfer"))
            self.assertAlmostEqual(p[2],resting_z+LAYOUT.lift_m)
        self.assertGreater(_path(1.,LAYOUT)[0][2],resting_z+.15)

    def test_contact_uses_triangle_intersection_at_stock_top(self):
        model = SimpleNamespace(joints={}, scene=SimpleNamespace(meshes=[],instances=[]))
        geometry = {}
        for k, sign in zip(FINGERS,(1.,-1.)):
            model.joints[k] = Joint("prismatic","left_gripper_link",k+"-finger",Pose(),(0,sign,0),0.,.05)
            model.scene.meshes.append(SimpleNamespace(id=k))
            model.scene.instances.append(SimpleNamespace(mesh_id=k,node_id=k+"-finger",local_pose=Pose()))
            geometry[k] = SimpleNamespace(positions=[(.02,-.010*sign,0.),(.08,.003*sign,.005),
                                                    (.08,.007*sign,-.005)], triangles=[(0,1,2)])
        with patch("phymeo_rendering.mesh_data.load_mesh", side_effect=lambda mesh: geometry[mesh.id]):
            tcp, contacts = _calibrate(model,LAYOUT)
        self.assertAlmostEqual(tcp,.064+.034)
        for row in contacts:
            expected = -.010 + (.065-.020)/(.080-.020)*.013
            self.assertAlmostEqual(row["sign"]*row["inner_y_at_zero_m"], expected)
            self.assertAlmostEqual(expected+row["closed_command_m"], LAYOUT.payload_size[1]/2)
            self.assertGreater(OPEN-row["closed_command_m"], .004)
            self.assertAlmostEqual(row["contact_at_zero_tool_m"][0], .065)

    def test_ik_reaches_independent_fk_pose_with_constant_tool_orientation(self):
        specs = (((0,.04,0),(0,1,0)), ((0,.106,0),(0,0,-1)), ((-.349,.020,0),(0,0,-1)),
                 ((.027,.070,0),(1,0,0)), ((.246,-.0005,0),(0,1,0)), ((.058,.0005,0),(1,0,0)))
        joints, links = {}, ["base_link"]
        for index,(name,(origin,axis)) in enumerate(zip(ARM,specs)):
            child = f"left_arm_link{index+1}"
            joints[name] = Joint("revolute",links[-1],child,Pose(origin),axis,-3.1,3.1)
            links.append(child)
        joints["left_gripper_joint"] = Joint("fixed",links[-1],"left_gripper_link",Pose((.104,0,0)),
                                             (0,0,0),0.,0.)
        links.append("left_gripper_link")
        model = Model(None,tuple(links),joints,0.,.07,{})
        q = np.array([.72,2.1,-2.1,0.,math.pi/2-.72,0.])
        base = Pose((-.09,.02,.86))
        contact = Pose((.098,0,0))
        hand = model.pose_nodes(dict(zip(ARM,q)),base)[-1].world_pose
        target = np.array(compose_pose(hand,contact).position_m)
        arm = _Arm(model,{},base,contact.position_m[0])
        solved, residual = arm.solve(target,q+[.025,-.02,.03,.015,-.02,.04])
        self.assertLess(residual,1e-6)
        result = model.pose_nodes(dict(zip(ARM,solved)),base)[-1].world_pose
        np.testing.assert_allclose(compose_pose(result,contact).position_m,target,atol=2e-6)
        self.assertGreater(abs(np.dot(result.orientation_xyzw,hand.orientation_xyzw)),1.-1e-7)


if __name__ == "__main__":
    unittest.main()
