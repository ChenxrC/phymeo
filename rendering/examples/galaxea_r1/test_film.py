"""Independent no-slip and joint-travel checks; no mesh conversion or GPU."""
import math
import unittest

import numpy as np

from galaxea_r1.film import motion, wheel_geometry
from galaxea_r1.model import Joint, Model
from phymeo_rendering.scene import Pose, compose_pose


def linkage():
    # Positions, axes and limits from the pinned r1_v2_1_0.urdf. Omit cameras
    # and visual assets. The rolling constraint should hold for any tread radius.
    joints = {}
    def add(name, parent, child, xyz, axis, limits, kind="revolute"):
        joints[name] = Joint(kind, parent, child, Pose(xyz), axis, *limits)
    for i, (x, y) in enumerate(((.21516,.28),(.21516,-.28),(-.28084,0.)), 1):
        steer = f"steer_motor_link{i}"
        add(f"steer_motor_joint{i}", "base_link", steer, (x,y,.12375), (0,0,1), (-1.5708,1.5708))
        add(f"wheel_motor_joint{i}", steer, f"wheel_motor_link{i}", (0,0,-.09525), (0,1,0),
            (-math.inf,math.inf), "continuous")
    parent = "base_link"
    for i, (xyz, axis, limits) in enumerate((
        ((-.032845,0,.301),(0,1,0),(-1.1345,1.8326)), ((0,0,.4),(0,1,0),(-2.7925,2.5307)),
        ((0,0,.3),(0,-1,0),(-1.8326,1.5708)), ((0,0,.09962),(0,0,1),(-3.0543,3.0543))), 1):
        child = f"torso_link{i}"
        add(f"torso_joint{i}", parent, child, xyz, axis, limits)
        parent = child
    for side, s in (("left",1),("right",-1)):
        left = s == 1
        base = f"{side}_arm_base_link"
        add(f"{side}_arm_base_joint", "torso_link4", base,
            (.00077694,.097234,.30285) if left else (.00077778,-.097235,.30285),
            (0,0,0), (0,0), "fixed")
        parent = base
        for i, (xyz, axis, limits) in enumerate((
            ((-.0011149 if left else -.0011145,s*.0446,0),(0,s,0),(-2.8798,2.8798)),
            ((0,s*.1061,0),(0,0,-s),(0,3.2289)),
            ((-.34928,.019998 if left else -.020002,0),(0,0,-s),(-3.3161,0)),
            ((.02735,s*.069767,0),(1,0,0),(-2.8798,2.8798)),
            ((.2463,-.00049894 if left else .00050106,0),(0,s,0),(-1.6581,1.6581)),
            ((.058249,.00050025 if left else -.00049975,0),(1,0,0),(-2.8798,2.8798))), 1):
            child = f"{side}_arm_link{i}"
            add(f"{side}_arm_joint{i}", parent, child, xyz, axis, limits)
            parent = child
        gripper = f"{side}_gripper_link"
        add(f"{side}_gripper_joint", parent, gripper, (.1039,0,0), (0,0,0), (0,0), "fixed")
        for i, direction in ((1,s),(2,-s)):
            z = .00012067 if direction == 1 else -.00012053
            add(f"{side}_gripper_finger_joint{i}", gripper, f"{side}_gripper_finger_link{i}",
                (.03689,.013453*direction,z), (0,direction,0), (0,.05), "prismatic")
    radius = .08
    return Model(None, ("base_link", *(j.child for j in joints.values())), joints, radius-.0285, radius, {})


class FilmKinematics(unittest.TestCase):
    def setUp(self):
        self.model = linkage()
        self.wheels = wheel_geometry(self.model)

    def sample(self, t):
        root, angles = motion(self.model, self.wheels, t)
        return {p.node_id:p.world_pose for p in self.model.pose_nodes(angles,root)}, angles

    def test_each_tread_contact_has_zero_planar_velocity_while_axle_moves(self):
        step = 1e-6
        for t in (.12,.28,.88,.93):
            before, _ = self.sample(t-step)
            center, _ = self.sample(t)
            after, _ = self.sample(t+step)
            for i in (1,2,3):
                name = f"wheel_motor_link{i}"
                x,y,z,w = center[name].orientation_xyzw
                # Freeze the physical tread point that touches the plane now.
                # Transport that same point through the full steered/spun FK.
                contact = compose_pose(Pose(orientation_xyzw=(-x,-y,-z,w)),
                                       Pose((0.,0.,-self.model.wheel_radius_m)))
                a = compose_pose(before[name],contact).position_m
                b = compose_pose(after[name],contact).position_m
                self.assertLess(math.dist(a[:2],b[:2])/(2*step), 1e-6, (t,name))
                self.assertGreater(math.dist(before[name].position_m,after[name].position_m)/(2*step), .1)
                self.assertAlmostEqual(compose_pose(center[name],contact).position_m[2], 0., places=10)

    def test_arms_reach_and_jaws_open_then_close_while_drive_holds_position(self):
        beginning, _ = self.sample(.45)
        end, _ = self.sample(.79)
        for name in ("base_link", *(f"wheel_motor_link{i}" for i in (1,2,3))):
            self.assertEqual(beginning[name],end[name])
        samples = [self.sample(t) for t in np.linspace(0.,1.,121)]
        for side in ("left","right"):
            wrist_positions = []
            for nodes,_ in (samples[0], samples[80]):
                shoulder = nodes[f"{side}_arm_base_link"]
                wrist = nodes[f"{side}_gripper_link"]
                delta = np.subtract(wrist.position_m, shoulder.position_m)
                x,y,z,w = shoulder.orientation_xyzw
                inverse = Pose(orientation_xyzw=(-x,-y,-z,w))
                wrist_positions.append(compose_pose(inverse,Pose(tuple(delta))).position_m)
            self.assertGreater(wrist_positions[1][0] - wrist_positions[0][0], .15)
            gaps = [math.dist(nodes[f"{side}_gripper_finger_link1"].position_m,
                              nodes[f"{side}_gripper_finger_link2"].position_m)
                    for nodes,_ in samples]
            self.assertGreater(max(gaps)-gaps[0],.070)
            self.assertAlmostEqual(gaps[0],gaps[-1],places=7)
            for i in range(1,7):
                values = [q[f"{side}_arm_joint{i}"] for _,q in samples]
                self.assertGreater(max(values)-min(values),.10)
        for _,q in samples:
            for name,value in q.items():
                joint = self.model.joints[name]
                self.assertLessEqual(joint.lower,value)
                self.assertLessEqual(value,joint.upper)


if __name__ == "__main__":
    unittest.main()
