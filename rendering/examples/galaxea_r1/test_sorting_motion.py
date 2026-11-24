"""Sorting decisions consume readings after station arrival and preserve part identity."""
from collections import Counter
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from galaxea_r1 import sorting_motion as motion
from galaxea_r1.sorting_layout import ITEM_KINDS, ITEM_NODES, LAYOUT
from phymeo_rendering.scene import NodePose, Pose, compose_pose


class _ControlFixture:
    """Exact tool-coordinate fixture for exercising the controller independent of IK."""
    floor_offset_m = .04
    center = .096
    joints = {j:SimpleNamespace(child=j+"-tip",origin=Pose()) for j in motion.FINGERS}

    def pose_nodes(self, joints, base):
        target = np.array([joints[k] for k in motion.ARM[:3]])
        rot = (0.,math.sqrt(.5),0.,math.sqrt(.5))
        hand = Pose(tuple(target+[0.,0.,self.center]),rot)
        nodes = [NodePose("left_gripper_link",hand)]
        for j,sign in zip(motion.FINGERS,(1.,-1.)):
            local = Pose((0.,sign*joints[j],0.))
            nodes.append(NodePose(self.joints[j].child, compose_pose(hand,local)))
        return nodes

    def contacts(self):
        return self.center,[dict(joint=j,sign=s,contact_at_zero_tool_m=[self.center,0.,0.],
                                 closed_command_m=LAYOUT.payload_size[1]/2)
                            for j,s in zip(motion.FINGERS,(1.,-1.))]


class SortingMotion(unittest.TestCase):
    def test_optical_decisions_precede_routing_and_own_the_saved_exposure(self):
        observed = ("hex","block","block","hex")  # Deliberately opposite to physical classes.
        reads = []
        def reader(sample, index):
            reads.append((sample,index))
            return observed[sample.active_index]
        robot = _ControlFixture()
        with patch.object(motion,"_calibrate",return_value=robot.contacts()), patch.object(motion,"_Arm") as arm:
            arm.return_value.solve.side_effect = lambda target,seed: (np.r_[target,[0.,0.,0.]],0.)
            samples, report = motion.build_sort_motion(robot,97,read_station=reader)
        self.assertEqual(len(samples),97)
        self.assertEqual(len(reads),4)
        self.assertFalse(report["classification_matches"])
        self.assertEqual([r["bin"] for r in report["routes"]],list(observed))
        self.assertEqual([r["slot_index"] for r in report["routes"]],[0,0,1,1])
        self.assertEqual(report["placed_count"],4)
        self.assertEqual(report["total_belt_travel_m"],4*LAYOUT.pitch)
        self.assertLess(report["max_conveyor_slip_m"],1e-12)
        self.assertLess(report["max_contact_gap_m"],1e-12)
        for i,((sample,index),cycle) in enumerate(zip(reads,report["cycles"])):
            self.assertIs(sample,samples[index])
            self.assertEqual(sample.active_index,i)
            self.assertEqual(sample.phase,"inspect")
            self.assertIsNone(sample.attached_index)
            self.assertEqual(tuple(sample.joints[j] for j in motion.FINGERS),(motion.OPEN,)*2)
            self.assertEqual(sample.belt_travel_m,samples[index-1].belt_travel_m)
            self.assertEqual(sample.belt_travel_m,samples[index+1].belt_travel_m)
            marks = cycle["proof_frames"]
            self.assertEqual(index,marks["stopped"])
            for j in range(i,4):
                pos = sample.payloads[j].position_m
                np.testing.assert_allclose(pos[:2], [LAYOUT.pick_xy[0],LAYOUT.pick_xy[1]+(j-i)*LAYOUT.pitch])
            self.assertEqual(samples[marks["pick"]].attached_index,i)
            lifted = samples[marks["lift"]].payloads[i].position_m[2]
            self.assertAlmostEqual(lifted,LAYOUT.support_z + LAYOUT.payload_size[2]/2 + LAYOUT.lift_m)
            self.assertIsNone(samples[marks["place"]].attached_index)
            self.assertEqual(samples[marks["release"]].placed_count,i+1)
            expected = tuple(cycle["center_m"])
            self.assertEqual(samples[marks["place"]].payloads[i].position_m,expected)
            self.assertEqual(samples[-1].payloads[i].position_m,expected)
            self.assertEqual(cycle["node"],ITEM_NODES[i])
        for j,pose in enumerate(samples[0].payloads):
            self.assertAlmostEqual(pose.position_m[1],LAYOUT.pick_xy[1]+(j+1)*LAYOUT.pitch)

    def test_capacity_validation_and_explicit_test_reader(self):
        occupied = Counter()
        motion._allocate(0,"block",occupied,LAYOUT)
        motion._allocate(1,"block",occupied,LAYOUT)
        with self.assertRaisesRegex(ValueError,"both slots"):
            motion._allocate(2,"block",occupied,LAYOUT)
        with self.assertRaisesRegex(ValueError,"unknown sort class"):
            motion._allocate(3,"other",occupied,LAYOUT)
        with self.assertRaisesRegex(ValueError,"explicit optical"):
            motion.build_sort_motion(None,97,read_station=None)
        reader = motion.scripted_classifier(ITEM_KINDS)
        self.assertTrue(reader.scripted_dry_run)
        self.assertEqual(reader(SimpleNamespace(active_index=2),12),ITEM_KINDS[2])

    def test_short_and_odd_frame_counts_include_each_event(self):
        for count in (33,61,97,420):
            cycles = motion._schedule(count,4)
            self.assertEqual(cycles[0][0],0)
            self.assertEqual(cycles[-1][-1],count-1)
            for i,edges in enumerate(cycles):
                self.assertTrue(np.all(np.diff(edges)>0))
                if i:
                    self.assertEqual(edges[0],cycles[i-1][-1])


if __name__ == "__main__":
    unittest.main()
