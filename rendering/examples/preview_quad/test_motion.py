"""HOST-only movie/identity/metric contracts, no rendering."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import motion
import score


class QuadMotionContracts(unittest.TestCase):
    def test_all_extents_are_single_bounce_and_never_select_quality_overlay(self):
        for name,size in (("720p",[1280,720]),("540p",[960,540]),("360p",[640,360])):
            p=motion.descriptor("quad2",name)
            self.assertEqual(p["extent"],size)
            self.assertEqual(p["settings"]["indirect_bounces"],1)
            self.assertEqual(p["settings"]["spatial_filter"],"guided_settling")
            self.assertEqual(p["settings"]["reflection_mode"],"single_bounce")
        with self.assertRaises(ValueError):
            motion.descriptor("quad2","quality720")

    def test_cache_records_reconstructed_terminal_blank_lines_without_claiming_exact_bytes(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"capture.py"
            path.write_bytes(b"capture()\n")
            old=sha256(b"capture()\n\n").hexdigest()
            evidence=motion.source_equivalence(path,old)
            self.assertFalse(evidence["exact_bytes"])
            self.assertEqual(evidence["original_sha256"],old)
            self.assertEqual(evidence["terminal_newline_count"],2)
            self.assertTrue(motion.source_equivalence(path,sha256(path.read_bytes()).hexdigest())["exact_bytes"])

    def test_cache_does_not_accept_changed_capture_logic_or_interior_whitespace(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"capture.py"
            path.write_bytes(b"capture()\n")
            for old in (b"other_capture()\n",b"capture( )\n",b"capture()\n# changed\n"):
                with self.assertRaisesRegex(ValueError,"neither exact"):
                    motion.source_equivalence(path,sha256(old).hexdigest())
    def test_actual_budget_does_not_relabel_the_legacy_counter(self):
        row=dict(frames=[dict(history={"progressive":{"batches":3,"primary_samples_per_pixel":48}})])
        motion.actual_budget(row,2,"bound")
        f=row["frames"][0]
        self.assertEqual(f["actual_accumulated_proposal_budget"],6)
        self.assertEqual(f["legacy_nominal_primary_samples"],48)
        self.assertEqual(f["history"]["progressive"]["primary_samples_per_pixel"],48)

    def test_cached_reference_is_not_counted_twice_in_aggregate(self):
        row=dict(preset="baseline16-720p",channels="rgb",cache={"receipt_sha256":"saved"},
                 frames=[{"capture_ms":20.},{"capture_ms":40.}])
        result=motion.aggregate_measurements([row,deepcopy(row)],quick=True)["baseline16-720p/rgb"]
        self.assertEqual(result["completed_captures"],2)
        self.assertEqual(result["distinct_measured_groups"],1)
        self.assertEqual(result["mean_ms"],30.)
        self.assertIsNone(result["latency_gate_passed"])

    def test_two_real_rounds_are_both_in_recent_rank_p95(self):
        a=dict(preset="quad2-720p",channels="rgb",receipt_sha256="a",
               frames=[{"capture_ms":float(i)} for i in range(1,301)])
        b=deepcopy(a)
        b["receipt_sha256"]="b"
        result=motion.aggregate_measurements([a,b],quick=False)["quad2-720p/rgb"]
        self.assertEqual(result["completed_captures"],600)
        self.assertEqual(result["p95_ms"],285.)
        self.assertFalse(result["latency_gate_passed"])

    def test_repeat_binding_rejects_frame_or_rgb_or_history_changes(self):
        f=dict(pose_index=60,frame_key={"tick":60},time_s=2.,state_sha256="s",camera_sha256="c",
               rgb_sha256="r",sensors={"depth":"d"},history={"progressive":{"batches":1}})
        a=dict(extent=[1280,720],frames=[f])
        motion.same_repeat(a,deepcopy(a))
        for key,value in (("rgb_sha256","bad"),("camera_sha256","bad"),("time_s",0.)):
            b=deepcopy(a)
            b["frames"][0][key]=value
            with self.assertRaises(ValueError):
                motion.same_repeat(a,b)
        b=deepcopy(a)
        b["frames"][0]["history"]["progressive"]["batches"]=2
        with self.assertRaises(ValueError):
            motion.same_repeat(a,b)

    def test_fixed_roi_identities_and_thresholds_match_q0(self):
        self.assertEqual(score.ROI_IDS,{"white-shell":(150,151,154),"fingers":(156,158),
            "rubber":(2,),"left-arm":(112,113,115,117),"wrist-shell":(125,142,149)})
        self.assertEqual(score.RMSE_ALLOWANCE,.005)
        self.assertEqual(score.TEMPORAL_PAIR,(59,60))

    @staticmethod
    def fixture():
        pose=dict(position_m=[0.,0.,0.],orientation_xyzw=[0.,0.,0.,1.])
        camera=dict(world_pose=deepcopy(pose),width=16,height=16,fx_px=50.,fy_px=50.,cx_px=8.,cy_px=8.)
        old=dict(camera=camera,frame={"nodes":[dict(node_id="finger",world_pose=deepcopy(pose))]})
        new=deepcopy(old)
        new["frame"]["nodes"][0]["world_pose"]["position_m"][0]=.02
        scene={"instances":[dict(id=156,node_id="finger",local_pose=deepcopy(pose))]}
        geo=dict(instance_id=np.full((16,16),156,dtype=np.uint32),
                 depth=np.ones((16,16),dtype=np.float32),valid=np.ones((16,16),dtype=np.uint8))
        return scene,old,new,geo

    def test_rigid_motion_correspondence_is_one_pixel_not_a_screen_roll(self):
        scene,old,new,g=self.fixture()
        pairs,info=score.rigid_correspondences(scene,old,new,g,g)
        self.assertGreater(info["accepted_pixels"],16)
        self.assertTrue(np.all(pairs[:,0]-pairs[:,1]==1))

    def test_new_background_is_not_a_finger_correspondence(self):
        scene,old,new,g=self.fixture()
        oldg=deepcopy(g)
        oldg["instance_id"][:]=0
        oldg["valid"][:]=0
        pairs,_=score.rigid_correspondences(scene,old,new,oldg,g)
        self.assertEqual(len(pairs),0)

    def test_true_reference_brightness_change_is_subtracted(self):
        before=np.full((2,2,3),.2)
        after=np.full((2,2,3),.3)
        pairs=np.array([[0,0],[1,1],[2,2],[3,3]])
        value=score.temporal_p95(before,after,before,after,pairs)
        self.assertAlmostEqual(value["p95"],0.)
        bad=score.temporal_p95(before,after+.05,before,after,pairs)
        self.assertAlmostEqual(bad["p95"],.05)


if __name__=="__main__":
    unittest.main()

