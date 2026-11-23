"""HOST registration and anti-stale controls for the real guided Panda motion."""
from copy import deepcopy
from dataclasses import asdict
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"examples"))


class GuidedMotionStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=importlib.import_module("guided_motion_study")

    def test_protocol_keeps_three_fixed_references_and_second_batch(self):
        p=self.study.frozen_profile()
        self.assertEqual(p["frames"],25)
        self.assertEqual(p["fps"],12)
        self.assertEqual(p["saved_batches"],[1,2,4])
        self.assertEqual(p["reference_poses"],[0,15,24])
        self.assertEqual(p["required_reveal_poses"],[15])
        self.assertEqual(p["seed"],0)
        self.assertEqual(p["reference_seed"],1)
        self.assertEqual(p["reference_batches"],16)
        self.assertEqual(p["display_error_space"],"inverse-sRGB display-linear")
        self.assertEqual(p["blocker_id"],202)
        self.assertEqual(p["right_finger_id"],158)
        self.assertEqual(p["msaa_samples"],1)
        self.assertEqual(p["expected_public_captures"],272)

    def test_real_scene_uses_original_fingers_board_light_and_material(self):
        import diffuse_bounce_study as original
        from phymeo_rendering.mesh_data import load_mesh
        with tempfile.TemporaryDirectory() as folder:
            cases=self.study.build_motion(Path(folder))
            old=original.build_cases(Path(folder)/"old")["red"]
            scene,frame,camera=cases[0]
            self.assertEqual(scene.materials,old[0].materials)
            self.assertEqual(scene.lights,old[0].lights)
            self.assertEqual([i for i in scene.instances if i.id!=202],
                             [i for i in old[0].instances if i.id!=202])
            instance=next(i for i in scene.instances if i.id==202)
            self.assertEqual(instance.node_id,"blocker")
            mesh=load_mesh(next(m for m in scene.meshes if m.id==instance.mesh_id))
            positions=np.asarray(mesh.positions)
            expected=2*np.asarray((.002,.030,.090),dtype="<f4").astype(np.float64)
            np.testing.assert_array_equal(np.ptp(positions,axis=0),expected)
            self.assertEqual((camera.width,camera.height),(1280,960))
            self.assertEqual((camera.fx_px,camera.fy_px,camera.near_m,camera.far_m),(1550.,1550.,.01,3.))
            self.assertEqual([n for n in frame.nodes if n.node_id!="blocker"],
                             [n for n in old[1].nodes if n.node_id!="blocker"])

    def test_trajectory_contains_25_distinct_actual_transforms_and_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            cases=self.study.build_motion(Path(folder))
            self.assertEqual(len(cases),25)
            cameras=[];blockers=[]
            for index,(scene,frame,camera) in enumerate(cases):
                self.assertEqual(frame.key.tick,index)
                self.assertEqual(frame.time_s,index/12)
                self.assertEqual(camera.frame_key,frame.key)
                self.assertEqual(frame.key.scene_key,scene.key)
                cameras.append(camera.world_pose.position_m)
                blockers.append(next(n.world_pose.position_m for n in frame.nodes if n.node_id=="blocker"))
            self.assertEqual(len(set(cameras)),25)
            self.assertEqual(len(set(blockers)),25)
            self.assertEqual(cameras[0],(.42,-.012,.24))
            self.assertEqual(cameras[-1],(.42,.012,.24))
            self.assertEqual(blockers[0],(.08,-.18,.14))
            self.assertEqual(blockers[-1],(.08,.18,.14))
            self.assertTrue(all(a[1]<b[1] for a,b in zip(blockers,blockers[1:])))

    @staticmethod
    def record(batch=1):
        from phymeo_rendering.outputs import Channel
        return dict(frame={"key":{"tick":15},"time_s":1.25},camera={"id":"actual-camera"},
                    batch=batch,plane_sha256={c.value:"plane-"+c.value for c in Channel},
                    raw_mean_sha256="actual-raw-mean")

    def test_same_inputs_and_raw_mean_are_bound_before_display_comparison(self):
        raw=self.record();filtered=deepcopy(raw)
        from phymeo_rendering.outputs import Channel
        filtered["plane_sha256"][Channel.RGB.value]="filtered-rgb"
        self.assertTrue(self.study.verify_pair(raw,filtered)["raw_mean_identical"])
        for field,value in (("frame",{"key":{"tick":16},"time_s":1.25}),
                            ("camera",{"id":"other-camera"}),("batch",2),
                            ("raw_mean_sha256","different-radiance")):
            bad=deepcopy(filtered);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.study.verify_pair(raw,bad)

    def test_every_geometry_plane_must_remain_byte_identical(self):
        from phymeo_rendering.outputs import Channel
        raw=self.record()
        for channel in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID):
            bad=deepcopy(raw);bad["plane_sha256"][channel.value]="changed"
            with self.subTest(channel=channel),self.assertRaises(ValueError):
                self.study.verify_pair(raw,bad)
        bad=deepcopy(raw);del bad["plane_sha256"][Channel.VALID.value]
        with self.assertRaises(ValueError):
            self.study.verify_pair(raw,bad)

    def test_second_and_fourth_batch_must_return_same_unfiltered_public_rgb(self):
        from phymeo_rendering.outputs import Channel
        for batch in (2,4):
            raw=self.record(batch)
            self.assertTrue(self.study.verify_pair(raw,deepcopy(raw))["rgb_identical"])
            bad=deepcopy(raw);bad["plane_sha256"][Channel.RGB.value]="blurred-later-frame"
            with self.subTest(batch=batch),self.assertRaises(ValueError):
                self.study.verify_pair(raw,bad)

    def test_reference_must_match_actual_pose_and_geometry_not_saved_old_pose(self):
        raw=self.record()
        reference=deepcopy(raw);reference["batch"]=16;reference["raw_mean_sha256"]="independent-seed"
        self.assertTrue(self.study.verify_reference(raw,reference))
        bad=deepcopy(reference);bad["frame"]["key"]["tick"]=14
        with self.assertRaises(ValueError):
            self.study.verify_reference(raw,bad)

    def test_actual_first_only_filter_receipt_requires_consistent_count(self):
        def diag(batch,mode="guided",count=16):
            return dict(spatial_filter=dict(mode=mode,scope="first-rgb-batch-only",
                last_capture_filtered=mode=="guided" and batch==1,filtered_captures=count,
                guide_source="same-exposure-gpu-direct-pass"))
        for batch in (1,2,4):
            self.assertTrue(self.study.verify_filter(diag(batch),mode="guided",batch=batch,filtered_count=16))
        for field,value in (("last_capture_filtered",False),("filtered_captures",15),("scope","every-batch")):
            bad=diag(1);bad["spatial_filter"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.study.verify_filter(bad,mode="guided",batch=1,filtered_count=16)

    def test_disabled_filter_cannot_claim_guide_or_hidden_display_mutation(self):
        d=dict(spatial_filter=dict(mode="none",scope="first-rgb-batch-only",last_capture_filtered=False,filtered_captures=0))
        self.assertTrue(self.study.verify_filter(d,mode="none",batch=1,filtered_count=0))
        d["spatial_filter"]["last_capture_filtered"]=True
        with self.assertRaises(ValueError):
            self.study.verify_filter(d,mode="none",batch=1,filtered_count=0)

    @staticmethod
    def geometry(ids,depth):
        ids=np.asarray(ids,dtype=np.uint32)
        return dict(INSTANCE_ID=ids,FORWARD_DEPTH=np.asarray(depth,dtype=np.float32),VALID=(ids!=0).astype(np.uint8))

    def test_reveal_separates_actual_right_finger_from_other_foreground(self):
        before=self.geometry([[202,202,202,202]],[[.2,.2,.2,.2]])
        now=self.geometry([[158,156,201,0]],[[.4,.4,.4,np.inf]])
        result=self.study.visibility_counts(before,now)
        self.assertEqual(result["revealed_foreground"],3)
        self.assertEqual(result["revealed_receivers"],{"156":1,"158":1})
        self.assertEqual(result["revealed_right_finger"],1)
        self.assertEqual(result["new_background_excluded"],1)

    def test_reveal_requires_previous_blocker_and_farther_finite_current_surface(self):
        before=self.geometry([[202,201,202,202]],[[.2,.2,.4,np.inf]])
        now=self.geometry([[158,158,158,158]],[[.4,.4,.2,.4]])
        self.assertEqual(self.study.visibility_counts(before,now)["revealed_right_finger"],1)
        self.assertEqual(self.study.visibility_counts(None,now)["revealed_foreground"],0)

    def test_media_names_are_lossless_per_pose_per_batch_and_reference_is_explicit(self):
        self.assertEqual(self.study.image_name("none",15,2),"none/frame-0015-b0002.png")
        self.assertEqual(self.study.image_name("guided",15,1),"guided/frame-0015-b0001.png")
        self.assertEqual(self.study.image_name("reference",15,16),"reference/frame-0015-b0016.png")
        for args in (("guided",25,1),("none",-1,1),("reference",14,16),("none",0,3)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                self.study.image_name(*args)


    def test_webp_probe_uses_supported_module_not_removed_animation_feature(self):
        class Features:
            @staticmethod
            def check_module(name):
                if name!="webp":raise AssertionError("wrong feature namespace")
                return True
        self.assertTrue(self.study.webp_available(Features))

if __name__=="__main__":
    unittest.main()