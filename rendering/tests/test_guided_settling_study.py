"""HOST registration/binding controls; synthetic fixtures are not device evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"examples"))


class GuidedSettlingStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=importlib.import_module("guided_settling_study")

    def test_fixed_protocol_includes_all_checkpoints_and_seventeenth_anchor(self):
        p=self.study.frozen_profile()
        self.assertEqual(p["poses"],[0,15,24])
        self.assertEqual(p["saved_batches"],[1,2,4,8,16,17])
        self.assertEqual(p["scored_batches"],[1,2,4,8,16])
        self.assertEqual(p["full_batches_per_pose"],17)
        self.assertEqual(p["reference_batches"],64)
        self.assertEqual(p["seed"],0)
        self.assertEqual(p["reference_seed"],1)
        self.assertEqual(p["fresh_pose"],15)
        self.assertEqual(p["expected_public_captures"],379)
        self.assertEqual(p["expected_raw_reads"],190)
        self.assertEqual(p["mode_order"],["raw","old","settling","reference","fresh-old","fresh-settling"])
        self.assertEqual(p["trajectory_scope"],"three entered states from the registered 25-pose input; not replay of the intervening motion")

    def test_generator_is_original_25_pose_input_not_synthetic_interpolation(self):
        import guided_motion_study as motion
        with tempfile.TemporaryDirectory() as directory:
            cases=self.study.build_inputs(Path(directory))
            self.assertEqual(cases,motion.build_motion(Path(directory)))
            self.assertEqual(len(cases),25)
            self.assertEqual([cases[i][1].key.tick for i in (0,15,24)],[0,15,24])

    @staticmethod
    def diagnostic(n,ordinal=0):
        k=min(n,15)
        weight=max(0.,1-math.sqrt((n-1)/15))
        actual=float(np.float32(1)-np.sqrt(np.float32(n-1)/np.float32(15))) if n<16 else 0.
        total=ordinal*1920+16*k*(k+1)//2
        count=ordinal*15+k
        return dict(spatial_filter=dict(mode="guided_settling",scope="settling-first-16-rgb-batches",
            last_capture_filtered=n<16,filtered_captures=count,last_filtered_fraction=actual,
            scheduled_filtered_fraction=weight,last_batch=n,filtered_primary_samples=total,
            last_filtered_primary_samples=16*n if n<16 else 0,guide_source="same-exposure-gpu-direct-pass"),
            device=dict(resident_settling_info=dict(settling_abi_version=1,enabled=1,last_batch=n,
                last_capture_filtered=int(n<16),last_filtered_fraction=actual,
                last_filtered_primary_samples=16*n if n<16 else 0,guide_valid=1,
                guide_draws=ordinal+1,filtered_rgb_captures=count,
                filtered_primary_samples=total,filter_dispatches=count)))

    def test_actual_fp32_weights_and_accumulated_filter_counts_are_checked(self):
        for n in (1,2,4,8,16,17):
            with self.subTest(n=n):self.assertTrue(self.study.verify_settling(self.diagnostic(n,2),n=n,ordinal=2))
        for field,value in (("last_filtered_fraction",1.),("last_batch",1),("filtered_primary_samples",32)):
            bad=self.diagnostic(2);bad["spatial_filter"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.study.verify_settling(bad,n=2,ordinal=0)

    def test_guide_must_be_drawn_once_per_actual_state_and_reused(self):
        for field,value in (("guide_draws",2),("guide_valid",0),("filter_dispatches",7),
                            ("filtered_primary_samples",128),("last_filtered_primary_samples",16)):
            bad=self.diagnostic(8);bad["device"]["resident_settling_info"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.study.verify_settling(bad,n=8,ordinal=0)

    def test_zero_tail_is_exactly_unfiltered_at_sixteen_and_seventeen(self):
        for n in (16,17):
            bad=self.diagnostic(n);bad["spatial_filter"]["last_capture_filtered"]=True
            with self.assertRaises(ValueError):self.study.verify_settling(bad,n=n,ordinal=0)
        with self.assertRaises(ValueError):self.study.verify_settling(self.diagnostic(1),n=0,ordinal=0)

    @staticmethod
    def row(n):
        from phymeo_rendering.outputs import Channel
        return dict(batch=n,frame={"key":{"tick":15},"time_s":1.25},camera={"id":"actual"},
            plane_sha256={c.value:c.value for c in Channel},raw_mean_sha256="same-real-raw")

    def test_first_and_zero_weight_tail_public_rgb_match_exactly(self):
        from phymeo_rendering.outputs import Channel
        for n in (1,16,17):
            raw=self.row(n);old=deepcopy(raw);new=deepcopy(raw)
            if n==1:
                old["plane_sha256"][Channel.RGB.value]="filtered-first"
                new["plane_sha256"][Channel.RGB.value]="filtered-first"
            self.assertTrue(self.study.compare_stage(raw,old,new)["raw_mean_identical"])
            new["plane_sha256"][Channel.RGB.value]="wrong-tail-or-first"
            with self.assertRaises(ValueError):self.study.compare_stage(raw,old,new)

    def test_intermediate_blend_changes_only_display_while_old_recovers_raw(self):
        from phymeo_rendering.outputs import Channel
        raw=self.row(4);old=deepcopy(raw);new=deepcopy(raw)
        new["plane_sha256"][Channel.RGB.value]="actual-display-blend"
        self.assertFalse(self.study.compare_stage(raw,old,new)["rgb_old_new_identical"])
        for field,value in (("raw_mean_sha256","changed-radiance"),("frame",{"key":{"tick":14}})):
            bad=deepcopy(new);bad[field]=value
            with self.assertRaises(ValueError):self.study.compare_stage(raw,old,bad)
        old["plane_sha256"][Channel.RGB.value]="still-filtered-old"
        with self.assertRaises(ValueError):self.study.compare_stage(raw,old,new)

    def test_same_state_guide_hash_cannot_silently_change_after_first(self):
        first=dict(raw_sha256="current-guide",metadata=dict(width=1280,height=960,format="RGBA8_UNORM",space="display-srgb-direct"))
        self.assertTrue(self.study.verify_cached_guide(first,deepcopy(first)))
        bad=deepcopy(first);bad["raw_sha256"]="stale-or-regenerated-guide"
        with self.assertRaises(ValueError):self.study.verify_cached_guide(first,bad)

    def source_fixture(self,root):
        source=root/"source";source.mkdir()
        cases=self.study.build_inputs(source/"geometry")
        protocol=dict(scene=asdict(cases[0][0]),poses=[dict(pose_id=i,frame=asdict(f),camera=asdict(c)) for i,(_,f,c) in enumerate(cases)],
            mesh_sha256={m.id:self.study.motion.original.load_mesh(m).source_sha256 for m in cases[0][0].meshes})
        text=json.dumps(protocol,allow_nan=False)
        (source/"protocol.json").write_text(text,encoding="utf-8")
        visibility=[]
        # Pure binding fixture: all-background arrays, not an alleged physical capture.
        arrays=dict(INSTANCE_ID=np.zeros((960,1280),np.uint32),FORWARD_DEPTH=np.full((960,1280),np.inf,np.float32),VALID=np.zeros((960,1280),np.uint8))
        for pose in (0,14,15,23,24):
            name=f"geometry/frame-{pose:04d}.npz";np.savez_compressed(source/name,**arrays)
            visibility.append(dict(pose_id=pose,geometry_file=name,geometry_sha256=sha256((source/name).read_bytes()).hexdigest()))
        summary=dict(status="complete",protocol_sha256=sha256((source/"protocol.json").read_bytes()).hexdigest(),visibility=visibility)
        (source/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
        return source,self.study.build_inputs(root/"current")

    def test_prior_geometry_binding_checks_every_required_current_and_predecessor(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cases=self.source_fixture(Path(folder))
            binding,geometry=self.study.bind_motion_source(cases,source)
            self.assertEqual(set(geometry),{0,14,15,23,24})
            self.assertTrue(binding["full_input_equal"])
            self.assertEqual(binding["predecessor_for_pose15"],14)

    def test_replaced_geometry_bytes_are_rejected_before_disocclusion_use(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cases=self.source_fixture(Path(folder))
            with (source/"geometry/frame-0014.npz").open("ab") as stream:stream.write(b"tampered")
            with self.assertRaises(ValueError):self.study.bind_motion_source(cases,source)

    def test_wrong_current_camera_or_source_clock_cannot_borrow_prior_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cases=self.source_fixture(Path(folder));cases=list(cases)
            s,f,c=cases[15];cases[15]=(s,replace(f,time_s=f.time_s+.001),c)
            with self.assertRaises(ValueError):self.study.bind_motion_source(cases,source)

    def test_sixteenth_anchor_never_relabels_reference_or_sampling_time(self):
        self.assertEqual(self.study.image_name("old",15,2),"old/pose-0015-b0002.png")
        self.assertEqual(self.study.image_name("settling",15,17),"settling/pose-0015-b0017.png")
        self.assertEqual(self.study.image_name("reference",15,64),"reference/pose-0015.png")
        for args in (("old",14,2),("reference",15,16),("settling",15,3)):
            with self.subTest(args=args),self.assertRaises(ValueError):self.study.image_name(*args)


    def test_actual_weight_uses_each_native_fp32_operation_not_one_final_cast(self):
        data=self.diagnostic(2)
        # Actual documented float divide, float sqrt, float subtraction.
        for d in (data["spatial_filter"],data["device"]["resident_settling_info"]):
            d["last_filtered_fraction"]=0.7418011426925659
        self.assertTrue(self.study.verify_settling(data,n=2,ordinal=0))
        wrong=deepcopy(data)
        for d in (wrong["spatial_filter"],wrong["device"]["resident_settling_info"]):
            d["last_filtered_fraction"]=0.7418010830879211
        with self.assertRaises(ValueError):self.study.verify_settling(wrong,n=2,ordinal=0)

if __name__=="__main__":unittest.main()