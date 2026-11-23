"""HOST controls for unfiltered progressive Panda replay and history invalidation."""
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))


class ProgressivePandaStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = importlib.import_module("progressive_panda_study")

    def test_registered_path_keeps_original_geometry_camera_exposure_and_one_bounce(self):
        import diffuse_bounce_study as original
        p = self.study.frozen_profile()
        self.assertEqual(p["original_profile"], original.frozen_profile())
        self.assertEqual(p["bounces"], 1)
        self.assertEqual(p["milestones"], [1,4,16,64])
        self.assertEqual(p["seeds"], [0,1,2,3])
        self.assertEqual(p["warmup_batches"], 0)
        self.assertEqual(p["noise_space"], "scene-linear-before-exposure")
        self.assertEqual(p["spatial_filter"], None)
        self.assertEqual(p["legacy_max_rgb8"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            base,reference,variants = self.study.build_inputs(Path(temporary))
            old = original.build_cases(Path(temporary))["red"]
            self.assertEqual(base,old)
            self.assertEqual(int(np.sum(reference["ids"]==156)),2058)
            self.assertEqual(int(np.sum(reference["ids"]==158)),2062)
            self.assertEqual(tuple(variants),("camera","blocker","board","lights"))

    def test_mutations_really_change_pose_or_direction_only_and_keep_original_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            (scene,frame,camera),_,variants = self.study.build_inputs(Path(temporary))
            moved = variants["camera"]
            self.assertEqual(moved[0],scene)
            self.assertEqual(moved[1].nodes,frame.nodes)
            self.assertNotEqual(moved[2].world_pose,camera.world_pose)
            self.assertEqual(moved[2].world_pose.position_m,(.42,.02,.24))
            for name,node_name,position in (("blocker","blocker",(0.,0.,.055)),
                                             ("board","board",(.06,0.,.037))):
                s,f,c = variants[name]
                self.assertEqual(s,scene)
                self.assertEqual(c.world_pose,camera.world_pose)
                changes = [(a,b) for a,b in zip(frame.nodes,f.nodes) if a != b]
                self.assertEqual(len(changes),1)
                self.assertEqual(changes[0][1].node_id,node_name)
                self.assertEqual(changes[0][1].world_pose.position_m,position)
            s,f,c = variants["lights"]
            self.assertNotEqual(s.lights,scene.lights)
            self.assertEqual(s.lights[0].illuminance,scene.lights[0].illuminance)
            self.assertEqual(s.lights[0].direction,(-2**-.5,0.,-2**-.5))
            self.assertEqual(s.instances,scene.instances)
            self.assertEqual(s.materials,scene.materials)
            self.assertEqual(f.nodes,frame.nodes)
            self.assertEqual(c.world_pose,camera.world_pose)
            self.assertEqual(self.study.reason_for("lights"),"lights")
            self.assertEqual(self.study.reason_for("board"),"transforms")

    @staticmethod
    def diagnostic(batches=1,reason="new_session",resets=1):
        return dict(progressive=dict(enabled=True,bounces=1,batches=batches,
            primary_samples_per_pixel=16*batches,batch_index=batches-1,
            last_reset_reasons=[reason],reset_count=resets,
            accumulation_space="scene-linear-before-exposure",storage="GPU RGBA32F mean",max_batches=4096))

    def test_history_receipt_requires_exact_sample_count_and_real_reset_reason(self):
        actual = self.diagnostic(1,"camera",2)
        self.assertTrue(self.study.verify_history(actual,batches=1,reason="camera",previous_resets=1))
        for field,value in (("batches",4),("batch_index",1),("primary_samples_per_pixel",32),
                            ("bounces",2),("storage","CPU RGB average"),("reset_count",1)):
            bad = deepcopy(actual); bad["progressive"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.verify_history(bad,batches=1,reason="camera",previous_resets=1)
        bad = deepcopy(actual); bad["progressive"]["last_reset_reasons"] = ["new_session"]
        with self.assertRaises(ValueError):
            self.study.verify_history(bad,batches=1,reason="camera",previous_resets=1)

    def test_zero_based_batch_index_is_not_sample_count(self):
        value = self.diagnostic(64)
        self.assertTrue(self.study.verify_history(value,batches=64))
        value["progressive"]["batch_index"] = 64
        with self.assertRaises(ValueError):
            self.study.verify_history(value,batches=64)

    def test_linear_mean_requires_owned_fp32_exact_stage_and_preserves_original_words(self):
        array = np.array((((.1,.2,.3,1.),(.5,.6,.7,1.)),),dtype="<f4")
        packet = dict(width=2,height=1,batches=64,primary_samples_per_pixel=1024,
                      format="RGBA32F_LE",space="scene-linear-before-exposure",data=array.tobytes())
        decoded = self.study.decode_linear(packet,width=2,height=1,batches=64)
        self.assertEqual(decoded.tobytes(),array.tobytes())
        self.assertEqual(decoded.shape,(1,2,4))
        for field,value in (("batches",1),("primary_samples_per_pixel",16),("format","RGB8"),
                            ("space","inverse-sRGB"),("data",bytearray(array.tobytes()))):
            bad=dict(packet);bad[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.decode_linear(bad,width=2,height=1,batches=64)
        wrong=array.copy();wrong[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            self.study.decode_linear(dict(packet,data=wrong.tobytes()),width=2,height=1,batches=64)

    @staticmethod
    def noise_data():
        result={}
        for seed,offset in enumerate((-.03,-.01,.01,.03)):
            result[seed]={}
            for batches in (1,4,16,64):
                a=np.full((4,4,3),.01,dtype=np.float64)
                a[:,:,0]=.3+offset/np.sqrt(batches)
                result[seed][batches]=a
        return result,np.ones((4,4),bool)

    def test_noise_is_paired_across_independent_seeds_at_same_registered_pixels(self):
        values,mask=self.noise_data()
        report=self.study.noise_report(values,mask)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["stages"]["16"]["rms_ratio_to_batch1"],.25,delta=1e-12)
        self.assertAlmostEqual(report["stages"]["64"]["rms_ratio_to_batch1"],.125,delta=1e-12)
        self.assertAlmostEqual(report["stages"]["64"]["mean_red_excess"],.29,delta=1e-12)
        # Population of sample pixels is fixed; a dark pixel is never removed.
        for seed in values:
            for batch in values[seed]:
                values[seed][batch][0,0]=0
        self.assertEqual(self.study.noise_report(values,mask)["pixels"],16)

    def test_lower_variance_does_not_hide_a_drifting_radiance_mean(self):
        values,mask=self.noise_data()
        for seed in values:
            values[seed][64] = values[seed][64]+.5
        report=self.study.noise_report(values,mask)
        self.assertFalse(report["passed"])
        self.assertFalse(report["mean_stability_passed"])

    def test_identical_sequence_cannot_masquerade_as_independent_seed_noise(self):
        values,mask=self.noise_data()
        for seed in values:
            values[seed] = {b:a.copy() for b,a in values[0].items()}
        with self.assertRaisesRegex(ValueError,"independent|variance|seed"):
            self.study.noise_report(values,mask)

    def test_nonfinite_or_missing_seed_stage_is_not_silently_dropped(self):
        values,mask=self.noise_data()
        del values[3][16]
        with self.assertRaises(ValueError):
            self.study.noise_report(values,mask)
        values,mask=self.noise_data();values[2][4][0,0,0]=np.nan
        with self.assertRaises(ValueError):
            self.study.noise_report(values,mask)

    @staticmethod
    def planes(value):
        from phymeo_rendering.outputs import Channel
        return {c:(f"{c.value}:{value}").encode() for c in Channel}

    def test_restart_matches_fresh_full_frame_and_rejects_stale_history(self):
        before=self.planes("old");current=self.planes("new")
        result=self.study.compare_restart(before,current,dict(current),geometry_changed=True)
        self.assertTrue(result["fresh_identical"])
        self.assertTrue(result["old_frame_rejected"])
        with self.assertRaises(ValueError):
            self.study.compare_restart(before,before,current,geometry_changed=True)
        with self.assertRaises(ValueError):
            self.study.compare_restart(before,dict(before),dict(before),geometry_changed=True)

    def test_light_changes_rgb_in_same_geometry_and_any_sensor_drift_fails(self):
        from phymeo_rendering.outputs import Channel
        before=self.planes("old");current=dict(before);current[Channel.RGB]=b"new light"
        self.assertTrue(self.study.compare_restart(before,current,dict(current),geometry_changed=False)["fresh_identical"])
        changed=dict(current);changed[Channel.INSTANCE_ID]=b"wrong ids"
        with self.assertRaises(ValueError):
            self.study.compare_restart(before,changed,dict(changed),geometry_changed=False)

    def test_legacy_to_first_batch_quantization_bound_is_one_rgb8_not_a_filter(self):
        a=np.full((4,4,3),20,np.uint8);b=a.copy();b[0,0,0]+=1
        self.assertEqual(self.study.compare_legacy(a,b)["max_rgb8"],1)
        b[0,0,0]+=1
        with self.assertRaises(ValueError):
            self.study.compare_legacy(a,b)


    def test_actual_light_receipt_accepts_declared_tuples_and_checks_submitted_fp32(self):
        from phymeo_rendering.scene import DirectionalLight
        light=DirectionalLight((2**-.5,0.,-2**-.5),(3.,3.,3.))
        diagnostic=dict(effective_directional_lights=[dict(direction=light.direction,illuminance=light.illuminance)],
            last_request=dict(effective_directional_lights=[dict(
                direction=[.7071067690849304,0.,-.7071067690849304],illuminance=[3.,3.,3.])]))
        self.assertTrue(self.study._light_binding(diagnostic,(light,)))
        bad=deepcopy(diagnostic)
        bad["last_request"]["effective_directional_lights"][0]["illuminance"][0]=2.
        with self.assertRaises(ValueError):
            self.study._light_binding(bad,(light,))

if __name__=="__main__":
    unittest.main()

