"""Falsifiable CPU reference controls; no device or rendering claim."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
_IMPORT_PATH=list(sys.path)
try:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from incoming import TemporalIncoming, decode_guide, STATUS
finally:
    sys.path[:]=_IMPORT_PATH
    del _IMPORT_PATH


def packet(tick=1,*,h=13,w=13):
    values=np.tile(np.linspace(1.,3.,w)[None,:,None],(h,1,3))
    normal=np.zeros((h,w,3));normal[...,2]=1
    return dict(incoming=values,normal_world=normal,depth=np.ones((h,w)),
        instance_id=np.ones((h,w),dtype=np.uint32),valid=np.ones((h,w),dtype=np.uint8),
        traced=np.ones((h,w),dtype=bool),camera_matrix=np.eye(4),instance_matrices={1:np.eye(4)},
        camera=dict(id="camera",width=w,height=h,fx_px=50.,fy_px=50.,cx_px=w/2,cy_px=h/2),
        key=dict(scene_key="scene",world_id="world",epoch=0,revision=tick-1,tick=tick,phase="accepted_endpoint"),
        time_s=tick/30,lighting_token="light",material_token="material",signal_token="true guide anchors")


class TemporalIncomingContracts(unittest.TestCase):
    def test_public_miss_is_infinite_depth_zero_id_and_not_history(self):
        p=packet();p["depth"][0,0]=np.inf;p["instance_id"][0,0]=0;p["valid"][0,0]=0;p["traced"][0,0]=False
        out=TemporalIncoming().update(p)
        self.assertEqual(out["age"][0,0],0)
        bad=deepcopy(p);bad["valid"][0,0]=1
        with self.assertRaises(ValueError):TemporalIncoming().update(bad)
        bad=deepcopy(p);bad["depth"][0,0]=np.nan
        with self.assertRaises(ValueError):TemporalIncoming().update(bad)
    def test_bound_presentation_adjacency_accepts_downsampled_physics_ticks_without_relabelling(self):
        a=packet();a["key"]["tick"]=2987
        a["sequence"]=dict(index=0,previous_key=None,source_sha256="a"*64)
        b=packet(2);b["key"]["tick"]=3040
        b["sequence"]=dict(index=1,previous_key=deepcopy(a["key"]),source_sha256="a"*64)
        engine=TemporalIncoming();engine.update(a);out=engine.update(b)
        self.assertEqual(out["diagnostics"]["current_tick"],3040)
        self.assertGreater(out["diagnostics"]["accepted_history_pixels"],16)
        self.assertEqual(out["diagnostics"]["reset_reasons"],[])
        for mutate in (lambda p:p["sequence"].update(index=2),
                       lambda p:p["sequence"]["previous_key"].update(tick=2988),
                       lambda p:p["sequence"].update(source_sha256="b"*64),
                       lambda p:p["key"].update(tick=2987)):
            bad=deepcopy(b);mutate(bad);engine=TemporalIncoming();engine.update(a)
            out=engine.update(bad)
            self.assertTrue(out["diagnostics"]["reset_reasons"])
            np.testing.assert_array_equal(out["incoming"],bad["incoming"])
    def test_first_frame_is_exact_and_does_not_claim_independent_spp(self):
        p=packet();out=TemporalIncoming().update(p)
        np.testing.assert_array_equal(out["incoming"],p["incoming"])
        self.assertEqual(out["diagnostics"]["reset_reasons"],["first_frame"])
        self.assertEqual(out["diagnostics"]["accepted_history_pixels"],0)
        self.assertNotIn("spp",out["diagnostics"])

    def test_identical_geometry_reuses_but_clamps_history_to_current_same_id_range(self):
        engine=TemporalIncoming(clamp_radius=1);a=packet();engine.update(a)
        b=packet(2);b["incoming"][6,6]=2.1
        result=engine.update(b)
        self.assertEqual(result["age"][6,6],2)
        self.assertGreater(result["incoming"][6,6,0],2.)
        self.assertLess(result["incoming"][6,6,0],2.1)
        c=packet(3);c["incoming"][:]=20.
        result=engine.update(c)
        np.testing.assert_array_equal(result["incoming"],c["incoming"])
        self.assertGreater(result["diagnostics"]["clamped_history_pixels"],0)

    def test_camera_translation_reprojects_one_pixel_using_actual_depth(self):
        engine=TemporalIncoming();engine.update(packet())
        b=packet(2);b["camera_matrix"][0,3]=.02
        out=engine.update(b)
        self.assertFalse(out["diagnostics"]["reset_reasons"])
        self.assertGreater(out["diagnostics"]["accepted_history_pixels"],16)
        self.assertEqual(int(out["previous_pixel_x"][6,6]),7)

    def test_lighting_epoch_world_and_size_reset_without_stale_blending(self):
        for mutate in (lambda p:p.update(lighting_token="other"),
                       lambda p:p["camera"].update(fx_px=51.),
                       lambda p:p["key"].update(epoch=1),lambda p:p["key"].update(world_id="other")):
            engine=TemporalIncoming();engine.update(packet());b=packet(2);mutate(b)
            out=engine.update(b)
            self.assertTrue(out["diagnostics"]["reset_reasons"])
            np.testing.assert_array_equal(out["incoming"],b["incoming"])
        engine=TemporalIncoming();engine.update(packet())
        self.assertIn("extent",engine.update(packet(2,h=15,w=15))["diagnostics"]["reset_reasons"])

    def test_jump_or_duplicate_clock_cannot_accumulate(self):
        engine=TemporalIncoming();engine.update(packet())
        self.assertIn("discontinuous_tick",engine.update(packet(3))["diagnostics"]["reset_reasons"])
        self.assertIn("discontinuous_tick",engine.update(packet(3))["diagnostics"]["reset_reasons"])

    def test_newly_revealed_other_id_never_inherits_blocker_history(self):
        engine=TemporalIncoming(scene_motion_policy="primary-only");a=packet()
        a["instance_matrices"][2]=np.eye(4);engine.update(a)
        b=packet(2);b["instance_matrices"][2]=np.eye(4);b["instance_id"][4:9,4:9]=2
        out=engine.update(b)
        self.assertEqual(out["age"][6,6],1)
        self.assertEqual(out["status"][6,6],STATUS["identity"])
        np.testing.assert_array_equal(out["incoming"][6,6],b["incoming"][6,6])

    def test_same_id_depth_disocclusion_and_normal_flip_reject(self):
        for modify,reason in ((lambda p:p["depth"].__setitem__((6,6),2.),"depth"),
                              (lambda p:p["normal_world"].__setitem__((6,6),[0.,0.,-1.]),"normal")):
            engine=TemporalIncoming();engine.update(packet());b=packet(2);modify(b)
            out=engine.update(b);self.assertEqual(out["status"][6,6],STATUS[reason])
            self.assertEqual(out["age"][6,6],1)

    def test_object_motion_default_resets_secondary_visibility_uncertainty(self):
        engine=TemporalIncoming();engine.update(packet());b=packet(2)
        b["instance_matrices"][1][0,3]=.02
        out=engine.update(b)
        self.assertIn("scene_transforms",out["diagnostics"]["reset_reasons"])
        exploratory=TemporalIncoming(scene_motion_policy="primary-only");exploratory.update(packet())
        out=exploratory.update(b)
        self.assertGreater(out["diagnostics"]["accepted_history_pixels"],16)
        self.assertTrue(out["diagnostics"]["unvalidated_secondary_visibility"])
        self.assertEqual(int(out["previous_pixel_x"][6,6]),5)

    def test_sh_fallback_or_untraced_guide_is_not_promoted_to_traced_history(self):
        engine=TemporalIncoming();engine.update(packet());b=packet(2);b["traced"][:]=False
        out=engine.update(b)
        self.assertEqual(out["diagnostics"]["accepted_history_pixels"],0)
        self.assertTrue(np.all(out["age"]==0))
        np.testing.assert_array_equal(out["incoming"],b["incoming"])

    def test_invalid_input_does_not_replace_accepted_history_and_inputs_are_owned(self):
        engine=TemporalIncoming();a=packet();engine.update(a);a["incoming"][:]=999.
        bad=packet(2);bad["incoming"][6,6,0]=np.nan
        with self.assertRaises(ValueError):engine.update(bad)
        out=engine.update(packet(2))
        self.assertEqual(out["diagnostics"]["previous_tick"],1)
        self.assertLess(out["incoming"].max(),4.)

    def test_guide_decoding_preserves_traced_and_outcome_flags(self):
        raw=np.zeros((1,2,4),dtype=np.float32);raw[...,:3]=1.
        code=512|(512<<10)
        raw[0,0,3]=code|(1<<20)|(3<<21);raw[0,1,3]=code
        decoded=decode_guide(raw)
        self.assertEqual(decoded["traced"].tolist(),[[True,False]])
        self.assertEqual(decoded["outcome"].tolist(),[[3,0]])
        self.assertGreater(decoded["normal_world"][0,0,2],.999)
        raw[0,0,3]=.5
        with self.assertRaises(ValueError):decode_guide(raw)


if __name__=="__main__":unittest.main()