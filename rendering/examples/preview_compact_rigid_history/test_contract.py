"""Presentation binding and independently calculable rigid motion controls."""
from copy import deepcopy
import importlib
import unittest


def frame(tick,time,revision=None,position=(0.,0.,0.)):
    return dict(key=dict(scene_key="panda",world_id="archive:"+"a"*64,epoch=0,
                        revision=tick-1 if revision is None else revision,tick=tick,phase="accepted_endpoint"),
                time_s=time,nodes=[dict(node_id="finger",world_pose=dict(position_m=position,orientation_xyzw=(0.,0.,0.,1.)))])


class RigidContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=importlib.import_module("preview_compact_rigid_history.state")

    def key(self,value,presentation):
        return dict(scene="panda",world=value["key"]["world_id"],epoch=0,phase="accepted_endpoint",
            revision=value["key"]["revision"],tick=value["key"]["tick"],time=value["time_s"],camera_id="c",
            extent=(97,95),lens=(80.,80.,48.5,47.5,.01,20.),camera_pose=(0.,0.,0.,0.,0.,0.,1.),
            nodes=tuple((n["node_id"],(*n["world_pose"]["position_m"],*n["world_pose"]["orientation_xyzw"])) for n in value["nodes"]),
            lights=(),exposure=1.,signal="asset-bound",presentation=presentation)

    def test_actual_1600hz_to_30hz_ticks_are_not_unit_tick(self):
        frames=[frame(3200,2.),frame(3253,2.033125,position=(.001,0.,0.)),frame(3307,2.066875,position=(.002,0.,0.))]
        bindings=self.m.bind_presentations(frames,"a"*64,start_index=59)
        state=self.m.RigidHistoryKey()
        for n,(f,b) in enumerate(zip(frames,bindings)):
            plan=state.prepare(self.key(f,b))
            self.assertEqual(plan["sequence"],n)
            self.assertEqual(plan["reasons"],["new_session"] if n==0 else [])
            state.commit(plan)
        self.assertEqual(bindings[1]["previous_key"]["tick"],3200)
        self.assertEqual(bindings[2]["previous_time_s"],2.033125)

    def test_current_full_frame_not_only_tick_is_bound(self):
        original=frame(53,.033125)
        binding=self.m.bind_presentations([original],"a"*64)[0]
        self.m.validate_presentation(original,binding)
        changed=deepcopy(original);changed["nodes"][0]["world_pose"]["position_m"]=(.1,0.,0.)
        with self.assertRaisesRegex(ValueError,"digest"):self.m.validate_presentation(changed,binding)

    def test_predecessor_complete_key_time_and_digest_must_match(self):
        fs=[frame(53,.033125),frame(107,.066875)]
        bindings=self.m.bind_presentations(fs,"a"*64)
        for field,value in (("previous_time_s",.03),("previous_frame_sha256","b"*64),
                            ("previous_key",dict(fs[0]["key"],phase="step_input"))):
            state=self.m.RigidHistoryKey();state.commit(state.prepare(self.key(fs[0],bindings[0])))
            bad=dict(bindings[1],**{field:value})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"predecessor"):
                state.prepare(self.key(fs[1],bad))
            self.assertEqual(state.prepare(self.key(fs[1],bindings[1]))["sequence"],1)

    def test_gap_requires_explicit_reset_not_fake_source_tick_step(self):
        fs=[frame(53,.033125),frame(107,.066875),frame(160,.1)]
        bs=self.m.bind_presentations(fs,"a"*64)
        state=self.m.RigidHistoryKey();state.commit(state.prepare(self.key(fs[0],bs[0])))
        with self.assertRaisesRegex(ValueError,"presentation gap"):state.prepare(self.key(fs[2],bs[2]))
        state.reset()
        self.assertEqual(state.prepare(self.key(fs[2],bs[2]))["sequence"],0)

    def test_repeated_presented_frame_is_allowed_and_sample_sequence_changes(self):
        f=frame(53,.033125);b=self.m.bind_presentations([f],"a"*64)[0]
        state=self.m.RigidHistoryKey();state.commit(state.prepare(self.key(f,b)))
        self.assertEqual(state.prepare(self.key(f,b))["sequence"],1)

    def test_lights_and_camera_identity_reset_without_ignoring_adjacency(self):
        fs=[frame(53,.033125),frame(107,.066875)];bs=self.m.bind_presentations(fs,"a"*64)
        for name,value in (("lights",((1.,0.,0.,1.,1.,1.),)),("camera_id","other"),("exposure",2.)):
            state=self.m.RigidHistoryKey();state.commit(state.prepare(self.key(fs[0],bs[0])))
            key=self.key(fs[1],bs[1]);key[name]=value
            self.assertIn(name,state.prepare(key)["reasons"])

    def test_failed_prepare_and_stale_commit_leave_history_unchanged(self):
        f=frame(53,.033125);b=self.m.bind_presentations([f],"a"*64)[0]
        state=self.m.RigidHistoryKey();plan=state.prepare(self.key(f,b));state.commit(plan)
        with self.assertRaises(ValueError):state.commit(plan)
        self.assertEqual(state.prepare(self.key(f,b))["sequence"],1)

    def test_rigid_translation_maps_current_world_point_back_to_previous(self):
        # Object moved +X .2; the same material point moved from x=.3 to .5.
        previous=self.m.translation_matrix((.1,0.,0.));current=self.m.translation_matrix((.3,0.,0.))
        self.assertAlmostEqual(self.m.map_rigid_point((.5,.2,-2.),current,previous)[0],.3)
        self.assertEqual(self.m.map_rigid_point((.5,.2,-2.),current,previous)[1:],(.2,-2.))

    def test_rigid_rotation_transports_normal_without_translation(self):
        current=(0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.,1.,0.,3.,2.,1.,1.)
        identity=self.m.translation_matrix((0.,0.,0.))
        self.assertEqual(self.m.map_rigid_normal((0.,1.,0.),current,identity),(1.,0.,0.))

    def test_reflection_scale_and_nonfinite_are_explicitly_unsupported(self):
        good=list(self.m.translation_matrix((0.,0.,0.)))
        for value in (-1.,2.,float("nan")):
            bad=good.copy();bad[0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.m.validate_rigid(bad)

    def test_local_color_gate_rejects_new_shadow_and_large_color_change(self):
        self.assertTrue(self.m.guide_matches((.2,.3,.4),(.21,.31,.39)))
        self.assertFalse(self.m.guide_matches((.2,.3,.4),(.01,.01,.01)))
        self.assertFalse(self.m.incoming_matches((1.,.01,.01),(.01,.01,.01),(.04,.04,.04)))
        self.assertTrue(self.m.incoming_matches((.03,.03,.03),(.01,.01,.01),(.04,.04,.04)))

    def test_exact_sha_types_and_bad_clock_are_rejected(self):
        with self.assertRaises(ValueError):self.m.bind_presentations([frame(53,.033125)],"x"*64)
        with self.assertRaises(ValueError):self.m.bind_presentations([frame(107,.066875),frame(53,.033125)],"a"*64)


if __name__=="__main__":unittest.main()
