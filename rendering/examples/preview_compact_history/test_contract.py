"""HOST contracts. Device evidence is produced separately by run_smoke."""
from copy import deepcopy
import importlib
import unittest


def declaration(tick=0, time=0.0):
    return dict(scene="s", world="w6", epoch=0, phase="authored", revision=tick,
                tick=tick, time=time, camera_id="c", extent=(97,95),
                lens=(80.,80.,48.5,47.5,.01,20.), camera_pose=(0.,0.,0.,0.,0.,0.,1.),
                nodes=(("a",(0.,0.,0.,0.,0.,0.,1.)),), lights=(), exposure=1., signal="bound")


class HistoryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module=importlib.import_module("preview_compact_history.state")

    def test_prepare_is_not_commit_and_failure_does_not_advance(self):
        state=self.module.HistoryKey()
        a=state.prepare(declaration())
        b=state.prepare(declaration())
        self.assertEqual(a,b)
        self.assertEqual(a["sequence"],0)
        self.assertEqual(a["reasons"],["new_session"])
        state.commit(a)
        self.assertEqual(state.prepare(declaration())["sequence"],1)

    def test_camera_motion_reprojects_but_geometry_change_resets(self):
        state=self.module.HistoryKey()
        state.commit(state.prepare(declaration()))
        moved=declaration(1,.01);moved["camera_pose"]=(.01,0.,0.,0.,0.,0.,1.)
        self.assertEqual(state.prepare(moved)["reasons"],[])
        moved["nodes"]=(("a",(.001,0.,0.,0.,0.,0.,1.)),)
        plan=state.prepare(moved)
        self.assertIn("nodes",plan["reasons"])
        self.assertEqual(plan["sequence"],0)

    def test_world_lens_lights_exposure_and_signal_are_invalidations(self):
        for key,value in (("world","w7"),("epoch",1),("phase","accepted"),("camera_id","other"),
                          ("extent",(98,95)),("lens",(81.,80.,48.5,47.5,.01,20.)),
                          ("lights",((1.,0.,0.,1.,1.,1.),)),("exposure",2.),("signal","other")):
            with self.subTest(key=key):
                state=self.module.HistoryKey();state.commit(state.prepare(declaration()))
                changed=declaration(1,.01);changed[key]=value
                self.assertIn(key,state.prepare(changed)["reasons"])

    def test_discontinuous_time_or_tick_resets(self):
        for tick,time in ((3,.01),(1,.2),(1,-.01)):
            state=self.module.HistoryKey();state.commit(state.prepare(declaration()))
            self.assertTrue(state.prepare(declaration(tick,time))["reasons"])

    def test_repeated_key_may_sample_again_but_cannot_claim_different_time(self):
        state=self.module.HistoryKey();state.commit(state.prepare(declaration()))
        self.assertEqual(state.prepare(declaration())["sequence"],1)
        with self.assertRaises(ValueError):state.prepare(declaration(0,.001))

    def test_bad_input_and_stale_commit_do_not_mutate(self):
        state=self.module.HistoryKey();a=state.prepare(declaration())
        bad=declaration();bad["time"]=float("nan")
        with self.assertRaises(ValueError):state.prepare(bad)
        state.commit(a)
        with self.assertRaises(ValueError):state.commit(a)
        self.assertEqual(state.prepare(declaration())["sequence"],1)

    def test_reset_restarts_sequence_and_source_owned(self):
        state=self.module.HistoryKey();source=declaration();plan=state.prepare(source)
        source["nodes"]=()
        self.assertNotEqual(plan["key"]["nodes"],())
        state.commit(plan);state.reset()
        self.assertEqual(state.prepare(declaration())["sequence"],0)
        self.assertEqual(state.prepare(declaration())["reasons"],["explicit"])

    def test_uint32_sequence_changes_actual_seed_and_replays(self):
        seeds=[self.module.sample_seed(7,n) for n in range(64)]
        self.assertEqual(seeds[0],7)
        self.assertEqual(len(set(seeds)),64)
        self.assertEqual(seeds,[self.module.sample_seed(7,n) for n in range(64)])
        with self.assertRaises(ValueError):self.module.sample_seed(0,2**32)

    def test_reprojection_analytic_translation_and_depth_rejection(self):
        # Camera +X translation sees the same world point move left.
        project=self.module.project_previous
        self.assertEqual(project((40.5,30.5),2.,(80.,80.,40.5,30.5),(.1,0.,0.)),(44.5,30.5,2.))
        self.assertTrue(self.module.accept_depth(2.,2.005))
        self.assertFalse(self.module.accept_depth(2.,2.007))
        self.assertFalse(self.module.accept_depth(float("inf"),2.))


if __name__=="__main__":unittest.main()
