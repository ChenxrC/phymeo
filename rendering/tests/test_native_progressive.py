"""HOST contracts for bounded scene-linear progressive capture; no GPU oracle."""
from dataclasses import replace
import unittest

from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import FrameKey, SceneFrame, NodePose, Pose


class ProgressiveContract(unittest.TestCase):
    def fixture(self, tick=0):
        key = FrameKey('scene', 'world', 0, tick, tick, 'authored')
        frame = SceneFrame(key, float(tick), (NodePose('node', Pose()),))
        camera = Camera('view', key, Pose(), 40, 30, 30., 30., 20., 15., .1, 10.)
        return frame, camera

    def test_public_defaults_and_admission(self):
        backend = NativeBackend()
        self.assertFalse(backend.progressive)
        self.assertEqual(backend.indirect_bounces, 1)
        for value in (None, 1, 'yes'):
            with self.assertRaises(TypeError): NativeBackend(progressive=value)
        for value in (True, 1., None):
            with self.assertRaises(TypeError): NativeBackend(indirect_bounces=value)
        for value in (0, 3):
            with self.assertRaises(ValueError): NativeBackend(indirect_bounces=value)
        for options in ({}, {'resident':True}, {'resident':True, 'shadow_mode':'ray_query'}):
            with self.assertRaises(UnsupportedRequestError): NativeBackend(progressive=True, **options)
        with self.assertRaises(UnsupportedRequestError): NativeBackend(indirect_bounces=2)
        options=dict(resident=True,shadow_mode='ray_query',indirect_mode='diffuse_bounce')
        with self.assertRaises(UnsupportedRequestError): NativeBackend(progressive=True,msaa_samples=4,**options)
        for bounces in (1,2):
            self.assertTrue(NativeBackend(progressive=True,indirect_bounces=bounces,**options).progressive)

    def test_static_ticks_accumulate_but_prepare_alone_never_commits(self):
        from phymeo_rendering.backends._native_progressive import ProgressiveHistory
        history=ProgressiveHistory(2)
        key,reasons=history.prepare(*self.fixture())
        self.assertEqual(reasons, ['new_session'])
        self.assertEqual(history.diagnostics()['batches'], 0)
        history.commit(key,reasons)
        key,reasons=history.prepare(*self.fixture(9))
        self.assertEqual(reasons, [])
        history.commit(key,reasons)
        info=history.diagnostics()
        self.assertEqual((info['batches'],info['batch_index'],info['primary_samples_per_pixel']),(2,1,32))
        self.assertEqual(info['accumulation_space'],'scene-linear-before-exposure')

    def test_every_camera_geometry_world_epoch_and_extent_change_resets(self):
        from phymeo_rendering.backends._native_progressive import ProgressiveHistory
        frame,camera=self.fixture()
        changed_key=replace(frame.key,world_id='other')
        changed_epoch=replace(frame.key,epoch=1)
        cases=[(frame,replace(camera,fx_px=31.),'camera'),
               (frame,replace(camera,world_pose=Pose((.01,0.,0.))),'camera'),
               (frame,replace(camera,id='other'),'camera'),
               (frame,replace(camera,width=41),'extent'),
               (replace(frame,nodes=(NodePose('node',Pose((1e-12,0.,0.))),)),camera,'transforms'),
               (replace(frame,key=changed_key),replace(camera,frame_key=changed_key),'world'),
               (replace(frame,key=changed_epoch),replace(camera,frame_key=changed_epoch),'epoch')]
        for new_frame,new_camera,reason in cases:
            history=ProgressiveHistory(1)
            history.commit(*history.prepare(frame,camera))
            key,reasons=history.prepare(new_frame,new_camera)
            self.assertIn(reason,reasons)
            history.commit(key,reasons)
            self.assertEqual(history.diagnostics()['batches'],1)

    def test_explicit_reset_and_bound_are_not_silent_rollover(self):
        from phymeo_rendering.backends._native_progressive import ProgressiveHistory
        history=ProgressiveHistory(1)
        history.commit(*history.prepare(*self.fixture()))
        history.reset()
        self.assertEqual(history.diagnostics()['batches'],0)
        self.assertEqual(history.prepare(*self.fixture())[1],['explicit_reset'])
        for _ in range(4096): history.commit(*history.prepare(*self.fixture()))
        with self.assertRaisesRegex(UnsupportedRequestError,'4096'):
            history.prepare(*self.fixture())
        # A changed physical state can start a new bounded history.
        frame,camera=self.fixture()
        self.assertIn('camera',history.prepare(frame,replace(camera,fx_px=31.))[1])

    def test_default_does_not_touch_progressive_extension(self):
        from phymeo_rendering.backends._native_progressive import configure_progressive
        class Untouchable:
            def __getattr__(self,name): raise AssertionError(name)
        self.assertIsNone(configure_progressive(Untouchable(),False,1))


if __name__=='__main__': unittest.main()
