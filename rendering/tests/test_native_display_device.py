"""Public display opt-in on software/hardware and every shipped panel grid."""
import os
from pathlib import Path
import unittest
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest,RenderPath
from phymeo_rendering.scene import FrameKey,Pose,SceneFrame,VisualScene
from phymeo_rendering.studio import StudioLighting
from test_reference_neutral import neutral_reference,code


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_DISPLAY_DLL'),'explicit native display device slot')
class NativeDisplayDeviceTests(unittest.TestCase):
    def test_public_modes_grids_constant_radiance_and_unchanged_sensors(self):
        library=Path(os.environ['PHYMEO_NATIVE_DISPLAY_DLL'])
        scene=VisualScene('display-empty',(),(),())
        key=FrameKey(scene.key,'world',0,0,0,'authored'); frame=SceneFrame(key,0.,())
        camera=Camera('fixed',key,Pose(),7,3,8.,8.,3.5,1.5,.1,10.)
        for mode in ('software','ray_query'):
            for grid in (2,4,8):
                results=[]
                for transform in ('clip','pbr-neutral'):
                    owner=NativeBackend(library_path=library,shader_directory=library.parent,
                        device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),
                        validation=True,resident=True,shadow_mode=mode,area_light_grid=grid,
                        studio=StudioLighting(panel_strength=0.,environment_radiance=.125),
                        exposure=16.,display_transform=transform).open_scene(scene)
                    try:
                        captured=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                        planes={plane.channel:plane.data for plane in captured.planes}; results.append(planes)
                        expected=255 if transform=='clip' else code(neutral_reference((.125,)*3,16.)[0])
                        self.assertTrue(all(abs(value-expected)<=1 for value in planes[Channel.RGB]))
                        self.assertEqual(owner.diagnostics()['display_transform'],transform)
                    finally: owner.close()
                    info=owner.diagnostics()['device']
                    self.assertTrue(all(info[key]==0 for key in ('live_contexts','live_frames','live_resources','validation_errors')))
                for channel in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID):
                    self.assertEqual(results[0][channel],results[1][channel])
                self.assertNotEqual(results[0][Channel.RGB],results[1][Channel.RGB])


    def test_public_fragment_all_variants_matches_unclipped_analytic_gray(self):
        from tempfile import TemporaryDirectory
        import math
        from test_native_ray_query_device import quad
        from phymeo_rendering.assets import Mesh,Material
        from phymeo_rendering.scene import Instance,NodePose,DirectionalLight
        library=Path(os.environ['PHYMEO_NATIVE_DISPLAY_DLL'])
        with TemporaryDirectory() as root:
            source=Path(root)/'gray.gltf'; quad(source,1.,1.)
            scene=VisualScene('gray-display',(Mesh('gray',str(source)),),
                (Material('gray',(.8,)*3,0.,1.),),(Instance(1,'gray','gray','gray'),),
                (DirectionalLight((0.,0.,-1.),(8.,)*3),))
            key=FrameKey(scene.key,'world',0,0,0,'authored')
            frame=SceneFrame(key,0.,(NodePose('gray',Pose((0.,0.,-2.))),))
            camera=Camera('fixed',key,Pose(),7,3,8.,8.,3.5,1.5,.1,10.)
            # Pixel center is exactly on normal incidence: diffuse+GGX with
            # alpha=1, F=.04, D=1/pi and both Smith factors one. No RGB inversion.
            import ctypes as ct
            linear=(.96*ct.c_float(.8).value+.01)/math.pi*8.
            for mode in ('software','ray_query'):
                for grid in (2,4,8):
                    results=[]
                    for transform in ('clip','pbr-neutral'):
                        owner=NativeBackend(library_path=library,shader_directory=library.parent,
                            device_index=int(os.environ.get('PHYMEO_NATIVE_DEVICE_INDEX','1')),
                            validation=True,resident=True,shadow_mode=mode,area_light_grid=grid,
                            exposure=2.,display_transform=transform).open_scene(scene)
                        try:
                            shot=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                            planes={plane.channel:plane.data for plane in shot.planes}; results.append(planes)
                            expected=255 if transform=='clip' else code(neutral_reference((linear,)*3,2.)[0])
                            self.assertTrue(all(abs(value-expected)<=2 for value in planes[Channel.RGB][30:33]))
                        finally: owner.close()
                        info=owner.diagnostics()['device']
                        self.assertTrue(all(info[name]==0 for name in ('live_contexts','live_frames','live_resources','validation_errors')))
                    for channel in (Channel.FORWARD_DEPTH,Channel.INSTANCE_ID,Channel.VALID):
                        self.assertEqual(results[0][channel],results[1][channel])
                    self.assertNotEqual(results[0][Channel.RGB],results[1][Channel.RGB])

if __name__=='__main__': unittest.main()
