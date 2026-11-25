"""Small optical/visibility regressions without graphics hardware or scene assets."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from _home_film_quality import _encode, _layered, _lens_camera, capture_film_frame, focus_distance
from _panda_scene import pose_matrix
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame


class FilmOpticsTests(unittest.TestCase):
    def setUp(self):
        self.key = FrameKey('room','optics',0,0,0,'authored')
        self.cam = Camera('lens',self.key,Pose(),80,40,90.,90.,40.,20.,.02,12.)

    def test_focus_integrates_fine_pixel_area_in_linear_display_light(self):
        y,x = np.indices((80,160))
        rgb = np.repeat((((x+y)%2)*255).astype('u1')[...,None],3,axis=2)
        linear,stats = _layered(rgb,np.ones((80,160),np.float32),self.cam,2,1.,.02)
        np.testing.assert_allclose(linear,.5,rtol=0,atol=1e-6)
        self.assertTrue(np.all(_encode(linear)==188))
        self.assertEqual(stats['requested_peak_coc_radius_px'],0.)

    def test_focused_near_surface_completely_blocks_defocused_background(self):
        rgb = np.zeros((40,80,3),np.uint8)
        rgb[:,:40,0] = 255
        rgb[:,41::2,2] = 255
        z = np.full((40,80),3.,np.float32)
        z[:,:40] = 1.
        result,stats = _layered(rgb,z,self.cam,1,1.,.09)
        self.assertTrue(np.all(_encode(result[:,:40])==[255,0,0]))
        self.assertLess(float(np.ptp(result[15:25,50:70,2])),.3)
        self.assertAlmostEqual(stats['requested_peak_coc_radius_px'],5.4,places=4)

    def test_near_silhouette_softens_over_observed_far_color(self):
        rgb = np.zeros((40,80,3),np.uint8)
        rgb[:,:40,0] = rgb[:,40:,2] = 255
        z = np.ones((40,80),np.float32)
        z[:,:40] = .45
        result,_ = _layered(rgb,z,self.cam,1,1.,.10)
        for x in (39,40):
            self.assertGreater(result[20,x,0],.1)
            self.assertGreater(result[20,x,2],.1)
        np.testing.assert_allclose(result[20,10],[1,0,0],atol=1e-5)
        np.testing.assert_allclose(result[20,69],[0,0,1],atol=1e-5)

    def test_physical_aperture_keeps_focus_plane_registered(self):
        def project(camera,p):
            m = pose_matrix(camera.world_pose)
            x,y,z = (np.asarray(p)-m[:3,3]) @ m[:3,:3]
            return np.array((camera.cx_px-camera.fx_px*x/z,camera.cy_px+camera.fy_px*y/z))
        for dx,dy in ((.025,.014),(-.015,.019)):
            lens = _lens_camera(self.cam,dx,dy,1.7)
            p = (.11,-.18,-1.7)
            self.assertAlmostEqual(focus_distance(self.cam,p),1.7)
            np.testing.assert_allclose(project(lens,p),project(self.cam,p),atol=1e-8)
            q = (.11,-.18,-.5)
            self.assertGreater(np.linalg.norm(project(lens,q)-project(self.cam,q)),1.)

    def test_visibility_comes_from_original_pinhole_pixel_centers(self):
        calls = []
        def capture(frame,request):
            c = request.camera
            calls.append((c,request.channels))
            count = c.width*c.height
            data = {Channel.RGB:bytes([117,68,201])*count,Channel.INSTANCE_ID:np.arange(count,dtype='<u4').tobytes()}
            return SimpleNamespace(frame_key=frame.key,time_s=frame.time_s,camera=c,
                    planes=[SimpleNamespace(channel=k,data=data[k]) for k in request.channels])
        rgb,ids,stats = capture_film_frame(SimpleNamespace(capture=capture),SceneFrame(self.key,0.,()),self.cam,
                                         focus_distance_m=1.,aperture_radius_m=0.,scale=2,inspect=True)
        self.assertEqual(rgb,bytes([117,68,201])*3200)
        self.assertEqual(calls[0][0].width,160)
        self.assertEqual(calls[1],(self.cam,(Channel.INSTANCE_ID,)))
        self.assertEqual(int(ids[-1,-1]),3199)
        self.assertEqual(stats['subpixels_per_output'],4)

    def test_window_backplate_preserves_foreground_and_off_axis_crop(self):
        from _home_showcase_room import with_home_backplate

        # Camera axes are world +X, +Z, -Y. The centre ray looks toward world +Y.
        quat = (2**-.5, 0., 0., 2**-.5)
        camera = Camera('window', self.key, Pose((-1., -1., 1.4), quat),
                        48, 36, 40., 40., 23.25, 17.75, .02, 12.)
        frame = SceneFrame(self.key, 0., ())
        channels = (Channel.FORWARD_DEPTH, Channel.RGB, Channel.INSTANCE_ID)
        request = CaptureRequest(camera, RenderPath.RASTER, channels)
        plate_id = 912
        ids = np.full((36, 48), 99, '<u4')
        ids[11:25, 13:33] = plate_id
        ids[15:23, 22:26] = 87  # An opaque object in front of the photographic plate.
        depth = np.where(ids == 87, .8, np.where(ids == plate_id, 5.3, 2.5)).astype('<f4')
        native_rgb = np.full((36, 48, 3), (17, 43, 103), np.uint8)
        source = dict(zip(channels, (depth, native_rgb, ids)))
        calls = []

        def capture(frame, request):
            cam = request.camera
            x = round(camera.cx_px - cam.cx_px)
            y = round(camera.cy_px - cam.cy_px)
            calls.append(request.channels)
            planes = tuple(ImagePlane(k, source[k][y:y+cam.height, x:x+cam.width].tobytes())
                           for k in request.channels)
            return Capture(frame.key, frame.time_s, cam, planes)

        y, x = np.indices((28, 40))
        photo = np.stack((7+5*x, 9+7*y, np.full_like(x, 210)), axis=-1).astype(np.uint8)
        scene = SimpleNamespace(instances=(SimpleNamespace(id=plate_id, mesh_id='home-window-view'),))
        with patch('_home_showcase_room.Image.open', return_value=Image.fromarray(photo)):
            session = with_home_backplate(SimpleNamespace(capture=capture), scene)

        result = session.capture(frame, request)
        self.assertEqual(tuple(p.channel for p in result.planes), channels)
        self.assertEqual(result.planes[0].data, depth.tobytes())
        self.assertEqual(result.planes[2].data, ids.tobytes())
        rgb = np.frombuffer(result.planes[1].data, 'u1').reshape(36, 48, 3)
        np.testing.assert_array_equal(rgb[ids != plate_id], native_rgb[ids != plate_id])

        # Independent ray at full pixel (18,16) intersects plate y=4.30.
        # The source's channel ramps identify horizontal UV, vertical UV and RGB order.
        wx = -1. + 5.3*(18.5-camera.cx_px)/40.
        wz = 1.4 + 5.3*(camera.cy_px-16.5)/40.
        tx = (.001+.998*(wx+6.05)/8.)*40-.5
        ty = (.999-.998*(wz+1.)/4.5)*28-.5
        np.testing.assert_allclose(rgb[16,18], (7+5*tx, 9+7*ty, 210), rtol=0, atol=1)

        # This asymmetric ROI includes both the silhouette and a margin of opaque wall.
        crop = replace(camera, width=34, height=26, cx_px=camera.cx_px-7, cy_px=camera.cy_px-5)
        cropped = session.capture(frame, replace(request, camera=crop, channels=(Channel.RGB,)))
        self.assertEqual(calls, [channels, (Channel.RGB, Channel.INSTANCE_ID)])
        self.assertEqual(tuple(p.channel for p in cropped.planes), (Channel.RGB,))
        np.testing.assert_array_equal(np.frombuffer(cropped.planes[0].data, 'u1').reshape(26,34,3),
                                      rgb[5:31,7:41])
