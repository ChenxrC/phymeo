"""A crop changes pixel addresses and the principal point, not the optical rays."""
from dataclasses import replace
import unittest

from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Capture, Channel, ImagePlane
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, SceneFrame
from phymeo_rendering.tiled import capture_tiled


class TileCameraTests(unittest.TestCase):
    def setUp(self):
        key=FrameKey("chart","world",0,7,7,"authored")
        self.frame=SceneFrame(key,.5,())
        cam=Camera("view",key,Pose(),7,5,11.,14.,2.25,3.75,.1,10.)
        self.req=CaptureRequest(cam,RenderPath.RASTER,(Channel.RGB,))
        self.cameras=[]

    def supports(self,request):
        return True

    def capture(self,frame,request):
        cam=request.camera
        self.cameras.append(cam)
        x0,y0=self.req.camera.cx_px-cam.cx_px,self.req.camera.cy_px-cam.cy_px
        data=bytes(int(val) for y in range(cam.height) for x in range(cam.width)
                   for val in (x+x0,y+y0,10*(y+y0)+x+x0))
        return Capture(frame.key,frame.time_s,cam,(ImagePlane(Channel.RGB,data),))

    def test_off_axis_camera_odd_size_has_exact_pixel_order_and_source_pose(self):
        direct=self.capture(self.frame,self.req)
        self.cameras.clear()
        tiled=capture_tiled(self,self.frame,self.req,3,2)
        self.assertEqual(tiled,direct)
        self.assertEqual([(c.width,c.height) for c in self.cameras],
                         [(3,2),(3,2),(1,2)]*2+[(3,1),(3,1),(1,1)])
        for c in self.cameras:
            self.assertEqual((c.fx_px,c.fy_px,c.world_pose,c.frame_key),
                             (11.,14.,self.req.camera.world_pose,self.frame.key))

    def test_incomplete_or_wrong_exposure_never_produces_full_image(self):
        original=self.capture
        def wrong(frame,request):
            result=original(frame,request)
            return replace(result,time_s=1.)
        self.capture=wrong
        with self.assertRaisesRegex(ValueError,"exposure|source"):
            capture_tiled(self,self.frame,self.req,3,2)
        self.assertEqual(len(self.cameras),1)

    def test_capacity_and_unsupported_last_crop_are_rejected_before_capture(self):
        for value in (0,-1,True,2.5):
            with self.assertRaises((ValueError,TypeError)):
                capture_tiled(self,self.frame,self.req,value,2)
        huge=replace(self.req,camera=replace(self.req.camera,width=4096,height=4096))
        with self.assertRaisesRegex(ValueError,"tile"):
            capture_tiled(self,self.frame,huge,1,1)
        self.supports=lambda req:req.camera.width!=1
        with self.assertRaisesRegex(Exception,"support"):
            capture_tiled(self,self.frame,self.req,3,2)
        self.assertEqual(self.cameras,[])
