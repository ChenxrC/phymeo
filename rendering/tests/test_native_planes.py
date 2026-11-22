"""Compact resident attachment reads; recorders are not GPU evidence."""
import ctypes as ct
import itertools
import math
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend, _parameters
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Pose, VisualScene
from test_native_resident import fixture, frame, Recorder


def parameters(mask=1):
    key=FrameKey('empty','world',0,0,0,'authored')
    camera=Camera('camera',key,Pose(),3,1,2.,2.,1.5,.5,.1,5.)
    channels=tuple(c for index,c in enumerate(Channel) if mask&(1<<index))
    return _parameters(VisualScene('empty',(),(),()),CaptureRequest(camera,RenderPath.RASTER,channels),0,1.)


class CompactPublication(unittest.TestCase):
    def test_all_fifteen_combinations_and_request_order_have_exact_content(self):
        scene,mesh=fixture()
        for mask in range(1,16):
            with self.subTest(mask=mask):
                device=Recorder(); calls=[]
                device.info=lambda: dict(max_width=4096,max_height=4096,max_vertices=2000000,
                    resident_max_instanced_triangles=1048576,resident_render_planes_available=True)
                def render(matrices,params,width,height,requested):
                    calls.append(requested)
                    self.assertEqual(params.counts[2],bool(requested&1))
                    pixels=width*height
                    parts=(bytes((10,20,30,255))*pixels,struct.pack('<f',math.inf)*pixels,
                           bytes(4*pixels),bytes(pixels))
                    return b''.join(raw for i,raw in enumerate(parts) if requested&(1<<i))
                device.render_planes=render
                device.render=lambda *_: self.fail('must use declared compact capability')
                with patch('phymeo_rendering.backends.native.require_library'), \
                     patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                     patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device):
                    owner=NativeBackend(resident=True).open_scene(scene)
                    value=frame(); camera=Camera('camera',value.key,Pose(),3,1,2.,2.,1.5,.5,.1,5.)
                    channels=tuple(reversed(tuple(c for i,c in enumerate(Channel) if mask&(1<<i))))
                    result=owner.capture(value,CaptureRequest(camera,RenderPath.RASTER,channels))
                    self.assertEqual(tuple(p.channel for p in result.planes),channels)
                    self.assertEqual(calls,[mask])
                    for plane in result.planes:
                        if plane.channel is Channel.RGB: self.assertEqual(plane.data,bytes((10,20,30))*3)
                    self.assertEqual(owner.diagnostics()['last_request']['native_attachment_mask'],mask)
                    owner.close()

    def test_sensor_only_does_not_allocate_rgb_conversion(self):
        from phymeo_rendering.backends.native import _image_planes
        with patch('phymeo_rendering.backends.native.bytearray',side_effect=AssertionError('unrequested RGB allocation'),create=True):
            planes=_image_planes(struct.pack('<f',1.)*3,3,(Channel.FORWARD_DEPTH,),2)
            self.assertEqual(len(planes[0].data),12)

    def test_old_dll_keeps_full_readback_capability_explicit(self):
        scene,mesh=fixture(); device=Recorder()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device):
            owner=NativeBackend(resident=True).open_scene(scene)
            value=frame(); camera=Camera('camera',value.key,Pose(),3,1,2.,2.,1.5,.5,.1,5.)
            owner.capture(value,CaptureRequest(camera,RenderPath.RASTER,(Channel.RGB,)))
            self.assertEqual(owner.diagnostics()['last_request']['native_attachment_mask'],15)
            self.assertEqual(owner.diagnostics()['readback_channels_mode'],'legacy all attachments')
            owner.close()


class CompactBoundary(unittest.TestCase):
    def owner(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._handle=123; owner._uploaded=True; owner._instance_count=0
        owner._last_info=dict(max_width=4096,max_height=4096)
        owner._output_buffer=None; owner._output_capacity=0; owner._has_captured=False
        calls=[]
        def draw(handle,matrices,count,params,width,height,mask,output,size):
            calls.append((mask,size)); output[:size]=bytes([len(calls)])*size; return 0
        owner._render_planes_function=draw
        return owner,calls

    def test_reused_buffer_never_changes_an_earlier_published_byte_string(self):
        owner,calls=self.owner(); matrices=(ct.c_float*0)()
        first=owner.render_planes(matrices,parameters(1),3,1,1)
        buffer=owner._output_buffer
        second=owner.render_planes(matrices,parameters(8),3,1,8)
        self.assertIs(buffer,owner._output_buffer)
        self.assertEqual(first,b'\1'*12); self.assertEqual(second,b'\2'*3)
        self.assertEqual(calls,[(1,12),(8,3)])

    def test_mask_projection_and_missing_symbol_fail_before_native_submission(self):
        owner,calls=self.owner(); matrices=(ct.c_float*0)()
        for mask in (0,16,-1,True):
            with self.assertRaises((ValueError,TypeError)): owner.render_planes(matrices,parameters(1),3,1,mask)
        with self.assertRaisesRegex(ValueError,'RGB'):
            owner.render_planes(matrices,parameters(1),3,1,2)
        self.assertEqual(calls,[])
        owner._render_planes_function=None
        with self.assertRaises(BackendUnavailableError): owner.render_planes(matrices,parameters(1),3,1,1)

    def test_readback_info_layout_and_live_snapshot_are_named(self):
        from phymeo_rendering.backends._native_resident import ResidentReadbackInfo, ResidentDevice
        value=ResidentReadbackInfo(); value.readback_abi_version=1
        value.memory_property_flags=14; value.persistent_mapped=1; value.last_channel_mask=8
        value.capacity_bytes=39; value.copied_bytes=3; value.map_calls=1
        owner=ResidentDevice.__new__(ResidentDevice); owner._remember_readback(value)
        self.assertEqual(ct.sizeof(value),64)
        self.assertEqual(owner._readback['memory_property_flags'],14)
        self.assertTrue(owner._readback['host_cached'])
        self.assertIn('last live',owner._readback['snapshot'])


if __name__=='__main__': unittest.main()
