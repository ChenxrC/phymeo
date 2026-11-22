"""Native studio ABI and source-axis checks; no graphics device is initialized."""
import ctypes as ct
import math
import unittest
from pathlib import Path
from unittest.mock import patch
from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.scene import VisualScene


class NativeStudioContract(unittest.TestCase):
    def test_shared_type_keeps_sapien_compatibility_and_native_opt_in(self):
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.backends.sapien import SapienStudio, SapienConfig
        from phymeo_rendering.pipeline import RenderPath
        self.assertIs(StudioLighting,SapienStudio)
        studio=StudioLighting()
        self.assertIsNone(NativeBackend().studio)
        self.assertIs(NativeBackend(resident=True,studio=studio).studio,studio)
        self.assertIs(SapienConfig(path=RenderPath.PATH_TRACE,studio=studio).studio,studio)
        with self.assertRaisesRegex(UnsupportedRequestError,'resident'):
            NativeBackend(studio=studio)
        with self.assertRaisesRegex(TypeError,'studio'):
            NativeBackend(resident=True,studio='neutral')

    def test_layout_and_default_environment_match_reference_rgbe(self):
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.backends._native_studio import pack_studio
        value,info=pack_studio(StudioLighting(environment_radiance=.13))
        self.assertEqual(ct.sizeof(value),144)
        self.assertEqual(tuple(value.environment),(.1298828125,)*3+(1.,))
        self.assertEqual(info['environment']['source_requested_radiance'],.13)
        self.assertEqual(len(value.panels),2)
        self.assertEqual(len(info['basis_sources']),3)

    def test_panel_axes_follow_shortest_rotation_and_actual_gl_shape(self):
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.backends._native_studio import pack_studio
        from phymeo_rendering.backends._sapien_studio import studio_recipe
        setting=StudioLighting(center_m=(.18,0.,.42))
        packed,info=pack_studio(setting)
        recipe=studio_recipe(setting)
        # Independent Rodrigues rotation of the entity +Y/+Z shape half axes.
        def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
        for actual,panel in zip(packed.panels,recipe['panels'],strict=True):
            d=panel['emission_direction_world']; axis=(0.,-d[2],d[1]); c=d[0]
            def rotate(v):
                once=cross(axis,v); twice=cross(axis,once)
                return tuple(v[i]+once[i]+twice[i]/(1+c) for i in range(3))
            width,height=panel['half_size_m']
            u=rotate((0.,width,0.)); v=rotate((0.,0.,height))
            for actual_axis,expected in ((actual.half_u,u),(actual.half_v,v)):
                for a,b in zip(actual_axis[:3],expected): self.assertAlmostEqual(a,b,places=6)
                self.assertEqual(actual_axis[3],0.)
            normal=cross(actual.half_u,actual.half_v)
            self.assertGreater(sum(a*b for a,b in zip(normal,d)),0.)
            self.assertAlmostEqual(math.hypot(*normal),width*height,places=6)
            self.assertEqual(actual.position[3],0.)
            self.assertEqual(actual.emission[3],0.)
            # Reflecting edgeX preserves the four actual centered rectangle corners.
            native={tuple(round(actual.position[i]+su*u[i]+sv*v[i],5) for i in range(3)) for su in (-1,1) for sv in (-1,1)}
            reference={tuple(round(actual.position[i]-su*u[i]+sv*v[i],5) for i in range(3)) for su in (-1,1) for sv in (-1,1)}
            self.assertEqual(native,reference)

    def test_panel_scale_preserves_power_and_zero_strength_is_explicit(self):
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.backends._native_studio import pack_studio
        small,_=pack_studio(StudioLighting())
        large,_=pack_studio(StudioLighting(panel_size_scale=2.))
        zero,_=pack_studio(StudioLighting(panel_strength=0.,environment_radiance=0.))
        for a,b,z in zip(small.panels,large.panels,zero.panels,strict=True):
            self.assertAlmostEqual(math.hypot(*b.half_u),2*math.hypot(*a.half_u),places=6)
            self.assertAlmostEqual(b.emission[0]*4,a.emission[0],places=6)
            self.assertEqual(tuple(z.emission),(0.,0.,0.,0.))
        self.assertEqual(tuple(zero.environment),(0.,0.,0.,1.))

    def test_default_never_calls_optional_setter(self):
        from test_native_resident import Recorder
        device=Recorder()
        device.set_studio=lambda *_: self.fail('default must not enable studio')
        with patch('phymeo_rendering.backends.native.require_library'),patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device):
            session=NativeBackend(resident=True).open_scene(VisualScene('empty',(),(),()))
            self.assertIsNone(session.diagnostics()['studio'])
            session.close()

    def test_studio_uploads_after_geometry_and_publishes_detached_evidence(self):
        from phymeo_rendering.studio import StudioLighting
        from test_native_resident import Recorder
        device=Recorder(); calls=[]
        old_upload=device.upload
        def upload(*args): calls.append('geometry'); old_upload(*args)
        device.upload=upload
        device.set_studio=lambda value: calls.append(('studio',bytes(value)))
        with patch('phymeo_rendering.backends.native.require_library'),patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device):
            session=NativeBackend(resident=True,studio=StudioLighting()).open_scene(VisualScene('empty',(),(),()))
            self.assertEqual(calls[0],'geometry'); self.assertEqual(calls[1][0],'studio')
            info=session.diagnostics(); self.assertFalse(info['path_trace'])
            info['studio']['panels'].clear()
            self.assertEqual(len(session.diagnostics()['studio']['panels']),2)
            session.close()

    def test_missing_optional_symbol_fails_explicitly_and_closes_owner(self):
        from phymeo_rendering.studio import StudioLighting
        from test_native_resident import Recorder
        device=Recorder()
        def fail(value): raise BackendUnavailableError('resident studio export unavailable')
        device.set_studio=fail
        with patch('phymeo_rendering.backends.native.require_library'),patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device):
            with self.assertRaisesRegex(BackendUnavailableError,'studio'):
                NativeBackend(resident=True,studio=StudioLighting()).open_scene(VisualScene('empty',(),(),()))
        self.assertTrue(device.closed)

    def test_private_boundary_rejects_missing_closed_and_late_setter(self):
        from phymeo_rendering.studio import StudioLighting
        from phymeo_rendering.backends._native_studio import pack_studio
        from phymeo_rendering.backends._native_resident import ResidentDevice
        value,_=pack_studio(StudioLighting())
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._handle=123; owner._studio_function=None; owner._has_captured=False
        with self.assertRaises(BackendUnavailableError): owner.set_studio(value)
        owner._studio_function=lambda *_: self.fail('late native call')
        owner._has_captured=True
        with self.assertRaises(RuntimeError): owner.set_studio(value)
        owner._has_captured=False; owner._handle=None
        with self.assertRaises(RuntimeError): owner.set_studio(value)


if __name__=='__main__': unittest.main()
