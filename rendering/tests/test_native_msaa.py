"""Optional MSAA contract and explicit ABI rejection; HOST tests do not claim pixels."""
import ctypes as ct
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend


def configure(target, samples):
    from phymeo_rendering.backends._native_msaa import configure_msaa
    return configure_msaa(target, samples)


class Function:
    def __init__(self, operation): self.operation=operation; self.calls=[]
    def __call__(self,*args):
        self.calls.append(args)
        return self.operation(*args) if callable(self.operation) else self.operation


def device(root, *, supported=True, compatible=True, version=1, status=0):
    from phymeo_rendering.backends._native_msaa import ResidentMsaaInfo
    state={'samples':1}
    def query(handle, pointer):
        value=ct.cast(pointer,ct.POINTER(ResidentMsaaInfo)).contents
        value.msaa_abi_version=1; value.requested_samples=state['samples']
        value.color_samples=state['samples']; value.sensor_samples=1
        value.color_sample_counts=5 if supported else 1
        value.depth_sample_counts=5 if supported else 1
        value.shader_compatible=int(compatible)
        value.max_width=4096; value.max_height=4096
        return 0
    def setter(handle,samples,*shaders):
        if status==0: state['samples']=samples
        return status
    functions=SimpleNamespace(pr_resident_msaa_abi_version=Function(version),
        pr_resident_msaa_info=Function(query),pr_resident_set_msaa=Function(setter))
    for stage in ('vert','frag'):
        (root/f'resident_msaa_resolve.{stage}.spv').write_bytes(bytes.fromhex('03022307')+bytes(16))
    return SimpleNamespace(_lib=functions,_handle=123,_error=lambda:'fixture MSAA rejection',
                           shader_directory=root,shader_sha256={})


class NativeMsaaContract(unittest.TestCase):
    def test_public_default_and_four_samples(self):
        self.assertEqual(NativeBackend().msaa_samples,1)
        self.assertEqual(NativeBackend(resident=True,msaa_samples=4).msaa_samples,4)
        self.assertEqual(NativeBackend(msaa_samples=1).msaa_samples,1)

    def test_strict_integer_and_explicit_sample_set(self):
        for value in (True,False,1.,4.,None,'4'):
            with self.subTest(value=value),self.assertRaises(TypeError):
                NativeBackend(resident=True,msaa_samples=value)
        for value in (0,2,8,-1):
            with self.subTest(value=value),self.assertRaises(ValueError):
                NativeBackend(resident=True,msaa_samples=value)

    def test_four_requires_resident(self):
        with self.assertRaisesRegex(UnsupportedRequestError,'resident'):
            NativeBackend(msaa_samples=4)

    def test_default_never_probes_optional_symbols(self):
        class Untouchable:
            def __getattr__(self,name): raise AssertionError('default touched '+name)
        result=configure(Untouchable(),1)
        self.assertEqual(result,dict(requested_samples=1,actual_color_samples=1,sensor_samples=1,
            resolve_space='none (single sample)',sensor_sampling='deterministic pixel center'))

    def test_fixed_info_layout_and_real_setter_arguments(self):
        from phymeo_rendering.backends._native_msaa import ResidentMsaaInfo
        self.assertEqual(ct.sizeof(ResidentMsaaInfo),128)
        with TemporaryDirectory() as root:
            target=device(Path(root)); result=configure(target,4)
            self.assertEqual(result['requested_samples'],4)
            self.assertEqual(result['actual_color_samples'],4)
            self.assertEqual(result['sensor_samples'],1)
            self.assertEqual(result['resolve_space'],'display-linear then IEC sRGB once')
            args=target._lib.pr_resident_set_msaa.calls[0]
            self.assertEqual(args[:2],(123,4)); self.assertEqual((args[3],args[5]),(20,20))
            self.assertEqual(set(target.shader_sha256),{'msaa_resolve_vert','msaa_resolve_frag'})

    def test_old_dll_missing_symbols_and_abi_version_reject(self):
        with TemporaryDirectory() as root:
            for missing in ('pr_resident_msaa_abi_version','pr_resident_msaa_info','pr_resident_set_msaa'):
                target=device(Path(root)); delattr(target._lib,missing)
                with self.subTest(missing=missing),self.assertRaisesRegex(BackendUnavailableError,'MSAA'):
                    configure(target,4)
            with self.assertRaises(BackendUnavailableError): configure(device(Path(root),version=2),4)

    def test_unsupported_samples_or_old_shader_reject_before_setter(self):
        with TemporaryDirectory() as root:
            for supported,compatible in ((False,True),(True,False)):
                target=device(Path(root),supported=supported,compatible=compatible)
                with self.assertRaises(BackendUnavailableError): configure(target,4)
                self.assertFalse(target._lib.pr_resident_set_msaa.calls)

    def test_missing_resolve_shader_and_actual_native_failure_reject(self):
        with TemporaryDirectory() as root:
            target=device(Path(root)); (Path(root)/'resident_msaa_resolve.frag.spv').unlink()
            with self.assertRaises(BackendUnavailableError): configure(target,4)
            with self.assertRaisesRegex(BackendUnavailableError,'fixture MSAA rejection'):
                configure(device(Path(root),status=-1),4)

    def test_public_diagnostics_and_failed_open_cleanup(self):
        from test_native_resident import fixture, Recorder
        scene,mesh=fixture()
        for fail in (False,True):
            with TemporaryDirectory() as root:
                recorder=Recorder(); raw=device(Path(root),status=-1 if fail else 0)
                for key in ('_lib','_handle','_error','shader_directory'):
                    setattr(recorder,key,getattr(raw,key))
                with patch('phymeo_rendering.backends.native.require_library'), \
                     patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
                     patch('phymeo_rendering.backends.native.ResidentDevice',return_value=recorder):
                    backend=NativeBackend(resident=True,msaa_samples=4)
                    if fail:
                        with self.assertRaises(BackendUnavailableError): backend.open_scene(scene)
                        self.assertTrue(recorder.closed)
                    else:
                        owner=backend.open_scene(scene)
                        self.assertEqual(owner.diagnostics()['msaa']['actual_color_samples'],4)
                        owner.close(); self.assertTrue(recorder.closed)


if __name__=='__main__': unittest.main()
