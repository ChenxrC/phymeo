"""Thin, explicit binding over the unchanged rigid-history session."""
import ctypes as ct
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from .build_source import verify,read,digest,save,policy


def check_seal(directory):
    root,manifest=verify(directory);seal=read(root/'lowres-seal.json')
    if manifest['contract'].get('lowres_filter')!=policy():raise ValueError('fixed lowres policy differs')
    if seal['manifest_sha256']!=digest(root/'source-manifest.json') or seal['dll_sha256']!=digest(root/'build/phymeo_native.dll'):
        raise ValueError('lowres seal changed')
    actual={p.name:digest(p) for p in (root/'build').glob('*.spv')}
    if actual!=seal['spv_sha256']:raise ValueError('actual lowres SPIR-V changed')
    return root,manifest


def backend_class(debug_directory):
    # The caller adds its frozen existing examples tree before this import.
    from preview_compact_rigid_history.runtime import RigidHistoryBackend
    debug_directory=Path(debug_directory)

    class LowresBackend(RigidHistoryBackend):
        def __init__(self,directory,**kwargs):
            check_seal(directory)
            super().__init__(directory,**kwargs)
            try:
                version=self.lib.pr_preview_lowres_filter_abi_version
                version.argtypes=[];version.restype=ct.c_uint32
                self.lowres_read=self.lib.pr_preview_lowres_filter_read
                self.lowres_read.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];self.lowres_read.restype=ct.c_int
            except AttributeError as error:raise RuntimeError('lowres DLL ABI absent; no unfiltered fallback') from error
            if version()!=1:raise RuntimeError('lowres DLL version differs')

        def open_scene(self,scene):
            session=super().open_scene(scene)
            return LowresSession(session,self,debug_directory)

    return LowresBackend


class LowresSession:
    def __init__(self,session,backend,debug_directory):
        self.session=session;self.backend=backend;self.debug_directory=Path(debug_directory)
        self._audited=set()

    def __getattr__(self,name):return getattr(self.session,name)

    def read_filtered_guide(self):
        s=self.session
        with s._lock:
            if s.closed or not s.last_rgb:raise RuntimeError('lowres read requires latest successful RGB')
            info=s._compact_info();w,h=info['guide_width'],info['guide_height']
            buffer=(ct.c_ubyte*(w*h*16))();device=s.underlying._delegate.device
            if self.backend.lowres_read(device._handle,buffer,ct.sizeof(buffer)):raise RuntimeError(device._error())
            return dict(width=w,height=h,format='RGBA32F_LE',capture_index=s.capture_count-1,data=bytes(buffer),
                scope='actual low-grid compute output before original full-rate K25; alpha original exact')

    def read_current_incoming(self):
        packet=self.session.read_current_incoming()
        if self.session.capture_count not in self._audited:
            self._save_debug()
            self._audited.add(self.session.capture_count)
        return packet

    def _save_debug(self):
        import numpy as np
        before=self.session.read_incoming_guide();after=self.read_filtered_guide()
        planes={'original':before,'filtered':after,**self.session.read_low_sensors()}
        w,h=before['width'],before['height']
        if (after['width'],after['height'])!=(w,h):raise RuntimeError('lowres debug extent changed')
        a=np.frombuffer(before['data'],dtype='<f4').reshape(-1,4)
        b=np.frombuffer(after['data'],dtype='<f4').reshape(-1,4)
        if not np.array_equal(a.view('<u4')[:,3],b.view('<u4')[:,3]):raise RuntimeError('filter changed original support/normal alpha')
        valid=np.frombuffer(planes['valid']['data'],dtype='u1')!=0
        ids=np.frombuffer(planes['id']['data'],dtype='<u4')
        supported=valid&(ids!=0)
        if not np.isfinite(b[supported]).all() or (b[supported,:3]<0).any():raise RuntimeError('nonfinite filtered supported incoming')
        changed=np.any(a.view('<u4')[:,:3]!=b.view('<u4')[:,:3],axis=1)
        mode='rigid-history' if self.backend.reuse_history else 'current'
        folder=self.debug_directory/mode/f'capture-{self.session.capture_count-1:04d}'
        folder.mkdir(parents=True,exist_ok=False)
        receipt=dict(schema='phymeo-preview-lowres-filter-debug-v1',width=w,height=h,policy=policy(),
            original_alpha_exact=True,changed_low_cells=int(changed.sum()),changed_visible_low_cells=int((changed&supported).sum()),
            max_abs_rgb_change=float(np.max(np.abs(a[supported,:3]-b[supported,:3]))) if supported.any() else None,
            actual=self.session.diagnostics()['actual'],files={},quality_claim=False)
        for name,value in planes.items():
            path=folder/(name+'.bin');path.write_bytes(value['data'])
            receipt['files'][name]={k:v for k,v in value.items() if k!='data'}
            receipt['files'][name].update(file=path.name,sha256=digest(path),bytes=len(value['data']))
        save(folder/'receipt.json',receipt)

    def diagnostics(self):
        result=self.session.diagnostics()
        result['lowres_filter']=deepcopy(policy())
        profile=result.get('device',{}).get('resident_profile')
        if profile is not None:profile['gpu_scope']='TLAS + original primary/GI + one low-grid filter + original K25/history + copies'
        return result
