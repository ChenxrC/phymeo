"""Explicit experimental edge rescue over a hash-bound private compact package."""
import ctypes as ct
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

_ACTIVE=None
_BASE=None
HERE=Path(__file__).resolve().parent

def activate(directory):
    global _ACTIVE,_BASE
    root=Path(directory).resolve(strict=True)
    if _ACTIVE is not None and _ACTIVE!=root:raise RuntimeError('one prepared package per process')
    manifest=json.loads((root/'source-manifest.json').read_text(encoding='utf-8'))
    if manifest['contract'].get('edge_rescue',{}).get('schema')!='phymeo-preview-edge-rescue-v1':
        raise ValueError('explicit edge source package required')
    for relative,expected in manifest['generated_sha256'].items():
        path=root/Path(relative.replace('\\','/'))
        if sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('edge prepared source changed: '+relative)
    if _BASE is None:
        for name in ('preview_sparse.runtime','preview_compact.runtime'):
            if name in sys.modules:raise RuntimeError('activate edge package before importing another compact adapter')
        previous=list(sys.path)
        try:
            sys.path.insert(0,str(root/'adapters'))
            _BASE=importlib.import_module('preview_compact.runtime')
        finally:sys.path[:]=previous
    _BASE.activate(root);_ACTIVE=root
    return root,manifest


def _validator():
    spec=importlib.util.spec_from_file_location('_edge_accounting',HERE/'build_source.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.validate_counts


class EdgeBackend:
    def __init__(self,directory,**kwargs):
        self.root,self.manifest=activate(directory)
        self.base=_BASE.CompactBackend(directory,**kwargs)
        self.library=self.base.library;self.lib=self.base.lib
        try:
            version=self.lib.pr_preview_edge_abi_version
            self.setter=self.lib.pr_preview_edge_set;self.query=self.lib.pr_preview_edge_info
        except AttributeError as e:raise RuntimeError('actual edge DLL ABI absent; no fallback') from e
        version.argtypes=[];version.restype=ct.c_uint32
        if version()!=1:raise RuntimeError('edge ABI version differs')
        self.setter.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t,ct.c_uint32];self.setter.restype=ct.c_int
        self.query.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t];self.query.restype=ct.c_int

    def open_scene(self,scene):
        session=self.base.open_scene(scene)
        try:
            normal=bool(getattr(session.underlying._delegate.device,'_normal_maps_set',False))
            filename='resident_edge_rescue_normal.frag.spv' if normal else 'resident_edge_rescue.frag.spv'
            data=(self.root/'build'/filename).read_bytes()
            if len(data)%4:raise ValueError('edge SPIR-V alignment')
            words=(ct.c_uint32*(len(data)//4)).from_buffer_copy(data)
            dev=session.underlying._delegate.device
            limit=self.manifest['contract']['edge_rescue']['max_invocations']
            if self.setter(dev._handle,words,len(data),limit):raise RuntimeError(dev._error())
            return EdgeSession(self,session,filename,sha256(data).hexdigest())
        except BaseException:
            session.close();raise


class EdgeSession:
    def __init__(self,backend,base,shader,shader_hash):
        self.backend,self.base=backend,base;self.shader,self.shader_hash=shader,shader_hash
        self._lock=threading.RLock();self.closed=False;self.last_edge=None
        self._validate=_validator()

    def supports(self,request):return self.base.supports(request)

    def _counts(self):
        values=(ct.c_uint32*16)();dev=self.base.underlying._delegate.device
        if self.backend.query(dev._handle,values,64):raise RuntimeError(dev._error())
        names=('selected_pixels','fragment_invocations','traced_invocations','reconstructed_rescue_pixels',
               'filtered_neighbors','missing_rescue','budget_exhausted','max_invocations')
        return dict(zip(names,map(int,values[8:])),invalid=int(values[0]))

    def capture(self,frame,request):
        with self._lock:
            if self.closed:raise RuntimeError('edge session closed')
            try:
                shot=self.base.capture(frame,request)
            except BaseException:
                self.last_edge=self._counts();raise
            counts=self._counts()
            if self.base.last_rgb:
                counts=self._validate(counts)
                if counts['max_invocations']!=self.backend.manifest['contract']['edge_rescue']['max_invocations']:
                    raise RuntimeError('edge actual bound differs from source declaration')
            self.last_edge=counts
            return shot

    def read_reconstructed_linear(self):
        with self._lock:
            result=self.base.read_reconstructed_linear()
            result['space']='scene-linear compact reconstruction with actual selected full-rate rescue; not uniform spp or a temporal mean'
            return result
    def read_incoming_guide(self):return self.base.read_incoming_guide()
    def read_full_attributes(self):return self.base.read_full_attributes()
    def read_low_sensors(self):return self.base.read_low_sensors()

    def read_rescue_incoming(self):
        with self._lock:
            if self.closed or not self.base.last_rgb:raise RuntimeError('rescue read requires latest successful RGB')
            info=self.base.last_compact;w,h=info['full_width'],info['full_height']
            data=(ct.c_ubyte*(w*h*16))();dev=self.base.underlying._delegate.device
            if self.backend.base.compact_read(dev._handle,5,data,ct.sizeof(data)):raise RuntimeError(dev._error())
            return dict(width=w,height=h,format='RGBA32F_LE',capture_index=self.base.capture_count-1,data=bytes(data),
                space='RGB actual per-pixel 16-proposal incoming before rescue filter; A oct-Ns bits0..19 and real-result-valid bit20; zero A unselected')

    def diagnostics(self):
        with self._lock:
            value=self.base.diagnostics();value['backend']='experimental-native-compact-edge-rescue-v1'
            value['edge_rescue']=deepcopy(self.last_edge)
            value['edge_shader']={'file':self.shader,'sha256':self.shader_hash}
            value['samples_scope']='compact low anchors *4 plus actual extra traced fragment invocations *16; not uniform spp or hardware ray count'
            if 'resident_profile' in value['device']:
                value['device']['resident_profile']['gpu_scope']='TLAS, primary, compact incoming, selection, actual full-rate selective trace draw, rescue incoming filter/merge, all copies'
            return value

    def close(self):
        with self._lock:
            if not self.closed:
                try:self.base.close()
                finally:self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()