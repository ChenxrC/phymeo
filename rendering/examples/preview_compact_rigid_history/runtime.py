"""Explicit short, per-ID rigid GPU history with bound presentation adjacency."""
import ctypes as ct
from copy import deepcopy
from hashlib import sha256
import struct

from preview_compact.runtime import CompactBackend,activate
from preview_compact_history.runtime import HistorySession
from .state import RigidHistoryKey,validate_presentation


class RigidHistoryBackend(CompactBackend):
    def __init__(self,*args,reuse_history=True,**kwargs):
        if type(reuse_history) is not bool:raise TypeError("reuse_history must be bool")
        super().__init__(*args,**kwargs)
        if self.manifest["contract"].get("rigid_history",{}).get("schema")!="phymeo-preview-compact-rigid-history-v1":
            raise ValueError("explicit per-ID rigid history package required")
        self.reuse_history=reuse_history
        signatures={"abi_version":([],ct.c_uint32),
            "setup":([ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t],ct.c_int),
            "arm":([ct.c_void_p,ct.c_uint32,ct.c_uint32],ct.c_int),
            "cancel":([ct.c_void_p],ct.c_int),"publish":([ct.c_void_p],ct.c_int),
            "info":([ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t],ct.c_int),
            "read":([ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.c_size_t],ct.c_int)}
        self.history_api={}
        try:
            for name,(arguments,result) in signatures.items():
                function=getattr(self.lib,"pr_preview_history_"+name)
                function.argtypes=arguments;function.restype=result;self.history_api[name]=function
            version=self.lib.pr_preview_rigid_history_abi_version
            version.argtypes=[];version.restype=ct.c_uint32
            self.rigid_info=self.lib.pr_preview_rigid_history_info
            self.rigid_info.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t];self.rigid_info.restype=ct.c_int
        except AttributeError as error:raise RuntimeError("rigid history DLL ABI missing; no static fallback") from error
        if version()!=1 or self.history_api["abi_version"]()!=1:raise RuntimeError("rigid history ABI differs")
        self.history_shader=(self.root/"build/preview_history.comp.spv").read_bytes()
        if len(self.history_shader)%4:raise ValueError("unaligned history SPIR-V")
        self.history_shader_sha256=sha256(self.history_shader).hexdigest()

    def open_scene(self,scene):
        from phymeo_rendering.backends.native import NativeBackend
        backend=NativeBackend(library_path=self.library,shader_directory=self.root/"build",resident=True,
            shadow_mode="ray_query",msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,
            progressive=True,spatial_filter="guided",area_light_grid=self.manifest["contract"]["area_light_grid"],**self.settings)
        backend._preview_sparse_requested=True
        underlying=backend.open_scene(scene)
        try:
            device=underlying._delegate.device
            code=(ct.c_uint32*(len(self.history_shader)//4)).from_buffer_copy(self.history_shader)
            if self.history_api["setup"](device._handle,code,len(self.history_shader)):raise RuntimeError(device._error())
            return RigidHistorySession(self,underlying)
        except BaseException:
            underlying.close();raise


class RigidHistorySession(HistorySession):
    def __init__(self,*args):
        super().__init__(*args)
        self.history=RigidHistoryKey()
        self._pending_presentation=None
        self.last_rigid=self._rigid_info()

    def _rigid_info(self):
        storage=(ct.c_uint32*16)();device=self.underlying._delegate.device
        if self.backend.rigid_info(device._handle,storage,64):raise RuntimeError(device._error())
        if (storage[0],storage[1],storage[4],storage[6],storage[7])!=(1,1,4,48,97):
            raise RuntimeError("rigid history configuration differs")
        return dict(abi_version=storage[0],enabled=bool(storage[1]),instance_count=storage[2],
            moved_instances=storage[3],age_cap=storage[4],committed_rgb=storage[5],history_pixel_stride=storage[6],marker=storage[7],
            rejected_lighting=storage[8],rejected_incoming=storage[9],moving_accepted_pixels=storage[10],
            lookup_misses=storage[11],rigid_table_upload_bytes=struct.unpack_from("<Q",bytes(storage),48)[0])

    def _declaration(self,frame,request):
        value=super()._declaration(frame,request)
        value["presentation"]=deepcopy(self._pending_presentation)
        return value

    def capture(self,frame,request,*,presentation=None):
        from phymeo_rendering.outputs import Channel
        with self._lock:
            if self.closed:raise RuntimeError("rigid history session closed")
            if Channel.RGB in request.channels:
                if presentation is None:raise ValueError("RGB requires a full bound presentation record")
                validate_presentation(frame,presentation)
            elif presentation is not None:validate_presentation(frame,presentation)
            self._pending_presentation=deepcopy(presentation)
            published=False
            try:
                result=super().capture(frame,request)
                published=True
                self.last_rigid=self._rigid_info()
                if self.last_rigid["lookup_misses"]:raise RuntimeError("visible instance missing from actual rigid table")
                return result
            except BaseException:
                # A failed post-publication diagnostic must not carry an
                # unreturned image into the next capture.
                if published:self.history.reset()
                self.last_rgb=False
                raise
            finally:self._pending_presentation=None

    def _read_history(self,plane):
        if plane==0:return super()._read_history(plane)
        with self._lock:
            if self.closed or not self.last_rgb:raise RuntimeError("history read requires latest successful RGB")
            value=self._history_info();width,height=value["width"],value["height"]
            storage=(ct.c_ubyte*(width*height*48))();device=self.underlying._delegate.device
            if self.backend.history_api["read"](device._handle,1,storage,ct.sizeof(storage)):raise RuntimeError(device._error())
            self.last_history=self._history_info()
            return dict(width=width,height=height,format="F32x4_U32x4_F32x4_LE",capture_index=self.capture_count-1,
                data=bytes(storage),scope="incoming/age; depthbits/octNs/uint32ID/support; current primary RGB/rho24 confidence guide")

    def diagnostics(self):
        with self._lock:
            value=super().diagnostics()
            value["backend"]="experimental-native-compact-rigid-history-v1"
            value["history"]["scope"]="rigid-primary-short-history"
            value["history"]["dynamic_scene"]="actual per-ID rigid correspondence; unobserved secondary visibility remains a bounded-history bias"
            value["history"]["age_scope"]="retained observations, capped EMA after 4; not independent spp"
            value["rigid_history"]=deepcopy(self.last_rigid)
            value["presentation"]=None if self.last_plan is None else deepcopy(self.last_plan["key"]["presentation"])
            value["unobserved_secondary_visibility"]=bool(self.last_rigid["moved_instances"])
            value["rigid_shader_sha256"]=self.backend.history_shader_sha256
            quality=value["contract"].get("surface_quality")
            if quality:
                value["surface_quality"]=deepcopy(quality)
                profile=value["device"].get("resident_profile")
                if profile is not None:
                    profile["gpu_scope"]="TLAS + primary panel shading + compact GI/filter + rigid history + copies"
                    if "contours" in quality:profile["gpu_scope"]+=" including four-ray candidate contour resolve"
            return value

    def close(self):
        with self._lock:
            if not self.closed:
                try:self.last_rigid=self._rigid_info()
                finally:super().close()
