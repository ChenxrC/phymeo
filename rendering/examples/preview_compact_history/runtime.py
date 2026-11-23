"""Explicit experimental GPU history; the ordinary CompactBackend is unchanged."""
import ctypes as ct
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import struct

from preview_compact.runtime import CompactBackend,CompactSession,activate,validate_counters
from .state import HistoryKey,sample_seed


COUNTERS=("eligible_pixels","accepted_pixels","rejected_projection","rejected_geometry",
          "rejected_support","clamped_pixels","age_sum","reset_pixels")


class HistoryBackend(CompactBackend):
    def __init__(self,*args,reuse_history=True,**kwargs):
        if type(reuse_history) is not bool:raise TypeError("reuse_history must be bool")
        super().__init__(*args,**kwargs)
        policy=self.manifest["contract"].get("history",{})
        if policy.get("schema")!="phymeo-preview-compact-history-v1":
            raise ValueError("explicit history source package required")
        self.reuse_history=reuse_history
        signatures={
            "abi_version":([],ct.c_uint32),
            "setup":([ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t],ct.c_int),
            "arm":([ct.c_void_p,ct.c_uint32,ct.c_uint32],ct.c_int),
            "cancel":([ct.c_void_p],ct.c_int),"publish":([ct.c_void_p],ct.c_int),
            "info":([ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t],ct.c_int),
            "read":([ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.c_size_t],ct.c_int),
        }
        self.history_api={}
        for name,(arguments,result) in signatures.items():
            try:fn=getattr(self.lib,"pr_preview_history_"+name)
            except AttributeError as error:raise RuntimeError("history DLL ABI missing; no fallback") from error
            fn.argtypes=arguments;fn.restype=result;self.history_api[name]=fn
        if self.history_api["abi_version"]()!=1:raise RuntimeError("history ABI mismatch")
        self.history_shader=(self.root/"build/preview_history.comp.spv").read_bytes()
        if len(self.history_shader)%4:raise ValueError("unaligned history SPIR-V")
        self.history_shader_sha256=sha256(self.history_shader).hexdigest()

    def open_scene(self,scene):
        from phymeo_rendering.backends.native import NativeBackend
        policy=self.manifest["contract"]
        backend=NativeBackend(library_path=self.library,shader_directory=self.root/"build",resident=True,
            shadow_mode="ray_query",msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,
            progressive=True,spatial_filter="guided",area_light_grid=policy["area_light_grid"],**self.settings)
        backend._preview_sparse_requested=True
        underlying=backend.open_scene(scene)
        try:
            device=underlying._delegate.device
            code=(ct.c_uint32*(len(self.history_shader)//4)).from_buffer_copy(self.history_shader)
            if self.history_api["setup"](device._handle,code,len(self.history_shader)):
                raise RuntimeError(device._error())
            return HistorySession(self,underlying)
        except BaseException:
            underlying.close()
            raise


class HistorySession(CompactSession):
    def __init__(self,*args):
        super().__init__(*args)
        self.history=HistoryKey()
        self.last_plan=None
        self.last_history=self._history_info()
        self.signal=sha256(json.dumps(asdict(self.underlying._delegate.scene),sort_keys=True).encode()).hexdigest()

    def _history_info(self):
        storage=(ct.c_uint32*32)()
        device=self.underlying._delegate.device
        if self.backend.history_api["info"](device._handle,storage,128):raise RuntimeError(device._error())
        if tuple(storage[:2])!=(1,1):raise RuntimeError("history device configuration differs")
        uniform_bytes,reads,read_bytes,requested_bytes=struct.unpack_from("<4Q",bytes(storage),96)
        return dict(abi_version=storage[0],enabled=bool(storage[1]),committed_rgb=storage[2],sequence=storage[3],
            last_reset=bool(storage[4]),width=storage[5],height=storage[6],age_cap=storage[7],read_slot=storage[8],
            valid=bool(storage[9]),armed=bool(storage[10]),reset_count=storage[11],
            **dict(zip(COUNTERS,storage[16:24])),parameter_upload_bytes=uniform_bytes,readbacks=reads,
            readback_bytes=read_bytes,requested_buffer_bytes=requested_bytes)

    def _declaration(self,frame,request):
        from phymeo_rendering.backends.native import _parameters
        delegate=self.underlying._delegate
        params=_parameters(delegate.scene,request,delegate.triangles,delegate.exposure,lights=delegate.effective_lights)
        camera=request.camera;key=frame.key
        return dict(scene=key.scene_key,world=key.world_id,epoch=key.epoch,phase=key.phase,revision=key.revision,
            tick=key.tick,time=frame.time_s,camera_id=camera.id,extent=(camera.width,camera.height),
            lens=(*tuple(params.intrinsics),camera.near_m,camera.far_m),
            camera_pose=(*camera.world_pose.position_m,*camera.world_pose.orientation_xyzw),
            nodes=tuple(sorted((node.node_id,(*node.world_pose.position_m,*node.world_pose.orientation_xyzw)) for node in frame.nodes)),
            lights=tuple((tuple(params.light_directions[i]),tuple(params.light_colors[i])) for i in range(params.counts[0])),
            exposure=float(params.options[0]),signal=self.signal)

    def capture(self,frame,request):
        from phymeo_rendering.outputs import Channel
        with self._lock:
            if self.closed:raise RuntimeError("history session closed")
            rgb=Channel.RGB in request.channels
            if request.camera.width*request.camera.height>(2**32-1)//25:
                raise ValueError("history pixel counters exceed uint32")
            plan=self.history.prepare(self._declaration(frame,request)) if rgb else None
            device=self.underlying._delegate.device
            self.last_rgb=False
            published=False
            try:
                if rgb:
                    self.underlying.reset_accumulation()
                    reset=bool(plan["reasons"]) or not self.backend.reuse_history
                    if self.backend.history_api["arm"](device._handle,int(reset),plan["sequence"]):
                        raise RuntimeError(device._error())
                result=self.underlying.capture(frame,request)
                info=self._info();compact=self._compact_info()
                if info["last_capture_rgb"]!=rgb or compact["last_capture_rgb"]!=rgb:
                    raise RuntimeError("history publication scope differs")
                if rgb:
                    validate_counters(info)
                    if self.backend.history_api["publish"](device._handle):raise RuntimeError(device._error())
                    published=True
                    self.history.commit(plan)
                    self.last_plan=deepcopy(plan)
                self.last_history=self._history_info()
                self.last_info=info;self.last_compact=compact
                self.capture_count+=1;self.last_rgb=rgb
                return result
            except BaseException:
                self.backend.history_api["cancel"](device._handle)
                # If diagnostics failed after a native publication, discard that
                # state at next arm rather than carrying an unreturned image.
                if published:self.history.reset()
                raise

    def reset_history(self):
        with self._lock:
            if self.closed:raise RuntimeError("history session closed")
            self.history.reset();self.last_rgb=False

    def set_directional_lights(self,lights):
        with self._lock:
            if self.closed:raise RuntimeError("history session closed")
            self.underlying.set_directional_lights(lights)

    def _read_history(self,plane):
        with self._lock:
            if self.closed or not self.last_rgb:raise RuntimeError("history read requires latest successful RGB")
            value=self._history_info();width,height=value["width"],value["height"]
            storage=(ct.c_ubyte*(width*height*(16 if plane==0 else 32)))()
            device=self.underlying._delegate.device
            if self.backend.history_api["read"](device._handle,plane,storage,ct.sizeof(storage)):
                raise RuntimeError(device._error())
            self.last_history=self._history_info()
            return dict(width=width,height=height,format="RGBA32F_LE" if plane==0 else "F32x4_U32x4_LE",
                capture_index=self.capture_count-1,data=bytes(storage),
                scope=("current demodulated compact reconstructed incoming RGB; A=actual traced-anchor support, not age" if plane==0 else
                       "committed filtered incoming RGB/retained-observation age; depth-float-bits/octNs/uint32-ID/support"))

    def read_current_incoming(self):return self._read_history(0)
    def read_history_incoming(self):return self._read_history(1)

    def diagnostics(self):
        with self._lock:
            value=super().diagnostics()
            value["backend"]="experimental-native-compact-history-v1"
            value["history"]=dict(mode="reprojection" if self.backend.reuse_history else "fresh-sequence-control",
                scope="static-geometry-moving-camera",actual=deepcopy(self.last_history),
                last_reset_reasons=[] if self.last_plan is None else deepcopy(self.last_plan["reasons"]),
                effective_seed=None if self.last_plan is None else sample_seed(self.backend.settings["indirect_seed"],self.last_plan["sequence"]),
                shader_sha256=self.backend.history_shader_sha256,
                age_scope="retained observations, capped EMA after 8; not independent spp",
                dynamic_scene="any node change resets all; no moving-Panda history gain claimed")
            if "resident_profile" in value["device"]:
                value["device"]["resident_profile"]["gpu_scope"]="TLAS + full primary + compact GI + K25 current incoming + barrier + GPU reprojection/clamp/recombine + copies"
            return value

    def close(self):
        with self._lock:
            if not self.closed:
                try:self.last_history=self._history_info()
                finally:super().close()
