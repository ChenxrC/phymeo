"""Explicit compact-raster ownership and honest single-exposure readbacks."""
import ctypes as ct
from copy import deepcopy
import threading

from preview_sparse.runtime import SparseBackend,SparseSession,activate


def validate_counters(value):
    if sum(value[k] for k in ("reconstructed_pixels","fallback_pixels","nondiffuse_pixels"))!=value["visible_pixels"]:
        raise ValueError("compact full-rate counter partition differs")
    if value["primary_proposals"]!=value["traced_anchor_pixels"]*value["paths_per_anchor"]:
        raise ValueError("compact low-raster proposal accounting differs")


class CompactBackend(SparseBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.manifest["contract"].get("compact",{}).get("schema")!="phymeo-preview-compact-v1":
            raise ValueError("explicit compact source package required")
        try:
            version=self.lib.pr_preview_compact_abi_version
            self.compact_query=self.lib.pr_preview_compact_info
            self.compact_read=self.lib.pr_preview_compact_read
        except AttributeError as error:
            raise RuntimeError("compact DLL ABI absent; no fallback") from error
        version.argtypes=[];version.restype=ct.c_uint32
        if version()!=1:raise RuntimeError("unsupported compact ABI")
        self.compact_query.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t]
        self.compact_query.restype=ct.c_int
        self.compact_read.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.c_size_t]
        self.compact_read.restype=ct.c_int

    def open_scene(self,scene):
        from phymeo_rendering.backends.native import NativeBackend
        policy=self.manifest["contract"]
        backend=NativeBackend(library_path=self.library,shader_directory=self.root/"build",resident=True,
            shadow_mode="ray_query",msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,
            progressive=True,spatial_filter="guided",area_light_grid=policy["area_light_grid"],**self.settings)
        backend._preview_sparse_requested=True
        underlying=backend.open_scene(scene)
        try:return CompactSession(self,underlying)
        except BaseException:
            underlying.close()
            raise


class CompactSession(SparseSession):
    def __init__(self,*args):
        self._lock=threading.RLock()
        super().__init__(*args)
        self.last_compact=self._compact_info()

    def _compact_info(self):
        data=(ct.c_uint32*16)()
        device=self.underlying._delegate.device
        if self.backend.compact_query(device._handle,data,64):raise RuntimeError(device._error())
        if tuple(data[:3])!=(1,1,4):raise RuntimeError("compact device configuration differs")
        value=dict(abi_version=data[0],enabled=bool(data[1]),tile=data[2],full_width=data[3],full_height=data[4],
                   guide_width=data[5],guide_height=data[6],last_capture_rgb=bool(data[7]),
                   rgb_captures=data[8],sensor_only_captures=data[9],successful_guide_draws=data[10],
                   primary_color_attachments=data[11],reconstruction_sampled_images=data[12])
        if value["guide_width"]!=(value["full_width"]+3)//4 or value["guide_height"]!=(value["full_height"]+3)//4:
            raise RuntimeError("compact device did not allocate the declared small guide")
        return value

    def capture(self,frame,request):
        with self._lock:
            if self.closed:raise RuntimeError("compact session closed")
            from phymeo_rendering.outputs import Channel
            rgb=Channel.RGB in request.channels
            if request.camera.width*request.camera.height>(2**32-1)//25:
                raise ValueError("compact extent would overflow exact K25 uint32 statistics")
            self.last_rgb=False
            if rgb:self.underlying.reset_accumulation()
            result=self.underlying.capture(frame,request)
            info=self._info();compact=self._compact_info()
            if info["last_capture_rgb"]!=rgb or compact["last_capture_rgb"]!=rgb:
                raise RuntimeError("compact capture scope differs")
            if rgb:validate_counters(info)
            self.last_info=info;self.last_compact=compact
            self.capture_count+=1;self.last_rgb=rgb
            return result

    def read_reconstructed_linear(self):
        with self._lock:return super().read_reconstructed_linear()

    def _read(self,plane):
        with self._lock:
            if self.closed or not self.last_rgb:raise RuntimeError("compact read requires latest successful RGB exposure")
            info=self._compact_info()
            full=plane==0
            width=info["full_width"] if full else info["guide_width"]
            height=info["full_height"] if full else info["guide_height"]
            bpp=16 if plane<2 else (1 if plane==4 else 4)
            storage=(ct.c_ubyte*(width*height*bpp))()
            device=self.underlying._delegate.device
            if self.backend.compact_read(device._handle,plane,storage,ct.sizeof(storage)):
                raise RuntimeError(device._error())
            return dict(width=width,height=height,format=("RGBA32F_LE","RGBA32F_LE","F32_LE","U32_LE","U8")[plane],
                        capture_index=self.capture_count-1,data=bytes(storage),
                        scope=("actual full-rate current SH and oct-Ns","actual compact incoming/normal/outcome flags",
                               "actual compact forward depth","actual compact instance IDs","actual compact validity")[plane])

    def read_incoming_guide(self):return self._read(1)
    def read_full_attributes(self):return self._read(0)
    def read_low_sensors(self):
        with self._lock:
            return {name:self._read(plane) for name,plane in (("depth",2),("id",3),("valid",4))}

    def diagnostics(self):
        with self._lock:
            value=super().diagnostics()
            value["backend"]="experimental-native-compact-raster-incoming-v1"
            value["compact"]=deepcopy(self.last_compact)
            value["samples_scope"]="final low-raster traced anchors times configured proposals, not all fragment invocations or hardware ray queries; full-rate reconstruction has no spp claim"
            if "resident_profile" in value["device"]:
                value["device"]["resident_profile"]["gpu_scope"]="TLAS + full primary MRT + true ceil(W/4)xceil(H/4) GI raster + full reconstruction + copies"
            return value

    def close(self):
        with self._lock:
            if not self.closed:
                try:self.last_compact=self._compact_info()
                finally:super().close()
