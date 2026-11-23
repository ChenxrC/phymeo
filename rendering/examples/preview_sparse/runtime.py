"""Explicit experimental session over a generated private transport/runtime.

This wrapper intentionally does not publish the inherited progressive or guided
sampling claims. A capture is a new one-hop spatial reconstruction, never a
batch of full-rate 16-spp pixels. It does not expose read_linear_mean().
"""
import ctypes as ct
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys


COUNTERS=("invalid","visible_pixels","traced_anchor_pixels","reconstructed_pixels",
          "fallback_pixels","nondiffuse_pixels","accepted_neighbors","primary_proposals")


def activate(directory):
    root=Path(directory).resolve()
    manifest=json.loads((root/"source-manifest.json").read_text(encoding="utf-8"))
    if manifest["contract"]["schema"]!="phymeo-preview-sparse-v1":raise ValueError("wrong sparse source manifest")
    runtime=root/"runtime"
    for name,module in tuple(sys.modules.items()):
        if name.startswith("phymeo_rendering"):
            path=getattr(module,"__file__",None)
            if path and not Path(path).resolve().is_relative_to(runtime):
                raise RuntimeError("activate private sparse runtime before importing any original renderer")
    for relative,expected in manifest["generated_sha256"].items():
        if relative.replace("\\","/").startswith("runtime/"):
            path=root/relative
            if sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError("private runtime source changed: "+relative)
    sys.path.insert(0,str(runtime))
    return root,manifest


class SparseBackend:
    def __init__(self,directory,*,device_index=1,studio=None,seed=0,exposure=8.,
                 display_transform="pbr-neutral",reflection_mode="single_bounce",validation=False):
        self.root,self.manifest=activate(directory)
        self.library=self.root/"build/phymeo_native.dll"
        if not self.library.is_file():raise FileNotFoundError(self.library)
        self.lib=ct.CDLL(str(self.library))
        try:
            version=self.lib.pr_preview_sparse_abi_version
            self.query=self.lib.pr_preview_sparse_info
        except AttributeError as error:
            raise RuntimeError("dedicated sparse ABI absent; no fallback") from error
        version.argtypes=[];version.restype=ct.c_uint32
        if version()!=1:raise RuntimeError("unsupported sparse ABI")
        self.query.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t];self.query.restype=ct.c_int
        self.library_sha256=sha256(self.library.read_bytes()).hexdigest()
        self.settings=dict(device_index=device_index,studio=studio,indirect_seed=seed,exposure=exposure,
                           display_transform=display_transform,reflection_mode=reflection_mode,validation=validation)

    def open_scene(self,scene):
        from phymeo_rendering.backends.native import NativeBackend
        policy=self.manifest["contract"]
        backend=NativeBackend(library_path=self.library,shader_directory=self.root/"build",resident=True,
            shadow_mode="ray_query",msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,
            progressive=True,spatial_filter="guided",area_light_grid=policy["area_light_grid"],**self.settings)
        backend._preview_sparse_requested=True
        underlying=backend.open_scene(scene)
        try:return SparseSession(self,underlying)
        except BaseException:
            underlying.close();raise


class SparseSession:
    def __init__(self,backend,underlying):
        self.backend,self.underlying=backend,underlying
        self.closed=False;self.last_rgb=False;self.capture_count=0;self.last_info=self._info()

    def _info(self):
        values=(ct.c_uint32*16)()
        device=self.underlying._delegate.device
        if self.backend.query(device._handle,values,64):raise RuntimeError(device._error())
        p=self.backend.manifest["contract"]
        revision=p.get("reconstruction_revision",1)
        estimator=1 if p.get("estimator","radiance")=="visibility-ratio" else 0
        flags=p["area_light_grid"]|(p["neighbor_candidates"]<<8)|((0 if revision==1 else revision)<<16)|(estimator<<24)
        if tuple(values[:4])!=(1,1,p["tile"],p["paths_per_anchor"]) or values[7]!=flags:
            raise RuntimeError("private DLL/configuration/source manifest mismatch")
        result=dict(abi_version=values[0],shader_compatible=values[1],tile=values[2],paths_per_anchor=values[3],
                    rgb_captures=values[4],sensor_only_captures=values[5],last_capture_rgb=bool(values[6]),
                    area_light_grid=values[7]&255,neighbor_candidates=(values[7]>>8)&255,
                    reconstruction_revision=((values[7]>>16)&255) or 1,
                    estimator="visibility-ratio" if (values[7]>>24)==1 else "radiance",**dict(zip(COUNTERS,values[8:])))
        if result["invalid"]:raise RuntimeError("failed sparse publication status")
        return result

    def supports(self,request):return self.underlying.supports(request)

    def capture(self,frame,request):
        if self.closed:raise RuntimeError("sparse session closed")
        from phymeo_rendering.outputs import Channel
        rgb=Channel.RGB in request.channels
        self.last_rgb=False
        if rgb:self.underlying.reset_accumulation()
        result=self.underlying.capture(frame,request)
        current=self._info()
        if current["last_capture_rgb"]!=rgb:raise RuntimeError("sparse capture scope mismatch")
        if rgb:
            summed=sum(current[k] for k in ("traced_anchor_pixels","reconstructed_pixels","fallback_pixels","nondiffuse_pixels"))
            if summed!=current["visible_pixels"] or current["primary_proposals"]!=current["traced_anchor_pixels"]*current["paths_per_anchor"]:
                raise RuntimeError("sparse actual counter partition mismatch")
        self.last_info=current;self.capture_count+=1;self.last_rgb=rgb
        return result

    def read_reconstructed_linear(self):
        if self.closed or not self.last_rgb:raise RuntimeError("requires latest successful RGB sparse exposure")
        raw=self.underlying.read_linear_mean()
        if raw["batches"]!=1 or raw["format"]!="RGBA32F_LE" or raw["space"]!="scene-linear-before-exposure":
            raise RuntimeError("private transport state differs")
        return dict(width=raw["width"],height=raw["height"],format=raw["format"],
                    space="scene-linear sparse reconstructed exposure before exposure/display; not a Monte Carlo mean",
                    capture_index=self.capture_count-1,data=raw["data"])

    def read_incoming_guide(self):
        if self.closed or not self.last_rgb:raise RuntimeError("requires latest successful RGB sparse exposure")
        device=self.underlying._delegate.device
        width,height=self.underlying._delegate.progressive_history.key["extent"]
        data=(ct.c_ubyte*(width*height*16))()
        read=self.backend.lib.pr_preview_sparse_read_guide
        read.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];read.restype=ct.c_int
        if read(device._handle,data,ct.sizeof(data)):raise RuntimeError(device._error())
        return dict(width=width,height=height,format="RGBA32F_LE",
                    space="RGB=declared incoming estimator at traced anchors, own SH elsewhere; A=numeric uint24: normal bits0..19, traced20; ratio debug any-miss21/any-hit22/Ng-clipped23 when declared by manifest",
                    estimator=self.backend.manifest["contract"].get("estimator","radiance"),
                    capture_index=self.capture_count-1,data=bytes(data))

    def diagnostics(self):
        result=self.underlying.diagnostics()
        device=deepcopy(result["device"])
        # Transport structs retain their physical allocation/dispatch counters,
        # but their old sampling fields do not describe this experiment.
        for key in tuple(device):
            if key.startswith(("resident_progressive","resident_spatial","resident_settling","resident_indirect")):
                del device[key]
        if "resident_profile" in device:
            device["resident_profile"]["gpu_scope"]="TLAS build/update + full-rate primary + sparse incoming draw + spatial reconstruction + copies; not ray-only cost"
        return dict(backend="experimental-native-sparse-incoming-v1",contract=deepcopy(self.backend.manifest["contract"]),
                    actual=deepcopy(self.last_info),device=device,library_sha256=self.backend.library_sha256,
                    shader_sha256=result["shader_sha256"],decoded_geometry_sha256=result["decoded_geometry_sha256"],
                    triangles=result["triangles"],normal_maps=result["normal_maps"],reflection=result["reflection"],
                    effective_directional_lights=result["effective_directional_lights"],
                    samples_scope="primary_proposals is the visible final-guide anchor count times configured proposals; not hardware ray queries or all raster invocations (overdraw/helper invocations are not counted); hemisphere-clipped proposals may trace no ray; no full-image spp claim",
                    history="none",closed=self.closed)

    def close(self):
        if not self.closed:
            try:
                self.last_info=self._info()
            finally:
                self.underlying.close();self.closed=True;self.last_rgb=False

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
