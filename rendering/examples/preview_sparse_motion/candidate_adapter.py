"""Explicit caller-side compact/sparse distinction; no GPU or shader implementation."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys


def require(value,message):
    if not value:raise ValueError(message)


def kind(policy):
    if "compact" not in policy:return "sparse"
    compact=policy["compact"]
    require(isinstance(compact,dict) and compact.get("schema")=="phymeo-preview-compact-v1"
            and compact.get("tile")==4 and compact.get("history")=="none",
            "invalid compact source contract")
    require(policy.get("schema")=="phymeo-preview-sparse-v1" and policy.get("tile")==4
            and policy.get("bounces")==1 and policy.get("neighbor_candidates")==25
            and policy.get("reconstruction_revision")==2
            and policy.get("paths_per_anchor") in (2,4,16)
            and policy.get("estimator") in ("radiance","visibility-ratio"),
            "unsupported compact parent source contract")
    return "compact"


def require_kind(policy,requested):
    require(requested in ("sparse","compact") and kind(policy)==requested,
            "explicit candidate backend differs from prepared source contract")


def backend_class(module,policy):
    name="CompactBackend" if kind(policy)=="compact" else "SparseBackend"
    value=getattr(module,name,None)
    require(value is not None,"explicit candidate runtime class absent: "+name)
    return value


def backend_name(policy):
    return ("experimental-native-compact-raster-incoming-v1" if kind(policy)=="compact"
            else "experimental-native-sparse-incoming-v1")


def check_compact(value,extent,expected_captures,actual):
    width,height=extent
    require(type(width) is int and type(height) is int and width>0 and height>0,"invalid public extent")
    expected=dict(abi_version=1,enabled=True,tile=4,full_width=width,full_height=height,
        guide_width=(width+3)//4,guide_height=(height+3)//4,last_capture_rgb=True,
        rgb_captures=expected_captures,sensor_only_captures=0,successful_guide_draws=expected_captures,
        primary_color_attachments=5,reconstruction_sampled_images=8)
    require(all(value.get(k)==v for k,v in expected.items()),"compact actual extent/draw/attachment counters differ")
    require(0<=actual["visible_pixels"]<=width*height and
            0<=actual["traced_anchor_pixels"]<=expected["guide_width"]*expected["guide_height"],
            "compact low-raster anchors or full-rate visible count exceeds actual allocation")


def load_runtime(path):
    """Import the explicit adapter and its sibling, without leaking a search path.

    Renderer activation happens in the caller after this returns. Existing modules
    from a different producer checkout are rejected, never silently substituted.
    """
    path=Path(path).resolve(strict=True);examples=path.parent.parent
    dependency=examples/"preview_sparse/runtime.py"
    if path.parent.name=="preview_compact":
        require(dependency.is_file(),"compact runtime needs its actual sibling sparse runtime")
        for name,module in tuple(sys.modules.items()):
            if name=="preview_sparse" or name.startswith("preview_sparse."):
                source=getattr(module,"__file__",None)
                if source:require(Path(source).resolve().is_relative_to(examples/"preview_sparse"),
                                  "different preview_sparse runtime already imported")
                else:
                    paths=getattr(module,"__path__",())
                    require(all(Path(p).resolve()==examples/"preview_sparse" for p in paths),
                            "different preview_sparse namespace already imported")
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(examples))
        spec=importlib.util.spec_from_file_location("bound_motion_candidate_runtime",path)
        require(spec is not None and spec.loader is not None,"runtime module unavailable")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=prior
    dependencies=[]
    if path.parent.name=="preview_compact":
        imported=sys.modules.get("preview_sparse.runtime")
        require(imported is not None and Path(imported.__file__).resolve()==dependency.resolve(),
                "compact imported a different sparse adapter")
        dependencies.append(dict(file=str(dependency.resolve()),sha256=sha256(dependency.read_bytes()).hexdigest()))
    module.motion_adapter_dependencies=dependencies
    return module