"""Three-way real-device check: original one-hop, bound sparse parent, compact.

No project execution occurs during source generation. This CLI is for the
serial execution owner; all output directories are exclusive.
"""
import argparse
import ctypes as ct
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

from preview_sparse.runtime import SparseSession
from .runtime import CompactBackend,activate


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def runtime_hashes(manifest):
    """Compare Windows package identities, not JSON path separator spelling."""
    result={}
    for relative,digest_value in manifest["generated_sha256"].items():
        key=relative.replace("\\","/").casefold()
        if not key.startswith("runtime/"):continue
        if key in result:raise ValueError("ambiguous runtime path aliases")
        if any(part in ("", ".", "..") for part in key.split("/")):
            raise ValueError("noncanonical runtime path")
        result[key]=digest_value
    if not result:raise ValueError("runtime binding is empty")
    return result


def cases(root,panda):
    import closed_room_bounce_calibration as room
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import Instance,NodePose,Pose
    scene,frame,camera,studio=room.build_public(root/"geometry")
    values=[("room",scene,frame,camera,studio,1.),
            ("odd",scene,frame,replace(camera,width=97,height=95,cx_px=48.5,cy_px=47.5),studio,1.)]
    chromatic=replace(scene,key="compact-chromatic",materials=tuple(replace(m,base_color=(.98,.01,.02) if i%2==0 else (.01,.05,.98)) for i,m in enumerate(scene.materials)))
    chromatic_frame=replace(frame,key=replace(frame.key,scene_key=chromatic.key))
    values.append(("chromatic",chromatic,chromatic_frame,replace(camera,frame_key=chromatic_frame.key),studio,1.))
    strip=write_mesh(root/"geometry","compact-thin",((-.01,.1,.2),(.01,.1,.2),(.01,.1,1.8),(-.01,.1,1.8)),((0.,-1.,0.),)*4,((0,1,2),(0,2,3)))
    thin=replace(scene,key="compact-thin",meshes=scene.meshes+(strip,),materials=scene.materials+(Material("thin",(.03,.95,.04),0.,1.),),
                 instances=scene.instances+(Instance(99,strip.id,"thin","thin"),))
    thin_frame=replace(frame,key=replace(frame.key,scene_key=thin.key),nodes=frame.nodes+(NodePose("thin",Pose()),))
    values.append(("thin",thin,thin_frame,replace(camera,frame_key=thin_frame.key,width=97,height=95,cx_px=48.5,cy_px=47.5),studio,1.))
    values.append(("zero-light",scene,frame,camera,replace(studio,panel_strength=0.,environment_radiance=0.),1.))
    if panda:
        import panda_render_performance as factory
        scenes,frames,panda_studio,_=factory.bind_scene(root/"panda-geometry")
        f,request=factory.make_inputs(scenes[1],frames)[60]
        values.append(("panda-60",scenes[1],f,request.camera,panda_studio,8.))
    return values


def sparse_owner(parent,compact_manifest,scene,settings):
    """Same verified Python runtime, original sparse DLL/SPVs; no activation swap."""
    from phymeo_rendering.backends.native import NativeBackend
    parent=Path(parent)
    raw=(parent/"source-manifest.json").read_bytes()
    if sha256(raw).hexdigest()!=compact_manifest["parent_manifest_sha256"]:
        raise ValueError("bound parent sparse manifest differs")
    manifest=json.loads(raw)
    if runtime_hashes(manifest)!=runtime_hashes(compact_manifest):
        raise ValueError("compact changed runtime semantics; cannot share process with sparse baseline")
    for relative,expected in manifest["generated_sha256"].items():
        if digest(parent/relative)!=expected:raise ValueError("bound parent source changed")
    library=parent/"build/phymeo_native.dll"
    lib=ct.CDLL(str(library))
    version=lib.pr_preview_sparse_abi_version;version.argtypes=[];version.restype=ct.c_uint32
    if version()!=1:raise ValueError("parent sparse ABI differs")
    query=lib.pr_preview_sparse_info
    query.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.c_size_t];query.restype=ct.c_int
    backend=NativeBackend(library_path=library,shader_directory=library.parent,resident=True,shadow_mode="ray_query",
        msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,progressive=True,spatial_filter="guided",
        area_light_grid=manifest["contract"]["area_light_grid"],**settings)
    backend._preview_sparse_requested=True
    owner=backend.open_scene(scene)
    try:
        return SparseSession(SimpleNamespace(manifest=manifest,query=query,lib=lib,library_sha256=digest(library)),owner)
    except BaseException:
        owner.close();raise


def run(args):
    prepared,manifest=activate(args.prepared)
    import numpy as np
    from PIL import Image
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-preview-compact-smoke-v1",status="started",contract=manifest["contract"],cases=[],
                source_manifest_sha256=digest(prepared/"source-manifest.json"),script_sha256=digest(__file__),
                full_library_sha256=digest(args.full_library),comparison_scope="same-state unfiltered original one-hop16, immutable sparse parent, true compact GI; radiance error is reported, not a quality pass")
    dump(out/"report.json",report)
    try:
        for name,scene,frame,camera,studio,exposure in cases(out,args.panda):
            target=out/name;target.mkdir()
            row=dict(name=name,input_frame=asdict(frame),camera=asdict(camera),outputs={})
            report["cases"].append(row)
            request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
            settings=dict(device_index=args.device_index,studio=studio,indirect_seed=args.seed,exposure=exposure,
                          display_transform="pbr-neutral",reflection_mode="single_bounce",validation=args.validation)
            def open_owner(label):
                if label=="compact":
                    return CompactBackend(prepared,device_index=args.device_index,studio=studio,seed=args.seed,
                        exposure=exposure,validation=args.validation).open_scene(scene)
                if label=="sparse":return sparse_owner(manifest["parent_prepared"],manifest,scene,settings)
                return NativeBackend(library_path=args.full_library,shader_directory=args.full_shaders or args.full_library.parent,
                    resident=True,shadow_mode="ray_query",msaa_samples=1,indirect_mode="diffuse_bounce",indirect_bounces=1,
                    progressive=True,spatial_filter="none",area_light_grid=manifest["contract"]["area_light_grid"],**settings).open_scene(scene)
            stored={};linear={}
            for label in ("full-rate","sparse","compact"):
                owner=open_owner(label)
                receipt={};row["outputs"][label]=receipt
                try:
                    start=perf_counter();capture=owner.capture(frame,request);elapsed=1000*(perf_counter()-start)
                    if capture.frame_key!=frame.key or capture.time_s!=frame.time_s or capture.camera!=camera:
                        raise ValueError("stale exposure identity")
                    planes={plane.channel:plane.data for plane in capture.planes};stored[label]=planes
                    for channel,data in planes.items():
                        (target/(label+"."+channel.value)).write_bytes(data)
                    Image.frombytes("RGB",(camera.width,camera.height),planes[Channel.RGB]).save(target/(label+".png"))
                    start=perf_counter()
                    raw=owner.read_linear_mean() if label=="full-rate" else owner.read_reconstructed_linear()
                    receipt["linear_read_ms"]=1000*(perf_counter()-start)
                    (target/(label+".rgba32f")).write_bytes(raw["data"])
                    linear[label]=np.frombuffer(raw["data"],"<f4").reshape(camera.height,camera.width,4).copy()
                    if not np.isfinite(linear[label]).all():raise ValueError("nonfinite published raw radiance")
                    receipt.update(capture_ms=elapsed,diagnostics=owner.diagnostics(),plane_sha256={k.value:sha256(v).hexdigest() for k,v in planes.items()},linear_sha256=sha256(raw["data"]).hexdigest())
                    if label=="compact":
                        start=perf_counter()
                        guide=owner.read_incoming_guide();attributes=owner.read_full_attributes();low=owner.read_low_sensors()
                        receipt["attribute_read_ms"]=1000*(perf_counter()-start)
                        for filename,data in (("low-guide",guide),("full-attributes",attributes),*(("low-"+k,v) for k,v in low.items())):
                            (target/(filename+".bin")).write_bytes(data["data"])
                        lw,lh=guide["width"],guide["height"]
                        if (lw,lh)!=((camera.width+3)//4,(camera.height+3)//4):raise ValueError("guide is not truly compact")
                        gids=np.frombuffer(low["id"]["data"],"<u4").reshape(lh,lw)
                        valid=np.frombuffer(low["valid"]["data"],"u1").reshape(lh,lw)
                        depth=np.frombuffer(low["depth"]["data"],"<f4").reshape(lh,lw)
                        normal=np.frombuffer(guide["data"],"<f4").reshape(lh,lw,4)
                        code=np.rint(normal[...,3]).astype(np.uint32)
                        traced=((code&(1<<20))!=0)&(valid!=0)
                        ys,xs=np.mgrid[:lh,:lw];outside=(4*xs+2>=camera.width)|(4*ys+2>=camera.height)
                        if np.any(valid[outside]) or np.any(gids[outside]) or np.any(traced[outside]):
                            raise ValueError("odd out-of-bounds anchor was falsely published")
                        if not np.array_equal(valid!=0,gids!=0) or not np.all(np.isfinite(depth[valid!=0])):
                            raise ValueError("low-resolution hit attributes disagree")
                        count=int(traced.sum())
                        if count!=receipt["diagnostics"]["actual"]["traced_anchor_pixels"]:
                            raise ValueError("low trace counter does not match actual guide")
                        receipt["low_geometry"]=dict(width=lw,height=lh,traced=count,odd_invalid_cells=int(outside.sum()),
                            traced_by_id={str(i):int(((gids==i)&traced).sum()) for i in np.unique(gids) if i},
                            guide_sha256=sha256(guide["data"]).hexdigest(),
                            attribute_sha256=sha256(attributes["data"]).hexdigest())
                        if name=="thin" and receipt["low_geometry"]["traced_by_id"].get("99",0)!=0:
                            raise ValueError("registered thin control unexpectedly contains a regular compact anchor")
                        if name=="odd":
                            changed=replace(camera,width=99,height=93,cx_px=49.5,cy_px=46.5)
                            owner.capture(frame,CaptureRequest(changed,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
                            try:owner.read_incoming_guide()
                            except RuntimeError:pass
                            else:raise ValueError("sensor-only capture exposed stale guide")
                            repeated=owner.capture(frame,request)
                            if {p.channel:p.data for p in repeated.planes}!=planes:
                                raise ValueError("compact state was not reproducible after sensor-only resize")
                            receipt["repeat_after_resize_exact"]=True
                finally:
                    original_error=sys.exc_info()[0] is not None
                    try:
                        owner.close();receipt["closed"]=owner.diagnostics()
                        device=receipt["closed"]["device"]
                        if device.get("closed") is not True or any(device.get(k,1) for k in ("live_contexts","live_frames","live_resources","validation_errors")):
                            raise ValueError("close/resource/validation check failed")
                    except BaseException as error:
                        receipt["close_error"]=dict(type=type(error).__name__,message=str(error))
                        if not original_error:raise
                    finally:dump(target/"receipt.json",row)
            row["differences"]={}
            for label in ("sparse","compact"):
                if any(stored[label][c]!=stored["full-rate"][c] for c in Channel if c is not Channel.RGB):
                    raise ValueError(label+" changed full-rate sensor bytes")
                delta=linear[label][...,:3].astype(float)-linear["full-rate"][...,:3]
                row["differences"][label]=dict(full_sensor_bytes_identical=True,raw_rgb_rmse=float(np.sqrt(np.mean(delta*delta))))
            if name=="zero-light" and any(np.any(value[...,:3]) for value in linear.values()):
                raise ValueError("zero-light control emitted radiance")
            dump(target/"receipt.json",row);dump(out/"report.json",report)
        report["status"]="complete"
    except BaseException as error:
        report["status"]="partial_failure";report["error"]=dict(type=type(error).__name__,message=str(error))
        raise
    finally:dump(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--full-library",type=Path,required=True)
    p.add_argument("--full-shaders",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device-index",type=int,default=1)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--panda",action="store_true")
    p.add_argument("--validation",action="store_true")
    run(p.parse_args())


if __name__=="__main__":main()
