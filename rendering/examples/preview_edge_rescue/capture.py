"""Actual GPU selective edge rescue, retaining the registered thin-edge evidence.

One prepared package per process. Existing independent seed1/seed2 references
are reused byte-for-byte. This is a new full-rate selective trace shader; no ROI-specific shading or
image correction are performed. This is a diagnostic, not an acceleration claim.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def bytes_hash(data):return sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def module(path,name):
    path=Path(path).resolve(strict=True)
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    previous=list(sys.path)
    try:
        sys.path.insert(0,str(path.parent));sys.modules[name]=value;spec.loader.exec_module(value)
    finally:sys.path[:]=previous
    return value


def source_contract(prepared,kind):
    manifest=read(prepared/"source-manifest.json");policy=manifest["contract"]
    require(policy.get("schema")=="phymeo-preview-sparse-v1" and policy.get("tile")==4 and
        policy.get("neighbor_candidates")==25 and policy.get("reconstruction_revision")==2 and
        policy.get("bounces")==1,"diagnostic source is explicitly limited to one-hop tile4/K25/R2")
    compact=policy.get("compact",{}).get("schema")=="phymeo-preview-compact-v1"
    require(compact and kind=="edge" and policy.get("edge_rescue",{}).get("schema")=="phymeo-preview-edge-rescue-v1","explicit adapter kind differs from source contract")
    path=prepared/"source/_shaders/resident_spatial.comp"
    expected={k.replace("\\","/"):v for k,v in manifest["generated_sha256"].items()}
    require(digest(path)==expected["source/_shaders/resident_spatial.comp"],"reconstruction source changed after generation")
    text=path.read_text(encoding="utf-8")
    tokens=(".003*min(a,b)","dot(normal,candidateN)<.98","incoming=sum/weightSum","vec3 incoming=own.rgb") if compact else (
        ".003*min(z,qz)","dot(normal,normalDecode(candidateCode))<.98","incoming=sum/total","vec3 incoming=own.rgb")
    require(all(x in text for x in tokens),"unsupported reconstruction gates; do not apply a stale CPU classifier")
    for name,value in manifest.get("compiled_sha256",{}).items():
        require(digest(prepared/"build"/name)==value,"sealed binary changed: "+name)
    return manifest


def plane(packet,shape,fmt,dtype,*,capture_index):
    import numpy as np
    require(packet["format"]==fmt and (packet["height"],packet["width"])==shape[:2]
        and packet["capture_index"]==capture_index,"actual GPU readback phase/format/extent differs")
    result=np.frombuffer(packet["data"],dtype)
    require(result.size==int(np.prod(shape)),"actual readback byte count differs")
    return result.reshape(shape).copy()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter",type=Path,default=HERE/"runtime.py")
    ap.add_argument("--kind",choices=("edge",),default="edge")
    ap.add_argument("--prepared",type=Path,required=True)
    ap.add_argument("--fixture",type=Path,required=True,help="bound original guided_motion_study.py")
    ap.add_argument("--reference",type=Path,required=True,help="existing complete thin3-ref directory")
    ap.add_argument("--prior",type=Path,help="optional earlier same-package thin3 report; actual RGB must match")
    ap.add_argument("--assets",type=Path,help="original licensed Panda assets; default fixture sibling rendering/assets/panda")
    ap.add_argument("--device-index",type=int,default=1)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(argv)
    args.prepared=args.prepared.resolve(strict=True);args.fixture=args.fixture.resolve(strict=True)
    args.reference=args.reference.resolve(strict=True);args.output=args.output.resolve()
    manifest=source_contract(args.prepared,args.kind)
    reference=read(args.reference/"summary.json")
    require(reference["schema"]=="phymeo-native-reveal-preview-v1" and reference["status"]=="complete"
        and reference["fixture_sha256"]==digest(args.fixture),"original thin reference/fixture binding differs")
    require(reference["reference_batches"]==64,"registered reference batch count differs")
    prior=read(args.prior/"summary.json") if args.prior else None
    if prior:
        require(prior["status"]=="complete" and prior["fixture_sha256"]==digest(args.fixture)
            and prior["prepared_sha256"]==digest(args.prepared/"source-manifest.json"),"prior evidence belongs to another package or fixture")
    # The existing loader imports only the explicit adapter, then restores sys.path.
    loader=module(HERE.parent/"preview_sparse_motion/candidate_adapter.py","edge_candidate_adapter")
    runtime=module(args.adapter,"bound_edge_runtime");runtime.activate(args.prepared)
    import numpy as np
    from PIL import Image
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    fixture=module(args.fixture,"edge_bound_motion_fixture")
    coverage=module(HERE/"coverage.py","edge_nominal_support")
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema="phymeo-edge-rescue-capture-v1",status="preparing",frames={},
        source=dict(prepared=str(args.prepared),manifest_sha256=digest(args.prepared/"source-manifest.json"),
            contract=manifest["contract"],adapter_file=str(args.adapter.resolve()),adapter_sha256=digest(args.adapter),
            fixture_file=str(args.fixture),fixture_sha256=digest(args.fixture),
            compute_sha256=digest(args.prepared/"source/_shaders/resident_spatial.comp"),
            trace_sha256=digest(args.prepared/"source/_shaders/resident_reflection.glsl"),
            estimator_sha256=digest(args.prepared/"source/_shaders/preview_sparse_ratio.glsl") if (args.prepared/"source/_shaders/preview_sparse_ratio.glsl").is_file() else None,
            reference_summary_sha256=digest(args.reference/"summary.json"),
            prior_summary_sha256=digest(args.prior/"summary.json") if args.prior else None,
            helper_sha256=digest(__file__),classifier_sha256=digest(HERE/"coverage.py")),
        protocol=dict(poses=[14,15],measured_pose=15,registered_receiver=158,blocker=202,expected_pixels=69,
            rule="previous blocker, current finite farther foreground; then fixed registered right-finger identity",
            support_scope="nominal source FP32 gates from actual GPU readbacks; near-boundary cases explicit",
            timing_scope="two completed four-channel captures; no warmup/throughput assertion; debug reads measured separately",
            model_scope="finite-reference displayed error; local transport is a different model, not an exact equivalent"),
        performance_claim=False,shader_changed=True)
    save(args.output/"summary.json",result)
    owner=None
    try:
        with TemporaryDirectory(prefix="phymeo-edge-support-") as temp:
            assets=(args.assets or args.fixture.parent.parent/"assets/panda").resolve(strict=True)
            visual=fixture.original.panda_visual_assets
            with patch.object(fixture.original,"panda_visual_assets",partial(visual,assets)):
                nodes=fixture.build_motion(Path(temp))
            scene=nodes[0][0]
            material_id=next(i.material_id for i in scene.instances if i.id==158)
            material=next(m for m in scene.materials if m.id==material_id)
            require(tuple(material.base_color)==(.7,.7,.7) and material.metallic==0 and material.roughness==1,
                "registered white diffuse receiver material differs")
            result["registered_material"]=asdict(material)
            save(args.output/"inputs.json",dict(scene=asdict(scene),poses={str(i):dict(frame=asdict(nodes[i][1]),camera=asdict(nodes[i][2])) for i in (14,15)}))
            result["inputs_sha256"]=digest(args.output/"inputs.json")
            backend=runtime.EdgeBackend(
                args.prepared,device_index=args.device_index,seed=0,studio=None,exposure=1.,
                display_transform="pbr-neutral",reflection_mode="environment")
            owner=backend.open_scene(scene);geometry={};images={}
            for i in (14,15):
                _,frame,camera=nodes[i];request=CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
                start=perf_counter();shot=owner.capture(frame,request);elapsed=(perf_counter()-start)*1000
                require(shot.frame_key==frame.key and shot.camera==camera,"capture returned another input")
                planes={p.channel:p.data for p in shot.planes};h,w=camera.height,camera.width
                sensors={c.value:bytes_hash(v) for c,v in planes.items() if c is not Channel.RGB}
                old=reference["frames"][str(i)]
                require(sensors==old["sensors"] and asdict(frame.key)==old["key"] and frame.time_s==old["time"],
                    "full geometry or source phase differs from independent reference")
                rgb=np.frombuffer(planes[Channel.RGB],np.uint8).reshape(h,w,3).copy();images[i]=rgb
                geometry[i]=dict(ids=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4").reshape(h,w).copy(),
                    depth=np.frombuffer(planes[Channel.FORWARD_DEPTH],"<f4").reshape(h,w).copy(),
                    valid=np.frombuffer(planes[Channel.VALID],np.uint8).reshape(h,w).copy())
                if prior:require(bytes_hash(planes[Channel.RGB])==prior["frames"][str(i)]["rgb_sha256"],"same-package actual RGB changed from prior evidence")
                Image.fromarray(rgb).save(args.output/f"pose-{i}.png")
                result["frames"][str(i)]=dict(key=asdict(frame.key),time=frame.time_s,sensors=sensors,
                    rgb_sha256=bytes_hash(planes[Channel.RGB]),png_sha256=digest(args.output/f"pose-{i}.png"),capture_ms=elapsed)
            diag=owner.diagnostics();result["diagnostics"]=diag
            require(diag["actual"]["rgb_captures"]==2 and diag["actual"]["invalid"]==0,"unexpected actual candidate work count")
            if prior:
                require(diag["library_sha256"]==prior["diagnostics"]["library_sha256"] and
                    diag["shader_sha256"]==prior["diagnostics"]["shader_sha256"],"candidate compiled code differs from prior")
            require(diag["decoded_geometry_sha256"]==reference["diagnostics"]["decoded_geometry_sha256"],"actual decoded mesh geometry differs")
            reveal=coverage.reveal_mask(geometry[14],geometry[15],blocker=202)
            mask=reveal&(geometry[15]["ids"]==158)
            require(int(mask.sum())==69,"registered right-finger reveal is not the original 69 pixels")
            saved_mask=np.array(Image.open(args.reference/"exposed-right-finger.png"))
            require(saved_mask.shape==mask.shape and np.array_equal(saved_mask,np.uint8(mask)*255),"actual reveal mask differs from saved reference")
            Image.fromarray(np.uint8(mask)*255).save(args.output/"exposed-right-finger.png")
            start=perf_counter();gp=owner.read_incoming_guide();lp=owner.read_reconstructed_linear()
            compact=True;lowshape=((h+3)//4,(w+3)//4) if compact else (h,w)
            guide=plane(gp,(*lowshape,4),"RGBA32F_LE","<f4",capture_index=1)
            linear=plane(lp,(h,w,4),"RGBA32F_LE","<f4",capture_index=1)
            require(np.isfinite(linear).all() and np.all(linear[...,:3]>=0),"invalid actual scene-linear readback")
            arrays=dict(ids=geometry[15]["ids"],depth=geometry[15]["depth"],valid=geometry[15]["valid"],guide=guide,compact=compact)
            packets={"guide":gp,"reconstructed_linear":lp}
            if compact:
                apacket=owner.read_full_attributes();packets["full_attributes"]=apacket
                arrays["attributes"]=plane(apacket,(h,w,4),"RGBA32F_LE","<f4",capture_index=1)
                low=owner.read_low_sensors()
                for name,key,fmt,dt in (("id","low_ids","U32_LE","<u4"),("depth","low_depth","F32_LE","<f4"),("valid","low_valid","U8","u1")):
                    packets[key]=low[name];arrays[key]=plane(low[name],lowshape,fmt,dt,capture_index=1)
            rescued_packet=owner.read_rescue_incoming()
            rescued=plane(rescued_packet,(h,w,4),"RGBA32F_LE","<f4",capture_index=1)
            packets["rescue_incoming"]=rescued_packet
            valid_rescue=(rescued[...,3].astype(np.uint32)&(1<<20))!=0
            require(int(valid_rescue[mask].sum())==69,"not all registered low-support pixels received genuine full-rate rescue")
            result["actual_rescue"]=dict(registered_pixels=69,valid_results=int(valid_rescue[mask].sum()),
                incoming_scope=rescued_packet["space"],device_counts=diag["edge_rescue"])
            result["debug_read_wall_ms"]=(perf_counter()-start)*1000
            result["actual_readbacks"]={name:dict(**{k:v for k,v in p.items() if k!="data"},bytes=len(p["data"]),sha256=bytes_hash(p["data"])) for name,p in packets.items()}
            references=[]
            for seed in (1,2):
                ref=np.array(Image.open(args.reference/f"seed-{seed}-pose-15.png"))
                binding=reference["references"][f"{seed}-15"]
                require(ref.shape==(h,w,3) and ref.dtype==np.uint8 and bytes_hash(ref.tobytes())==binding["rgb_sha256"] and binding["spp"]==1024,
                    "independent reference pixels or actual spp differs")
                references.append(ref)
            baseline=np.array(Image.open(args.reference/"pose-15.png"))
            require(bytes_hash(baseline.tobytes())==reference["frames"]["15"]["rgb_sha256"],"old full-rate output changed")
            pixels=[]
            for y,x in zip(*np.where(mask)):
                row=coverage.inspect_pixel(int(x),int(y),**arrays)
                row["actual_rescue_incoming"]=rescued[y,x,:3].astype(float).tolist()
                row["actual_rescue_valid"]=bool(valid_rescue[y,x])
                row["actual_scene_linear_total"]=linear[y,x,:3].astype(float).tolist()
                row["actual_rgb8"]=images[15][y,x].tolist()
                row["reference_rgb8"]=[ref[y,x].tolist() for ref in references]
                row["display_errors"]=[coverage.display_error(images[15][y:y+1,x],ref[y:y+1,x]) for ref in references]
                pixels.append(row)
            result["coverage"]=dict(pixels=69,classes=dict(Counter(p["classification"] for p in pixels)),
                accepted_neighbor_counts=dict(Counter(str(p["accepted_count"]) for p in pixels)),
                boundary_uncertain_pixels=sum(p["boundary_uncertain"] for p in pixels),
                scope="original K25 support classification plus actual extra-pass incoming and valid bit; all registered69 genuinely rescued")
            result["groups"]={}
            for name in ("all","geometry_fallback","supported_reconstruction"):
                selected=np.ones(len(pixels),bool) if name=="all" else np.array([p["classification"]==name for p in pixels])
                current=images[15][mask][selected]
                result["groups"][name]=dict(pixels=int(selected.sum()),
                    candidate=[coverage.display_error(current,ref[mask][selected]) for ref in references],
                    original_fullrate16_guided=[coverage.display_error(baseline[mask][selected],ref[mask][selected]) for ref in references])
            result["reference_disagreement"]=coverage.display_error(references[0][mask],references[1][mask])
            save(args.output/"pixels.json",pixels);result["pixels_sha256"]=digest(args.output/"pixels.json")
            archive={k:v for k,v in arrays.items() if isinstance(v,np.ndarray)}
            archive.update(previous_ids=geometry[14]["ids"],previous_depth=geometry[14]["depth"],previous_valid=geometry[14]["valid"],
                registered_mask=mask,selected_scene_linear_total=linear[mask],selected_rescue_incoming=rescued[mask])
            np.savez_compressed(args.output/"evidence.npz",**archive)
            result["evidence_sha256"]=digest(args.output/"evidence.npz")
            result["interpretation"]=["No HDR/studio in this fixture: unsupported diffuse receivers use zero incoming, which cannot by itself explain a positive light bias.",
                "Supported same-ID/depth/Ns neighbors need not share secondary visibility; recorded source samples and signed errors separate that question from fallback.",
                "Local120 uses both 0.12m guide ray tmax and 1-smoothstep hit fade; beyond-radius directions are misses. Unlimited comparison must match compact/path/seed/configuration; old sparse P2 is confounded.",
                "Two independent 1024-spp displayed references are finite estimates; this report does not claim an exact light-transport oracle."]
            result["status"]="actual_readbacks_bound"
    except BaseException as error:
        result["status"]="failed";result["error"]=str(error);raise
    finally:
        try:
            if owner is not None:
                owner.close();result["closed_diagnostics"]=owner.diagnostics()
        except Exception as error:
            result["cleanup_error"]=str(error)
            if result["status"]!="failed":
                result["status"]="failed";result["error"]="cleanup failed: "+str(error)
                raise
        finally:
            save(args.output/"summary.json",result)
    print(str(args.output/"summary.json"))


if __name__=="__main__":main()
