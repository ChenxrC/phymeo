"""Bounded real-device sparse comparison; invoked only by the execution owner.

python -m preview_sparse.run_smoke --prepared NEW_SOURCE --baseline-library DLL --output NEW_RESULT
Add --panda for one real 720p Panda frame after the small room/odd/chromatic/thin/black controls.
The full-rate baseline uses one bounce / 16 proposals, no display denoiser.
Image error is recorded, not silently converted into quality acceptance.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from .runtime import activate,SparseBackend


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def run(args):
    prepared,manifest=activate(args.prepared)
    # Every rendering import follows activation of the hashed private copy.
    import numpy as np
    from PIL import Image
    import closed_room_bounce_calibration as room
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import Instance,NodePose,Pose
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath

    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(schema="phymeo-preview-sparse-smoke-v1",status="started",contract=manifest["contract"],cases=[],
                baseline_library_sha256=sha256(args.baseline_library.read_bytes()).hexdigest(),
                source_manifest_sha256=sha256((prepared/"source-manifest.json").read_bytes()).hexdigest(),
                error_scope="same-state full-rate 16-proposal one-hop baseline; raw difference includes reduced sampling, spatial bias and RGB8 rho quantization; no automatic quality pass")
    dump(out/"report.json",result)
    scene,frame,camera,studio=room.build_public(out/"geometry")
    cases=[("room",scene,frame,camera,studio),
           ("odd",scene,frame,replace(camera,width=97,height=95,cx_px=48.5,cy_px=47.5),studio)]
    saturated=replace(scene,key=scene.key+":chromatic",materials=tuple(replace(m,base_color=(.98,.01,.02) if i%2==0 else (.01,.05,.98)) for i,m in enumerate(scene.materials)))
    chromatic_frame=replace(frame,key=replace(frame.key,scene_key=saturated.key))
    cases.append(("chromatic",saturated,chromatic_frame,replace(camera,frame_key=chromatic_frame.key),studio))
    strip=write_mesh(out/"geometry","thin-receiver",((-.01,.1,.2),(.01,.1,.2),(.01,.1,1.8),(-.01,.1,1.8)),
                     ((0.,-1.,0.),)*4,((0,1,2),(0,2,3)))
    thin=replace(scene,key=scene.key+":thin",meshes=scene.meshes+(strip,),
                 materials=scene.materials+(Material("thin-material",(.03,.95,.04),0.,1.),),
                 instances=scene.instances+(Instance(99,strip.id,"thin-material","thin"),))
    thin_frame=replace(frame,key=replace(frame.key,scene_key=thin.key),nodes=frame.nodes+(NodePose("thin",Pose()),))
    cases.append(("thin",thin,thin_frame,replace(camera,frame_key=thin_frame.key,width=97,height=95,cx_px=48.5,cy_px=47.5),studio))
    cases.append(("zero-light",scene,frame,camera,replace(studio,panel_strength=0.,environment_radiance=0.)))
    if manifest["contract"].get("guide_outcome_flags",False):
        # Independently registered geometry: open single back wall, then a
        # closed six-wall room with black terminals, then a red rear terminal.
        # The camera/receiver are identical; these are real ray-query controls.
        def ratio_case(name,ids,red_front=False):
            instances=tuple(i for i in scene.instances if i.id in ids)
            mesh_ids={i.mesh_id for i in instances}
            materials=[]
            for material in scene.materials:
                if material.id not in {i.material_id for i in instances}:continue
                white=material.id=="back"
                red=red_front and material.id=="front"
                materials.append(replace(material,base_color=(.6,.6,.6) if white else ((.8,.01,.01) if red else (0.,0.,0.)),
                                         metallic=0. if white or red else 1.,roughness=1.))
            value=replace(scene,key=name,instances=instances,meshes=tuple(m for m in scene.meshes if m.id in mesh_ids),materials=tuple(materials))
            node_ids={i.node_id for i in instances}
            value_frame=replace(frame,key=replace(frame.key,scene_key=value.key),nodes=tuple(n for n in frame.nodes if n.node_id in node_ids))
            return name,value,value_frame,replace(camera,frame_key=value_frame.key),replace(studio,panel_strength=0.,environment_radiance=.25)
        cases.extend((ratio_case("ratio-open",{5}),ratio_case("ratio-closed",set(range(1,7))),
                      ratio_case("ratio-colored-hit",set(range(1,7)),True)))
    if args.panda:
        import panda_render_performance as factory
        scenes,frames,panda_studio,trace=factory.bind_scene(out/"panda-geometry")
        panda_frame,panda_request=factory.make_inputs(scenes[1],frames)[60]
        cases.append(("panda-60",scenes[1],panda_frame,panda_request.camera,panda_studio))
        result["panda_trace_sha256"]=sha256(trace.read_bytes()).hexdigest()
    try:
        for name,current,current_frame,current_camera,current_studio in cases:
            target=out/name;target.mkdir()
            exposure=8. if name=="panda-60" else 1.
            baseline=NativeBackend(library_path=args.baseline_library,shader_directory=args.baseline_shaders or args.baseline_library.parent,
                device_index=args.device_index,resident=True,shadow_mode="ray_query",studio=current_studio,
                area_light_grid=manifest["contract"]["area_light_grid"],msaa_samples=1,
                reflection_mode="single_bounce",indirect_mode="diffuse_bounce",indirect_bounces=1,
                indirect_seed=0,progressive=True,spatial_filter="none",exposure=exposure,
                display_transform="pbr-neutral",validation=args.validation).open_scene(current)
            try:
                sparse=SparseBackend(prepared,device_index=args.device_index,studio=current_studio,seed=0,
                    exposure=exposure,validation=args.validation).open_scene(current)
            except BaseException:
                baseline.close()
                raise
            row=dict(name=name,frame=asdict(current_frame),camera=asdict(current_camera),outputs={})
            request=CaptureRequest(current_camera,RenderPath.RASTER,tuple(Channel))
            captures={};raws={}
            try:
                for label,session in (("full-rate",baseline),("sparse",sparse)):
                    start=perf_counter();shot=session.capture(current_frame,request)
                    elapsed=1000*(perf_counter()-start)
                    if shot.frame_key!=current_frame.key or shot.camera!=current_camera or shot.time_s!=current_frame.time_s:
                        raise ValueError("stale device exposure")
                    planes={p.channel:p.data for p in shot.planes};captures[label]=planes
                    rgb=planes[Channel.RGB]
                    Image.frombytes("RGB",(current_camera.width,current_camera.height),rgb).save(target/(label+".png"))
                    for channel,data in planes.items():
                        if channel is not Channel.RGB:(target/(label+"."+channel.value)).write_bytes(data)
                    read_start=perf_counter()
                    raw=session.read_linear_mean() if label=="full-rate" else session.read_reconstructed_linear()
                    raw_ms=1000*(perf_counter()-read_start)
                    (target/(label+".rgba32f")).write_bytes(raw["data"])
                    raw_array=np.frombuffer(raw["data"],"<f4").reshape(current_camera.height,current_camera.width,4)
                    if not np.isfinite(raw_array).all():raise ValueError("nonfinite raw exposure")
                    raws[label]=raw_array.copy()
                    row["outputs"][label]=dict(capture_ms=elapsed,linear_read_ms=raw_ms,
                        plane_sha256={k.value:sha256(v).hexdigest() for k,v in planes.items()},
                        linear_sha256=sha256(raw["data"]).hexdigest(),diagnostics=session.diagnostics())
                if any(captures["full-rate"][c]!=captures["sparse"][c] for c in Channel if c is not Channel.RGB):
                    raise ValueError("sparse changed actual depth/ID/VALID")
                guide_start=perf_counter();guide=sparse.read_incoming_guide()
                row["guide_read_ms"]=1000*(perf_counter()-guide_start)
                (target/"incoming-guide.rgba32f").write_bytes(guide["data"])
                guides=np.frombuffer(guide["data"],"<f4").reshape(current_camera.height,current_camera.width,4)
                codes=np.rint(guides[...,3]).astype(np.uint32)
                ids=np.frombuffer(captures["sparse"][Channel.INSTANCE_ID],"<u4").reshape(current_camera.height,current_camera.width)
                traced=(codes&(1<<20))!=0
                row["traced_anchor_pixels_by_id"]={str(i):int(((ids==i)&traced).sum()) for i in np.unique(ids) if i}
                row["visible_pixels_by_id"]={str(i):int((ids==i).sum()) for i in np.unique(ids) if i}
                if name.startswith("ratio-"):
                    selected=(ids==5)&traced
                    if not np.any(selected):raise ValueError("ratio control has no actual receiver anchors")
                    flag=codes[selected]
                    any_miss=(flag&(1<<21))!=0
                    any_hit=(flag&(1<<22))!=0
                    clipped=(flag&(1<<23))!=0
                    values=guides[selected,:3]
                    row["ratio_control"]=dict(receiver_anchor_pixels=int(selected.sum()),
                        any_miss_pixels=int(any_miss.sum()),any_hit_pixels=int(any_hit.sum()),
                        any_geometric_clip_pixels=int(clipped.sum()),
                        incoming_min_rgb=values.min(axis=0).tolist(),incoming_max_rgb=values.max(axis=0).tolist())
                    if name=="ratio-open":
                        if not np.all(any_miss) or np.any(any_hit) or np.any(clipped) or not np.all(values==np.float32(.25)):
                            raise ValueError("fully open constant environment did not reproduce exact existing SH/IBL")
                    else:
                        if np.any(any_miss) or not np.all(any_hit):
                            raise ValueError("closed receiver failed actual no-sky-miss control")
                        row["ratio_control"]["sky_contribution_zero_from_actual_miss_flags"]=True
                        if name=="ratio-colored-hit" and not np.any(values[:,0]>values[:,1]):
                            raise ValueError("true red terminal produced no chromatic incoming")
                rgb_a=np.frombuffer(captures["full-rate"][Channel.RGB],"u1").astype(float)
                rgb_b=np.frombuffer(captures["sparse"][Channel.RGB],"u1").astype(float)
                difference=raws["sparse"][...,:3].astype(float)-raws["full-rate"][...,:3]
                row["paired_difference"]=dict(rgb8_rmse=float(np.sqrt(np.mean((rgb_a-rgb_b)**2))),
                    raw_rgb_rmse=float(np.sqrt(np.mean(difference**2))),sensor_bytes_identical=True)
                if name=="zero-light" and (np.any(rgb_a) or np.any(rgb_b) or np.any(raws["sparse"][...,:3])):
                    raise ValueError("zero-light scene emitted nonzero radiance")
                if name=="thin":
                    if row["visible_pixels_by_id"].get("99",0)==0:raise ValueError("thin control is not actually visible")
                    if manifest["contract"]["tile"]==4 and row["traced_anchor_pixels_by_id"].get("99",0)!=0:
                        raise ValueError("registered tile4 thin control unexpectedly intersects anchor lattice")
                    row["thin_scope"]="one visible receiving strip; actual traced-anchor coverage reported, no assumption that fallback is accurate GI"
                if name=="odd":
                    # A sensor-only extent change must not alter the next same-
                    # state exposure. Independent repeated capture is bit-exact.
                    sensor_camera=replace(current_camera,width=99,height=93,cx_px=49.5,cy_px=46.5)
                    sparse.capture(current_frame,CaptureRequest(sensor_camera,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
                    try:sparse.read_reconstructed_linear()
                    except RuntimeError:pass
                    else:raise ValueError("sensor-only capture exposed stale reconstructed RGB")
                    repeated=sparse.capture(current_frame,request)
                    repeated_planes={p.channel:p.data for p in repeated.planes}
                    if repeated_planes!=captures["sparse"]:raise ValueError("same seed/geometry did not reproduce after sensor-only resize")
                    row["repeat_after_sensor_resize_bytes_identical"]=True
            finally:
                original_failure=sys.exc_info()[0] is not None
                close_errors=[]
                for label,session in (("sparse",sparse),("full_rate",baseline)):
                    try:
                        session.close()
                    except BaseException as error:
                        close_errors.append(dict(stage="close",owner=label,type=type(error).__name__,message=str(error)))
                for label,session in (("sparse",sparse),("full_rate",baseline)):
                    try:
                        row["closed_"+label]=session.diagnostics()
                        value=row["closed_"+label]["device"]
                        if value.get("closed") is not True or any(value.get(k,1) for k in ("live_contexts","live_frames","live_resources","validation_errors")):
                            raise ValueError("native resource/validation close check failed")
                    except BaseException as error:
                        close_errors.append(dict(stage="closed diagnostics",owner=label,type=type(error).__name__,message=str(error)))
                row["close_errors"]=close_errors
                dump(target/"receipt.json",row)
                if close_errors and not original_failure:
                    raise ValueError("native close/validation checks failed; original diagnostics retained in case receipt")
            result["cases"].append(row);dump(out/"report.json",result)
        result["status"]="complete"
    except BaseException as error:
        result["status"]="partial_failure";result["error"]=dict(type=type(error).__name__,message=str(error))
        raise
    finally:dump(out/"report.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--baseline-library",type=Path,required=True)
    parser.add_argument("--baseline-shaders",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--validation",action="store_true")
    parser.add_argument("--panda",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
