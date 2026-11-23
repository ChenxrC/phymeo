"""Actual public-session probe for the isolated preview quad shader overlay.

The default 3-pose comparison is a diagnostic, not a 300-frame/30 FPS claim.
Legacy backend counters remain16; actual experimental proposal budget is bound
to the overlay's named SPIR-V constants and hashes, never inferred from counters.
"""
import argparse
import ctypes as ct
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_overlay import TARGET, digest, require, save, verify_overlay


def subgroup_capabilities(device_index):
    """Query core Vulkan properties on the SAME enumeration index as NativeBackend."""
    class Application(ct.Structure):
        _fields_ = [("sType",ct.c_uint32),("pNext",ct.c_void_p),("pApplicationName",ct.c_char_p),
                    ("applicationVersion",ct.c_uint32),("pEngineName",ct.c_char_p),
                    ("engineVersion",ct.c_uint32),("apiVersion",ct.c_uint32)]
    class Create(ct.Structure):
        _fields_ = [("sType",ct.c_uint32),("pNext",ct.c_void_p),("flags",ct.c_uint32),
                    ("pApplicationInfo",ct.POINTER(Application)),("enabledLayerCount",ct.c_uint32),
                    ("ppEnabledLayerNames",ct.c_void_p),("enabledExtensionCount",ct.c_uint32),
                    ("ppEnabledExtensionNames",ct.c_void_p)]
    class Subgroup(ct.Structure):
        _fields_ = [("sType",ct.c_uint32),("pNext",ct.c_void_p),("subgroupSize",ct.c_uint32),
                    ("supportedStages",ct.c_uint32),("supportedOperations",ct.c_uint32),
                    ("quadOperationsInAllStages",ct.c_uint32)]
    class Identity(ct.Structure):
        _fields_ = [("sType",ct.c_uint32),("pNext",ct.c_void_p),("deviceUUID",ct.c_ubyte*16),
                    ("driverUUID",ct.c_ubyte*16),("deviceLUID",ct.c_ubyte*8),
                    ("deviceNodeMask",ct.c_uint32),("deviceLUIDValid",ct.c_uint32)]
    class Properties(ct.Structure):
        # Oversized output tail avoids reproducing the unrelated full limits struct.
        # sType/pNext and the properties start retain the official pointer alignment.
        _fields_ = [("sType",ct.c_uint32),("pNext",ct.c_void_p),("properties",ct.c_ubyte*4096)]
    loader = ct.WinDLL("vulkan-1.dll") if os.name == "nt" else ct.CDLL("libvulkan.so.1")
    loader.vkCreateInstance.argtypes = [ct.POINTER(Create),ct.c_void_p,ct.POINTER(ct.c_void_p)]
    loader.vkCreateInstance.restype = ct.c_int32
    loader.vkDestroyInstance.argtypes = [ct.c_void_p,ct.c_void_p]
    loader.vkDestroyInstance.restype = None
    loader.vkEnumeratePhysicalDevices.argtypes = [ct.c_void_p,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_void_p)]
    loader.vkEnumeratePhysicalDevices.restype = ct.c_int32
    loader.vkGetPhysicalDeviceProperties2.argtypes = [ct.c_void_p,ct.POINTER(Properties)]
    loader.vkGetPhysicalDeviceProperties2.restype = None
    application = Application(sType=0, pApplicationName=b"phymeo-preview-quad-probe", apiVersion=(1<<22)|(1<<12))
    create = Create(sType=1, pApplicationInfo=ct.pointer(application))
    instance = ct.c_void_p()
    require(loader.vkCreateInstance(ct.byref(create),None,ct.byref(instance)) == 0, "Vulkan1.1 instance unavailable")
    try:
        count=ct.c_uint32()
        require(loader.vkEnumeratePhysicalDevices(instance,ct.byref(count),None)==0, "cannot enumerate physical devices")
        require(0<=device_index<count.value, "physical device index is unavailable")
        devices=(ct.c_void_p*count.value)()
        require(loader.vkEnumeratePhysicalDevices(instance,ct.byref(count),devices)==0, "device enumeration changed")
        subgroup=Subgroup(sType=1000094000)
        identity=Identity(sType=1000071004,pNext=ct.cast(ct.pointer(subgroup),ct.c_void_p))
        properties=Properties(sType=1000059001,pNext=ct.cast(ct.pointer(identity),ct.c_void_p))
        loader.vkGetPhysicalDeviceProperties2(devices[device_index],ct.byref(properties))
        name=bytes(properties.properties)[20:276].split(b"\0",1)[0].decode("utf-8")
        report=dict(device_index=device_index,device_name=name,device_uuid=bytes(identity.deviceUUID).hex(),
                    subgroup_size=subgroup.subgroupSize,supported_stages=subgroup.supportedStages,
                    supported_operations=subgroup.supportedOperations,
                    quad_operations_in_all_stages=bool(subgroup.quadOperationsInAllStages),
                    required_operations=dict(basic=1,ballot=8,quad=128),required_stage_fragment=16)
        require(report["subgroup_size"]>=4 and report["subgroup_size"]%4==0 and
                report["supported_stages"]&16 and report["supported_operations"]&137==137,
                "selected device lacks fragment BASIC+BALLOT+QUAD")
        return report
    finally:
        loader.vkDestroyInstance(instance,None)


def run_owner(args, scene, studio, inputs, shaders, name, *, samples, seed, batches, expected_frag, device_uuid):
    import numpy as np
    from PIL import Image
    import panda_motion_preview as preview
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.outputs import Channel
    folder=args.output/name
    folder.mkdir()
    session=None
    record=dict(name=name,proposal_budget=samples,seed=seed,batches_per_pose=batches,
                records=[],status="opening",
                legacy_counter_scope="public backend metadata remains16; use proposal_budget for this experimental shader")
    images={}
    problem=None
    try:
        config=preview.preset("preview720")
        config["settings"].update(indirect_seed=seed,spatial_filter=args.filter)
        opened=perf_counter()
        session=NativeBackend(library_path=args.library,shader_directory=shaders,
                              device_index=args.device_index,studio=studio,exposure=8.,
                              display_transform="pbr-neutral",validation=args.validation,
                              **config["settings"]).open_scene(scene)
        record["open_ms"]=(perf_counter()-opened)*1000
        for i in (57,58,59):
            f,req=inputs[i]
            preview.verify_capture(session.capture(f,req),f,req)
        baseline=session.diagnostics()["device"]["resident_profile"].copy()
        for pose in args.poses:
            f,req=inputs[pose]
            session.reset_accumulation()
            times=[]
            for n in range(batches):
                started=perf_counter()
                shot=session.capture(f,req)
                times.append((perf_counter()-started)*1000)
                preview.verify_capture(shot,f,req)
            diag=session.diagnostics()
            require(diag["device"]["device_uuid"]==device_uuid, "capability probe and actual native device differ")
            require(diag["shader_sha256"]["frag"]==expected_frag, "actual fragment shader differs from bound overlay")
            require(diag["indirect"]["bounces"]==1 and diag["progressive"]["batches"]==batches,
                    "prototype requires a single diffuse bounce and exact batches")
            require(diag["spatial_filter"]["mode"]==args.filter, "display filter differs")
            planes={p.channel:p.data for p in shot.planes}
            rgb=np.frombuffer(planes[Channel.RGB],np.uint8).reshape(args.height,args.width,3).copy()
            png=folder/f"pose-{pose:04d}.png"
            Image.fromarray(rgb).save(png)
            started=perf_counter()
            packet=session.read_linear_mean()
            read_ms=(perf_counter()-started)*1000
            require(packet["format"]=="RGBA32F_LE" and packet["space"]=="scene-linear-before-exposure" and
                    packet["width"]==args.width and packet["height"]==args.height and packet["batches"]==batches,
                    "raw linear mean metadata mismatch")
            raw=np.frombuffer(packet["data"],"<f4").reshape(args.height,args.width,4)
            require(np.isfinite(raw).all(),"nonfinite raw mean")
            if args.save_linear:
                (folder/f"pose-{pose:04d}.rgba32f").write_bytes(packet["data"])
            row=dict(pose_index=pose,frame_key=asdict(f.key),time_s=f.time_s,
                     state_sha256=sha256(preview.canonical(asdict(f))).hexdigest(),
                     camera_sha256=sha256(preview.canonical(asdict(req.camera))).hexdigest(),
                     capture_ms=times,completed_mean_ms=sum(times)/len(times),
                     png=png.name,png_sha256=digest(png),rgb_sha256=sha256(planes[Channel.RGB]).hexdigest(),
                     sensor_sha256={c.value:sha256(v).hexdigest() for c,v in planes.items() if c!=Channel.RGB},
                     actual_primary_proposals_per_eligible_pixel_per_batch=samples,
                     actual_accumulated_primary_proposal_budget=samples*batches,
                     legacy_reported_primary_samples=diag["progressive"]["primary_samples_per_pixel"],
                     raw_mean_sha256=sha256(packet["data"]).hexdigest(),raw_read_ms=read_ms,
                     raw_mean_rgb=[float(x) for x in raw[:,:,:3].mean(axis=(0,1),dtype=np.float64)],
                     raw_mean_max=[float(x) for x in raw[:,:,:3].max(axis=(0,1))],
                     backend=diag)
            record["records"].append(row)
            images[pose]=rgb
            save(folder/"receipt.json",record)
        after=session.diagnostics()["device"]["resident_profile"]
        record["profile"]=preview.profile_delta(baseline,after,len(args.poses)*batches)
        record["status"]="complete"
    except BaseException as error:
        record.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        problem=error
    finally:
        if session is not None:
            try:
                session.close()
                record["closed"]=session.diagnostics()["device"]
                preview.verify_close(record["closed"])
            except BaseException as error:
                record["close_error"]=str(error)
                if problem is None:
                    problem=error
                    record["status"]="failed"
        save(folder/"receipt.json",record)
    if problem is not None:
        raise problem
    return record,images


def main():
    import numpy as np
    import panda_render_performance as bench
    import panda_motion_preview as preview
    from phymeo_rendering.outputs import Channel
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library",type=Path,required=True)
    ap.add_argument("--base-shaders",type=Path,required=True)
    ap.add_argument("--overlay",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device-index",type=int,default=1)
    ap.add_argument("--poses",nargs="+",type=int,default=[59,60,149])
    ap.add_argument("--filter",choices=("none","guided_settling"),default="guided_settling")
    ap.add_argument("--reference-batches",type=int,choices=(0,64),default=0,
                    help="64 adds real independent seed1 1024-proposal matched reference at every pose")
    ap.add_argument("--resolution",choices=("720p","360p"),default="720p")
    ap.add_argument("--save-linear",action="store_true")
    ap.add_argument("--validation",action="store_true",default=True)
    ap.add_argument("--no-validation",action="store_false",dest="validation")
    args=ap.parse_args()
    require(args.poses and len(set(args.poses))==len(args.poses) and all(0<=i<300 for i in args.poses),
            "poses must be unique sealed episode indices")
    args.output=args.output.resolve()
    args.library=args.library.resolve(strict=True)
    args.base_shaders=args.base_shaders.resolve(strict=True)
    args.overlay=args.overlay.resolve(strict=True)
    args.width,args.height=(1280,720) if args.resolution=="720p" else (640,360)
    manifest=verify_overlay(args.overlay)
    require(args.validation or manifest["standalone_spirv_validation"]["status"]=="passed",
            "without standalone spirv-val, first actual pipeline validation layers are mandatory")
    require({p.name:digest(p) for p in args.base_shaders.glob("*.spv")}==manifest["base_shaders"],
            "baseline shader directory differs from the overlay's complete bound base set")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema="phymeo-preview-quad-validation-v1",status="preparing",
                overlay_manifest_sha256=digest(args.overlay/"overlay.json"),overlay=manifest,
                library=dict(file=str(args.library),sha256=digest(args.library)),
                poses=args.poses,extent=[args.width,args.height],filter=args.filter,
                measurement_scope="same-state public four-channel captures; raw read/hash/PNG/profile outside capture interval",
                quality_claim=False,performance_claim=False,runs=[])
    save(args.output/"summary.json",report)
    try:
        report["subgroup"]=subgroup_capabilities(args.device_index)
        with TemporaryDirectory(prefix="phymeo-preview-quad-") as temp:
            scenes,frames,studio,trace=bench.bind_scene(Path(temp)/"geometry")
            declaration=preview.preset("preview720" if args.resolution=="720p" else "fast360")
            inputs=preview.make_inputs(scenes[1],frames,declaration,"detail")
            inputs={i:(f,replace(req,channels=tuple(Channel))) for i,(f,req) in inputs.items()}
            save(args.output/"inputs.json",dict(scene=asdict(scenes[1]),studio=asdict(studio),
                trace_sha256=digest(trace),poses=[dict(index=i,frame=asdict(inputs[i][0]),
                camera=asdict(inputs[i][1].camera)) for i in args.poses]))
            report["inputs_sha256"]=digest(args.output/"inputs.json")
            old,old_images=run_owner(args,scenes[1],studio,inputs,args.base_shaders,"baseline16",
                samples=16,seed=0,batches=1,expected_frag=manifest["base_shaders"][TARGET],
                device_uuid=report["subgroup"]["device_uuid"])
            report["runs"].append(old)
            save(args.output/"summary.json",report)
            new,new_images=run_owner(args,scenes[1],studio,inputs,args.overlay,"quad",
                samples=manifest["samples_per_eligible_pixel"],seed=0,batches=1,
                expected_frag=manifest["shaders"][TARGET],device_uuid=report["subgroup"]["device_uuid"])
            report["runs"].append(new)
            for a,b in zip(old["records"],new["records"],strict=True):
                require(all(a[k]==b[k] for k in ("pose_index","frame_key","time_s","state_sha256",
                                                "camera_sha256","sensor_sha256")),
                        "paired actual source/camera/full-frame sensor bytes differ")
            report["full_frame_sensors_exact"]=True
            def linear(image):
                v=image.astype(np.float64)/255.
                return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
            report["paired_difference"]={str(i):dict(
                display_linear_rmse=float(np.sqrt(np.mean((linear(new_images[i])-linear(old_images[i]))**2))),
                max_rgb8_difference=int(np.abs(new_images[i].astype(int)-old_images[i].astype(int)).max()),
                scope="difference from noisy16-proposal matched input; NOT an accuracy oracle")
                for i in args.poses}
            if args.reference_batches:
                ref,reference=run_owner(args,scenes[1],studio,inputs,args.base_shaders,"reference",
                    samples=16,seed=1,batches=args.reference_batches,
                    expected_frag=manifest["base_shaders"][TARGET],device_uuid=report["subgroup"]["device_uuid"])
                report["runs"].append(ref)
                for a,b in zip(old["records"],ref["records"],strict=True):
                    require(a["sensor_sha256"]==b["sensor_sha256"],"reference geometry differs")
                report["reference_errors"]={str(i):dict(
                    baseline_display_linear_rmse=float(np.sqrt(np.mean((linear(old_images[i])-linear(reference[i]))**2))),
                    quad_display_linear_rmse=float(np.sqrt(np.mean((linear(new_images[i])-linear(reference[i]))**2))),
                    reference_seed=1,reference_primary_proposals=1024,
                    scope="finite noisy reference; full image includes background; no ROI quality gate")
                    for i in args.poses}
            report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        save(args.output/"summary.json",report)
    print(json.dumps(dict(status=report["status"],output=str(args.output),
                          sensors_exact=report.get("full_frame_sensors_exact")),ensure_ascii=False))


if __name__=="__main__":
    main()

