"""Compare native Vulkan presets on identical measured Panda exposures.

Run with the rendering/src directory on PYTHONPATH. Writes into a new directory.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

from PIL import Image
import numpy as np
import panda_realtime_film as film
from _panda_scene import write_mesh
from panda_msaa_compare import TRACE_SHA256, _hash
from phymeo_rendering.assets import TextureMap, TexturedMaterial
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.backends.sapien import SapienStudio
from phymeo_rendering.environment import RadianceEnvironment
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath

SIZE = (1280, 720)
MEASURE = tuple(range(33, 65))
WARMUP = tuple(range(27, 33))
HERO = 60
GI = dict(progressive=True, indirect_mode="diffuse_bounce")
TWO = dict(GI, indirect_bounces=2)
CASES = [
    ("software", "软件 BVH 阴影", dict(shadow_mode="software")),
    ("common", "常用 · 硬件阴影", {}),
    ("aa4", "4× MSAA", dict(msaa_samples=4)),
    ("dense4", "面光源 4×4", dict(area_light_grid=4)),
    ("dense8", "面光源 8×8", dict(area_light_grid=8)),
    ("normal", "橡胶台面法线贴图", dict(normal=True)),
    ("mirror", "近场镜面反射", dict(reflection_mode="single_bounce")),
    ("gi1", "一次漫反射", GI),
    ("gi2", "二次漫反射", TWO),
    ("denoise", "二次漫反射＋引导降噪", dict(TWO, spatial_filter="guided_settling")),
    ("high", "高画质 · 综合光照", dict(TWO, spatial_filter="guided_settling",
         reflection_mode="single_bounce", area_light_grid=8, normal=True)),
]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bind_scene(directory):
    trace = film.ROOT / "tests/fixtures/panda_film/trace.json"
    if _hash(trace) != TRACE_SHA256:
        raise ValueError("Panda recording identity differs")
    scene, frames = film.load_episode(directory, trace, photographic=True,
                                      surface_roughness=True, local_light_scale=.125)
    # Give the grasped payload a real eligible specular receiver in every preset.
    materials = tuple(replace(m, base_color=(.75, .42, .18), metallic=1., roughness=.07)
                      if m.id == "oxide-red" else m for m in scene.materials)
    # Supply exactly the same metric UVs in both variants of the rubber tabletop.
    top = next(m for m in scene.meshes if m.id == "tabletop")
    mesh = load_mesh(top)
    uv = []
    for (x,y,z), (nx,ny,nz) in zip(mesh.positions, mesh.normals, strict=True):
        a,b = (x,y) if abs(nz)>.5 else ((x,z) if abs(ny)>.5 else (y,z))
        uv.append((.5+a/.35, .5-b/.35))
    mapped = write_mesh(directory, "rubber-with-uv", mesh.positions, mesh.normals, mesh.triangles, texcoords=uv)
    scene = replace(scene, key="performance-workshop", materials=materials,
                    meshes=tuple(replace(m, uri=mapped.uri) if m.id==top.id else m for m in scene.meshes))
    tex = film.ROOT / "results/normal-maps/textures/rubber-normal.png"
    mapped_materials = tuple(TexturedMaterial(m.id, m.base_color, m.metallic, m.roughness,
                            normal_map=TextureMap(tex.as_uri(), _hash(tex)))
                             if m.id=="rubber-mat" else m for m in materials)
    normal = replace(scene, key=scene.key+"-normal", materials=mapped_materials)
    environment = film.ROOT/"examples/materials/university_workshop_1k.hdr"
    env = RadianceEnvironment(environment.as_uri(), _hash(environment),
                              strength=.0625, yaw_degrees=25., face_size=256)
    studio = SapienStudio(center_m=(.18,0,.42), scale_m=.85, panel_strength=.125, environment_map=env)
    return (scene,normal), frames, studio, trace


def make_inputs(scene, frames):
    result = {}
    for i in WARMUP + MEASURE:
        f = replace(frames[i], key=replace(frames[i].key, scene_key=scene.key))
        c = film.exposure_camera(SimpleNamespace(camera="detail"), f, i, len(frames), *SIZE)
        result[i] = (f, CaptureRequest(c, RenderPath.RASTER, tuple(Channel)))
    return result


def timed(session, item):
    f, req = item
    start = perf_counter()
    shot = session.capture(f, req)
    ms = (perf_counter() - start)*1000
    if shot.frame_key != f.key or shot.time_s != f.time_s or shot.camera != req.camera:
        raise RuntimeError("stale exposure")
    return shot, ms


def sensor_digest(shot):
    planes = {p.channel:p.data for p in shot.planes}
    return {c.value:sha256(planes[c]).hexdigest() for c in Channel if c!=Channel.RGB}


def run_case(mode, context, *, still=False):
    name, label, options = mode
    out, dll, device, scenes, all_inputs, studio = context
    normal = options.get("normal",False)
    settings = dict(shadow_mode="ray_query", area_light_grid=2, msaa_samples=1,
                    reflection_mode="environment", indirect_mode="none", indirect_bounces=1,
                    progressive=False, spatial_filter="none")
    settings.update((k,v) for k,v in options.items() if k!="normal")
    camera_inputs = all_inputs[int(normal)]
    b = NativeBackend(library_path=dll, shader_directory=dll.parent, device_index=device,
                      resident=True, studio=studio, exposure=8., display_transform="pbr-neutral",
                      validation=False, **settings)
    opening = perf_counter()
    session = b.open_scene(scenes[int(normal)])
    record = dict(open_ms=1000*(perf_counter()-opening),capture_ms=[])
    try:
        for i in WARMUP:
            timed(session, camera_inputs[i])
        initial=session.diagnostics()["device"]["resident_profile"].copy()
        frame_ids = (HERO,)*64 if still else MEASURE
        rgb = None
        sensors = []
        for j,i in enumerate(frame_ids):
            shot, ms = timed(session, camera_inputs[i])
            record["capture_ms"].append(ms)
            if not still and i == HERO or still and j == len(frame_ids)-1:
                planes = {p.channel:p.data for p in shot.planes}
                rgb = planes[Channel.RGB]
                ids=np.frombuffer(planes[Channel.INSTANCE_ID],"<u4")
                record.update(hero_sensor_sha256=sensor_digest(shot), hero_pixels={str(k):int((ids==k).sum()) for k in (1,2)})
                if min(record["hero_pixels"].values())<16:
                    raise RuntimeError("specular cube or normal-mapped tabletop not visible")
            if not still:
                sensors.append(sensor_digest(shot))
        diag=session.diagnostics()
        profile=diag["device"]["resident_profile"]
        def delta(k):
            return profile[k]-initial[k]
        n=len(frame_ids)
        if delta("captures")!=n or delta("timestamp_samples")!=n:
            raise RuntimeError("device timestamp/capture count mismatch")
        for key in ("area_light_grid","shadow_mode"):
            if diag[key]!=settings[key]:raise RuntimeError("backend did not select requested "+key)
        if (diag["msaa"]["actual_color_samples"]!=settings["msaa_samples"] or
            diag["reflection"]["mode"]!=settings["reflection_mode"] or
            diag["indirect"]["mode"]!=settings["indirect_mode"] or bool(diag["normal_maps"])!=normal):
            raise RuntimeError("backend feature selection differs")
        if settings["progressive"]:
            hist=diag["progressive"]
            if hist["batches"]!=(64 if still else 1) or diag["indirect"]["bounces"]!=settings["indirect_bounces"]:
                raise RuntimeError("incorrect accumulation or bounce count")
        if settings["spatial_filter"]!="none" and diag["spatial_filter"]["mode"]!=settings["spatial_filter"]:
            raise RuntimeError("requested filter not active")
        record.update(device=diag["device"]["device_name"], uuid=diag["device"]["device_uuid"],
                      triangles=diag["triangles"], geometries=diag["decoded_geometry_sha256"],
                      rgb_sha256=sha256(rgb).hexdigest(), sensors_sha256=sha256(json.dumps(sensors,sort_keys=True).encode()).hexdigest(),
                      gpu_mean_ms={k:delta(k)/n/1e6 for k in ("gpu_render_ns","gpu_copy_ns","gpu_total_ns")},
                      vulkan_timestamps=n, backend_passes=diag["last_passes"], shaders=diag["shader_sha256"])
    finally:
        session.close()
    closed=session.diagnostics()["device"]
    if not closed["closed"] or any(closed[key] for key in ("live_contexts","live_frames","live_resources","validation_errors")):
        raise RuntimeError("unclean native session")
    return record, rgb


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device-index",type=int,default=1)
    args=ap.parse_args()
    dll=args.library.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    png=out/"images"
    png.mkdir()
    cases=[dict(id=name,label=label,overrides=kw,rounds=[],image=f"images/{name}.png") for name,label,kw in CASES]
    with TemporaryDirectory(prefix="phymeo-performance-") as folder:
        scenes,frames,studio,trace=bind_scene(Path(folder)/"geometry")
        inputs=[make_inputs(s,frames) for s in scenes]
        binding=film.input_binding(scenes[0], [inputs[0][i] for i in WARMUP+MEASURE], trace)
        result=dict(width=SIZE[0],height=SIZE[1],source_sha256=_hash(Path(__file__)),library_sha256=_hash(dll),
                    warmup=WARMUP,pose_indices=MEASURE,hero_pose=HERO,channels=[c.value for c in Channel],
                    timing="synchronous capture to owned four-channel CPU planes; no open/upload, warmup, hashes, PNG or physics; GPU times are inclusive timestamps over measured captures",
                    source_trace_sha256=binding["source_excerpt_sha256"],frames=binding["frames"],
                    texture_sha256=scenes[1].materials[[m.id for m in scenes[1].materials].index("rubber-mat")].normal_map.sha256,
                    ordering="one forward and one reverse round; each configuration starts a fresh session",
                    cases=cases)
        context=(out,dll,args.device_index,scenes,inputs,studio)
        geom=None
        for round_id,indices in enumerate((range(len(CASES)),reversed(range(len(CASES))))):
            for index in indices:
                mode=CASES[index]; row=cases[index]
                r,rgb=run_case(mode,context)
                geom = geom or r["sensors_sha256"]
                if r["sensors_sha256"]!=geom:raise RuntimeError("motion depth, instance ID or VALID changed")
                if row["rounds"] and r["rgb_sha256"]!=row["rounds"][0]["rgb_sha256"]:
                    raise RuntimeError("same captured hero differs between rounds")
                if not row["rounds"]:Image.frombytes("RGB",SIZE,rgb).save(out/row["image"])
                row["rounds"].append(r)
                result["device"]=r["device"]
                print(f"round {round_id+1} | {row['id']} | {sum(r['capture_ms'])/len(MEASURE):.2f} ms",flush=True)
                save(out/"benchmark.json",result)
        record,rgb=run_case(CASES[-1],context,still=True)
        if record["hero_sensor_sha256"]!=cases[-1]["rounds"][0]["hero_sensor_sha256"]:
            raise RuntimeError("still sensor mismatch")
        Image.frombytes("RGB",SIZE,rgb).save(png/"still.png")
        result["still"]=dict(record,image="images/still.png",batches=64,spp=1024,
                             meaning="identical fixed pose, high preset, total of 64 public captures per finished image")
        result["total_timed_captures"]=sum(len(r["capture_ms"]) for c in cases for r in c["rounds"])+64
        save(out/"benchmark.json",result)


if __name__=="__main__":
    main()
