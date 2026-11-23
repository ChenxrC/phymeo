"""Fixed Panda gripper stills: red transfer, equal-linear-Y grey, and occlusion.

Manufacturer hand/finger triangle geometry and metre units are preserved. The
uniform white paint and static bench placement are authored optical controls,
not an archived robot motion or a physical grasp. Each public capture owns all
four channels. Only published sRGB8 is available: saved display-linear values
are its inverse sRGB, NOT unclipped radiance or an HDR attachment.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np

from _panda_scene import box, look_at, pose_matrix
from normal_map_film import _json, compare_sensors, verify_close, SENSORS
from panda_msaa_compare import _hash, _implementation_files, _timing
from phymeo_rendering.assets import Material
from phymeo_rendering.cameras import Camera
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.panda_visuals import panda_visual_assets
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

ROOT = Path(__file__).resolve().parent
STATES = ("red", "grey", "red-blocked", "grey-blocked")
MODES = ("none", "diffuse_bounce")
RED = (.8, .01, .01)
CROP = (384, 288, 896, 672)
EYE = np.array((.42, 0., .24))
FORWARD = np.array((-.42, 0., -.15))/math.hypot(.42, .15)
RIGHT = np.array((0., 1., 0.))
UP = np.cross(RIGHT, FORWARD)
RECEIVERS = (156, 158)


def frozen_profile():
    return dict(states=list(STATES), modes=list(MODES), size=[1280, 960], seed=0,
        cosine_samples=16, bounces=1, warmup=3, repeats=9, msaa_samples=1,
        channels=[ch.value for ch in Channel], crop_xyxy=list(CROP), receiver_ids=list(RECEIVERS),
        camera_eye_m=EYE.tolist(), camera_target_m=[0., 0., .09], focal_px=1550.,
        hand_origin_m=[0., 0., .18], hand_quaternion_xyzw=[1., 0., 0., 0.],
        finger_displacement_m=.035, urdf_finger_origin_z_m=.0584,
        board_size_m=[.22, .26, .006], board_centre_m=[0., 0., .037],
        blocker_size_m=[.11, .20, .002], blocker_z_m=.055, parked_blocker_z_m=-.2,
        white_reflectance=[.7, .7, .7], red_reflectance=list(RED), grey_reflectance=.177954,
        luminance_coefficients=[.2126, .7152, .0722], environment=None,
        directional_travel=[1/math.sqrt(2), 0., -1/math.sqrt(2)], illuminance=[3., 3., 3.],
        display_transform="pbr-neutral", exposure=1., shadow_mode="ray_query", validation=False,
        roi_rule="same pre-GPU nearest original finger_01 triangles; local z [0.012,0.032]m, world normal.x>=0.5, barycentric min>=0.02",
        minimum_roi_pixels_per_finger=128, depth_error_max_m=.0001,
        linear_scope="inverse IEC sRGB of published RGB8: quantized display-linear, not raw HDR or radiance",
        transfer_expression="T=((on-off)_red-(on-off)_grey) dot (1,-0.5,-0.5)",
        transfer_linear_min=.005, transfer_rgb8_min=3., blocked_max_fraction=.5, blocked_linear_allowance=.002,
        pixel_policy="all registered pixels retained, including zero energy and 16-sample noise; no post-image pruning",
        occlusion_scope="physical black screen can both intercept transfer paths and shade the board; not an isolated secondary-visibility-only test",
        timing="public synchronous RGB+forward-depth+ID+VALID owned bytes; three warmups excluded; nine static repeats",
        limitation="one diffuse bounce, 16 stable cosine samples, no denoising or temporal accumulation; no physical motion")


def equal_luminance_grey(rgb):
    values = np.asarray(rgb, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("linear reflectance must be three finite values in [0,1]")
    return float(values @ np.array((.2126, .7152, .0722)))


def build_cases(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meshes, _, instances = panda_visual_assets()
    instances = tuple(replace(i, material_id="study-white") for i in instances
                      if i.node_id in ("panda_hand", "panda_leftfinger", "panda_rightfinger"))
    used = {i.mesh_id for i in instances}
    meshes = tuple(m for m in meshes if m.id in used)
    board = box(directory, "study-board-mesh", (.22, .26, .006))
    blocker = box(directory, "study-blocker-mesh", (.11, .20, .002))
    instances += (Instance(201, board.id, "study-board", "board"),
                  Instance(202, blocker.id, "study-black", "blocker"))
    light = DirectionalLight((1/math.sqrt(2), 0., -1/math.sqrt(2)), (3., 3., 3.))
    base_nodes = (NodePose("panda_hand", Pose((0., 0., .18), (1., 0., 0., 0.))),
        NodePose("panda_leftfinger", Pose((0., -.035, .1216), (1., 0., 0., 0.))),
        NodePose("panda_rightfinger", Pose((0., .035, .1216), (1., 0., 0., 0.))))
    result = {}
    for state in STATES:
        color = RED if state.startswith("red") else (equal_luminance_grey(RED),)*3
        scene = VisualScene(f"panda-diffuse-study-{state.split('-')[0]}-v1", meshes+(board, blocker),
            (Material("study-white", (.7, .7, .7), 0., 1.), Material("study-board", color, 0., 1.),
             Material("study-black", (0., 0., 0.), 0., 1.)), instances, (light,))
        nodes = base_nodes+(NodePose("board", Pose((0., 0., .037))),
                           NodePose("blocker", Pose((0., 0., .055 if state.endswith("blocked") else -.2))))
        frame = SceneFrame(FrameKey(scene.key, state, 0, 0, 0, "authored"), 0., nodes)
        camera = Camera("fixed-gripper-camera", frame.key, look_at(tuple(EYE), (0., 0., .09), (0, 0, 1)),
                        1280, 960, 1550., 1550., 640., 480., .01, 3.)
        result[state] = (scene, frame, camera)
    return result


def project(point):
    """Independent registered camera basis; no renderer projection helper."""
    relative = np.asarray(point, float)-EYE
    depth = relative @ FORWARD
    if np.any(depth <= 0) or not np.isfinite(relative).all():
        raise ValueError("reference point must be finite and in front of camera")
    return np.stack((640+1550*(relative@RIGHT)/depth, 480-1550*(relative@UP)/depth, depth), axis=-1)


def camera_ray(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    d = FORWARD+(x[..., None]-640)/1550*RIGHT-(y[..., None]-480)/1550*UP
    return d/np.linalg.norm(d, axis=-1, keepdims=True)


def intersect_triangles(origins, directions, triangles):
    """FP64, two-sided Moller-Trumbore nearest positive hit; chunked ray storage."""
    origins, directions, triangles = (np.asarray(x, dtype=float) for x in (origins, directions, triangles))
    if (origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape
        or triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
        or not all(np.isfinite(x).all() for x in (origins, directions, triangles))):
        raise ValueError("finite ray arrays and triangle triples required")
    n = len(origins)
    result = dict(triangle=np.full(n, -1, int), distance=np.full(n, np.inf), barycentric=np.zeros((n, 3)))
    if not len(triangles):
        return result
    v0 = triangles[:, 0]; e1 = triangles[:, 1]-v0; e2 = triangles[:, 2]-v0
    for begin in range(0, n, 64):
        sl = slice(begin, min(begin+64, n))
        p = np.cross(directions[sl, None, :], e2)
        determinant = np.sum(e1*p, axis=-1)
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=np.abs(determinant)>1e-14)
        relative = origins[sl, None, :]-v0
        u = np.sum(relative*p, axis=-1)*inverse
        q = np.cross(relative, e1)
        v = np.sum(directions[sl, None, :]*q, axis=-1)*inverse
        t = np.sum(e2*q, axis=-1)*inverse
        valid = (np.abs(determinant)>1e-14) & (u>=0) & (v>=0) & (u+v<=1) & (t>0)
        distances = np.where(valid, t, np.inf)
        index = np.argmin(distances, axis=1); rows = np.arange(len(index))
        nearest = distances[rows, index]; hit = np.isfinite(nearest)
        result["triangle"][sl] = np.where(hit, index, -1)
        result["distance"][sl] = nearest
        result["barycentric"][sl] = np.stack((1-u[rows,index]-v[rows,index], u[rows,index], v[rows,index]), axis=-1)
    return result


def _world_geometry(scene, frame):
    nodes = {n.node_id: n.world_pose for n in frame.nodes}
    meshes = {m.id: load_mesh(m) for m in scene.meshes}
    arrays = {k: [] for k in ("triangles", "local", "normals", "ids")}
    transforms = {}
    for instance in scene.instances:
        matrix = pose_matrix(nodes[instance.node_id]) @ pose_matrix(instance.local_pose)
        transforms[instance.id] = matrix
        data = meshes[instance.mesh_id]; indices = np.asarray(data.triangles, int)
        local = np.asarray(data.positions)[indices]
        arrays["local"].append(local)
        arrays["triangles"].append(local @ matrix[:3,:3].T+matrix[:3,3])
        arrays["normals"].append(np.asarray(data.normals)[indices] @ matrix[:3,:3].T)
        arrays["ids"].append(np.full(len(indices), instance.id, np.uint32))
    return {k:np.concatenate(v) for k,v in arrays.items()}, transforms


def register_reference(scene, frame, camera):
    expected_pose = look_at(tuple(EYE), (0., 0., .09), (0, 0, 1))
    if (camera.world_pose != expected_pose or camera.frame_key != frame.key or frame.time_s != 0.
        or (camera.width, camera.height, camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px) !=
           (1280, 960, 1550., 1550., 640., 480.)):
        raise ValueError("reference requires the frozen camera and same current frame")
    geometry, transforms = _world_geometry(scene, frame)
    output = dict(ids=np.zeros((960,1280),np.uint32), depth=np.full((960,1280),np.inf),
                  local_points=np.zeros((960,1280,3)), normal_x=np.zeros((960,1280)))
    projected = project(geometry["triangles"])
    boxes = (projected[:,:,:2].min(axis=1), projected[:,:,:2].max(axis=1))
    for identity in RECEIVERS:
        matrix = transforms[identity]
        corners = np.array(list(itertools.product((-.012,.012), (0.,.028), (.012,.032))))
        screen = project(corners @ matrix[:3,:3].T+matrix[:3,3])
        lo = np.floor(screen[:,:2].min(axis=0)).astype(int)
        hi = np.ceil(screen[:,:2].max(axis=0)).astype(int)
        if np.any(lo < (0,0)) or np.any(hi >= (1280,960)):
            raise ValueError("registered local finger box outside image")
        xx, yy = np.meshgrid(np.arange(lo[0],hi[0])+ .5,np.arange(lo[1],hi[1])+ .5)
        directions = camera_ray(xx.ravel(), yy.ravel())
        candidates = np.flatnonzero(np.all(boxes[1]>=lo,axis=1) & np.all(boxes[0]<=hi,axis=1))
        hits = intersect_triangles(np.broadcast_to(EYE,directions.shape), directions, geometry["triangles"][candidates])
        hit = hits["triangle"] >= 0
        indices = candidates[np.maximum(hits["triangle"],0)]
        bary = hits["barycentric"]
        local = np.sum(geometry["local"][indices]*bary[:,:,None],axis=1)
        normals = np.sum(geometry["normals"][indices]*bary[:,:,None],axis=1)
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-30)
        selected = (hit & (geometry["ids"][indices]==identity) & (local[:,2]>=.012) &
                    (local[:,2]<=.032) & (normals[:,0]>=.5) & (bary.min(axis=1)>=.02))
        if int(selected.sum()) < 128:
            raise ValueError(f"registered original finger {identity} has insufficient support: {selected.sum()}")
        rows, cols = yy.ravel().astype(int)[selected], xx.ravel().astype(int)[selected]
        output["ids"][rows,cols] = identity
        output["depth"][rows,cols] = hits["distance"][selected]*(directions[selected]@FORWARD)
        output["local_points"][rows,cols] = local[selected]
        output["normal_x"][rows,cols] = normals[selected,0]
        if np.any((cols<CROP[0]) | (cols>=CROP[2]) | (rows<CROP[1]) | (rows>=CROP[3])):
            raise ValueError("registered receiver pixels outside the frozen 1:1 crop")
    return output


def display_linear(rgb):
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.shape[-1] != 3:
        raise ValueError("published RGB8 triples required; no implied raw HDR")
    s = value.astype(np.float64)/255.
    return np.where(s<=.04045, s/12.92, ((s+.055)/1.055)**2.4)


def verify_roi(reference, ids, depth, valid):
    mask = reference["ids"] != 0
    if (ids.shape != mask.shape or depth.shape != mask.shape or valid.shape != mask.shape
        or not mask.any() or np.any(ids[mask] != reference["ids"][mask])
        or np.any(valid[mask] != 1) or not np.isfinite(depth[mask]).all()):
        raise ValueError("registered foreground ID/valid/depth binding differs; no ROI pruning allowed")
    error = float(np.max(np.abs(depth[mask]-reference["depth"][mask])))
    if error > .0001:
        raise ValueError("registered foreground primary depth exceeds 0.1 mm")
    return dict(pixels=int(mask.sum()), max_depth_error_m=error)


def compare_appearance(images, registered_ids):
    if tuple(images) != STATES:
        raise ValueError("all four registered states required in order")
    result = {}
    vector = np.array((1., -.5, -.5))
    for identity in RECEIVERS:
        mask = registered_ids == identity
        if int(mask.sum()) < 128:
            raise ValueError("too few registered pixels; no intensity filtering allowed")
        values = {}
        for suffix in ("", "-blocked"):
            red, grey = images["red"+suffix], images["grey"+suffix]
            if not np.array_equal(red["none"][mask], grey["none"][mask]):
                raise ValueError("direct baseline changed with board colour on identical receiver pixels")
            differences = []
            encoded = []
            for pair in (red,grey):
                differences.append(display_linear(pair["diffuse_bounce"][mask])-display_linear(pair["none"][mask]))
                encoded.append(pair["diffuse_bounce"][mask].astype(float)-pair["none"][mask].astype(float))
            linear = (differences[0]-differences[1])@vector
            rgb8 = (encoded[0]-encoded[1])@vector
            values["blocked" if suffix else "unblocked"] = dict(linear_red_excess=float(linear.mean()),
                rgb8_red_excess=float(rgb8.mean()), all_pixels=int(mask.sum()),
                min_linear=float(linear.min()), max_linear=float(linear.max()),
                zero_off_pixels=int(np.sum(np.all(red["none"][mask]==0,axis=1))),
                fraction_positive=float(np.mean(linear>0)))
        a,b = values["unblocked"],values["blocked"]
        color = a["linear_red_excess"] >= .005 and a["rgb8_red_excess"] >= 3.
        blocked = b["linear_red_excess"] <= .5*a["linear_red_excess"]+.002
        result[str(identity)] = dict(unblocked_linear_red_excess=a["linear_red_excess"],
            blocked_linear_red_excess=b["linear_red_excess"], color_transfer_passed=bool(color),
            occlusion_passed=bool(blocked), states=values)
    return dict(passed=all(x["color_transfer_passed"] and x["occlusion_passed"] for x in result.values()),
                receivers=result, expression=frozen_profile()["transfer_expression"],
                scope=frozen_profile()["linear_scope"])


def verify_indirect(backend, mode):
    top = backend.get("indirect", {})
    data = backend.get("device", {}).get("resident_indirect_info")
    if top.get("mode") != mode or top.get("seed") != 0:
        raise ValueError("actual indirect mode/seed differs")
    if mode == "none":
        if data is not None or top.get("samples_per_pixel") != 0 or top.get("bounces") != 0:
            raise ValueError("none unexpectedly enables indirect rays")
    else:
        if top.get("samples_per_pixel") != 16 or top.get("bounces") != 1 or not isinstance(data,dict):
            raise ValueError("actual cosine sample/bounce evidence differs")
        for key,value in (("indirect_abi_version",1),("supported",1),("shader_compatible",1),
            ("enabled",1),("samples_per_pixel",16),("seed",0),("max_bounces",1),("set_calls",1),
            ("captures",12),("rgb_captures",12),("receiver_count",10)):
            if data.get(key) != value:
                raise ValueError(f"actual retained indirect evidence differs: {key}")
    return True


def _capture(root, state, mode, case, backend):
    from PIL import Image
    scene, frame, camera = case
    folder = root/state; folder.mkdir(exist_ok=True)
    stem = mode.replace("_", "-")
    result = dict(state=state,mode=mode,status="opening",capture_ms=[],repeated_hashes=[])
    session = None; first = None
    try:
        session = backend.open(scene)
        request = CaptureRequest(camera, tuple(Channel), RenderPath.RASTER)
        for index in range(12):
            start = perf_counter(); capture = session.capture(frame, request); duration = (perf_counter()-start)*1000
            if capture.frame_key != frame.key or capture.time_s != 0. or capture.camera != camera:
                raise ValueError("published exposure differs from registered input")
            planes = {plane.channel:plane.data for plane in capture.planes}
            if tuple(planes) != tuple(Channel):
                raise ValueError("four-channel publication missing or reordered")
            hashes = {ch.value:sha256(data).hexdigest() for ch,data in planes.items()}
            if first is None:
                first = planes
            elif any(planes[ch] != first[ch] for ch in Channel):
                raise ValueError("same static frame/seed does not repeat byte-for-byte")
            if index >= 3:
                result["capture_ms"].append(duration)
                result["repeated_hashes"].append(hashes)
        rgb = np.frombuffer(first[Channel.RGB],np.uint8).reshape(960,1280,3)
        Image.fromarray(rgb).save(folder/f"{stem}.png")
        np.savez_compressed(folder/f"{stem}.npz", display_linear=display_linear(rgb).astype("<f4"),
                            **{ch.name:np.frombuffer(first[ch],np.uint8) for ch in SENSORS})
        result.update(status="complete",timing=_timing(result["capture_ms"]),
            png=f"{state}/{stem}.png",png_sha256=_hash(folder/f"{stem}.png"),
            data=f"{state}/{stem}.npz",data_sha256=_hash(folder/f"{stem}.npz"),
            plane_sha256=hashes,backend_before_close=session.diagnostics())
    except BaseException as error:
        result.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        original = sys.exc_info()[0] is not None
        failure = None
        if session is not None:
            try:
                session.close(); result["backend"] = session.diagnostics()
                result["close"] = verify_close(result["backend"])
            except Exception as error:
                failure = error
                result["close_error"] = dict(type=type(error).__name__,message=str(error))
        _json(folder/f"{stem}-receipt.json",result)
        if failure is not None and not original:
            raise failure
    return result


def _load(root, receipt):
    from PIL import Image
    if _hash(root/receipt["png"]) != receipt["png_sha256"] or _hash(root/receipt["data"]) != receipt["data_sha256"]:
        raise ValueError("saved image or sensor archive hash differs")
    with Image.open(root/receipt["png"]) as image:
        rgb = np.array(image.convert("RGB"))
    with np.load(root/receipt["data"],allow_pickle=False) as data:
        sensors = {ch:data[ch.name].tobytes() for ch in SENSORS}
        if not np.array_equal(data["display_linear"],display_linear(rgb).astype("<f4")):
            raise ValueError("saved display-linear data differs from inverse sRGB")
    if sha256(rgb.tobytes()).hexdigest() != receipt["plane_sha256"][Channel.RGB.value]:
        raise ValueError("saved PNG no longer equals actual public RGB")
    if any(sha256(v).hexdigest() != receipt["plane_sha256"][ch.value] for ch,v in sensors.items()):
        raise ValueError("saved sensor bytes differ from actual public readback")
    return rgb,sensors


def _present(root, images, passed):
    from PIL import Image,ImageDraw,ImageFont
    font = ImageFont.load_default(size=22)
    overview = Image.new("RGB",(2562,4032),(22,24,29))
    for row,state in enumerate(STATES):
        pair = Image.new("RGB",(2562,1008),(22,24,29))
        crop_pair = Image.new("RGB",(1026,432),(22,24,29))
        for col,mode in enumerate(MODES):
            image = Image.fromarray(images[state][mode])
            label = f"{state.upper()} | {'DIRECT ONLY' if mode=='none' else 'ONE DIFFUSE BOUNCE / 16 SAMPLES'}"
            if not passed:
                label += " | OPTICAL GATE FAILED"
            pair.paste(image,(col*1282,48))
            ImageDraw.Draw(pair).text((col*1282+12,17),label,fill="white",font=font)
            crop = image.crop(CROP); crop.save(root/f"{state}-{mode.replace('_','-')}-roi.png")
            crop_pair.paste(crop,(col*514,48))
            crop_label = f"{state.upper()} | {'OFF' if col==0 else 'ON / 16 SAMPLES'}"
            if not passed:
                crop_label = "GATE FAILED | "+crop_label
            ImageDraw.Draw(crop_pair).text((col*514+8,17),crop_label,fill="white",font=font)
        pair.save(root/f"{state}-pair.png"); crop_pair.save(root/f"{state}-roi-pair.png")
        overview.paste(pair,(0,row*1008))
    overview.save(root/"overview.png")


def _panda_provenance(scene):
    mesh = next(m for m in scene.meshes if m.id.startswith("panda_visual_"))
    root = Path(url2pathname(urlsplit(mesh.uri).path)).parent.parent
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(manifest=dict(file=str(manifest_path),sha256=_hash(manifest_path)),
        selected_instances=[asdict(i) for i in scene.instances if 150<=i.id<=158],
        licenses=[dict(file=row["file"],sha256=_hash(root/row["file"]),
                       text=(root/row["file"]).read_text(encoding="utf-8")) for row in manifest["licenses"]],
        visual_placement_sources=manifest["visual_placement_sources"],
        change="Original metre mesh geometry/local placements; uniform authored neutral white coating replaces all hand/finger material factors; static hand placement is not an archived physical state")

def run(output, library, *, device_index=1, shader_directory=None):
    from phymeo_rendering.backends.native import NativeBackend
    output,library = Path(output).resolve(),Path(library).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(schema="phymeo-diffuse-bounce-study-v1",status="preparing",physics_steps=0)
    try:
        cases = build_cases(output/"geometry")
        reference = register_reference(*cases["red"])
        np.savez_compressed(output/"reference.npz",**reference)
        shaders = library.parent if shader_directory is None else Path(shader_directory).resolve()
        scene = cases["red"][0]
        profile = frozen_profile()
        protocol = dict(schema="phymeo-diffuse-bounce-study-protocol-v1",profile=profile,
            cases={name:dict(scene=asdict(s),frame=asdict(f),camera=asdict(c)) for name,(s,f,c) in cases.items()},
            reference_sha256=_hash(output/"reference.npz"),panda_provenance=_panda_provenance(scene),
            reference_pixels={str(i):int(np.sum(reference["ids"]==i)) for i in RECEIVERS},
            source="pinned original Panda hand/finger surface geometry; authored uniform white paint and static bench placement; no physics",
            script_sha256=_hash(__file__),implementation=_implementation_files(),
            helpers={name:_hash(ROOT/name) for name in ("_panda_scene.py","normal_map_film.py","panda_msaa_compare.py")},
            mesh_sha256={m.id:load_mesh(m).source_sha256 for m in scene.meshes},
            library=dict(file=str(library),sha256=_hash(library)),
            shaders={p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))},device_index=device_index,
            comparison="same ROI in all states; off/on differs only indirect_mode; red/grey differs only board base colour; blocked changes only blocker pose")
        _json(output/"protocol.json",protocol)
        receipts = {}
        for state,case in cases.items():
            receipts[state] = {}
            for mode in MODES:
                backend = NativeBackend(library_path=library,shader_directory=shaders,device_index=device_index,
                    resident=True,validation=False,studio=None,exposure=1.,display_transform="pbr-neutral",
                    shadow_mode="ray_query",msaa_samples=1,indirect_mode=mode,indirect_seed=0)
                receipts[state][mode] = _capture(output,state,mode,case,backend)
        if (protocol["script_sha256"] != _hash(__file__) or protocol["implementation"] != _implementation_files()
            or protocol["library"]["sha256"] != _hash(library)
            or protocol["shaders"] != {p.name:_hash(p) for p in sorted(shaders.glob("*.spv"))}
            or protocol["mesh_sha256"] != {m.id:load_mesh(m).source_sha256 for m in scene.meshes}
            or any(_hash(ROOT/name)!=digest for name,digest in protocol["helpers"].items())):
            raise ValueError("registered source/assets/device program changed")
        images = {}; geometry = {}; state_sensors = {}; base = receipts["red"]["none"]["backend"]
        for state in STATES:
            images[state] = {}; raw = {}
            for mode in MODES:
                r = receipts[state][mode]; backend = r["backend"]; d = backend["device"]
                verify_close(backend); verify_indirect(backend,mode)
                for key,value in (("library_sha256",protocol["library"]["sha256"]),("exposure",1.),
                                  ("display_transform","pbr-neutral"),("shadow_mode","ray_query")):
                    if backend.get(key) != value:
                        raise ValueError(f"actual backend setting differs: {key}")
                if (backend.get("studio") is not None or backend.get("environment") is not None
                    or d.get("device_uuid") != base["device"].get("device_uuid") or not d.get("device_uuid")
                    or d.get("resident_captures") != 12 or d.get("resident_upload_calls") != 1
                    or backend.get("msaa",{}).get("actual_color_samples") != 1):
                    raise ValueError("actual environment/device/capture count/MSAA differs")
                baseline = receipts[state]["none"]["backend"]
                for key in ("scene_key","decoded_geometry_sha256","triangles","light_count","texture_assets","studio","environment"):
                    if backend.get(key) != baseline.get(key):
                        raise ValueError(f"paired immutable geometry/material/light receipt differs: {key}")
                images[state][mode],raw[mode] = _load(output,r)
            sensors = compare_sensors(raw["none"],raw["diffuse_bounce"])
            ids = np.frombuffer(raw["none"][Channel.INSTANCE_ID],"<u4").reshape(960,1280)
            depth = np.frombuffer(raw["none"][Channel.FORWARD_DEPTH],"<f4").reshape(960,1280)
            valid = np.frombuffer(raw["none"][Channel.VALID],np.uint8).reshape(960,1280)
            roi = verify_roi(reference,ids,depth,valid)
            if not np.array_equal(images[state]["none"][ids==0],images[state]["diffuse_bounce"][ids==0]):
                raise ValueError("background changed under indirect surface toggle")
            geometry[state] = dict(sensors=sensors,roi=roi,background_identical=True)
            state_sensors[state] = raw["none"]
        for suffix in ("", "-blocked"):
            compare_sensors(state_sensors["red"+suffix],state_sensors["grey"+suffix])
        appearance = compare_appearance(images,reference["ids"])
        _present(output,images,appearance["passed"])
        report.update(status="complete" if appearance["passed"] else "optical_gate_failed",appearance=appearance,
            geometry=geometry,protocol_sha256=_hash(output/"protocol.json"),physical_device=base["device"],
            receipts={state:{mode:dict(file=f"{state}/{mode.replace('_','-')}-receipt.json",
                sha256=_hash(output/state/f"{mode.replace('_','-')}-receipt.json"),timing=r["timing"],close=r["close"])
                for mode,r in rows.items()} for state,rows in receipts.items()},
            interpretation="static colour transfer study; quantized display-linear comparison only; fixed optical gates never alter saved images")
    except BaseException as error:
        report.update(status="failed",error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        _json(output/"summary.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--shader-directory",type=Path)
    parser.add_argument("--device-index",type=int,default=1)
    args = parser.parse_args()
    result = run(args.output,args.library,device_index=args.device_index,shader_directory=args.shader_directory)
    print(json.dumps(dict(status=result["status"],output=str(args.output.resolve())),ensure_ascii=False))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()





