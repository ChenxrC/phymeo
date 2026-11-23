"""Q0: source-normal audit and one-variable Panda lighting comparisons.

Examples (run from an environment with the tested rendering package on PYTHONPATH):
  python panda_appearance_diagnosis.py --output NEW --source-only
  python panda_appearance_diagnosis.py --library DLL --output NEW --batches 1 --seeds 0
  python panda_appearance_diagnosis.py --library DLL --output NEW
  python panda_appearance_diagnosis.py --library DLL --output NEW --ablations --reference --motion

Every output directory must be new. The default is frame 60, seeds 0/1,
16/1024 native samples, original 2x2 versus 8x8 panel integration. No material
or manufacturer normal is edited. Optional ablations are diagnostic controls,
not proposed appearance improvements. Native convergence and external PT are
reported separately; neither completion nor a dense light grid means quality
acceptance. CPU normal images are explicitly reconstructed attributes, not
new GPU attachments. No physics is executed.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ROI_IDS = {"white-shell": (150, 151, 154), "fingers": (156, 158), "rubber": (2,),
           "left-arm": (112, 113, 115, 117), "wrist-shell": (125, 142, 149)}
CASES = ("baseline", "dense-panels", "no-directional", "no-panels", "no-normal", "wide-panels", "soft-studio")
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def frozen_profile():
    return dict(frame_index=60, size=[1280, 720], checkpoints=[1, 64], seeds=[0, 1],
                candidate="dense-panels", ablations=["no-directional", "no-panels", "no-normal"],
                ablation_reference="dense-panels", registered_regions={k:list(v) for k,v in ROI_IDS.items()},
                display_candidates={"wide-panels":"panel size x2, area x4, radiance /4; same centers and power",
                                    "soft-studio":"wide-panels plus no directional lights; combined display preset, not a one-variable attribution"},
                change="only area_light_grid: 2 -> 8; same radiance, geometry and authored normals",
                samples_per_batch=16, exposure=8., display_transform="pbr-neutral",
                filter="none: prevent a spatial filter from hiding a stable lighting boundary",
                motion_indices=list(range(33, 65)), playback_fps=30,
                attribution="source audit + normal images + paired light ablations; correlation alone is not a cause")


def case_settings(name):
    if name not in CASES:
        raise ValueError("unknown Q0 case")
    return dict(shadow_mode="ray_query", area_light_grid=2 if name == "baseline" else 8,
                msaa_samples=1, reflection_mode="single_bounce", indirect_mode="diffuse_bounce",
                indirect_bounces=2, progressive=True, spatial_filter="none")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_same_sensors(a, b):
    if set(a) != {"depth", "id", "valid"} or set(b) != set(a):
        raise ValueError("all three actual sensor planes are required")
    for key in a:
        if not isinstance(a[key], bytes) or a[key] != b[key]:
            raise ValueError("sensor bytes differ: " + key)
    return True


def require_linear_mean(record, width, height, batches):
    expected = dict(format="RGBA32F_LE", space="scene-linear-before-exposure", width=width,
                    height=height, batches=batches, primary_samples_per_pixel=16*batches)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("public raw mean metadata differs")
    data = record.get("data")
    if not isinstance(data, bytes) or len(data) != width*height*16:
        raise ValueError("public raw mean byte length differs")
    array = np.frombuffer(data, "<f4").reshape(height, width, 4)
    if not np.isfinite(array).all() or np.any(array[..., :3] < 0):
        raise ValueError("raw mean is nonfinite or negative")
    return array


def interpolate_normals(normals, screen_weights, depths):
    values = np.asarray(normals, dtype=float)
    weights = np.asarray(screen_weights, dtype=float) / np.asarray(depths, dtype=float)
    result = weights @ values
    size = float(np.linalg.norm(result))
    if not math.isfinite(size) or size == 0:
        raise ValueError("degenerate interpolated normal")
    return result / size


def normal_statistics(positions, normals, triangles):
    p, n, t = np.asarray(positions, float), np.asarray(normals, float), np.asarray(triangles, int)
    n = n / np.linalg.norm(n, axis=1)[:, None]
    corner = n[t]
    cross = np.cross(p[t[:, 1]]-p[t[:, 0]], p[t[:, 2]]-p[t[:, 0]])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths == 0) or not np.isfinite(corner).all():
        raise ValueError("invalid admitted triangle or normal")
    face = cross / lengths[:, None]
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ikj->ik", face, corner), -1, 1)))
    within = np.degrees(np.arccos(np.clip(np.sum(corner*corner[:, [1,2,0]], axis=2), -1, 1)))
    by_position = {}
    for point, normal in zip(p, n, strict=True):
        by_position.setdefault(tuple(point), set()).add(tuple(normal))
    return dict(triangles=len(t), vertices=len(p),
                geometric_to_authored_angle_deg_max=float(angle.max()),
                geometric_to_authored_angle_deg_p50=float(np.median(angle)),
                within_triangle_normal_angle_deg_max=float(within.max()),
                within_triangle_normal_angle_deg_p50=float(np.median(within)),
                coincident_positions_with_distinct_normals=sum(len(v)>1 for v in by_position.values()))


def source_corner_normals(xml, row):
    """Independent pinned COLLADA attribute reconstruction, not the preparation tool.

    Resolve the NORMAL accessor and its own indices, source node matrix, and
    declared retained source-triangle runs. Return corners in prepared order.
    """
    root = ET.fromstring(xml)
    geometry = root.find(".//c:geometry[@id='" + row["source_geometry"] + "']/c:mesh", NS)
    node = root.find(".//c:node[@id='" + row["source_node"] + "']", NS)
    if geometry is None or node is None:
        raise ValueError("source geometry/node binding absent")
    matrix_node = node.find("c:matrix", NS)
    if matrix_node is None:
        raise ValueError("source node matrix absent")
    matrix = np.asarray([float(x) for x in matrix_node.text.split()]).reshape(4, 4)
    if not np.array_equal(matrix.reshape(-1), row["source_matrix_row_major"]):
        raise ValueError("source matrix binding differs")
    primitive = geometry.findall("c:triangles", NS)[row["source_primitive"]]
    inputs = primitive.findall("c:input", NS)
    normal = next(i for i in inputs if i.attrib["semantic"] == "NORMAL")
    stride = 1 + max(int(i.attrib["offset"]) for i in inputs)
    index = np.asarray([int(x) for x in primitive.find("c:p", NS).text.split()]).reshape(-1, 3, stride)
    if len(index) != row["source_triangle_count"]:
        raise ValueError("source triangle count differs")
    source = geometry.find("c:source[@id='" + normal.attrib["source"].removeprefix("#") + "']", NS)
    accessor = source.find("c:technique_common/c:accessor", NS)
    if int(accessor.attrib.get("stride", 1)) != 3 or int(accessor.attrib.get("offset", 0)) != 0:
        raise ValueError("unsupported normal accessor")
    values = np.asarray([float(x) for x in source.find("c:float_array", NS).text.split()]).reshape(-1, 3)
    inverse_transpose = np.linalg.inv(matrix[:3, :3]).T
    # Deliberately use scalar sums and hypot like the declared normalization,
    # but resolve source arrays/indices independently of prepare_panda_visuals.
    converted = []
    for value in values:
        vector = [sum(float(inverse_transpose[a,b])*float(value[b]) for b in range(3)) for a in range(3)]
        length = math.hypot(*vector)
        if not length:
            raise ValueError("zero source normal")
        converted.append([x/length for x in vector])
    selected = [i for first, stop in row["source_triangle_runs"] for i in range(first, stop)]
    if selected != sorted(set(selected)) or not selected or selected[-1] >= len(index):
        raise ValueError("invalid retained source ordinal mapping")
    return np.asarray(converted, dtype="<f4")[index[selected, :, int(normal.attrib["offset"])]]


def audit_assets(asset_dir):
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.panda_visuals import panda_visual_assets

    meshes, _, instances = panda_visual_assets(asset_dir)  # Existing full pin/license validation.
    manifest = json.loads((asset_dir/"manifest.json").read_text(encoding="utf-8"))
    sources = {}
    source_rows = []
    for row in manifest["sources"]:
        path = asset_dir/row["archive_file"]
        if digest(path) != row["archive_sha256"] or not 0 < row["bytes"] <= 16_000_000:
            raise ValueError("source archive identity/size differs")
        with gzip.open(path, "rb") as stream:
            data = stream.read(row["bytes"]+1)
        if len(data) != row["bytes"] or sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("uncompressed source identity differs")
        sources[row["file"]] = data
        source_rows.append({k: row[k] for k in ("file", "sha256", "archive_sha256", "url")})
    lookup = {m.id: m for m in meshes}
    rows = []
    for row in manifest["meshes"]:
        mesh = load_mesh(lookup[row["id"]])
        expected = source_corner_normals(sources[row["source_file"]], row)
        actual = np.asarray(mesh.normals, dtype="<f4")[np.asarray(mesh.triangles)]
        if expected.shape != actual.shape:
            raise ValueError("normal corner count differs: " + row["id"])
        mismatch = expected.view("<u4") != actual.view("<u4")
        error = float(np.max(np.abs(expected.astype(float)-actual)))
        # Report exact byte mismatch, do not silently repair/re-normalize assets.
        if error > 2**-23:
            raise ValueError("source normal conversion exceeds one unit-scale FP32 ULP: " + row["id"])
        rows.append(dict(id=row["id"], source_file=row["source_file"],
                         source_geometry=row["source_geometry"], prepared_mesh_sha256=row["sha256"],
                         instances=[i.id for i in instances if i.mesh_id==row["id"]],
                         source_normal_corner_components=len(actual.reshape(-1)),
                         source_normal_nonidentical_components=int(mismatch.sum()),
                         source_normal_max_abs_difference=error,
                         **normal_statistics(mesh.positions, mesh.normals, mesh.triangles)))
    return dict(manifest_sha256=digest(asset_dir/"manifest.json"), sources=source_rows, meshes=rows,
                comparison="independent NORMAL indices + inverse transpose + normalization + retained source ordinals; FP32 component comparison",
                threshold="at most 2^-23 absolute per normalized component; exact mismatches also reported",
                conclusion="no evidence of replacement by face normals if all source comparisons pass; this does not prove the manufacturer mesh is visually smooth")


def crop_regions(ids, margin=6):
    result = {}
    h, w = ids.shape
    for name, identities in ROI_IDS.items():
        ys, xs = np.nonzero(np.isin(ids, identities))
        if not len(xs):
            raise ValueError("registered identity ROI missing: " + name)
        result[name] = dict(ids=list(identities), pixels=len(xs),
                            box=[max(0,int(xs.min())-margin), max(0,int(ys.min())-margin),
                                 min(w,int(xs.max())+1+margin), min(h,int(ys.max())+1+margin)])
    return result


def reconstruct_normals(scene, frame, camera, sensors, directory):
    """Perspective CPU source attributes, accepted only at actual ID/depth pixels.

    No claim of Vulkan subpixel-edge equivalence: barycentric edge pixels and
    unmatched depth are unknown/black. Normals use authored vertex normals,
    not a geometric smoothing operation. Texture-perturbed normals are absent.
    """
    from _panda_scene import pose_matrix
    from phymeo_rendering.mesh_data import load_mesh

    w, h = camera.width, camera.height
    ids = np.frombuffer(sensors["id"], "<u4").reshape(h,w)
    depth = np.frombuffer(sensors["depth"], "<f4").reshape(h,w)
    normal_regions = tuple(name for name in ROI_IDS if name!="rubber")
    chosen = {identity for name in normal_regions for identity in ROI_IDS[name]}
    node = {n.node_id: pose_matrix(n.world_pose) for n in frame.nodes}
    meshes = {m.id: m for m in scene.meshes}
    cam = pose_matrix(camera.world_pose)
    eye, rotation = cam[:3,3], cam[:3,:3]
    gmap, smap = np.zeros((h,w,3)), np.zeros((h,w,3))
    matched = np.zeros((h,w), bool)
    best = np.full((h,w), np.inf)
    for instance in scene.instances:
        if instance.id not in chosen:
            continue
        mesh = load_mesh(meshes[instance.mesh_id])
        transform = node[instance.node_id] @ pose_matrix(instance.local_pose)
        p = np.asarray(mesh.positions) @ transform[:3,:3].T + transform[:3,3]
        n = np.asarray(mesh.normals)
        n /= np.linalg.norm(n,axis=1)[:,None]
        n = n @ np.linalg.inv(transform[:3,:3])
        cp = (p-eye) @ rotation
        z = -cp[:,2]
        uv = np.column_stack((camera.cx_px+camera.fx_px*cp[:,0]/z,
                              camera.cy_px-camera.fy_px*cp[:,1]/z))
        for indices in mesh.triangles:
            t = np.asarray(indices)
            if np.any(z[t] <= camera.near_m) or np.any(z[t] >= camera.far_m):
                continue
            v = uv[t]
            lo = np.maximum(np.ceil(v.min(axis=0)-.5).astype(int), [0,0])
            hi = np.minimum(np.floor(v.max(axis=0)-.5).astype(int)+1, [w,h])
            if np.any(hi<=lo):
                continue
            x0,y0 = lo; x1,y1 = hi
            local_id = ids[y0:y1,x0:x1] == instance.id
            if not local_id.any():
                continue
            a,b = v[1]-v[0],v[2]-v[0]
            determinant = a[0]*b[1]-a[1]*b[0]
            if determinant == 0:
                continue
            yy,xx = np.mgrid[y0:y1,x0:x1]
            px,py = xx+.5-v[0,0],yy+.5-v[0,1]
            s = (px*b[1]-py*b[0])/determinant
            q = (a[0]*py-a[1]*px)/determinant
            bary = np.stack((1-s-q,s,q),axis=-1)
            inside = local_id & (bary.min(axis=-1)>1e-5)
            weights = bary/z[t]
            denominator = weights.sum(axis=-1)
            candidate_depth = np.divide(1.,denominator,out=np.full_like(denominator,np.inf),where=denominator!=0)
            defect = np.abs(candidate_depth-depth[y0:y1,x0:x1])
            accepted = inside & (defect<=.002) & (defect<best[y0:y1,x0:x1])
            if not accepted.any():
                continue
            shading = weights @ n[t]
            lengths = np.linalg.norm(shading,axis=-1)
            accepted &= lengths>0
            shading = np.divide(shading,lengths[...,None],out=np.zeros_like(shading),where=lengths[...,None]>0)
            geometric = np.cross(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            geometric /= np.linalg.norm(geometric)
            if np.dot(geometric,eye-p[t].mean(axis=0)) < 0:
                geometric = -geometric
            point = np.divide(weights @ p[t],denominator[...,None],
                              out=np.zeros_like(shading),where=denominator[...,None]!=0)
            shading *= np.where(np.sum(shading*(eye-point),axis=-1)<0,-1.,1.)[...,None]
            gmap[y0:y1,x0:x1][accepted] = geometric
            smap[y0:y1,x0:x1][accepted] = shading[accepted]
            matched[y0:y1,x0:x1][accepted] = True
            best[y0:y1,x0:x1][accepted] = defect[accepted]
    for name,array in (("cpu-geometric-normal",gmap),("cpu-authored-shading-normal",smap)):
        image = np.rint(np.clip((array+1)*127.5,0,255)).astype("u1")
        image[~matched] = 0
        Image.fromarray(image).save(directory/(name+".png"))
    Image.fromarray(matched.astype("u1")*255).save(directory/"cpu-normal-valid.png")
    material_lookup = {m.id:m for m in scene.materials}
    factors, roughness = np.zeros((h,w,3),dtype="u1"),np.zeros((h,w),dtype="u1")
    material_rows = []
    for instance in scene.instances:
        material = material_lookup[instance.material_id]
        mask = ids==instance.id
        factors[mask] = np.rint(np.asarray(material.base_color)*255).astype("u1")
        roughness[mask] = round(material.roughness*255)
        material_rows.append(dict(instance_id=instance.id,mesh_id=instance.mesh_id,
                                  material=asdict(material),pixels=int(mask.sum())))
    Image.fromarray(factors).save(directory/"declared-linear-basecolor-factor.png")
    Image.fromarray(roughness).save(directory/"declared-roughness-factor.png")
    save(directory/"material-bindings.json",material_rows)
    np.savez_compressed(directory/"cpu-normals.npz",geometric=gmap.astype("<f4"),
                        authored_shading=smap.astype("<f4"),valid=matched,depth_defect_m=best.astype("<f4"))
    angle = np.degrees(np.arccos(np.clip(np.sum(gmap*smap,axis=-1),-1,1)))
    rows = {}
    for name in normal_regions:
        mask = np.isin(ids,ROI_IDS[name])
        active = mask & matched
        if not active.any():
            raise ValueError("CPU normal reconstruction has no supported pixels: "+name)
        rows[name] = dict(actual_pixels=int(mask.sum()),reconstructed_pixels=int(active.sum()),
                          angle_p50_deg=float(np.median(angle[active])),angle_p99_deg=float(np.quantile(angle[active],.99)),
                          depth_defect_max_m=float(best[active].max()))
    return dict(scope="CPU source attributes bound to actual sensor ID and <=2mm forward depth; not GPU attachments",
                edge_scope="pixels within 1e-5 screen barycentric distance of a source triangle edge are unknown; no assumed Vulkan subpixel precision",
                absent="normal-map perturbation, base-color texture sampling, per-lobe direct/indirect/specular GPU attachments",
                image_encoding="world normal XYZ -> (N+1)/2 RGB8; black unknown; material factor images are linear parameters, not sRGB colors or sampled textures",
                regions=rows)


def _planes(shot):
    from phymeo_rendering.outputs import Channel
    p = {p.channel:p.data for p in shot.planes}
    return p[Channel.RGB], dict(depth=p[Channel.FORWARD_DEPTH],id=p[Channel.INSTANCE_ID],valid=p[Channel.VALID])


def _close(session):
    session.close()
    result = session.diagnostics()
    d = result["device"]
    if d.get("closed") is not True or any(d.get(k,1) for k in ("live_contexts","live_frames","live_resources","validation_errors")):
        raise RuntimeError("native session did not cleanly close")
    return result


def _case_scene(name, scenes, studio):
    scene = scenes[0 if name=="no-normal" else 1]
    if name in ("no-directional","soft-studio"):
        scene = replace(scene,key=scene.key+":no-directional",lights=())
    if name=="no-panels":
        studio = replace(studio,panel_strength=0.)
    if name in ("wide-panels","soft-studio"):
        studio = replace(studio,panel_size_scale=2.)
    return scene,studio


def _item(scene, frames, index):
    import panda_realtime_film as film
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    frame = replace(frames[index],key=replace(frames[index].key,scene_key=scene.key))
    camera = film.exposure_camera(SimpleNamespace(camera="detail"),frame,index,len(frames),1280,720)
    return frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))


def _backend(args,studio,case,seed):
    from phymeo_rendering.backends.native import NativeBackend
    return NativeBackend(library_path=args.library,shader_directory=args.shader_directory or args.library.parent,
                         device_index=args.device_index,resident=True,studio=studio,exposure=8.,
                         display_transform="pbr-neutral",indirect_seed=seed,validation=args.validation,
                         **case_settings(case))


def _capture_case(args,scene,frames,studio,case,seed,out,reference_sensors):
    out.mkdir()
    frame,request = _item(scene,frames,60)
    session = _backend(args,studio,case,seed).open_scene(scene)
    row = dict(case=case,seed=seed,settings=case_settings(case),frame=asdict(frame),camera=asdict(request.camera),
               lights=[asdict(x) for x in scene.lights],studio=asdict(studio),checkpoints=[],capture_ms=[])
    latest = None
    try:
        for n in range(1,max(args.batches)+1):
            start = perf_counter(); shot = session.capture(frame,request)
            row["capture_ms"].append(1000*(perf_counter()-start))
            if shot.frame_key!=frame.key or shot.camera!=request.camera or shot.time_s!=frame.time_s:
                raise ValueError("stale capture")
            rgb,sensors = _planes(shot)
            if reference_sensors is not None:
                require_same_sensors(reference_sensors,sensors)
            else:
                reference_sensors = sensors
            if n not in args.batches:
                continue
            diag = session.diagnostics()
            if (diag["area_light_grid"]!=case_settings(case)["area_light_grid"] or
                diag["progressive"]["batches"]!=n or diag["indirect"]["bounces"]!=2 or
                diag["spatial_filter"]["mode"]!="none"):
                raise ValueError("actual diagnostic mode differs")
            raw_record = session.read_linear_mean()
            raw = require_linear_mean(raw_record,1280,720,n)
            if session.diagnostics()["progressive"]["batches"]!=n:
                raise ValueError("raw readback advanced samples")
            stem = f"batch-{n:03d}"
            Image.frombytes("RGB",(1280,720),rgb).save(out/(stem+".png"))
            (out/(stem+".rgba32f")).write_bytes(raw_record["data"])
            for key,value in sensors.items():
                (out/(stem+"."+key)).write_bytes(value)
            ids = np.frombuffer(sensors["id"],"<u4").reshape(720,1280)
            crops = crop_regions(ids)
            metrics = {}
            rgbarray = np.frombuffer(rgb,"u1").reshape(720,1280,3)
            for name,region in crops.items():
                Image.fromarray(rgbarray).crop(region["box"]).save(out/(stem+"-"+name+".png"))
                mask = np.isin(ids,region["ids"])
                metrics[name] = dict(pixels=int(mask.sum()),raw_mean_rgb=raw[...,:3][mask].mean(axis=0,dtype=float).tolist())
            checkpoint = dict(batch=n,primary_samples_per_pixel=16*n,rgb_sha256=sha256(rgb).hexdigest(),
                              raw_sha256=sha256(raw_record["data"]).hexdigest(),
                              sensor_sha256={k:sha256(v).hexdigest() for k,v in sensors.items()},
                              crops=crops,metrics=metrics,diagnostics=diag)
            row["checkpoints"].append(checkpoint)
            latest = (sensors,frame,request.camera)
            save(out/"receipt.json",row)
    finally:
        row["closed_diagnostics"] = _close(session)
        save(out/"receipt.json",row)
    return row,reference_sensors,latest


def _board(out,rows):
    """No resampling: original-pixel crops, padding only, explicit labels."""
    tiles = []
    for row in rows:
        for cp in row["checkpoints"]:
            path = out/f"{row['case']}-seed-{row['seed']}"/f"batch-{cp['batch']:03d}.png"
            with Image.open(path) as source:
                for name,region in cp["crops"].items():
                    tile = source.crop(region["box"]).convert("RGB")
                    tiles.append((f"{row['case']} seed {row['seed']} {16*cp['batch']} spp | {name}",tile))
    if not tiles:
        return
    width = max(t.width for _,t in tiles)
    board = Image.new("RGB",(width,sum(t.height+28 for _,t in tiles)),(30,30,30))
    draw = ImageDraw.Draw(board); y=0
    for text,tile in tiles:
        draw.text((4,y+5),text,fill=(255,255,255)); board.paste(tile,(0,y+28)); y+=tile.height+28
    board.save(out/"original-pixel-crops.png")


def paired_observations(out,rows,sensors):
    """Paired differences, not ground-truth errors or automatic attribution."""
    ids=np.frombuffer(sensors["id"],"<u4").reshape(720,1280)
    available={(r["case"],r["seed"],cp["batch"]) for r in rows for cp in r["checkpoints"]}
    pairs=[]
    for case,seed,batch in sorted(available):
        current=(case,seed,batch)
        if seed==0 and (case,1,batch) in available:
            pairs.append(("independent-seed-disagreement",current,(case,1,batch)))
        base="baseline" if case=="dense-panels" else "dense-panels"
        if case not in ("baseline","soft-studio") and (base,seed,batch) in available:
            pairs.append(("one-variable-effect",(base,seed,batch),current))
        if case=="soft-studio" and ("wide-panels",seed,batch) in available:
            pairs.append(("one-variable-effect",("wide-panels",seed,batch),current))
    observations=[]
    for kind,a,b in pairs:
        def read(identity):
            case,seed,batch=identity
            stem=out/f"{case}-seed-{seed}"/f"batch-{batch:03d}"
            with Image.open(str(stem)+".png") as image:
                rgb=np.asarray(image,dtype=float)
            raw=np.fromfile(str(stem)+".rgba32f",dtype="<f4").reshape(720,1280,4)[...,:3].astype(float)
            return rgb,raw
        ar,al=read(a);br,bl=read(b)
        regions={}
        for name,identities in ROI_IDS.items():
            mask=np.isin(ids,identities)
            rgbdiff=br[mask]-ar[mask];rawdiff=bl[mask]-al[mask]
            regions[name]=dict(pixels=int(mask.sum()),rgb8_rmse=float(np.sqrt(np.mean(rgbdiff**2))),
                               rgb8_max_abs=float(np.abs(rgbdiff).max()),
                               raw_rgb_rmse=float(np.sqrt(np.mean(rawdiff**2))),
                               raw_mean_delta_rgb=rawdiff.mean(axis=0).tolist())
        observations.append(dict(kind=kind,a=list(a),b=list(b),regions=regions))
    return dict(scope="seed disagreement estimates stochastic variation; grid/ablation differences measure influence, not correctness or causality",pairs=observations)


def implementation_sources():
    result={}
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,"__file__",None)
        if filename and (name.startswith("phymeo_rendering") or name in
                         ("panda_render_performance","panda_realtime_film","_panda_scene","_panda_textures")):
            path=Path(filename)
            if path.is_file():
                result[name]=dict(path=str(path.resolve()),sha256=digest(path))
    return result


def _external(args,scenes,frames,studio,out):
    """A separate explicitly matched common-subset comparison, not an old film."""
    from phymeo_rendering.backends.sapien import SapienBackend,SapienConfig
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    scene = scenes[0]  # Both reference and its native comparator have no normal map.
    row,_,_ = _capture_case(args,scene,frames,studio,"dense-panels",1,out/"reference-native-no-normal",None)
    frame,request = _item(scene,frames,60)
    request = CaptureRequest(request.camera,RenderPath.PATH_TRACE,(Channel.RGB,))
    config = SapienConfig(path=RenderPath.PATH_TRACE,device=args.reference_device,samples_per_pixel=1024,
                          max_bounces=6,seed=1,denoiser="none",studio=studio,sampling_mode="shifted-halton",
                          exposure=8.,display_transform="pbr-neutral")
    session = SapienBackend(config).open_scene(scene)
    result = dict(native_common_subset=row,config=asdict(config),
                  scope="matched public scene/material/camera/HDR/exposure/display with normal maps disabled on BOTH; different PT transport depth/BRDF/MIS remains a model difference")
    # Enum conversion is explicit, never stringify an unknown object.
    result["config"]["path"] = config.path.value
    try:
        start=perf_counter();shot=session.capture(frame,request)
        result["capture_ms"]=1000*(perf_counter()-start)
        if shot.frame_key!=frame.key or shot.camera!=request.camera:
            raise ValueError("stale PT reference")
        rgb=next(p.data for p in shot.planes if p.channel is Channel.RGB)
        Image.frombytes("RGB",(1280,720),rgb).save(out/"sapien-common-subset-1024spp.png")
        result["rgb_sha256"]=sha256(rgb).hexdigest()
        result["diagnostics"]=session.diagnostics()
    finally:
        session.close()
        result["closed_diagnostics"]=session.diagnostics()
        save(out/"external-reference.json",result)
    return result


def _motion(args,scenes,frames,studio,out):
    ffmpeg,ffprobe=shutil.which("ffmpeg"),shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("--motion requires ffmpeg and ffprobe; stills are preserved")
    indices=frozen_profile()["motion_indices"]
    # Labels do not reveal which grid is denser; mapping remains in the receipt.
    mapping={"X":args.motion_candidate,"Y":"baseline"}
    receipts=[];sensors_by_frame={}
    for label,case in mapping.items():
        path=out/("motion-"+label);path.mkdir()
        scene,effective_studio=_case_scene(case,scenes,studio)
        session=_backend(args,effective_studio,case,0).open_scene(scene)
        row=dict(label=label,case=case,frames=[])
        try:
            for j,index in enumerate(indices):
                frame,request=_item(scene,frames,index)
                start=perf_counter();shot=session.capture(frame,request)
                ms=1000*(perf_counter()-start)
                if shot.frame_key!=frame.key or shot.camera!=request.camera:
                    raise ValueError("stale motion capture")
                rgb,sensors=_planes(shot)
                if index in sensors_by_frame:
                    require_same_sensors(sensors_by_frame[index],sensors)
                else:
                    sensors_by_frame[index]=sensors
                if session.diagnostics()["progressive"]["batches"]!=1:
                    raise ValueError("motion must reset the static accumulation each exposure")
                Image.frombytes("RGB",(1280,720),rgb).save(path/f"{j:04d}.png")
                row["frames"].append(dict(index=index,source_time_s=frame.time_s,playback_time_s=j/30,
                    frame=asdict(frame),camera=asdict(request.camera),capture_ms=ms,
                    rgb_sha256=sha256(rgb).hexdigest(),sensor_sha256={k:sha256(v).hexdigest() for k,v in sensors.items()}))
                save(path/"receipt.json",row)
        finally:
            row["closed_diagnostics"]=_close(session);save(path/"receipt.json",row)
        target=out/("blind-"+label+".mp4")
        command=[ffmpeg,"-hide_banner","-loglevel","error","-nostdin","-n","-framerate","30",
                 "-i",str(path/"%04d.png"),"-frames:v",str(len(indices)),"-vf","setsar=1",
                 "-c:v","libx264","-crf","15","-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
        subprocess.run(command,check=True)
        probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-count_frames","-show_streams","-of","json",str(target)],text=True))
        stream=probe["streams"][0]
        if int(stream["nb_read_frames"])!=len(indices) or stream["avg_frame_rate"]!="30/1" or stream["sample_aspect_ratio"]!="1:1":
            raise ValueError("motion media count/clock/SAR differs")
        row.update(video_sha256=digest(target),codec_probe=probe)
        receipts.append(row)
    result=dict(label_mapping=mapping,scope="32 distinct saved poses, physical clock 30 Hz; encoded playback rate is NOT renderer FPS; PNGs remain primary",receipts=receipts)
    save(out/"blind-motion-receipt.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path)
    parser.add_argument("--shader-directory",type=Path)
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--validation",action="store_true")
    parser.add_argument("--source-only",action="store_true")
    parser.add_argument("--batches",type=int,nargs="+",default=[1,64],choices=[1,64])
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1],choices=[0,1])
    parser.add_argument("--ablations",action="store_true")
    parser.add_argument("--cases",nargs="+",choices=CASES,default=["baseline","dense-panels"])
    parser.add_argument("--reference",action="store_true")
    parser.add_argument("--reference-device",default="cuda:0")
    parser.add_argument("--motion",action="store_true")
    parser.add_argument("--motion-candidate",choices=["dense-panels","wide-panels","soft-studio"],default="dense-panels")
    args=parser.parse_args(argv)
    if not args.source_only and args.library is None:
        parser.error("--library is required unless --source-only")
    args.batches=sorted(set(args.batches));args.seeds=sorted(set(args.seeds))
    args.cases=list(dict.fromkeys(args.cases+(frozen_profile()["ablations"] if args.ablations else [])))
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    if args.library:
        args.library=args.library.resolve()
    result=dict(schema="phymeo-panda-appearance-diagnosis-v1",status="started",profile=frozen_profile(),
                actual_checkpoints=args.batches,actual_seeds=args.seeds,actual_cases=args.cases,
                source_sha256=digest(__file__),cases=[],
                conclusion="unclassified: inspect original pixels and paired evidence; no automatic claim of improvement or asset fault")
    try:
        result["assets"]=audit_assets(ROOT/"assets/panda")
        save(args.output/"source-normal-audit.json",result["assets"])
        if not args.source_only:
            import panda_render_performance as factory
            import panda_realtime_film as film
            scenes,frames,studio,trace=factory.bind_scene(args.output/"geometry")
            result["factory_sha256"]=digest(factory.__file__)
            result["library_sha256"]=digest(args.library)
            result["input_binding"]=film.input_binding(scenes[1],[_item(scenes[1],frames,60)],trace)
            result["scene"]=asdict(scenes[1])
            result["source_trace_path"]=str(trace)
            sensors=None;latest=None
            for case in args.cases:
                scene,effective_studio=_case_scene(case,scenes,studio)
                for seed in args.seeds:
                    row,sensors,latest=_capture_case(args,scene,frames,effective_studio,case,seed,
                        args.output/f"{case}-seed-{seed}",sensors)
                    result["cases"].append(row)
                    save(args.output/"report.json",result)
            normal_dir=args.output/"normals";normal_dir.mkdir()
            # Baseline geometry is identical in all ablations, including no-normal.
            frame,request=_item(scenes[1],frames,60)
            result["normal_images"]=reconstruct_normals(scenes[1],frame,request.camera,sensors,normal_dir)
            _board(args.output,result["cases"])
            result["paired_observations"]=paired_observations(args.output,result["cases"],sensors)
            if args.reference:
                result["external_reference"]=_external(args,scenes,frames,studio,args.output)
            if args.motion:
                result["blind_motion"]=_motion(args,scenes,frames,studio,args.output)
            result["missing_debug_channels"]=["GPU geometric/shading normals", "evaluated basecolor/roughness", "isolated direct/indirect/specular radiance"]
        result["status"]="complete"
    except Exception as error:
        result["status"]="partial_failure"
        result["error"]=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        result["implementation_sources"]=implementation_sources()
        save(args.output/"report.json",result)
    return result


if __name__=="__main__":
    main()
