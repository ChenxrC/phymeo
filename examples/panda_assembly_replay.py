"""Film recorded dual-Panda assembly configurations, using one XML per episode.

python examples/panda_assembly_replay.py --episode pin.xml pin.npz \
    --episode screw.xml screw.npz --backend vulkan --library /path/libphymeo_floor.so --out movie

The NPZ must contain time_s and qpos. Optional model_sha256 binds the XML.
The overview and hole close-up use the same recorded sample, never a physics step.
Use --stills-only for first/middle/final images or --backend mujoco for local OpenGL.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(xml, trajectory):
    model = mj.MjModel.from_xml_path(str(xml.resolve()))
    source = digest(xml)
    with np.load(trajectory, allow_pickle=False) as data:
        q = np.asarray(data["qpos"], dtype=np.float64)
        t = np.asarray(data["time_s"], dtype=np.float64)
        binding = None
        if "model_sha256" in data:
            binding = data["model_sha256"].item()
            if isinstance(binding, bytes):
                binding = binding.decode("ascii")
            if binding != source:
                raise ValueError(f"qpos model_sha256 does not match {xml}")
    if (t.ndim != 1 or not len(t) or q.ndim != 2 or q.shape != (len(t), model.nq)
            or not np.isfinite(q).all() or not np.isfinite(t).all() or not (np.diff(t) > 0).all()):
        raise ValueError(f"{trajectory}: expected strictly ordered finite time_s and matching qpos[N,{model.nq}]")
    for joint in range(model.njnt):
        kind = model.jnt_type[joint]
        if kind in (mj.mjtJoint.mjJNT_BALL, mj.mjtJoint.mjJNT_FREE):
            start = model.jnt_qposadr[joint] + (3 if kind == mj.mjtJoint.mjJNT_FREE else 0)
            if np.any(abs(np.linalg.norm(q[:,start:start+4], axis=1)-1.) > 1e-3):
                raise ValueError(f"{trajectory}: nonunit recorded quaternion in joint {joint}")
    return model, q, t, dict(xml=str(xml), xml_sha256=source, trajectory=str(trajectory),
        trajectory_sha256=digest(trajectory), trajectory_binds_xml=binding is not None,
        nq=model.nq, recorded_samples=len(t), recorded_start_s=t[0], recorded_end_s=t[-1])


def visible(model, groups):
    rgba = model.geom_rgba.copy()
    has_material = model.geom_matid >= 0
    # Non-default geom RGBA overrides its named material in MuJoCo.
    inherited = has_material & np.all(rgba == np.array([.5,.5,.5,1.],dtype=rgba.dtype),axis=1)
    rgba[inherited] = model.mat_rgba[model.geom_matid[inherited]]
    ids = np.flatnonzero(np.isin(model.geom_group,groups) & (rgba[:,3] > 0.))
    if not len(ids):
        raise ValueError("selected groups contain no visible geometry")
    return ids, rgba


def _cad(model, i):
    """Compiled CAD coordinates are in geom space. Preserve discontinuous corner normals."""
    va, nv = int(model.mesh_vertadr[i]), int(model.mesh_vertnum[i])
    na, nn = int(model.mesh_normaladr[i]), int(model.mesh_normalnum[i])
    fa, nf = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i])
    xyz = model.mesh_vert[va:va+nv]
    normal = model.mesh_normal[na:na+nn]
    face = model.mesh_face[fa:fa+nf]
    corner = model.mesh_facenormal[fa:fa+nf]
    if not len(face) or face.min() < 0 or face.max() >= nv:
        raise ValueError(f"invalid faces in visual mesh {i}")
    points = xyz[face]
    flat = np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    length = np.linalg.norm(flat,axis=1)
    keep = length > 0.
    if not np.any(keep):
        raise ValueError(f"empty visual mesh {i}")
    flat = flat[keep]/length[keep,None]
    points, corner = points[keep], corner[keep]
    n = np.broadcast_to(flat[:,None],points.shape).copy()
    valid = (corner >= 0) & (corner < nn)
    n[valid] = normal[corner[valid]]
    lengths = np.linalg.norm(n,axis=2)
    bad = lengths < 1e-8
    n[bad] = np.broadcast_to(flat[:,None],points.shape)[bad]
    n /= np.linalg.norm(n,axis=2)[:,:,None]
    vertices = np.concatenate((points,n),axis=2).reshape(-1,6).astype("f4")
    vertices, index = np.unique(vertices,axis=0,return_inverse=True)
    return vertices, index.reshape(-1).astype("u4")


class Geometries:
    def __init__(self, model, ids, colors, plane_half):
        from _bimanual_floor_scene import _Atlas, _transform
        atlas = _Atlas()
        slots, shape = [], []
        for g in ids:
            kind, size = model.geom_type[g], model.geom_size[g]
            if colors[g,3] < .9999:
                raise ValueError(f"{model.geom(g).name} is translucent; select the MuJoCo backend or an opaque visual group")
            if kind == mj.mjtGeom.mjGEOM_MESH:
                i = int(model.geom_dataid[g])
                key = ("compiled",i)
                if key not in atlas.slots:
                    v, f = _cad(model,i)
                    atlas.slots[key] = len(atlas.ranges)
                    atlas.ranges.append((atlas.nv,len(v),atlas.ni,len(f)))
                    atlas.nv += len(v)
                    atlas.ni += len(f)
                    atlas.vertices.append(v)
                    atlas.indices.append(f)
                slot, local = atlas.slots[key], _transform()
            elif kind == mj.mjtGeom.mjGEOM_PLANE:
                # Truncate the infinite support only for photography; its top stays on the source plane.
                slot = atlas.mesh(("box",))
                local = _transform((0.,0.,-.006),scale=(max(size[0],plane_half),max(size[1],plane_half),.006))
            else:
                slot, scale = atlas.shape(kind,size)
                local = _transform(scale=scale)
            shape.append(local)
            slots.append(slot)
        order = np.argsort(slots,kind="stable")
        self.geom = ids[order]
        self.local = np.asarray(shape,"f4")[order]
        self.vertices = np.ascontiguousarray(np.vstack(atlas.vertices),dtype="f4")
        self.indices = np.concatenate(atlas.indices).astype("u4")
        self.meshes = np.asarray(atlas.ranges,dtype="u4")
        self.instance_mesh = np.asarray(slots,"u4")[order]
        self.instance_id = self.geom.astype("u4")+1
        self.instance_color = np.ascontiguousarray(colors[self.geom],dtype="f4")
        counts = np.bincount(self.instance_mesh.astype(int),minlength=len(atlas.ranges))
        self.draws = np.column_stack((np.cumsum(counts)-counts,counts)).astype("u4")

    def transform(self, data):
        world = self.local.copy()
        rot = data.geom_xmat[self.geom].reshape(-1,3,3)
        world[:,:3,:3] = rot @ self.local[:,:3,:3]
        world[:,:3,3] = (rot @ self.local[:,:3,3,None])[:,:,0] + data.geom_xpos[self.geom]
        return world


def axes(azimuth, elevation):
    az, el = np.deg2rad((azimuth,elevation))
    back = np.array((np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)))
    right = np.array((-np.sin(az),np.cos(az),0.))
    return np.stack((right,np.cross(back,right),back))


def camera_vp(target, distance, azimuth, elevation, fov, aspect):
    r = axes(azimuth,elevation)
    eye = target+distance*r[2]
    view = np.eye(4)
    view[:3,:3], view[:3,3] = r, -r@eye
    n, f = .004, max(32.,distance*4)
    fy = 1/math.tan(math.radians(fov)/2)
    proj = np.array(((fy/aspect,0,0,0),(0,-fy,0,0),(0,0,f/(n-f),f*n/(n-f)),(0,0,-1,0)))
    return np.ascontiguousarray((proj@view).T,dtype="f4")


def overview(model, data, qpos, ids, args, panel_ratio, fov):
    robot = np.array([model.body(model.geom_bodyid[i]).name or "" for i in ids], dtype=str)
    include = (model.body_weldid[model.geom_bodyid[ids]] != 0) | np.char.startswith(robot,"left_panda_") | np.char.startswith(robot,"right_panda_")
    moving = ids[include &
                 (model.geom_type[ids] != mj.mjtGeom.mjGEOM_PLANE)]
    if not len(moving):
        raise ValueError("no moving robot or assembly geometry in the selected visual groups")
    box = model.geom_aabb[moving].reshape(-1,2,3)
    signs = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    corners = box[:,None,0,:] + signs*box[:,None,1,:]
    r = axes(args.azimuth,args.elevation)
    lo, hi = np.full(3,np.inf), np.full(3,-np.inf)
    for q in qpos:
        data.qpos[:] = q
        mj.mj_kinematics(model,data)
        points = corners @ data.geom_xmat[moving].reshape(-1,3,3).transpose(0,2,1) + data.geom_xpos[moving,None,:]
        projected = points.reshape(-1,3)@r.T
        lo, hi = np.minimum(lo,projected.min(axis=0)), np.maximum(hi,projected.max(axis=0))
    radius = (hi-lo)/2.
    distance = radius[2] + 1.13*max(radius[1],radius[0]/panel_ratio)/math.tan(math.radians(fov)/2)
    return .5*(lo+hi)@r, float(max(.10,distance))


def assembly_motion(model, data, qpos, entry, tip, part):
    """Signed screw coordinates in the moving board frame at *every* logged state.

    The angle uses the runner's HOME mounting orientation and the projected
    workpiece rotation in the board XY plane. Unwrapping precedes movie frame
    selection, including when only three stills are requested.
    """
    axis = mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,"assembly_pin_axis")
    if min(entry,tip,axis,part)<0 or model.site_bodyid[axis]!=part or model.site_bodyid[tip]!=part:
        raise ValueError("recorded assembly requires both pin sites on the independent assembly_part body")
    joint = mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,"assembly_part_free")
    if joint<0 or model.jnt_type[joint]!=mj.mjtJoint.mjJNT_FREE or model.jnt_bodyid[joint]!=part or model.nq!=21:
        raise ValueError("expected 14 Panda hinges and the independently recorded free workpiece")
    data.qpos[:] = model.key("home").qpos
    mj.mj_kinematics(model,data)
    hole, driver = model.site("assembly_hole").id, model.site("assembly_driver").id
    home_rotation = data.site_xmat[hole].reshape(3,3).T @ data.site_xmat[driver].reshape(3,3)
    rel = np.empty((len(qpos),3))
    twist = np.empty(len(qpos))
    for i,q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_kinematics(model,data)
        board = data.site_xmat[entry].reshape(3,3)
        rel[i] = board.T @ (data.site_xpos[tip]-data.site_xpos[entry])
        part_rotation = data.site_xmat[hole].reshape(3,3).T @ data.site_xmat[axis].reshape(3,3)
        projected = (part_rotation @ home_rotation.T)[:,0]
        if projected[:2] @ projected[:2] < .2:
            raise ValueError("workpiece X axis has no defined azimuth in the board plane")
        twist[i] = np.arctan2(projected[1],projected[0])
    unwrapped = np.unwrap(twist)
    return rel, unwrapped-unwrapped[0], rel[:,2]-rel[0,2]


def gl_info():
    """Read the renderer of the current GL context if exposed by its dispatch library."""
    for name in ("GL","OpenGL","opengl32"):
        lib = ctypes.util.find_library(name)
        if lib:
            gl = ctypes.CDLL(lib)
            query = gl.glGetString
            query.argtypes, query.restype = [ctypes.c_uint], ctypes.c_char_p
            value = query(0x1F01)
            if value:
                return value.decode(errors="replace")
    return "OpenGL renderer string unavailable"


def centered_crop(image, width):
    left = (image.width-width)//2
    return image.crop((left,0,left+width,image.height))


def run(args):
    from bimanual_training_floor_reel import FloorDevice
    if len(args.episode) > 2 or args.width%2 or args.height%2 or min(args.width,args.height)<160:
        raise ValueError("specify one or two episodes and even positive film dimensions >=160")
    if not 0 < args.fps <= 120 or not np.isfinite(args.playback_rate) or args.playback_rate <= 0:
        raise ValueError("fps and playback rate must be positive")
    if (not all(-89 < e < 89 for e in [args.elevation,args.detail_elevation])
            or not np.isfinite([args.azimuth,args.detail_azimuth,args.detail_distance]).all() or args.detail_distance <= .01):
        raise ValueError("invalid camera orientation or distance")
    if args.backend == "vulkan" and not args.library:
        raise ValueError("--backend vulkan requires --library pointing to libphymeo_floor")
    args.out.mkdir(parents=True,exist_ok=False)
    movie = args.out/"assembly.mp4"
    manifest = dict(backend=args.backend, fps=args.fps, playback_rate=args.playback_rate, size=[args.width,args.height],
                    semantics="Nearest recorded qpos. No scoring or physics steps during replay.", episodes=[])
    fp = None
    writer = None
    frames_total = 0
    try:
        if not args.stills_only:
            ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
            if not ffmpeg:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            fp = (args.out/"encode.log").open("xb")
            writer = subprocess.Popen([str(ffmpeg),"-nostdin","-hide_banner","-loglevel","error","-n",
                "-f","rawvideo","-pix_fmt","rgb24","-s",f"{args.width}x{args.height}","-r",str(args.fps),
                "-i","pipe:0","-an","-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-preset","fast",
                "-movflags","+faststart",str(movie)], stdin=subprocess.PIPE,stderr=fp)
        for chapter,(xml,trajectory) in enumerate(args.episode,1):
            model,q,t,report = load_episode(xml,trajectory)
            report["movie_start_frame"] = frames_total
            ids,colors = visible(model,args.groups)
            data = mj.MjData(model)
            focus_kind = "body" if args.focus_body else "site"
            focus_name = args.focus_body or args.focus_site or "assembly_hole"
            focus = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY if args.focus_body else mj.mjtObj.mjOBJ_SITE,focus_name)
            if focus < 0 and (args.focus_body or args.focus_site):
                raise ValueError(f"unknown {focus_kind}: {focus_name}")
            split = round(.61*args.width) if focus >= 0 else args.width
            fov = model.vis.global_.fovy if args.backend=="mujoco" else 42.
            target,dist = overview(model,data,q,ids,args,split/args.height,fov)
            model_name = bytes(model.names).split(b"\0",1)[0].decode("utf-8")
            known = {"dual_panda_pin":"Pin insertion", "dual_panda_screw_proxy":"Passive helical guide / 1.5 mm per turn"}
            official_experiment = model_name in known
            entry = mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,"assembly_entry")
            tip = mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,"assembly_pin_tip")
            part = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,"assembly_part")
            free_model = official_experiment and part>=0
            screw = model_name=="dual_panda_screw_proxy"
            if official_experiment and not free_model and screw:
                known[model_name] = "Legacy rigid tool / pitch diagnostic"
            report.update(detail=None if focus<0 else {focus_kind:focus_name,"distance_m":args.detail_distance},
                          model_name=model_name, stage=known.get(model_name,"Assembly replay"),
                          overview_target_m=target.tolist(), overview_distance_m=dist, groups=args.groups)
            if official_experiment:
                if free_model:
                    relative,turn,feed = assembly_motion(model,data,q,entry,tip,part)
                    report["model_scope"] = ("14 Panda hinges; structural collision-proxy appearance. Board mounted to left wrist. "
                        "Right wrist drives the independently simulated assembly_part through a compliant 6D tool mount.")
                    report["measurement"] = ("Free workpiece relative to the moving board entry. Positive axial is inward. "
                        "Positive turn is about board +Z. Turn and feed changes start at the first recorded state. "
                        "Azimuth is unwrapped using all time_s samples, assuming less than half a turn between adjacent samples.")
                    if screw:
                        report["guide_model"] = ("Pre-engaged passive 1.5 mm/rev positive lead and kinetic rotational drag. "
                            "Reciprocal generalized loads computed on the host by the dynamics runner. No modeled thread-surface contact.")
                else:
                    report["model_scope"] = "Legacy rigid wrist tools; structural collision proxies; no screw constraint."
            with ExitStack() as stack:
                if args.backend == "vulkan":
                    gpu = FloorDevice(args.library,args.device_index,args.width,args.height)
                    stack.callback(gpu.close)
                    geom = Geometries(model,ids,colors,plane_half=max(2.,dist))
                    report["geometry"] = gpu.upload(geom)
                    report["device"] = gpu.info()
                    report["appearance"] = "Compiled geometry with opaque base colours and Vulkan shadow mapping."
                else:
                    model.vis.global_.offwidth = max(args.width,model.vis.global_.offwidth)
                    model.vis.global_.offheight = max(args.height,model.vis.global_.offheight)
                    renderer = stack.enter_context(mj.Renderer(model,height=args.height,width=args.width))
                    render_options = mj.MjvOption()
                    render_options.geomgroup[:] = np.isin(np.arange(6),args.groups)
                    render_options.sitegroup[:] = 0
                    report["device"] = dict(backend="MuJoCo OpenGL",renderer=gl_info())
                n = max(1, math.ceil(float(t[-1]-t[0])*args.fps/args.playback_rate)+1)
                selected = {0,n//2,n-1}
                indices = sorted(selected) if args.stills_only else range(n)
                report["frames"] = []
                for i in indices:
                    stamp = min(t[-1],t[0]+i*args.playback_rate/args.fps)
                    j = int(np.searchsorted(t,stamp))
                    k = j-1 if j>0 and stamp-t[j-1]<t[j]-stamp else j
                    data.qpos[:] = q[k]
                    data.time = t[k]
                    mj.mj_kinematics(model,data)
                    if args.backend=="vulkan":
                        transforms = geom.transform(data)
                    else:
                        mj.mj_comPos(model,data)
                        mj.mj_camlight(model,data)
                    def shoot(center,distance,az,el):
                        if args.backend=="vulkan":
                            vp = camera_vp(center,distance,az,el,fov,args.width/args.height)
                            pixels,_ = gpu.render(transforms,vp)
                            return Image.fromarray(pixels[:,:,:3],"RGB")
                        cam = mj.MjvCamera()
                        cam.lookat[:],cam.distance = center,distance
                        cam.azimuth,cam.elevation = (az+180.)%360.,-el
                        renderer.update_scene(data,camera=cam,scene_option=render_options)
                        return Image.fromarray(renderer.render(),"RGB")
                    frame = Image.new("RGB",(args.width,args.height))
                    frame.paste(centered_crop(shoot(target,dist,args.azimuth,args.elevation),split),(0,0))
                    if focus >= 0:
                        center = (data.xpos if args.focus_body else data.site_xpos)[focus]
                        detail = shoot(center,args.detail_distance,args.detail_azimuth,args.detail_elevation)
                        frame.paste(centered_crop(detail,args.width-split),(split,0))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,args.width,24),fill=(22,32,37))
                    draw.text((8,5),f"{report['stage']}  /  Episode {chapter}    SIM {t[k]:.3f} s    {args.playback_rate:g}x",
                              fill=(241,244,245))
                    if focus>=0:
                        draw.text((split+8,5),f"DETAIL  /  {focus_name}",fill=(237,216,168))
                    frame_info = dict(chapter_frame=i,source_sample=k,sim_time_s=float(t[k]))
                    if official_experiment:
                        draw.rectangle((0,args.height-43,args.width,args.height),fill=(22,32,37))
                        scope = "Panda structural proxies  /  free part + compliant 6D wrist mount" if free_model else "Legacy Panda proxies / rigid wrist tools"
                        draw.text((8,args.height-36),scope,fill=(216,224,225))
                        if free_model and screw:
                            draw.text((8,args.height-18),"Pre-engaged lead + turning drag; no thread surfaces",fill=(216,224,225))
                    if official_experiment and entry>=0 and tip>=0:
                        pos = relative[k] if free_model else data.site_xmat[entry].reshape(3,3).T @ (data.site_xpos[tip]-data.site_xpos[entry])
                        frame_info["tip_in_entry_frame_m"] = pos.tolist()
                        line = f"Tip from entry: axial {1000*pos[2]:+.1f} mm    lateral {1000*np.linalg.norm(pos[:2]):.1f} mm"
                        if free_model:
                            frame_info.update(turn_from_first_rad=float(turn[k]), axial_feed_from_first_m=float(feed[k]))
                            motion = f"Travel from start: {1000*feed[k]:+.2f} mm"
                            if screw:
                                motion = f"Part turn {np.rad2deg(turn[k]):+.1f} deg    measured feed {1e6*feed[k]:+.0f} um"
                            draw.text((args.width-8,args.height-30),motion,anchor="rm",fill=(244,220,168))
                        draw.text((args.width-8,args.height-12),line,anchor="rm",fill=(237,216,168))
                    if i in selected:
                        frame.save(args.out/f"episode-{chapter:02d}-frame-{i:05d}.png")
                    if writer is not None:
                        writer.stdin.write(frame.tobytes())
                    report["frames"].append(frame_info)
                    frames_total += 1
                manifest["episodes"].append(report)
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("FFmpeg failed; see encode.log")
            manifest["movie"] = dict(path=movie.name,sha256=digest(movie),frames=frames_total,duration_s=frames_total/args.fps)
    finally:
        if writer is not None and writer.poll() is None:
            writer.stdin.close()
            writer.wait()
        if fp is not None:
            fp.close()
    (args.out/"replay.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=Path,nargs=2,action="append",required=True,metavar=("XML","NPZ"))
    p.add_argument("--backend",choices=("vulkan","mujoco"),default="vulkan")
    p.add_argument("--library",type=Path)
    p.add_argument("--device-index",type=int,default=0)
    p.add_argument("--out",type=Path,required=True,help="new output directory")
    p.add_argument("--width",type=int,default=1280)
    p.add_argument("--height",type=int,default=720)
    p.add_argument("--fps",type=int,default=30)
    p.add_argument("--playback-rate",type=float,default=1.)
    p.add_argument("--stills-only",action="store_true")
    p.add_argument("--ffmpeg",type=Path)
    p.add_argument("--groups",type=int,nargs="+",choices=range(6),default=[0,1,2])
    p.add_argument("--azimuth",type=float,default=-78.)
    p.add_argument("--elevation",type=float,default=32.)
    focus = p.add_mutually_exclusive_group()
    focus.add_argument("--focus-body")
    focus.add_argument("--focus-site")
    p.add_argument("--detail-distance",type=float,default=.29)
    p.add_argument("--detail-azimuth",type=float,default=-50.)
    p.add_argument("--detail-elevation",type=float,default=22.)
    run(p.parse_args())


if __name__ == "__main__":
    main()
