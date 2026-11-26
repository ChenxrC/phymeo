"""Film the measured frictional grasp of a free pin with two Panda robots.

Input is the original MJCF and one or two independently recorded trajectories.
Replay the recorded contact-solver input for each force report when available.
Legacy records show the published endpoint and label the force from the preceding step.
"""
import argparse
from contextlib import closing
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _panda_assembly_appearance import build_appearance
from panda_assembly_replay import axes, digest
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.photography import StudioRenderer, studio_lighting
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import Pose


WIDTH, HEIGHT = 1600, 900
PANEL = (800, 712)
SENSOR = (1000, 890)
PADS = ("right_left_pad", "right_right_pad")
PHASES = dict(initial="初始", settle="静置", approach="接近", descend="下降", close="闭指",
              lift="提升", transfer="移送", insert="插入", seat="落座",
              release="释放", clear="撤回", retreat="撤回", complete="指令结束")


@dataclass
class Trace:
    path: Path
    sha256: str
    time: np.ndarray
    qpos: np.ndarray
    phase: np.ndarray
    pads: np.ndarray
    support: np.ndarray | None
    seat: np.ndarray | None
    published_time: np.ndarray
    contact_aligned: bool
    arm_targets: np.ndarray | None


def read_trace(path, model, xml_hash):
    with np.load(path, allow_pickle=False) as z:
        for name in ("time_s", "qpos", "phase", "pad_forces", "model_sha256"):
            if name not in z:
                raise ValueError(f"{path.name} lacks {name}")
        binding = z["model_sha256"].item()
        if isinstance(binding,bytes):
            binding = binding.decode("ascii")
        if binding != xml_hash:
            raise ValueError(f"{path.name} does not match the supplied MJCF")
        t, q = np.asarray(z["time_s"], float), np.asarray(z["qpos"], float)
        ph, pad = z["phase"], np.asarray(z["pad_forces"], float)
        n = t.size
        if t.shape != (n,) or n < 2 or not np.all(np.diff(t) > 0):
            raise ValueError("require at least two increasing simulation timestamps")
        if q.shape != (n, model.nq) or ph.shape != (n,) or ph.dtype.kind != "U" or pad.shape != (n, 2):
            raise ValueError("qpos, Unicode phase and measured two-pad forces must share the source clock")
        if not all(np.isfinite(a).all() for a in (t, q, pad)):
            raise ValueError("state or force telemetry contains nonfinite values")
        def optional(name):
            if name not in z:
                return None
            arr = np.asarray(z[name], float)
            if arr.shape != (n,) or not np.isfinite(arr).all():
                raise ValueError(f"{name} is not finite, timestamp-aligned force telemetry")
            return arr
        published_time = t
        keys = {"contact_time_s", "contact_qpos", "contact_phase"}
        present = keys.intersection(z.files)
        if present and present != keys:
            raise ValueError(f"{path.name} lacks contact-input fields {sorted(keys-present)}")
        if present:
            t = np.asarray(z["contact_time_s"], float)
            q = np.asarray(z["contact_qpos"], float)
            ph = z["contact_phase"]
            if (t.shape != (n,) or q.shape != (n,model.nq) or ph.shape != (n,)
                    or ph.dtype.kind != "U" or not np.isfinite(t).all() or not np.isfinite(q).all()):
                raise ValueError("invalid per-report contact input clock, positions or phases")
            # Row zero is the initial state. Every other row measures exactly
            # one completed step and refers to the clock of that step's input.
            if (ph[0] != "initial" or t[0] != published_time[0]
                    or not np.allclose(t[1:],published_time[:-1],rtol=0,atol=1e-10)):
                raise ValueError("contact solver input clock does not match the recorded step sequence")
        targets = None
        if "arm_target_q" in z:
            targets = np.asarray(z["arm_target_q"],float)
            if targets.shape != (n,14) or not np.isfinite(targets).all():
                raise ValueError("expected fourteen finite arm position targets per report")
        return Trace(path, digest(path), t, q, ph, pad, optional("support_n"), optional("seat_n"),
                     published_time, bool(present), targets)


def source_schedule(time, count):
    movie_clock = np.linspace(time[0], time[-1], count)
    hi = np.clip(np.searchsorted(time, movie_clock), 1, len(time)-1)
    schedule = np.where(movie_clock-time[hi-1] <= time[hi]-movie_clock, hi-1, hi)
    schedule[0], schedule[-1] = 0, len(time)-1
    return schedule


def exposure_frame(appearance, data, trace, k, epoch=0):
    # A force report's input pose is a distinct exposure phase, even when
    # both poses happen to coincide (for example with a stationary hand).
    frame = appearance.frame(data,k,trace.time[k])
    phase = "step_input" if trace.contact_aligned and k else "accepted_endpoint"
    return replace(frame,key=replace(frame.key,epoch=epoch,phase=phase))


def checks_for(trace, schedule):
    # Take a real exposure from every controller phase that appears on screen.
    p = trace.phase[schedule]
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1])+1]
    ends = np.r_[starts[1:]-1, len(p)-1]
    return {0, len(p)-1, *map(int, (starts+ends)//2)}


class Rig:
    def __init__(self, model):
        self.pads = [model.geom(n).id for n in PADS]
        self.hand = model.site("right_grasp_center").id
        self.part = model.site("part_grasp_center").id
        self.tip = model.site("assembly_pin_tip").id
        self.entry = model.site("assembly_entry").id
        for geom, body, joint in zip(self.pads, ("leftfinger", "rightfinger"),
                                     ("finger_joint1", "finger_joint2"), strict=True):
            bid = model.body("right_panda_"+body).id
            jid = model.joint("right_panda_"+joint).id
            if (model.geom_bodyid[geom] != bid or model.jnt_bodyid[jid] != bid or
                    model.jnt_type[jid] != mj.mjtJoint.mjJNT_SLIDE):
                raise ValueError("active pad and CAD finger must share the driven finger body")
        part = model.body("assembly_part").id
        if model.body_jntnum[part] != 1 or model.jnt_type[model.body_jntadr[part]] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("replay needs a genuine free workpiece")

    def detail_angle(self, d):
        # Look across the jaw axis so both moving fingers can be distinguished.
        jaw = d.geom_xpos[self.pads[0]]-d.geom_xpos[self.pads[1]]
        side = np.cross(jaw, [0., 0., 1.])
        if side[1] > 0:
            side = -side
        return float(np.degrees(np.arctan2(side[1], side[0]))) if np.linalg.norm(side) > 1e-6 else -65.

    def detail(self, key, d, angle):
        # Even a dropped part and an empty departing hand use their own positions.
        p = np.vstack((d.geom_xpos[self.pads], d.site_xpos[[self.part, self.hand, self.tip]]))
        mid = (p.min(axis=0)+p.max(axis=0))/2
        radius = np.max(np.linalg.norm(p-mid, axis=1))+.032
        return camera(key, "fingers-and-workpiece", mid, max(.245, radius/.27), angle, 22., 36.)


def camera(key, name, target, distance, az, el, fov):
    rotation = axes(az, el)
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, rotation.T.flatten())
    origin = target + distance*rotation[2]
    width, height = SENSOR
    focal = height/(2*np.tan(np.deg2rad(fov)/2))
    return Camera(name, key, Pose(tuple(origin), tuple(quat[[1,2,3,0]])),
                  width, height, focal, focal, width/2, height/2, .004, 32.)


def expose(session, frame, cam, audit):
    channels = (Channel.RGB, Channel.INSTANCE_ID) if audit else (Channel.RGB,)
    shot = session.capture(frame, CaptureRequest(cam, RenderPath.RASTER, channels))
    if (shot.frame_key, shot.camera, shot.time_s) != (frame.key, cam, frame.time_s):
        raise ValueError("native exposure returned a different state, camera or time")
    planes = {plane.channel: plane.data for plane in shot.planes}
    rgb = Image.frombytes("RGB", SENSOR, planes[Channel.RGB]).resize(PANEL, Image.Resampling.LANCZOS)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], "<u4") if audit else None
    return rgb, ids


def choose_detail_angle(session, appearance, rig, data, model, trace):
    """Compare full-fixture camera placements on actual grasp and insertion samples."""
    data.qpos[:] = trace.qpos[0]
    mj.mj_kinematics(model,data)
    across = rig.detail_angle(data)
    angles = dict.fromkeys(round(a,2) for a in (across, across+36., across-36., -62., 125.))
    scores = {a:[] for a in angles}
    telemetry_checks = []
    indices = []
    for phase in ("close","lift","insert"):
        candidates = np.flatnonzero(trace.phase==phase)
        if len(candidates):
            indices.append(int(candidates[2*len(candidates)//3]))
    if not indices:
        indices = [0,len(trace.time)-1]
    apparatus = appearance.tool_ids
    handle_ids = [apparatus[n] for n in ("assembly_pin","part_handle","part_shoulder")]
    for k in indices:
        data.qpos[:] = trace.qpos[k]
        mj.mj_kinematics(model,data)
        frame = exposure_frame(appearance,data,trace,k)
        for a in angles:
            cam = rig.detail(frame.key,data,a)
            shot = session.capture(frame,CaptureRequest(cam,RenderPath.RASTER,(Channel.INSTANCE_ID,)))
            if (shot.frame_key,shot.time_s,shot.camera) != (frame.key,frame.time_s,cam):
                raise ValueError("camera scouting returned a mismatched exposure")
            pixels = np.frombuffer(shot.planes[0].data,"<u4")
            pads = [int(np.count_nonzero(pixels==apparatus[n])) for n in PADS]
            workpiece = int(np.isin(pixels,handle_ids).sum())
            shaft = int(np.count_nonzero(pixels==apparatus["assembly_pin"]))
            # Prioritize angles that keep the workpiece and both fingers visible
            # across phases. The shaft can be legitimately occluded inside the hole.
            s = np.log1p(min(pads)) + .65*np.log1p(max(pads)) + np.log1p(workpiece) + .4*np.log1p(shaft)
            scores[a].append(s)
            telemetry_checks.append(dict(sample=k,azimuth_deg=a,pad_pixels=pads,
                                         workpiece_pixels=workpiece,shaft_pixels=shaft))
    return max(angles,key=lambda a:(min(scores[a]),sum(scores[a]))),telemetry_checks


def make_fonts(path):
    if path is None:
        return [ImageFont.load_default(size=s) for s in (26, 22, 19)]
    return [ImageFont.truetype(str(path), size=s) for s in (26, 22, 19)]


def caption(image, fonts, chinese, trace, k, speed, control, depth, matched_targets):
    draw = ImageDraw.Draw(image)
    large, body, small = fonts
    def pair(zh, en):
        return zh if chinese else en
    title = pair("双 Panda  ·  自由销抓取与插装", "Dual Panda   /   Free pin grasp and insertion")
    role = pair("闭指指令 · 任务记录", "Gripper commands  /  task record")
    if control:
        role = (pair("张指对照 · 同一关节目标", "Open fingers  /  matched arm targets") if matched_targets
                else pair("保持张指 · 对照记录", "Open fingers  /  control record"))
    draw.text((26, 27), title, font=large, fill="#f1f3ee", anchor="lm")
    alignment = (pair("初态 · 尚无求解载荷", "Initial pose · no solved forces") if k==0 else
                 pair("求解入步姿态 · 同步接触力", "Solver input pose and forces") if trace.contact_aligned else
                 pair("步末姿态 · 力来自前一步", "Endpoint pose · forces from previous step"))
    draw.text((WIDTH//2, 27), alignment, font=small, fill="#b5c9c2", anchor="mm")
    draw.text((WIDTH-26, 27), role, font=body, fill="#d0e4d9", anchor="rm")
    y = 61+PANEL[1]
    draw.line((WIDTH//2, 61, WIDTH//2, y), fill="#243a38", width=2)
    draw.text((26, y+23), pair("工装全景", "Full fixture"), font=body, fill="white", anchor="lm")
    draw.text((PANEL[0]+23, y+23), pair("右手双指与自由销", "Right fingers and free pin"),
              font=body, fill="white", anchor="lm")
    draw.text((WIDTH-26, y+23), f"t = {trace.time[k]:.3f} s   ·   {speed:.2f}×",
              font=body, fill="#ffc981", anchor="rm")
    phase = str(trace.phase[k])
    label = PHASES.get(phase.lower(), phase) if chinese else phase
    phase_title = pair("同程参考阶段   ", "Reference phase   ") if control else pair("指令阶段   ", "Command phase   ")
    draw.text((26, y+62), phase_title+label, font=body,
              fill="#d0e4d9", anchor="lm")
    forces = pair("两指垫法向力   ", "Pad normal loads   ")
    forces += f"{trace.pads[k,0]:.2f} / {trace.pads[k,1]:.2f} N" if k else "—"
    draw.text((PANEL[0]+23, y+62), forces, font=body, fill="#ffc981", anchor="lm")
    def load(arr):
        if k==0:
            return "—"
        return pair("未记录", "n/a") if arr is None else f"{arr[k]:.2f} N"
    values = pair("取件托架   ", "Pickup rest   ")+load(trace.support)
    values += pair("      落座   ", "      Seat   ")+load(trace.seat)
    draw.text((PANEL[0]+23, y+99), values, font=small, fill="#b5c9c2", anchor="lm")
    draw.text((26, y+99), pair("销尖相对孔口平面 · 沿孔轴   ", "Tip relative to entry plane, along hole axis   ")+f"{depth*1000:+.2f} mm",
              font=small, fill="#b5c9c2", anchor="lm")


def run(args):
    if not 1 <= args.fps <= 60 or not 2 <= args.seconds <= 60 or not 0 < args.exposure <= 16:
        raise ValueError("invalid fps, total duration or exposure")
    model = mj.MjModel.from_xml_path(str(args.scene.resolve()))
    xml_hash = digest(args.scene)
    traces = [read_trace(args.trace, model, xml_hash)]
    matched_targets = False
    if args.control_trace is not None:
        traces.append(read_trace(args.control_trace, model, xml_hash))
        if traces[0].sha256 == traces[1].sha256:
            raise ValueError("a control chapter needs its own trajectory")
        targets = traces[0].arm_targets
        matched_targets = bool(targets is not None and np.any(targets != targets[0])
            and np.array_equal(traces[0].published_time,traces[1].published_time)
            and np.array_equal(targets,traces[1].arm_targets))
    rig = Rig(model)
    args.out.mkdir(parents=True, exist_ok=False)
    appearance = build_appearance(model, args.out/"geometry")
    ids = appearance.tool_ids
    for name in (*PADS, "assembly_pin", "part_handle", "part_shoulder", "pickup_base",
                 "board_carrier", "board_edge_clamp", *(f"board_wall_{i}" for i in range(4))):
        if name not in ids:
            raise ValueError(f"physical fixture is missing its visual instance: {name}")
    lights = studio_lighting()
    renderer = StudioRenderer(library_path=args.library.resolve(), shader_directory=args.library.resolve().parent,
        studio=replace(lights, center_m=(0.,0.,.5), scale_m=1.05, panel_size_scale=.82,
                       panel_strength=.24, environment_map=replace(lights.environment_map, strength=.042)),
        exposure=args.exposure, panel_samples=args.panel_samples, device_index=args.device_index)
    fonts = make_fonts(args.font)
    total = round(args.fps*args.seconds)
    counts = [total] if len(traces)==1 else [total//2,total-total//2]
    if min(counts) < 2:
        raise ValueError("each chapter needs enough video frames to show both endpoints")
    journal = dict(model_sha256=xml_hash, fps=args.fps, size=[WIDTH, HEIGHT], sensor=SENSOR,
                   appearance=appearance.provenance, complete_fixture=True,
                   matched_arm_targets_verified=matched_targets, chapters=[])
    part = model.body("assembly_part").id
    data = mj.MjData(model)
    proc = None
    finished = False
    movie = args.out/"panda-grasp-insert.mp4"
    try:
        with closing(renderer.open_scene(appearance.scene)) as session:
            for chapter,(trace,count) in enumerate(zip(traces,counts,strict=True)):
                schedule = source_schedule(trace.time,count)
                checks = checks_for(trace,schedule)
                speed = float((trace.time[-1]-trace.time[0])*args.fps/(count-1))
                data.qpos[:] = trace.qpos[0]
                mj.mj_kinematics(model,data)
                if chapter==0:
                    if args.detail_azimuth is None:
                        az,scouting = choose_detail_angle(session,appearance,rig,data,model,trace)
                        journal["detail_camera_scouting"] = scouting
                    else:
                        az = args.detail_azimuth
                    data.qpos[:] = trace.qpos[0]
                    mj.mj_kinematics(model,data)
                chapter_log = dict(trajectory=trace.path.name, trajectory_sha256=trace.sha256,
                    sample_count=len(trace.time), role="open-finger control" if chapter else "task",
                    pose_time_source="contact_time_s" if trace.contact_aligned else "time_s",
                    pose_source="contact_qpos" if trace.contact_aligned else "qpos",
                    phase_source="contact_phase" if trace.contact_aligned else "phase",
                    force_alignment="recorded solver input" if trace.contact_aligned else "previous step; only endpoint pose available",
                    playback_rate=speed, detail_azimuth_deg=az, visible_id_checks=[], frames=[])
                base_positions = [data.xpos[model.body(side+"_panda_link0").id]
                                  for side in ("left", "right")]
                wide_target = np.mean(base_positions,axis=0)+[0.,0.,.38]
                for local,source in enumerate(schedule):
                    if args.stills_only and local not in checks:
                        continue
                    k = int(source)
                    data.qpos[:] = trace.qpos[k]
                    mj.mj_kinematics(model,data)
                    # Each chapter has its own epoch, including repeated source indices across chapters.
                    frame = exposure_frame(appearance,data,trace,k,chapter)
                    wide_cam = camera(frame.key,"overview",wide_target,2.04,-80.,17.,42.)
                    close_cam = rig.detail(frame.key,data,az)
                    audit = local in checks
                    wide,wide_ids = expose(session,frame,wide_cam,audit)
                    detail,close_ids = expose(session,frame,close_cam,audit)
                    img = Image.new("RGB",(WIDTH,HEIGHT),"#172b2b")
                    img.paste(wide,(0,61))
                    img.paste(detail,(PANEL[0],61))
                    tip_rel = data.site_xmat[rig.entry].reshape(3,3).T @ (data.site_xpos[rig.tip]-data.site_xpos[rig.entry])
                    caption(img,fonts,args.font is not None,trace,k,speed,bool(chapter),tip_rel[2],matched_targets)
                    chapter_log["frames"].append(dict(output_frame=sum(counts[:chapter])+local,
                        source_sample=k,time_s=float(trace.time[k]),phase=str(trace.phase[k]),
                        published_endpoint_time_s=float(trace.published_time[k]),
                        pad_forces_n=trace.pads[k].tolist() if k else None,part_position_m=data.xpos[part].tolist(),
                        tip_in_entry_frame_m=tip_rel.tolist(),
                        support_n=None if k==0 or trace.support is None else float(trace.support[k]),
                        seat_n=None if k==0 or trace.seat is None else float(trace.seat[k])))
                    if audit:
                        visible = {name:int((close_ids==ids[name]).sum()) for name in
                                   (*PADS,"assembly_pin","part_handle","part_shoulder","board_carrier","board_edge_clamp")}
                        visible["board"] = int(np.isin(close_ids,[ids[f"board_wall_{i}"] for i in range(4)]).sum())
                        overview = {s:int(np.isin(wide_ids,v).sum()) for s,v in appearance.robot_ids.items()}
                        if min(overview.values()) <= 0:
                            raise ValueError(f"full fixture framing lost an arm in sample {k}: {overview}")
                        chapter_log["visible_id_checks"].append(dict(source_sample=k, close=visible, overview=overview))
                        img.save(args.out/f"chapter-{chapter+1}-frame-{local:04d}.png")
                    if not args.stills_only:
                        if proc is None:
                            proc = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg","-v","error","-nostdin","-n",
                                "-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(args.fps),
                                "-i","pipe:0","-c:v","libx264","-preset","fast","-crf","16","-pix_fmt","yuv420p",
                                "-movflags","+faststart",str(movie)], stdin=subprocess.PIPE)
                        proc.stdin.write(img.tobytes())
                journal["chapters"].append(chapter_log)
                print(f"{trace.path.name}: {len(chapter_log['frames'])} exposures for each camera", flush=True)
            dev = session.diagnostics()
            journal["renderer"] = dict(device=dev["device"]["device_name"],shadow_mode=dev["shadow_mode"],
                                       msaa=dev["msaa"]["actual_color_samples"],light_grid=dev["area_light_grid"])
        finished = True
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
            finally:
                rc = proc.wait()
            if finished and rc != 0:
                raise RuntimeError(f"FFmpeg failed ({rc})")
    journal["movie_sha256"] = None if args.stills_only else digest(movie)
    (args.out/"film.json").write_text(json.dumps(journal,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene","trace","library","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--control-trace",type=Path,help="optional recorded open-finger negative control for the same XML")
    p.add_argument("--font",type=Path,help="Chinese font (TTF/TTC); captions otherwise use English")
    p.add_argument("--fps",type=int,default=24)
    p.add_argument("--seconds",type=float,default=16.,help="total film duration including optional control")
    p.add_argument("--device-index",type=int)
    p.add_argument("--exposure",type=float,default=6.)
    p.add_argument("--detail-azimuth",type=float,help="optional fixed detail azimuth in degrees")
    p.add_argument("--panel-samples",type=int,choices=(4,16,64),default=16)
    p.add_argument("--stills-only",action="store_true",help="render first, last and middle of each recorded command phase")
    run(p.parse_args())
