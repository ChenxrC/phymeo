"""Photograph the exact states of an existing dual-Panda assembly replay.

Supply the original replay.json and its two XML/NPZ pairs in chapter order.
The archive binds the simulation clock and original inputs to every exposure.
"""
import argparse
from contextlib import ExitStack, closing
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _panda_assembly_appearance import build_appearance
from panda_assembly_replay import assembly_motion, axes, digest, load_episode
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.photography import StudioRenderer, studio_lighting
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import Pose


WIDTH, HEIGHT = 1920, 1080
PANEL = (960, 912)
SENSOR = (1440, 1368)


def camera(key, name, target, distance, azimuth, elevation, fov):
    rotation = axes(azimuth, elevation)
    eye = target + distance * rotation[2]
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, rotation.T.flatten())
    pose = Pose(tuple(eye), tuple(quat[[1, 2, 3, 0]]))
    width, height = SENSOR
    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
    return Camera(name, key, pose, width, height, focal, focal, width/2, height/2, .004, 24.)


def expose(session, frame, cam, check):
    channels = (Channel.RGB, Channel.INSTANCE_ID) if check else (Channel.RGB,)
    shot = session.capture(frame, CaptureRequest(cam, RenderPath.RASTER, channels))
    if (shot.frame_key, shot.time_s, shot.camera) != (frame.key, frame.time_s, cam):
        raise ValueError("Vulkan returned a different state or camera")
    planes = {plane.channel: plane.data for plane in shot.planes}
    image = Image.frombytes("RGB", SENSOR, planes[Channel.RGB]).resize(PANEL, Image.Resampling.LANCZOS)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], "<u4") if check else None
    return image, ids


def caption(frame, font, chapter, screw, index, speed, now, relative, turn, feed):
    chinese = font is not None
    def face(size):
        return ImageFont.truetype(str(font), size) if chinese else ImageFont.load_default(size=size)
    top, body = face(31), face(26)
    draw = ImageDraw.Draw(frame)
    title = ("已啮合旋拧" if screw else "插销入孔") if chinese else (
        "Pre-engaged screw" if screw else "Pin insertion")
    draw.text((32, 33), f"0{chapter}   /   {title}", font=top, fill="#f2f4ef", anchor="lm")
    draw.text((WIDTH-32, 33), f"PHYMEO    /    DUAL PANDA      ·      {speed:g}×", font=body,
              fill="#bcd3cd", anchor="rm")
    draw.line((WIDTH//2, 72, WIDTH//2, 72+PANEL[1]), fill="#35534e", width=2)
    labels = (("原装夹全景", "孔口剖视 · 隐去支杆") if chinese else ("Full fixture", "Cutaway · carrier hidden"))
    for x, label in zip((32, WIDTH//2+32), labels):
        draw.text((x, HEIGHT-52), label, font=body, fill="#cee2da", anchor="lm")
    draw.text((WIDTH//2-30, HEIGHT-52), f"t = {now:.3f} s", font=body, fill="white", anchor="rm")
    if screw:
        value = (f"{np.rad2deg(turn[index]):+.1f}°    /    {1e6*feed[index]:+.0f} µm")
    else:
        value = f"{relative[index,2]*1000:+.2f} mm"
    label = ("转角 / 进给" if screw else "销尖深度") if chinese else ("Turn / feed" if screw else "Tip depth")
    draw.text((WIDTH-32, HEIGHT-52), f"{label}    {value}", font=body, fill="#ffc981", anchor="rm")


def run(args):
    book = json.loads(args.reference.read_bytes())
    if len(book["episodes"]) != len(args.episode):
        raise ValueError("episode count differs from original film")
    args.out.mkdir(parents=True, exist_ok=False)
    journal = dict(reference_sha256=digest(args.reference), fps=book["fps"], size=[WIDTH, HEIGHT],
                   camera_raster=list(SENSOR), panel_output=list(PANEL),
                   sample_clock="same archived qpos index and time as original film", chapters=[])
    lighting = studio_lighting()
    renderer = StudioRenderer(library_path=args.library.resolve(), shader_directory=args.library.resolve().parent,
                              studio=replace(lighting, center_m=(0., 0., .50), scale_m=1.05,
                                             panel_size_scale=.82, panel_strength=.24,
                                             environment_map=replace(lighting.environment_map, strength=.042)),
                              exposure=args.exposure, panel_samples=64, device_index=args.device_index)
    proc = None
    frame_count = 0
    try:
        for ordinal, (inputs, old) in enumerate(zip(args.episode, book["episodes"], strict=True), 1):
            model, q, t, evidence = load_episode(*inputs)
            if not evidence["trajectory_binds_xml"] or any(evidence[k] != old[k] for k in
                    ("trajectory_sha256", "xml_sha256", "recorded_samples", "nq")):
                raise ValueError("inputs differ from recorded S4000 episode")
            data = mj.MjData(model)
            entry = model.site("assembly_entry").id
            rel, turn, feed = assembly_motion(model, data, q, entry, model.site("assembly_pin_tip").id,
                                              model.body("assembly_part").id)
            appearance = build_appearance(model, args.out/"geometry")
            detail_ids = appearance.tool_ids
            carrier = model.geom("board_carrier").id
            if model.geom_contype[carrier] or model.geom_conaffinity[carrier]:
                raise ValueError("cutaway is defined for the recorded noncolliding carrier only")
            cutaway = replace(appearance.scene, key=appearance.scene.key+"-cutaway",
                              instances=tuple(x for x in appearance.scene.instances
                                              if x.id != detail_ids["board_carrier"]))
            with ExitStack() as stack:
                session = stack.enter_context(closing(renderer.open_scene(appearance.scene)))
                detail_session = stack.enter_context(closing(renderer.open_scene(cutaway)))
                screw = old["model_name"] == "dual_panda_screw_proxy"
                checks = {0, len(old["frames"])//2, len(old["frames"])-1}
                rows = []
                last, previous = None, None
                for i, saved in enumerate(old["frames"]):
                    if args.stills_only and i not in checks:
                        continue
                    k = saved["source_sample"]
                    if (k < 0 or k >= len(q) or saved["sim_time_s"] != t[k] or
                        not np.allclose(saved["tip_in_entry_frame_m"], rel[k], atol=1e-9, rtol=0) or
                        abs(saved["turn_from_first_rad"]-turn[k]) > 1e-8 or
                        abs(saved["axial_feed_from_first_m"]-feed[k]) > 1e-9):
                        raise ValueError("archived frame does not match measured workpiece state")
                    if k != last or i in checks:
                        data.qpos[:] = q[k]
                        mj.mj_kinematics(model, data)
                        frame = appearance.frame(data, k, t[k])
                        wide_cam = camera(frame.key, "overview", np.array([0.,0.,.49]), 1.95,-81.,17.,40.)
                        detail_frame = replace(frame, key=replace(frame.key, scene_key=cutaway.key))
                        close_cam = camera(detail_frame.key, "detail", data.site_xpos[entry]+[0.,0.,.016],
                                           .238, -62., 33., 37.)
                        wide, overview_ids = expose(session, frame, wide_cam, i in checks)
                        close, ids = expose(detail_session, detail_frame, close_cam, i in checks)
                        previous = Image.new("RGB", (WIDTH,HEIGHT), "#172b2b")
                        previous.paste(wide,(0,72))
                        previous.paste(close,(PANEL[0],72))
                        if i in checks:
                            visible = {key:int(np.isin(overview_ids, values).sum()) for key, values
                                       in appearance.robot_ids.items()}
                            visible["pin"] = int((ids==detail_ids["assembly_pin"]).sum())
                            visible["board"] = int(np.isin(ids, [v for n,v in detail_ids.items()
                                                                  if n.startswith("board_wall_")]).sum())
                            if min(visible.values()) <= 0:
                                raise ValueError(f"camera obscures the task: {visible}")
                            rows.append(dict(frame=i, sample=k, time_s=float(t[k]), visible_pixels=visible))
                        last = k
                    image = previous.copy()
                    caption(image,args.font,ordinal,screw,k,book["playback_rate"],t[k],rel,turn,feed)
                    if i in checks:
                        image.save(args.out/f"episode-{ordinal}-{i:03d}.png")
                    if not args.stills_only:
                        if proc is None:
                            proc = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg", "-v","error","-nostdin","-n",
                                "-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(book["fps"]),
                                "-i","pipe:0","-c:v","libx264","-preset","fast","-crf","16",
                                "-pix_fmt","yuv420p","-movflags","+faststart",str(args.out/"assembly-studio.mp4")],
                                stdin=subprocess.PIPE)
                        proc.stdin.write(image.tobytes())
                    frame_count += 1
                d = session.diagnostics()
                journal["chapters"].append(dict(inputs=evidence, appearance=appearance.provenance, checks=rows,
                        device=d["device"]["device_name"], msaa=d["msaa"]["actual_color_samples"],
                        area_light_grid=d["area_light_grid"], shadows=d["shadow_mode"],
                        detail_view="Cutaway hides only board_carrier (contype=conaffinity=0)",
                        exposure=args.exposure))
                print(f"Chapter {ordinal} complete", flush=True)
    finally:
        if proc is not None:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("FFmpeg encoding failed")
    journal["frames"] = frame_count
    if not args.stills_only:
        journal["movie_sha256"] = digest(args.out/"assembly-studio.mp4")
    (args.out/"studio.json").write_text(json.dumps(journal, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", nargs=2, type=Path, action="append", required=True, metavar=("XML","NPZ"))
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--device-index", type=int)
    p.add_argument("--font", type=Path, help="Chinese font file for Chinese titles; defaults to English")
    p.add_argument("--exposure", type=float, default=6.)
    p.add_argument("--stills-only", action="store_true")
    run(p.parse_args())
