"""Device acceptance for the recorded 720p Panda motion, jaw opening and shell lighting."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from preview_compact_rigid_history.state import frame_digest
from preview_lowres_filter._helpers import digest, read, save


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "artifacts/rendering/panda-surface-fix/sequence/rigid-history/frames"
SIZE = (1280, 720)
SHELL = (150, 151, 154)
JAW = ("panda_leftfinger", "panda_rightfinger")


def rgb(planes):
    from phymeo_rendering.outputs import Channel
    return np.frombuffer(planes[Channel.RGB], "u1").reshape(720, 1280, 3)


def geometry(planes):
    from phymeo_rendering.outputs import Channel
    return (np.frombuffer(planes[Channel.INSTANCE_ID], "<u4").reshape(720, 1280),
            np.frombuffer(planes[Channel.FORWARD_DEPTH], "<f4").reshape(720, 1280))


def sensor_hashes(planes):
    from phymeo_rendering.outputs import Channel
    return {ch.value: sha256(data).hexdigest() for ch, data in planes.items() if ch != Channel.RGB}


def rgb_difference(a, b, mask):
    if not mask.any():
        return dict(pixels=0)
    diff = a[mask].astype(float) - b[mask]
    return dict(pixels=int(mask.sum()), changed_pixels=int(np.any(diff != 0, axis=1).sum()),
                mean_delta=float(diff.mean()), mean_absolute=float(np.abs(diff).mean()),
                max_abs=float(np.abs(diff).max()),
                darker_pixels=int((diff.mean(axis=1) < 0).sum()))


def opening(inputs):
    gaps = []
    for i in range(300):
        positions = {n.node_id: n.world_pose.position_m for n in inputs[i][0].nodes}
        gaps.append(math.dist(positions[JAW[0]], positions[JAW[1]]))
    # Choose entirely from the recorded physical trajectory, before reading colors.
    increase = np.subtract(gaps[6:], gaps[:-6])
    i = int(np.argmax(increase))
    if increase[i] <= 1e-4:
        raise ValueError("recorded fingers have no opening over six presented intervals")
    return i, i + 6, gaps


def reveal(scene, inputs, start, end, first, last):
    from preview_compact_rigid_history.run_smoke import matrices, pose_matrix
    f0, c0 = inputs[start]; f1, c1 = inputs[end]
    old_ids, old_z = geometry(first); ids, z = geometry(last)
    finger_ids = [item.id for item in scene.instances if item.node_id in JAW]
    old_jaw = np.isin(old_ids, finger_ids)
    jaw = np.isin(ids, finger_ids)
    if not old_jaw.any() or not jaw.any():
        raise ValueError("opening geometry is outside the registered views")
    yy, xx = np.where(jaw)
    region = np.zeros_like(jaw)
    region[max(0, yy.min()-8):min(720, yy.max()+9), max(0, xx.min()-8):min(1280, xx.max()+9)] = True
    old_transform = matrices(scene, f0, np)
    new_transform = matrices(scene, f1, np)
    c0, c1 = c0.camera, c1.camera
    old_view = np.linalg.inv(pose_matrix(c0.world_pose, np))
    new_camera = pose_matrix(c1.world_pose, np)
    found = np.zeros_like(jaw)
    # Transport the newly visible physical surface to the old camera. Its old
    # depth must have been strictly behind a visible jaw at the projected point.
    for identity in np.unique(ids[region & ~jaw]):
        if identity == 0:
            continue
        y, x = np.where(region & (ids == identity))
        depth = z[y, x].astype(float)
        points = np.stack(((x+.5-c1.cx_px)*depth/c1.fx_px, -(y+.5-c1.cy_px)*depth/c1.fy_px,
                           -depth, np.ones_like(depth)))
        previous = old_view @ old_transform[int(identity)] @ np.linalg.inv(new_transform[int(identity)]) @ new_camera @ points
        projected_z = -previous[2]
        valid = projected_z > c0.near_m
        safe_z = np.where(valid, projected_z, 1.)
        u = c0.cx_px+c0.fx_px*previous[0]/safe_z
        v = c0.cy_px-c0.fy_px*previous[1]/safe_z
        # Exclude projections indistinguishable from raster pixel-boundary
        # ties with the precision of the uploaded transforms and center depth.
        valid &= (abs(u-np.rint(u)) > .002) & (abs(v-np.rint(v)) > .002)
        px, py = np.floor(u).astype(int), np.floor(v).astype(int)
        valid &= (px >= 0) & (px < 1280) & (py >= 0) & (py < 720)
        i = np.flatnonzero(valid)
        hidden = old_jaw[py[i], px[i]] & (old_z[py[i], px[i]] + .001 < projected_z[i])
        found[y[i[hidden]], x[i[hidden]]] = True
    return found, jaw, region


def near_jaw_surfaces(jaw, region, current):
    # Register a receiver band from geometry only: nearby depth and projected
    # proximity to a visible finger. No color or AO value selects these pixels.
    ids, depth = geometry(current)
    mask = np.zeros_like(jaw)
    padded_jaw = np.pad(jaw, 8, constant_values=False)
    padded_z = np.pad(depth, 8, constant_values=0.)
    finite = region & ~jaw & (ids != 0)
    for dy in (-8,-4,0,4,8):
        for dx in (-8,-4,0,4,8):
            local = np.s_[8+dy:8+dy+720, 8+dx:8+dx+1280]
            neighbor = finite & padded_jaw[local]
            mask[neighbor] |= abs(depth[neighbor] - padded_z[local][neighbor]) <= .03
    return mask


def board(path, panels, mask, padding=7):
    y, x = np.where(mask)
    if not len(y):
        return
    box = (max(0, int(x.min())-padding), max(0, int(y.min())-padding),
           min(1280, int(x.max())+padding+1), min(720, int(y.max())+padding+1))
    w, h = box[2]-box[0], box[3]-box[1]
    result = Image.new("RGB", (w * len(panels), h+25), "#19232d")
    draw = ImageDraw.Draw(result)
    for index, (label, image) in enumerate(panels):
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        result.paste(Image.fromarray(image).crop(box), (w*index, 25))
        draw.text((w*index+4, 6), label, fill="white")
    result.save(path)


def encode(raw, old, output):
    movie = output / "stable-motion.mp4"
    compare = output / "old-new.mp4"
    # The raw input is a sequential disk spool. Limit decoder/encoder buffering
    # for the double-width comparison instead of retaining a movie in RAM.
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
              "-filter_threads", "1", "-filter_complex_threads", "1", "-threads", "2"]
    movie_input = ["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
                   "-framerate", "30", "-i", str(raw)]
    options = ["-frames:v", "300", "-c:v", "libx264", "-threads", "2", "-x264-params", "rc-lookahead=6:sync-lookahead=0",
               "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart"]
    subprocess.run(common + movie_input + options + [str(movie)], check=True)
    subprocess.run(common + ["-framerate", "30", "-start_number", "0", "-i", str(old/"%04d.png")] +
                   movie_input + ["-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]"] +
                   options + [str(compare)], check=True)
    results = {}
    for path, size in ((movie, SIZE), (compare, (2560, 720))):
        p = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,duration",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
        stream = json.loads(p.stdout)["streams"][0]
        if ((stream["width"], stream["height"]) != size or stream["nb_read_frames"] != "300"
                or stream["avg_frame_rate"] != "30/1" or abs(float(stream["duration"])-10.) > .001):
            raise ValueError(f"unexpected encoded stream: {stream}")
        results[path.name] = dict(stream=stream, sha256=digest(path))
    return results


def shell_light_probe(args, scene_data, session, expose):
    from phymeo_rendering.scene import DirectionalLight
    from .scene import prepare_scene

    raw_root = args.output / "raw-scene"
    raw_root.mkdir(exist_ok=True)
    raw = prepare_scene(raw_root, args.source_inputs, smooth=False, camera="sequence", studio_look="soft-studio")
    directions = ((-.8, .0, -.6), (.8, .0, -.6))
    pictures, hashes, scores = {}, [], {}
    for label, (scene, studio, inputs, bindings, _) in (("authored", raw), ("smooth", scene_data)):
        # A white directional key at equal energy from two sides of the robot.
        # The soft panels are disabled only for this separate diagnostic.
        controlled = replace(studio, panel_strength=0., environment_map=None, environment_radiance=.04)
        with session(args.package, scene, controlled) as owner:
            for j, direction in enumerate(directions):
                owner.set_directional_lights((DirectionalLight(direction, (.65, .65, .65)),))
                p, _ = expose(owner, inputs, bindings, 60, True)
                hashes.append(sensor_hashes(p))
                ids, z = geometry(p); shell = np.isin(ids, SHELL)
                if not shell.any():
                    raise ValueError("no registered white shell at frame 60")
                image = rgb(p)
                pictures[label, j] = image
                # Curvature of displayed luminance along rows and columns on
                # continuous-depth interiors; the same ROI is used for all four.
                luma = image @ np.array([.2126, .7152, .0722])
                residuals = []
                for axis in (0, 1):
                    index = np.moveaxis(ids, axis, 0); depth = np.moveaxis(z, axis, 0)
                    light = np.moveaxis(luma, axis, 0); mid = np.moveaxis(shell, axis, 0)[1:-1]
                    inside = mid & (index[:-2] == index[1:-1]) & (index[2:] == index[1:-1])
                    d0, d1, d2 = depth[:-2][inside], depth[1:-1][inside], depth[2:][inside]
                    continuous = (abs(d1-d0) < .005*d1) & (abs(d1-d2) < .005*d1)
                    residuals.append(np.abs((light[:-2]-2*light[1:-1]+light[2:])[inside][continuous]))
                values = np.concatenate(residuals)
                scores[f"{label}_{j}"] = dict(interior_samples=int(values.size), curvature_p95=float(np.quantile(values, .95)),
                                              curvature_mean=float(values.mean()))
    board(args.output/"white-shell-lights.png", [(f"{k[0]} / key {k[1]+1}", v) for k,v in pictures.items()], shell)
    return dict(frame=60, shell_ids=SHELL, directions=directions, illumination=(.65,.65,.65),
                controls="panels off, constant environment 0.04, equal-energy directional key", scores=scores,
                sources=raw[-1], same_sensors=all(h == hashes[0] for h in hashes),
                smooth_light_response=rgb_difference(pictures["smooth",0], pictures["smooth",1], shell),
                smooth_vs_authored=[rgb_difference(pictures["smooth",j], pictures["authored",j], shell) for j in (0,1)])


def acceptance_checks(report):
    opening_data = report["opening"]
    shell = report["white_shell"]
    response = opening_data["ao_on_minus_off"]["finger_neighborhood"]
    return dict(
        freeze=report["freeze"]["unique"] == 1 and report["freeze"]["timed_match"],
        timing_30_fps=report["timing"]["fps"] >= 30 and report["timing"]["p95_ms"] <= 33.3,
        newly_visible=opening_data["current_vs_fresh_revealed"]["pixels"] > 0,
        sensor_equality=opening_data["sensor_equal"] and shell["same_sensors"],
        motion_freshness=opening_data["current_vs_fresh_entire"]["changed_pixels"] == 0
                         and all(opening_data["seed_sequence_rgb_equal"].values()),
        contact_darkening=response["changed_pixels"] > 0 and response["mean_delta"] < 0,
        light_response=shell["smooth_light_response"]["changed_pixels"] > 0,
        shell_smoother_in_both_lights=all(
            shell["scores"][f"smooth_{key}"]["curvature_p95"] < shell["scores"][f"authored_{key}"]["curvature_p95"]
            for key in (0, 1)),
    )


def validate(args):
    from .build import check_package
    from .runtime import activate, open_session
    _, policy = activate(args.package)
    _, control = check_package(args.unoccluded_package)
    if (policy["parent_seal_sha256"] != control["parent_seal_sha256"]
            or policy["stable_lighting"]["ao_radius_um"] <= 0
            or control["stable_lighting"]["ao_radius_um"] != 0):
        raise ValueError("AO comparison requires a positive radius and a zero-radius control from the same parent")
    from panda_motion_preview import verify_capture, verify_close
    from phymeo_rendering.outputs import Channel
    from .scene import prepare_scene

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"report.json").exists():
        raise FileExistsError("validation already published a report in this directory")
    report = dict(status="running", channels="timed owner RGB only", source_sha256=digest(args.source_inputs),
                  package=str(args.package.resolve()), unoccluded_package=str(args.unoccluded_package.resolve()),
                  old_frames=str(args.old_frames.resolve()), close_checks=[])
    def checkpoint():
        save(args.output/"report.json", report)

    @contextmanager
    def session(package, scene, studio):
        owner = open_session(package, scene, studio, device_index=args.device_index)
        try:
            yield owner
        finally:
            owner.close()
            device = owner.diagnostics()["device"]
            report["close_checks"].append(dict(name=device["device_name"], errors=device["validation_errors"],
                                               live_resources=device["live_resources"]))
            verify_close(device)

    def expose(owner, inputs, bindings, i, sensors=False):
        frame, request = inputs[i]
        request = replace(request, channels=tuple(Channel) if sensors else (Channel.RGB,))
        start = perf_counter()
        shot = owner.capture(frame, request, presentation=bindings[i])
        elapsed = (perf_counter()-start)*1000
        verify_capture(shot, frame, request)
        return {plane.channel: plane.data for plane in shot.planes}, elapsed

    checkpoint()
    try:
        movie_root = args.output/"scene"
        movie_root.mkdir(exist_ok=True)
        data = prepare_scene(movie_root, args.source_inputs, smooth=True, camera="sequence", studio_look="soft-studio")
        scene, studio, inputs, bindings, normals = data
        report["normal_preparation"] = normals
        old_input = read(args.old_frames.parents[1]/"inputs.json")
        original = read(args.source_inputs)
        if (old_input["camera"] != "sequence" or old_input["studio_look"] != "soft-studio"
                or old_input["trace_sha256"] != original["trace_sha256"]
                or old_input["studio"] != original["studio"]):
            raise ValueError("comparison frames require the identical registered sequence, lighting and recording")
        by_index = {r["pose_index"]: r for r in old_input["actual_full_inputs"]}
        for i in range(300):
            frame, request = inputs[i]
            original_frame, original_camera = by_index[i]["frame"], by_index[i]["camera"]
            # Geometry-derived shading normals give the scene a new identity.
            # Compare every other field, including the complete source tick,
            # poses and camera registration, with the archived exposure.
            archived_frame = replace(frame, key=replace(frame.key, scene_key=original_frame["key"]["scene_key"]))
            archived_camera = replace(request.camera, frame_key=replace(request.camera.frame_key,
                                      scene_key=original_camera["frame_key"]["scene_key"]))
            if (frame.key.scene_key != scene.key or request.camera.frame_key != frame.key
                    or frame_digest(archived_frame) != frame_digest(original_frame)
                    or frame_digest(archived_camera) != frame_digest(original_camera)
                    or (request.camera.width, request.camera.height) != SIZE
                    or not (args.old_frames/f"{i:04}.png").is_file()):
                raise ValueError(f"comparison frame binding or existing PNG differs at {i}")

        i0, i1, gaps = opening(inputs)
        raw_video = args.output / "stable-motion.rgb24"
        times, publication_hashes = [], []
        with session(args.package, scene, studio) as owner, raw_video.open("xb") as stream:
            for i in range(10):
                expose(owner, inputs, bindings, i)
            owner.reset_history()
            for i in range(300):
                planes, ms = expose(owner, inputs, bindings, i)
                times.append(ms); frame = planes[Channel.RGB]
                stream.write(frame); publication_hashes.append(sha256(frame).hexdigest())
            report["device"] = owner.diagnostics()["device"]
        mean = float(np.mean(times))
        report["timing"] = dict(frame_ms=times, mean_ms=mean, fps=1000./mean, p50_ms=float(np.median(times)),
                                p95_ms=sorted(times)[284], under_33_3_ms=sum(ms <= 33.3 for ms in times),
                                measured="single warm round of 300 public synchronous 720p RGB captures; hashing, raw spool and encoding outside timers")
        report["rgb_sha256"] = publication_hashes
        checkpoint()
        report["videos"] = encode(raw_video, args.old_frames, args.output)
        raw_video.unlink()
        checkpoint()
        with session(args.package, scene, studio) as owner:
            hashes = [sha256(expose(owner,inputs,bindings,60)[0][Channel.RGB]).hexdigest() for _ in range(32)]
            report["freeze"] = dict(frame=60, captures=32, rgb_sha256=hashes, unique=len(set(hashes)),
                                    timed_match=hashes[0] == publication_hashes[60])
        checkpoint()

        observed = {}
        with session(args.package, scene, studio) as owner:
            for i in range(i1+1):
                p, _ = expose(owner, inputs, bindings, i, i in (i0,i1))
                if i in (i0,i1):
                    observed[i] = p
        with session(args.package, scene, studio) as owner:
            fresh, _ = expose(owner, inputs, bindings, i1, True)
        noao = {}
        with session(args.unoccluded_package, scene, studio) as owner:
            for i in (i0, i1):
                owner.reset_history()
                noao[i], _ = expose(owner, inputs, bindings, i, True)
        visible, jaw, region = reveal(scene, inputs, i0, i1, observed[i0], observed[i1])
        ids, _ = geometry(observed[i1])
        whole = np.ones_like(jaw)
        contact = near_jaw_surfaces(jaw, region, observed[i1])
        compare = {label: rgb_difference(rgb(observed[i1]), rgb(noao[i1]), mask) for label,mask in
                   (("whole",whole), ("fingers",jaw), ("finger_neighborhood",region), ("near_finger_receivers",contact),
                    ("revealed",visible), ("white_shell",np.isin(ids,SHELL)))}
        compare["fingers_before_opening"] = rgb_difference(rgb(observed[i0]), rgb(noao[i0]), np.isin(geometry(observed[i0])[0],
                                           [item.id for item in scene.instances if item.node_id in JAW]))
        report["opening"] = dict(frames=[i0,i1], jaw_separation_m=gaps[i0:i1+1], trajectory_separation_m=gaps,
            method="largest six-interval positive jaw separation change, newly visible surface projected by its own recorded rigid motion into prior camera; prior finger at least 1 mm nearer",
            current_vs_fresh_entire=rgb_difference(rgb(observed[i1]),rgb(fresh),whole),
            current_vs_fresh_revealed=rgb_difference(rgb(observed[i1]),rgb(fresh),visible),
            sensor_equal=(sensor_hashes(fresh) == sensor_hashes(observed[i1]) and
                          all(sensor_hashes(noao[i]) == sensor_hashes(observed[i]) for i in (i0,i1))),
            seed_sequence_rgb_equal={str(i): sha256(p[Channel.RGB]).hexdigest()==publication_hashes[i] for i,p in observed.items()},
            receiver_ids={str(int(identity)):int((ids[visible] == identity).sum()) for identity in np.unique(ids[visible])},
            ao_on_minus_off=compare)
        board(args.output/"opening.png", [("Before",rgb(observed[i0])),("Current / AO",rgb(observed[i1])),
              ("Fresh / AO",rgb(fresh)),("Current / no AO",rgb(noao[i1])),("Newly visible",visible.astype("u1")*255)], region)
        checkpoint()
        report["white_shell"] = shell_light_probe(args,data,session,expose)
        report["checks"] = acceptance_checks(report)
        if not all(report["checks"].values()):
            raise RuntimeError("one or more measured gates remain open")
        report["status"]="complete"
        checkpoint()
    except Exception as error:
        report["error"]=str(error)
        if report["status"] != "complete":
            report["status"]="partial"
        checkpoint()
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("package","unoccluded-package","source-inputs","output"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--old-frames",type=Path,default=OLD)
    cli.add_argument("--device-index",type=int,default=1)
    validate(cli.parse_args())
