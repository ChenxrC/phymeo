"""Replay eight saved CUDA worlds; diagnostic control is not a trained policy.

Only saved endpoints are drawn. Native kinematics supplies geom/site transforms;
there is no integration, motion interpolation, or modification of saved q.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import textwrap
from contextlib import suppress
from pathlib import Path

import mujoco as mj
import numpy as np


WIDTH, HEIGHT, FPS = 1920, 1080, 30
TILE_WIDTH, TILE_HEIGHT, ROW_HEIGHT = 480, 360, 448


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(root):
    root = root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == "phymeo-manipulation-control-v1", "manifest schema")
    require(manifest["mode"] == "cuda" and manifest["h"] == 0.001, "actual CUDA .001 s input required")
    require(manifest["controller"] == "diagnostic-native-pd", "diagnostic controller required")
    require(manifest["status"] in ("COMPLETE", "FAILED_PREFIX"), "closed trajectory required")
    steps, requested = manifest["steps"], manifest["requested_steps"]
    require(type(steps) is int and type(requested) is int and 0 <= steps <= requested,
            "invalid accepted/requested counts")
    require(manifest["status"] != "COMPLETE" or steps == requested, "incomplete run marked complete")
    worlds = manifest["worlds"]
    require(len(worlds) == 8 and all(type(row["world"]) is int for row in worlds), "eight worlds required")
    order = [row["world"] for row in worlds]
    require(len(set(order)) == 8 and len({row["key"] for row in worlds}) == 8, "duplicate world/model identity")
    require(manifest["state_file"] == "states.jsonl.gz", "state filename")
    state_path = (root / manifest["state_file"]).resolve(strict=True)
    require(state_path.is_relative_to(root) and sha(state_path) == manifest["state_sha256"], "state path/hash")
    with gzip.open(state_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    expected_steps = list(range(0, steps + 1, 10)) + ([steps] if steps % 10 else [])
    require(len(records) == len(expected_steps), "missing or extra saved endpoints")
    models, data, bounds, cameras = [], [], [], []
    for row in worlds:
        require(type(row["task_success"]) is bool, "task success must be independent boolean metadata")
        for name in ("key", "robot_kind", "goal_kind", "instruction_en"):
            require(type(row[name]) is str and row[name].strip(), "invalid world label")
        source = (root / row["source_file"]).resolve(strict=True)
        require(source.is_relative_to(root) and sha(source) == row["source_sha256"], "source path/hash")
        model = mj.MjModel.from_xml_string(source.read_text(encoding="utf-8"))
        require(all(type(row[name]) is int for name in ("nq", "nv", "na")), "native dimensions must be integers")
        require((model.nq, model.nv, model.na) == (row["nq"], row["nv"], row["na"]), "source dimensions")
        require(model.opt.timestep == manifest["h"], "source timestep mismatch")
        require(model.nsite > 0, "all original target sites must be present")
        models.append(model)
        data.append(mj.MjData(model))
        bounds.append([])
    geometries = []
    for ordinal, (record, step) in enumerate(zip(records, expected_steps, strict=True)):
        require(type(record["ordinal"]) is int and record["ordinal"] == ordinal, "endpoint ordinal")
        require(type(record["step"]) is int and record["step"] == step, "endpoint sampling/timeline")
        require(all(type(state["world"]) is int for state in record["worlds"]) and [state["world"] for state in record["worlds"]] == order, "world order changed")
        frame = []
        for index, (state, model, datum) in enumerate(zip(record["worlds"], models, data, strict=True)):
            require(type(state["tick"]) is int and state["tick"] == step, "tick is not accepted step")
            require(type(state["success"]) is bool and type(state["stage"]) is str and state["stage"],
                    "invalid recorded controller/evaluation state")
            require(type(state["time"]) in (int, float) and np.isfinite(state["time"])
                    and abs(state["time"] - step * manifest["h"]) <= 2e-6, "physical clock mismatch")
            for target, name in (("qpos", "q"), ("qvel", "v"), ("act", "act")):
                value = np.asarray(state[name], dtype=np.float64)
                require(value.shape == getattr(datum, target).shape and np.isfinite(value).all(), "invalid state")
                getattr(datum, target)[:] = value
            datum.time = state["time"]
            before = datum.qpos.copy()
            mj.mj_kinematics(model, datum)
            require(before.tobytes() == datum.qpos.tobytes(), "kinematics modified saved q")
            frame.append((datum.geom_xpos.copy(), datum.geom_xmat.copy(),
                          datum.site_xpos.copy(), datum.site_xmat.copy()))
            # Infinite support remains visible, but must not dominate camera framing.
            finite = model.geom_type != int(mj.mjtGeom.mjGEOM_PLANE)
            centers = np.concatenate((datum.geom_xpos[finite], datum.site_xpos))
            radius = np.r_[np.linalg.norm(model.geom_size[finite], axis=1) + model.geom_size[finite, 1],
                           np.linalg.norm(model.site_size, axis=1) + model.site_size[:, 1]]
            bounds[index].append((np.min(centers - radius[:, None], axis=0),
                                  np.max(centers + radius[:, None], axis=0)))
        geometries.append(frame)
    for samples in bounds:
        lower, upper = np.min([pair[0] for pair in samples], axis=0), np.max([pair[1] for pair in samples], axis=0)
        camera = mj.MjvCamera()
        camera.lookat[:] = (lower + upper) / 2
        camera.distance = max(0.75, float(np.linalg.norm(upper - lower)) * 1.65)
        camera.azimuth, camera.elevation = 125, -32
        cameras.append(camera)
    return manifest, records, models, geometries, cameras


def filters(output, manifest, records, speed):
    font = "C\\:/Windows/Fonts/segoeui.ttf"
    layers = []

    def label(text, x, y, size, color="white", enable=None):
        name = f"label-{len(layers):03d}.txt"
        (output / name).write_text(text, encoding="utf-8")
        value = (f"drawtext=fontfile='{font}':textfile='{name}':expansion=none:"
                 f"x={x}:y={y}:fontsize={size}:fontcolor={color}:line_spacing=3")
        layers.append(value + (f":enable='{enable}'" if enable else ""))

    label("8 heterogeneous GPU environments", 24, 16, 30)
    label("Diagnostic controller - NOT a trained policy", 24, 57, 21, "0xe5b76d")
    label("FAILED PREFIX" if manifest["status"] == "FAILED_PREFIX" else "COMPLETE PHYSICS WINDOW",
          1440, 25, 18, "0xe5b76d")
    seconds = manifest["steps"] * manifest["h"]
    label(f"1 GPU | MuJoCo-TYX | h = 0.001 s | {manifest['steps']} / {manifest['requested_steps']} accepted steps",
          24, 1015, 19)
    label(f"Actual simulation {seconds:.3f} s | Playback {len(records) / FPS:.3f} s | Average speed {speed:.4f}x | Saved endpoints only",
          24, 1048, 17, "0xa4b4ca")
    for index, row in enumerate(manifest["worlds"]):
        x, y = index % 4 * TILE_WIDTH + 10, 100 + index // 4 * ROW_HEIGHT
        label(f"{index + 1:02d}  {row['robot_kind']}  |  {row['goal_kind']}  |  world {row['world']}", x, y + 363, 17)
        label(textwrap.fill(row["instruction_en"], width=61), x, y + 386, 14, "0xa4b4ca")
        start = 0
        for stop in range(1, len(records) + 1):
            state = records[start]["worlds"][index]
            tag = (state["stage"], state["success"])
            if stop < len(records):
                next_state = records[stop]["worlds"][index]
                if (next_state["stage"], next_state["success"]) == tag:
                    continue
            label(f"Stage: {tag[0]} | Recorded goal: {'MET' if tag[1] else 'not met'}", x, y + 423, 13,
                  "0x75d5b6" if tag[1] else "0xa4b4ca", f"between(n,{start},{stop - 1})")
            start = stop
    return ",".join(layers)


def render_geometry(renderer, blank_data, model, geometry, camera):
    renderer.update_scene(blank_data, camera=camera)
    scene = renderer.scene
    geom_pos, geom_rot, site_pos, site_rot = geometry
    for kinds, sizes, colors, positions, rotations in (
        (model.geom_type, model.geom_size, model.geom_rgba, geom_pos, geom_rot),
        (model.site_type, model.site_size, model.site_rgba, site_pos, site_rot),
    ):
        for kind, size, color, position, rotation in zip(kinds, sizes, colors, positions, rotations, strict=True):
            mj.mjv_initGeom(scene.geoms[scene.ngeom], int(kind), size, position, rotation, color)
            scene.ngeom += 1
    return renderer.render()


def render(root, output, ffmpeg, ffprobe):
    root, output = root.resolve(strict=True), output.resolve()
    require(not output.exists(), "fresh output directory required")
    manifest, records, models, geometries, cameras = read_run(root)
    output.mkdir(parents=True)
    duration = len(records) / FPS
    speed = manifest["steps"] * manifest["h"] / duration
    blank = mj.MjModel.from_xml_string(
        f'<mujoco><visual><global offwidth="{TILE_WIDTH}" offheight="{TILE_HEIGHT}"/>'
        '<headlight ambient=".55 .55 .55" diffuse=".65 .65 .65" specular=".25 .25 .25"/>'
        '<rgba haze=".04 .06 .10 1"/></visual><worldbody>'
        '<light pos="0 -2 5" dir="0 0 -1" directional="true"/></worldbody></mujoco>'
    )
    datum = mj.MjData(blank)
    mj.mj_kinematics(blank, datum)
    mj.mj_camlight(blank, datum)
    video = output / "phymeo-eight-manipulation-cuda.mp4"
    command = [str(ffmpeg.resolve(strict=True)), "-hide_banner", "-loglevel", "warning", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-vf", filters(output, manifest, records, speed),
               "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(video)]
    raw_hash = hashlib.sha256()
    renderer = mj.Renderer(blank, height=TILE_HEIGHT, width=TILE_WIDTH,
                           max_geom=max(model.ngeom + model.nsite for model in models) + 16)
    try:
        with (output / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log, cwd=output)
            try:
                for ordinal, geometry in enumerate(geometries):
                    canvas = np.full((HEIGHT, WIDTH, 3), (12, 19, 31), dtype=np.uint8)
                    for index, model in enumerate(models):
                        tile = render_geometry(renderer, datum, model, geometry[index], cameras[index])
                        x, y = index % 4 * TILE_WIDTH, 100 + index // 4 * ROW_HEIGHT
                        canvas[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = tile
                    progress = records[ordinal]["step"] / max(1, manifest["steps"])
                    canvas[1000:1004, 24:24 + int((WIDTH - 48) * progress)] = (68, 207, 189)
                    raw = canvas.tobytes()
                    raw_hash.update(raw)
                    process.stdin.write(raw)
                process.stdin.close()
                require(process.wait() == 0, "encoding failed; inspect ffmpeg.log")
            finally:
                with suppress(BrokenPipeError):
                    process.stdin.close()
                if process.poll() is None:
                    process.wait()
    finally:
        renderer.close()
    result = subprocess.run([str(ffprobe.resolve(strict=True)), "-v", "error", "-count_frames",
                             "-select_streams", "v:0", "-show_entries",
                             "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(video)],
                            check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    require(int(stream["nb_read_frames"]) == len(records) and stream["r_frame_rate"] == "30/1"
            and (stream["width"], stream["height"]) == (WIDTH, HEIGHT) and abs(float(stream["duration"]) - duration) <= 1 / FPS, "encoded frame mismatch")
    poster = output / "preview.png"
    subprocess.run([str(ffmpeg.resolve(strict=True)), "-hide_banner", "-loglevel", "error", "-n",
                    "-i", str(video), "-vf", f"select=eq(n\\,{len(records) // 2})", "-frames:v", "1", str(poster)], check=True)
    receipt = dict(schema="phymeo-manipulation-render-v1", status=manifest["status"],
                   controller=manifest["controller"], trained_policy=False, physics="saved CUDA endpoints only",
                   native_use="offline kinematics and camera/light transforms; no mj_step or mj_forward", frames=len(records),
                   frame_steps=[record["step"] for record in records],
                   frame_world_times=[[state["time"] for state in record["worlds"]] for record in records],
                   simulation_seconds=manifest["steps"] * manifest["h"], playback_seconds=duration,
                   playback_fps=FPS, average_playback_speed=speed, task_success=[row["task_success"] for row in manifest["worlds"]],
                   camera="fixed full-trajectory finite geoms and all target sites; support plane still drawn",
                   renderer_sha256=sha(Path(__file__)), world_keys=[row["world"] for row in manifest["worlds"]], source_manifest_sha256=sha(root / "manifest.json"), state_sha256=manifest["state_sha256"],
                   source_sha256=[row["source_sha256"] for row in manifest["worlds"]],
                   raw_rgb_sha256=raw_hash.hexdigest(), video_sha256=sha(video), preview_sha256=sha(poster),
                   video_bytes=video.stat().st_size, probe=probe)
    (output / "render-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "ffmpeg", "ffprobe"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.output, args.ffmpeg, args.ffprobe)
