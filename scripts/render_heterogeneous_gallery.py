"""Render saved CUDA endpoints with native FK only and one offline GL context.

No physics stepping, interpolation, or q modification. The default camera is fixed
over the full trajectory. Optional tracking changes the camera only and retains
that fixed full-trajectory view in an inset.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] in ("PASS", "FUNCTIONAL_PASS"), "complete functional run required")
    require(manifest["mode"] == "cuda", "video requires actual CUDA trajectory")
    require(manifest["steps"] == 256 and manifest["h"] == 1 / 128, "fixed gallery timeline")
    worlds = manifest["worlds"]
    require(len(worlds) == len({r["world"] for r in worlds}) == 32, "32 unique worlds required")
    require(len({r["physical_sha256"] for r in worlds}) == 32, "32 physical structures required")
    state_file = root / manifest["state_file"]
    require(sha(state_file) == manifest["state_sha256"], "saved states SHA mismatch")
    with gzip.open(state_file, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    require(len(records) == 257, "all 257 actual endpoint frames required")
    order = [r["world"] for r in worlds]
    models, geometries, bounds = [], [], []
    for row in worlds:
        source = root / row["source_file"]
        require(source.resolve().is_relative_to(root.resolve()), "source path escapes run")
        require(sha(source) == row["source_sha256"], "source XML hash mismatch")
        model = mj.MjModel.from_xml_string(source.read_text(encoding="utf-8"))
        require(
            (model.nq, model.nv, model.na) == (row["nq"], row["nv"], row["na"]), "source dimensions"
        )
        models.append(model)
        geometries.append([])
        bounds.append([])
    for frame, record in enumerate(records):
        require(record["ordinal"] == frame, "missing/reordered endpoint")
        require([r["world"] for r in record["worlds"]] == order, "world identity order changed")
        for index, (state, model) in enumerate(zip(record["worlds"], models, strict=True)):
            require(state["tick"] == frame, "frame is not the actual tick")
            data = mj.MjData(model)
            for field, key in (("qpos", "q"), ("qvel", "v"), ("act", "act")):
                value = np.asarray(state[key], dtype=np.float64)
                require(
                    value.shape == getattr(data, field).shape and np.isfinite(value).all(),
                    "invalid state",
                )
                getattr(data, field)[:] = value
            data.time = state["time"]
            require(abs(data.time - frame / 128) < 2e-6, "saved time is not the common timeline")
            before = data.qpos.copy()
            mj.mj_forward(model, data)  # FK / graphics only, never mj_step.
            np.testing.assert_array_equal(data.qpos, before)
            geometries[index].append((data.geom_xpos.copy(), data.geom_xmat.copy()))
            # Bounding sphere for every primitive, conservative for capsule/box.
            radius = np.linalg.norm(model.geom_size, axis=1) + model.geom_size[:, 1]
            bounds[index].append(
                (data.geom_xpos - radius[:, None], data.geom_xpos + radius[:, None])
            )
    cameras = []
    for samples in bounds:
        lower = np.min([v[0].min(axis=0) for v in samples], axis=0)
        upper = np.max([v[1].max(axis=0) for v in samples], axis=0)
        camera = mj.MjvCamera()
        camera.lookat[:] = (lower + upper) / 2
        camera.distance = max(0.75, float(np.linalg.norm(upper - lower)) * 1.65)
        camera.azimuth, camera.elevation = 125, -24
        cameras.append(camera)
    return manifest, worlds, models, geometries, cameras


def filters(worlds, height, qualification, tracking_overview):
    # Fixed labels do not depend on simulation output; ffmpeg receives argv, no shell.
    font = "C\\:/Windows/Fonts/segoeui.ttf"
    bold = "C\\:/Windows/Fonts/segoeuib.ttf"
    camera_label = (
        "Tracking camera + fixed overview inset" if tracking_overview else "Fixed cameras"
    )
    layers = [
        f"drawtext=fontfile='{bold}':text='PHY MEO  /  HETEROGENEOUS CUDA':x=28:y=17:fontsize=30:fontcolor=white",
        f"drawtext=fontfile='{font}':text='{qualification}':x=1430:y=24:fontsize=19:fontcolor=0xe5b76d",
        f"drawtext=fontfile='{font}':text='32 models | 32 worlds | 1 GPU | MuJoCo-TYX | {camera_label}':x=28:y=58:fontsize=20:fontcolor=0xa4b4ca",
        f"drawtext=fontfile='{font}':text='Saved endpoint tick %{{n}} / 256    h = 1/128 s    2.00 s simulation':x=28:y={height - 64}:fontsize=19:fontcolor=white",
        f"drawtext=fontfile='{font}':text='Offline playback 0.234375x | source buckets, not a fused kernel | independent worlds, no cross-world contacts':x=28:y={height - 34}:fontsize=17:fontcolor=0x9bacc3",
    ]
    for i, row in enumerate(worlds):
        label = f"{i + 1:02d}  " + row["key"].removeprefix(f"gallery-{i:02d}-").replace("-", " ")
        label = label.replace("'", "").replace(":", " ")
        layers.append(
            f"drawtext=fontfile='{font}':text='{label}':x={i % 8 * 240 + 10}:y={100 + i // 8 * 240 + 213}:fontsize=13:fontcolor=white"
        )
    return ",".join(layers)


def render_geometry(renderer, data, model, positions, rotations, camera):
    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    for geom in range(model.ngeom):
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            int(model.geom_type[geom]),
            model.geom_size[geom],
            positions[geom],
            rotations[geom],
            model.geom_rgba[geom],
        )
        scene.ngeom += 1
    return renderer.render()


def render(root, output, ffmpeg, ffprobe, tracking_overview=False):
    require(not output.exists(), "fresh rendering directory required")
    manifest, worlds, models, geometries, cameras = read_run(root)
    output.mkdir(parents=True)
    blank = mj.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="240" offheight="208"/>'
        '<headlight ambient=".55 .55 .55" diffuse=".65 .65 .65" specular=".25 .25 .25"/>'
        '<rgba haze=".04 .06 .10 1"/></visual><worldbody>'
        '<light pos="0 -2 5" dir="0 0 -1" directional="true"/></worldbody></mujoco>'
    )
    data = mj.MjData(blank)
    mj.mj_forward(blank, data)
    width, height = 1920, 1140
    video = output / "phymeo-32-models-cuda.mp4"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        filters(worlds, height, manifest["qualification"], tracking_overview),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    frame_hash = hashlib.sha256()
    colors = ((57, 168, 222), (236, 155, 71), (87, 189, 127), (175, 115, 221))
    renderer = mj.Renderer(blank, height=208, width=240, max_geom=1024)
    try:
        with (output / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log)
            try:
                for frame in range(257):
                    canvas = np.empty((height, width, 3), dtype=np.uint8)
                    canvas[:] = (12, 19, 31)
                    for i, model in enumerate(models):
                        positions, rotations = geometries[i][frame]
                        camera = cameras[i]
                        if tracking_overview:
                            radius = np.linalg.norm(model.geom_size, axis=1) + model.geom_size[:, 1]
                            lower = np.min(positions - radius[:, None], axis=0)
                            upper = np.max(positions + radius[:, None], axis=0)
                            camera = mj.MjvCamera()
                            camera.lookat[:] = (lower + upper) / 2
                            camera.distance = max(0.35, float(np.linalg.norm(upper - lower)) * 1.3)
                            camera.azimuth, camera.elevation = 125, -24
                        tile = render_geometry(renderer, data, model, positions, rotations, camera)
                        if tracking_overview:
                            overview = render_geometry(
                                renderer, data, model, positions, rotations, cameras[i]
                            )[::3, ::3]
                            tile[4:76, 156:238] = (74, 91, 115)
                            tile[5:75, 157:237] = overview
                        x, y = (i % 8) * 240, 100 + (i // 8) * 240
                        canvas[y : y + 208, x : x + 240] = tile
                        canvas[y : y + 3, x + 3 : x + 237] = colors[i // 8]
                        canvas[y + 208 : y + 238, x : x + 240] = (23, 34, 50)
                    # Progress bar is saved tick count, not a synthesized motion frame.
                    canvas[height - 83 : height - 79, 28 : 28 + int((width - 56) * frame / 256)] = (
                        68,
                        207,
                        189,
                    )
                    raw = canvas.tobytes()
                    frame_hash.update(raw)
                    process.stdin.write(raw)
                process.stdin.close()
                require(process.wait() == 0, "ffmpeg encoding failed; inspect ffmpeg.log")
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.wait()
    finally:
        renderer.close()
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    require(
        int(stream["nb_read_frames"]) == 257
        and (stream["width"], stream["height"]) == (width, height),
        "encoded frames differ",
    )
    poster = output / "preview.png"
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-ss",
            "3",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(poster),
        ],
        check=True,
    )
    receipt = dict(
        schema="phymeo-gallery-render-v1",
        physics="saved CUDA accepted states only",
        native_use="offline FK and rendering; zero mj_step calls",
        frames=257,
        simulation_seconds=2,
        playback_fps=30,
        playback_speed=30 / 128,
        camera=(
            "per-frame camera fitted to actual geometry; fixed full-trajectory inset"
            if tracking_overview
            else "per-world full-trajectory fixed bounds; no recentering"
        ),
        tracking_overview=tracking_overview,
        source_manifest_sha256=sha(root / "manifest.json"),
        state_sha256=manifest["state_sha256"],
        raw_rgb_sha256=frame_hash.hexdigest(),
        video_sha256=sha(video),
        video_bytes=video.stat().st_size,
        probe=probe,
    )
    (output / "render-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--tracking-overview", action="store_true")
    args = parser.parse_args()
    render(args.input, args.output, args.ffmpeg, args.ffprobe, args.tracking_overview)
