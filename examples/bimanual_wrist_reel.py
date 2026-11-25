"""Film complete, synchronized successful episodes at source speed or in a 30–60 s cut.

Run inside the Phymeo rendering environment, for example:
    python examples/bimanual_wrist_reel.py --evaluation runs/carry/episodes/holdout \
        --output movies/carry.mp4 --cache-dir training-cache/camera --seconds 40

Alternatively use --episode path/to/123-trained.json (repeat for multiple episodes).
The original XML and NPZ must accompany each JSON under the same basename.
For a complete 16 ms decision trace at physical speed, pass --real-time --fps 64.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco as mj
import numpy as np

from bimanual_wrist_render import LEGACY_MARKER_LAYOUT, _digest, _ffmpeg, record


_TOOLS = {"single-pad": ("pad",), "twin-pad": ("pad_a", "pad_b"),
          "four-contact": ("pad_a", "pad_b", "pad_c", "pad_d")}


def _physical_pads(xml, receipt):
    model = mj.MjModel.from_xml_path(str(xml))
    names = {model.geom(i).name for i in range(model.ngeom)}
    present = {name for name in names if name and name.startswith(("left_pad", "right_pad"))}
    for profile, suffixes in _TOOLS.items():
        pads = {f"{side}_{suffix}": side for side in ("left", "right") for suffix in suffixes}
        if set(pads) == present:
            break
    else:
        raise ValueError(f"unsupported or incomplete wrist geometry in {xml}: {sorted(present)}")
    for metadata in (receipt, receipt["final_metrics"]):
        if metadata.get("tool_profile", profile) != profile:
            raise ValueError(f"recorded tool profile disagrees with the physical pads in {xml}")
    ids = {}
    for name, side in pads.items():
        geom = model.geom(name).id
        wrist = model.site(f"{side}_tool").id
        if (model.geom_bodyid[geom] != model.site_bodyid[wrist]
                or model.geom_type[geom] != mj.mjtGeom.mjGEOM_SPHERE
                or not (model.geom_contype[geom] or model.geom_conaffinity[geom])):
            raise ValueError(f"{name} is not a physical spherical pad on the {side} tool in {xml}")
        ids[name] = int(geom) + 1  # The archived camera renderer publishes instance ID = geom ID + 1.
    return profile, ids


def _episodes(args):
    paths = sorted(args.evaluation.glob("*-trained.json")) if args.evaluation else args.episode
    selected, population = [], []
    for path in paths:
        path = path.resolve(strict=True)
        row = json.loads(path.read_text(encoding="utf-8"))
        job = row["job"]
        eligible = (row["success"] is True and row["final_metrics"]["success"] is True
                    and row["terminated"] is True and row["truncated"] is False
                    and job["arms"] == "both" and job["black"] is False)
        if args.evaluation and job["label"] != "trained":
            raise ValueError(f"unexpected policy label in {path}")
        population.append(dict(receipt=str(path), seed=row["seed"], success=row["success"],
                               task=row["task"], label=job["label"]))
        if not eligible:
            if args.evaluation:
                continue
            raise ValueError(f"{path} is not a successful two-arm, unmasked episode")
        xml, trace = (path.with_suffix(ext).resolve(strict=True) for ext in (".xml", ".npz"))
        with np.load(trace, allow_pickle=False) as archive:
            times = archive["time_s"]
            duration = float(times[-1] - times[0])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid simulation duration in {trace}")
        if (not isinstance(row["arm_dofs"], list) or sorted(row["arm_dofs"]) != [6, 7]
                or _digest(xml) != row["scene_sha256"]):
            raise ValueError(f"arm configuration or original XML binding is inconsistent in {path}")
        tool, pad_ids = _physical_pads(xml, row)
        if tool == "four-contact" and row.get("trajectory_sha256") != _digest(trace):
            raise ValueError(f"four-contact episode must bind its complete trajectory: {trace}")
        selected.append((path, row, duration, tool, pad_ids))
    if not selected:
        raise ValueError("no successful episode with two active arms and unmasked RGB was supplied")
    if len({r["task"] for r in population}) != 1 or len({p for p, *_ in selected}) != len(selected):
        raise ValueError("select distinct complete episodes from one task per film")
    return selected, population


def _allocate(durations, total):
    if total < 2 * len(durations):
        raise ValueError("too many episodes to show both endpoints")
    available = total - 2 * len(durations)
    shares = [available * d / sum(durations) for d in durations]
    counts = [2 + int(share) for share in shares]
    order = sorted(range(len(counts)), key=lambda i: shares[i] % 1., reverse=True)
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def _visibility_and_clock(film, required):
    coverage = {name: dict(instance_id=i, pixels=0) for name, i in required.items()}
    pairs = {}
    for_side = {side: [n for n in required if n.startswith(side + "_")] for side in ("left", "right")}
    for frame in film["captures"]:
        views = frame["views"]
        if set(views) != {"overview", "left_wrist_camera", "right_wrist_camera"}:
            raise ValueError("missing a camera from the retained step")
        for camera, view in views.items():
            if (view["time_s"], view["tick"]) != (frame["time_s"], frame["tick"]):
                raise ValueError("camera time disagrees with retained trajectory")
            pixels = view.get("visible_geom_pixels", {})
            for pad, evidence in coverage.items():
                n = pixels.get(pad, 0)
                if n > evidence.get("pixels", 0):
                    evidence.update(pixels=n, camera=camera, tick=frame["tick"])
            for side, names in for_side.items():
                minimum = min(pixels.get(n, 0) for n in names)
                if minimum > pairs.get(side, {}).get("smaller_visible_area_px", 0):
                    pairs[side] = dict(smaller_visible_area_px=minimum, camera=camera, tick=frame["tick"])
    if len(required) == 4 and any(pairs.get(s, {}).get("smaller_visible_area_px", 0) < 8
                                 for s in ("left", "right")):
        raise ValueError(f"the sampled views do not clearly show both contact pads on each tool: {pairs}")
    if len(required) == 8:
        hidden = [n for n, evidence in coverage.items() if evidence["pixels"] < 8]
        if hidden:
            raise ValueError(f"four-contact pads without an identifiable 8-pixel exposure: {hidden}; {coverage}")
    return dict(per_pad=coverage, simultaneous_twin_pad_views=pairs if len(required) == 4 else {},
                simultaneous_all_pads_per_tool=pairs)


def _verify_movie(path, fps, expected):
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe", path=str(Path(_ffmpeg()).parent))
    if ffprobe:
        command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                   "-show_entries", "stream=codec_name,nb_read_packets,duration,width,height",
                   "-of", "json", str(path)]
        actual = json.loads(subprocess.check_output(command, text=True))["streams"][0]
        if (actual["codec_name"] != "h264" or (actual["width"], actual["height"]) != (1352, 828)
                or int(actual["nb_read_packets"]) != expected
                or abs(float(actual["duration"]) - expected/fps) > .5/fps):
            raise ValueError(f"encoded reel differs from the verified clip schedule: {actual}")
        return actual
    # The portable imageio distribution includes ffmpeg but not always ffprobe.
    command = [_ffmpeg(), "-v", "error", "-progress", "pipe:1", "-nostats", "-i", str(path),
               "-map", "0:v:0", "-c:v", "copy", "-f", "null", "-"]
    progress = subprocess.check_output(command, text=True)
    counters = dict(line.split("=", 1) for line in progress.splitlines() if "=" in line)
    if counters.get("progress") != "end" or int(counters["frame"]) != expected:
        raise ValueError("encoded reel packet count differs from verified clip schedule")
    return dict(verification="FFmpeg full video packet count", packets=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluation", type=Path, help="folder of frozen holdout *-trained.json receipts")
    source.add_argument("--episode", type=Path, action="append", help="complete episode receipt, repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--seconds", type=int, default=40, choices=range(30, 61), metavar="30..60")
    timing.add_argument("--real-time", action="store_true", help="1 simulated second per video second")
    parser.add_argument("--fps", type=int, default=30, choices=range(1, 121), metavar="1..120")
    args = parser.parse_args()
    output = args.output.resolve()
    report = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or report.exists():
        raise ValueError("choose a fresh .mp4 filename and companion .json")
    selected, population = _episodes(args)
    durations = [episode[2] for episode in selected]
    counts = ([max(1, math.ceil(d * args.fps - 1e-9)) + 1 for d in durations] if args.real_time
              else _allocate(durations, args.seconds * args.fps))
    parts = output.with_name(output.stem + "-clips")
    parts.mkdir(parents=True, exist_ok=True)
    chapters, offset = [], 0
    for i, ((path, row, duration, tool, pad_ids), count) in enumerate(zip(selected, counts, strict=True)):
        speed = 1. if args.real_time else duration * args.fps / (count - 1)
        config = "+".join(str(n) for n in row["arm_dofs"])
        title = f"{'Carry' if row['task'] == 'carry' else 'Push T'}   /   {config} DOF   /   {tool}   /   {row['job']['label']}"
        clip = parts / f"clip-{i:03d}.mp4"
        evidence = clip.with_suffix(".frames.json")
        model, trace = path.with_suffix(".xml"), path.with_suffix(".npz")
        with np.load(trace, allow_pickle=False) as archive:
            times = archive["time_s"]
            clock = np.arange(count) * speed / args.fps
            selected_states = np.searchsorted(times - times[0], clock + 1e-9, side="right") - 1
            selected_states[-1] = len(times) - 1
            if len(np.unique(selected_states)) != len(times):
                raise ValueError("too few movie frames; select fewer episodes, a longer film, or a higher fps")
        if not clip.exists() and not evidence.exists():
            record(argparse.Namespace(model=model, trajectory=trace, episode_receipt=path, title=title,
                                      output=clip, fps=args.fps, playback_speed=speed, cache_dir=args.cache_dir,
                                      library=args.library, device_index=args.device_index,
                                      inspect_all_states=tool == "four-contact"))
        film = json.loads(evidence.read_text(encoding="utf-8"))
        if (film["episode_outcome"]["receipt_sha256"] != _digest(path)
                or film["model_sha256"] != _digest(model) or film["trace_sha256"] != _digest(trace)
                or film["video_sha256"] != _digest(clip) or film["fps"] != args.fps
                or film["movie_frames"] != count or film["playback_speed"] != speed
                or film["episode_outcome"]["success"] is not True
                or film["fiducials"] != row.get("fiducials", False)
                or film["marker_layout_version"] != row.get("marker_layout_version", LEGACY_MARKER_LAYOUT)
                or film["presentation_source_indices"][0] != 0
                or film["presentation_source_indices"][-1] != film["source_samples"] - 1):
            raise ValueError(f"clip receipt disagrees with this reel's inputs or clock: {clip}")
        if len(set(film["presentation_source_indices"])) != film["source_samples"]:
            raise ValueError("some recorded states were skipped; choose fewer episodes, more seconds, or a higher fps")
        coverage = _visibility_and_clock(film, pad_ids)
        chapters.append(dict(first_frame=offset, frames=count, playback_speed=speed, seed=row["seed"],
                             arm_dofs=row["arm_dofs"], tool_profile=tool, fiducials=film["fiducials"],
                             simulated_duration_s=duration, marker_layout_version=film["marker_layout_version"],
                             model_sha256=film["model_sha256"], trajectory_sha256=film["trace_sha256"],
                             clip=clip.name, receipt=str(path), clip_manifest_sha256=_digest(evidence),
                             visible_pad_pixels=coverage))
        offset += count
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        scratch = Path(scratch)
        concat_file = parts / "concat.ffconcat"
        concat_file.write_text("ffconcat version 1.0\n" + "".join(
            f"file {c['clip']}\nduration {c['frames']/args.fps:.12f}\n" for c in chapters), encoding="utf-8")
        movie = scratch / "joined.mp4"
        subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1",
                        "-i", str(concat_file), "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
                        str(movie)], check=True)
        checked = _verify_movie(movie, args.fps, offset)
        manifest = dict(task=selected[0][1]["task"], seconds=offset/args.fps, fps=args.fps,
                        frame_count=offset, sha256=_digest(movie), verification=checked, chapters=chapters,
                        episode_inputs=population, segments_directory=parts.name,
                        edit="complete episodes in input order; hard cuts; no motion interpolation")
        manifest_path = scratch / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(movie, output)
        os.replace(manifest_path, report)
    print(f"{output}  ({offset} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
