"""Encode completed RGB PNG batches, with no renderer or source-image edits.

Requires ffmpeg (including libx264) and ffprobe on PATH. Only labels occupy
new pixels; all five source groups retain their original size and framing.
RGB values already carry the capture's sRGB display transform. Conversion is
BT.709 matrix / full RGB to limited YUV, without another transfer function.
H.264 is lossy and 4:2:0 subsamples chroma; original PNGs remain authoritative.

FFmpeg references: https://ffmpeg.org/ffmpeg-filters.html#scale and
https://ffmpeg.org/ffmpeg-formats.html#mov_002c-mp4_002c-ismv .
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

_WIDTH, _HEIGHT, _BAND, _DIVIDER = 768, 576, 48, 2
_GROUPS = ("single", "studio", "materials-wide", "materials-detail", "copper-detail")
_NAMES = ("panda-studio.mp4", "panda-lighting.mp4", "panda-materials.mp4", "panda-surface.mp4")
# Original, fixed 5x7 bitmap lettering: reproducible without installed fonts.
_GLYPHS = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    ":": (0, 4, 4, 0, 4, 4, 0),
    "A": (14, 17, 17, 31, 17, 17, 17),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (30, 17, 17, 17, 17, 17, 30),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15),
    "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 25, 21, 19, 19, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "W": (17, 17, 17, 21, 21, 27, 17),
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command):
    result = subprocess.run(
        [str(arg) for arg in command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(
            f"{Path(command[0]).name} failed ({result.returncode}): {result.stderr[-12000:]}"
        )
    return result.stdout


def _sources(root, count):
    result = {}
    expected = [f"{index:04d}.png" for index in range(count)]
    for group in _GROUPS:
        directory = root / group
        names = sorted(path.name for path in directory.glob("*.png"))
        if names != expected:
            raise ValueError(f"{group}: require exactly contiguous PNGs 0000..{count - 1:04d}")
        frames = []
        for name in names:
            path = directory / name
            with path.open("rb") as stream:
                header = stream.read(33)
            if (
                len(header) != 33
                or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[8:16] != b"\x00\x00\x00\rIHDR"
            ):
                raise ValueError(f"invalid PNG IHDR: {path}")
            width, height, depth, kind, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", header[16:29]
            )
            # The public writer uses opaque RGB, never silently flatten alpha.
            if (
                (width, height, depth, kind) != (_WIDTH, _HEIGHT, 8, 2)
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise ValueError(f"{path}: require 768x576 opaque 8-bit RGB PNG")
            frames.append({"file": f"{group}/{path.name}", "sha256": _sha(path)})
        result[group] = frames
    return result


def _band(path, width, labels, *, divider=False):
    pixels = bytearray(bytes((22, 24, 29)) * width * _BAND)

    def paint(x, y, color):
        start = (y * width + x) * 3
        pixels[start : start + 3] = bytes(color)

    for left, text in labels:
        if left + len(text) * 18 > width:
            raise ValueError("label does not fit its external band")
        for index, char in enumerate(text):
            for row, bits in enumerate(_GLYPHS[char]):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        for dy in range(3):
                            for dx in range(3):
                                paint(
                                    left + index * 18 + col * 3 + dx,
                                    13 + row * 3 + dy,
                                    (235, 237, 240),
                                )
    if divider:
        for y in range(_BAND):
            for x in range(_WIDTH, _WIDTH + _DIVIDER):
                paint(x, y, (64, 64, 64))
    path.write_bytes(f"P6\n{width} {_BAND}\n255\n".encode("ascii") + pixels)


def _faststart(path):
    positions = {}
    length = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < length:
            offset = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("truncated MP4 box header")
            size, kind = struct.unpack(">I4s", header)
            minimum = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("truncated MP4 extended box")
                size = struct.unpack(">Q", extended)[0]
                minimum = 16
            elif size == 0:
                size = length - offset
            if size < minimum or offset + size > length:
                raise ValueError("invalid MP4 box extent")
            positions.setdefault(kind.decode("ascii", errors="replace"), offset)
            stream.seek(offset + size)
    if not ("moov" in positions and "mdat" in positions and positions["moov"] < positions["mdat"]):
        raise ValueError("MP4 is not faststart: moov must precede mdat")
    return {"moov_offset": positions["moov"], "mdat_offset": positions["mdat"]}


def _verify(path, ffmpeg, ffprobe, fps, count, width, height):
    data = json.loads(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-show_frames",
                "-show_entries",
                "stream=index,codec_type,codec_name,profile,codec_tag_string,pix_fmt,width,height,"
                "avg_frame_rate,r_frame_rate,time_base,duration,duration_ts,nb_frames,nb_read_frames,"
                "color_range,color_space,color_transfer,color_primaries,sample_aspect_ratio:"
                "frame=best_effort_timestamp_time:format=duration,format_name",
                "-of",
                "json",
                path,
            ]
        )
    )
    streams = data.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise ValueError(f"{path.name}: require exactly one video stream and no audio")
    stream = streams[0]
    required = {
        "codec_name": "h264",
        "codec_tag_string": "avc1",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "iec61966-2-1",
        "color_primaries": "bt709",
        "sample_aspect_ratio": "1:1",
    }
    for key, expected in required.items():
        if stream.get(key) != expected:
            raise ValueError(f"{path.name}: {key}={stream.get(key)!r}, expected {expected!r}")
    if (
        int(stream.get("nb_frames", -1)) != count
        or int(stream.get("nb_read_frames", -1)) != count
        or Fraction(stream["avg_frame_rate"]) != fps
        or Fraction(stream["r_frame_rate"]) != fps
    ):
        raise ValueError(f"{path.name}: frame count/rate mismatch")
    wanted_duration = Fraction(count, fps)
    actual_duration = int(stream["duration_ts"]) * Fraction(stream["time_base"])
    if actual_duration != wanted_duration:
        raise ValueError(f"{path.name}: stream duration mismatch")
    if abs(float(data["format"]["duration"]) - float(wanted_duration)) > 0.000002:
        raise ValueError(f"{path.name}: container duration mismatch")
    frames = data.get("frames", [])
    if len(frames) != count or any(
        abs(Fraction(frame["best_effort_timestamp_time"]) - Fraction(i, fps))
        > Fraction(1, 1_000_000)
        for i, frame in enumerate(frames)
    ):
        raise ValueError(f"{path.name}: noncontiguous presentation timestamps")
    # Decode every frame through the real decoder, with errors made fatal.
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ]
    )
    return {
        "stream": stream,
        "format": data["format"],
        "decoded_frames": len(frames),
        "contiguous_timestamps": True,
        "complete_decode": True,
        "faststart": _faststart(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def encode_movies(frames_dir, output_dir, fps, count):
    """Encode/verify four movies from a complete batch; return JSON-ready evidence.

    At fps=24,count=108: studio/lighting/surface last 4.5 s; materials lasts 9 s.
    Existing destination movies are refused. All four are encoded and checked
    in a temporary directory before publishing. No PNG is created or modified.
    """
    for name, value in (("fps", fps), ("count", count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    root, destination = Path(frames_dir).resolve(), Path(output_dir).resolve()
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        raise RuntimeError("ffmpeg and ffprobe must both be available on PATH")
    sources = _sources(root, count)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in _NAMES):
        raise FileExistsError("refusing to replace existing Panda movies")
    report = {
        "schema": "phymeo-panda-movie-encode-v1",
        "fps": fps,
        "source_frames_per_group": count,
        "sources": sources,
        "source_color": "already display-encoded sRGB RGB8",
        "conversion": "BT.709 matrix; full RGB to limited YUV420; sRGB transfer retained",
        "encoding": "CPU libx264 High profile, CRF 18, preset slow, no audio, faststart",
        "labels": "original fixed 5x7 bitmap glyphs, 3x scale, external 48px band",
        "source_images_unchanged": False,
        "tools": {name: _run([exe, "-version"]).splitlines()[0] for name, exe in tools.items()},
        "movies": {},
    }
    with tempfile.TemporaryDirectory(prefix=".panda-encode-", dir=destination) as temporary:
        temp = Path(temporary)
        lighting_width = 2 * _WIDTH + _DIVIDER
        _band(
            temp / "lighting.ppm",
            lighting_width,
            ((24, "SINGLE DIRECTIONAL"), (_WIDTH + _DIVIDER + 24, "STUDIO")),
            divider=True,
        )
        _band(
            temp / "surface.ppm",
            lighting_width,
            ((24, "RED FINISH"), (_WIDTH + _DIVIDER + 24, "POLISHED COPPER")),
            divider=True,
        )
        _band(temp / "wide.ppm", _WIDTH, ((24, "MATERIALS: WIDE"),))
        _band(temp / "detail.ppm", _WIDTH, ((24, "MATERIALS: DETAIL"),))

        def input_args(group):
            return ["-framerate", str(fps), "-start_number", "0", "-i", root / group / "%04d.png"]

        def label_args(name):
            return ["-loop", "1", "-framerate", str(fps), "-i", temp / name]

        def rgb(index, label):
            return f"[{index}:v]format=rgb24,trim=end_frame={count},setpts=N/({fps}*TB),setsar=1[{label}]"

        final = (
            f"setpts=N/({fps}*TB),setsar=1,scale=in_range=full:out_range=limited:"
            "out_color_matrix=bt709:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,"
            "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709[out]"
        )
        split_graph = ";".join(
            (
                rgb(0, "left"),
                rgb(1, "right"),
                rgb(2, "title"),
                "[left]pad=770:576:0:0:color=0x404040[padded]",
                "[padded][right]hstack=inputs=2:shortest=1[pictures]",
                "[title][pictures]vstack=inputs=2:shortest=1[combined]",
                "[combined]" + final,
            )
        )
        materials_graph = ";".join(
            (
                rgb(0, "wide"),
                rgb(1, "detail"),
                rgb(2, "wide_title"),
                rgb(3, "detail_title"),
                "[wide_title][wide]vstack=inputs=2:shortest=1[wide_labeled]",
                "[detail_title][detail]vstack=inputs=2:shortest=1[detail_labeled]",
                "[wide_labeled][detail_labeled]concat=n=2:v=1:a=0[combined]",
                "[combined]" + final,
            )
        )
        jobs = (
            (
                _NAMES[0],
                input_args("studio"),
                rgb(0, "plain") + ";[plain]" + final,
                count,
                _WIDTH,
                _HEIGHT,
                [{"group": "studio", "first_frame": 0, "frames": count}],
            ),
            (
                _NAMES[1],
                input_args("single") + input_args("studio") + label_args("lighting.ppm"),
                split_graph,
                count,
                lighting_width,
                _HEIGHT + _BAND,
                [{"left": "single", "right": "studio", "first_frame": 0, "frames": count}],
            ),
            (
                _NAMES[2],
                input_args("materials-wide")
                + input_args("materials-detail")
                + label_args("wide.ppm")
                + label_args("detail.ppm"),
                materials_graph,
                2 * count,
                _WIDTH,
                _HEIGHT + _BAND,
                [
                    {"group": "materials-wide", "first_frame": 0, "frames": count},
                    {"group": "materials-detail", "first_frame": count, "frames": count},
                ],
            ),
            (
                _NAMES[3],
                input_args("materials-detail")
                + input_args("copper-detail")
                + label_args("surface.ppm"),
                split_graph,
                count,
                lighting_width,
                _HEIGHT + _BAND,
                [
                    {
                        "left": "materials-detail",
                        "right": "copper-detail",
                        "first_frame": 0,
                        "frames": count,
                        "left_label": "RED FINISH",
                        "right_label": "POLISHED COPPER",
                    }
                ],
            ),
        )
        for name, inputs, graph, frames, width, height, segments in jobs:
            output = temp / name
            command = [
                tools["ffmpeg"],
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-xerror",
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-profile:v",
                "high",
                "-x264-params",
                "colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-fps_mode",
                "cfr",
                "-frames:v",
                str(frames),
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "iec61966-2-1",
                "-video_track_timescale",
                str(fps * 1000),
                "-movflags",
                "+faststart+write_colr",
                "-map_metadata",
                "-1",
                "-y",
                output,
            ]
            _run(command)
            verified = _verify(
                output, tools["ffmpeg"], tools["ffprobe"], fps, frames, width, height
            )
            report["movies"][name] = {
                "file": name,
                "segments": segments,
                "filter_graph": graph,
                **verified,
            }
        # Bind the completed batch again after encoding; concurrent replacement
        # is a failed input transaction, never evidence for these movies.
        if _sources(root, count) != sources:
            raise RuntimeError("source PNG batch changed while encoding")
        for name in _NAMES:
            if (destination / name).exists():
                raise FileExistsError(f"destination appeared while encoding: {name}")
        for name in _NAMES:
            os.replace(temp / name, destination / name)
    report["source_images_unchanged"] = True
    return report
