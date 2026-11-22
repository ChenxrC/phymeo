"""Encode a real capture series and compare every decoded frame with its GPU readback."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from subprocess import PIPE, Popen, check_output, run

import numpy as np
from PIL import Image


def encode(directory, movie):
    journal = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    size, count, fps = tuple(journal["size"]), journal["frames"], journal["fps"]
    names = [directory / "frames" / f"{i:04d}.png" for i in range(count)]
    if sorted((directory / "frames").glob("*.png")) != names or len(journal["frame_sha256"]) != count:
        raise ValueError("PNG sequence and capture journal disagree")
    movie.parent.mkdir(parents=True, exist_ok=True)
    report_path = movie.with_suffix(".decoded.json")
    if movie.exists() or report_path.exists():
        raise FileExistsError("output already exists")
    run(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-n","-framerate",str(fps),
         "-i",str(directory / "frames/%04d.png"),"-vf",
         "scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p",
         "-an","-c:v","libx264","-crf","17","-preset","medium","-color_range","tv",
         "-colorspace","bt709","-color_primaries","bt709","-color_trc","iec61966-2-1",
         "-movflags","+faststart",str(movie)],check=True)
    info = json.loads(check_output(["ffprobe","-v","error","-show_streams","-show_frames",
                                   "-show_entries","stream=width,height,codec_name,pix_fmt,avg_frame_rate,time_base,nb_frames,duration_ts,color_space,color_range,color_transfer,color_primaries:frame=pts",
                                   "-select_streams","v","-of","json",str(movie)]))
    stream, = info["streams"]
    if ((stream["width"],stream["height"]) != size or Fraction(stream["avg_frame_rate"]) != fps
            or len(info["frames"]) != count or int(stream["nb_frames"]) != count
            or stream["duration_ts"] * Fraction(stream["time_base"]) != Fraction(count,fps)
            or any(Fraction(f["pts"])*Fraction(stream["time_base"]) != Fraction(i,fps)
                   for i,f in enumerate(info["frames"]))):
        raise ValueError("encoded stream differs from source clock or dimensions")
    width,height = size
    errors = []
    with Popen(["ffmpeg","-nostdin","-v","error","-i",str(movie),"-f","rawvideo","-pix_fmt","rgb24","-"],stdout=PIPE) as decoder:
        try:
            for path, expected in zip(names,journal["frame_sha256"]):
                with Image.open(path) as im:
                    if im.mode != "RGB" or im.size != size or sha256(im.tobytes()).hexdigest() != expected:
                        raise ValueError(f"input differs from GPU capture: {path.name}")
                    original = np.asarray(im,dtype=np.int16)
                raw = decoder.stdout.read(width*height*3)
                decoded = np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3)
                errors.append(float(np.mean(np.abs(original-decoded))))
            if decoder.stdout.read(1) or decoder.wait() != 0:
                raise ValueError("extra frames or incomplete decoder")
        finally:
            decoder.stdout.close()
    report = dict(movie=movie.name,video_sha256=sha256(movie.read_bytes()).hexdigest(),stream=stream,
                  source_excerpt_sha256=journal["source_excerpt_sha256"],frames_compared=count,
                  source_journal_sha256=sha256((directory/"capture.json").read_bytes()).hexdigest(),
                  max_frame_mean_abs_error_rgb8=max(errors),mean_abs_error_rgb8=sum(errors)/count,
                  per_frame_mean_abs_error_rgb8=errors)
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"{movie.name} {count} frames; worst frame mean abs RGB8 error {max(errors):.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir",type=Path)
    parser.add_argument("movie",type=Path)
    args = parser.parse_args()
    encode(args.capture_dir,args.movie)
