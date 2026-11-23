"""Show the same recorded Panda frames before and after low-rate diffuse filtering."""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from preview_compact_rigid_history import preview, preview_assess


def dataset(directory, filtered):
    base = directory.resolve()
    summary = preview.read(base / "summary.json")
    if summary["camera"] != "detail" or summary["studio_look"] != "recorded":
        raise ValueError("the two movies require the registered camera and lighting")
    row, = (r for r in summary["runs"] if r["name"] == "rigid-history")
    root, receipt = preview_assess.load_row(base, row)
    contract = receipt["backend"]["contract"]
    if contract["paths_per_anchor"] != 8 or ("lowres_filter" in contract) != filtered:
        raise ValueError("incorrect eight-proposal exposure")
    frames = receipt["frames"]
    if tuple(f["pose_index"] for f in frames) != preview.POSES:
        raise ValueError("incomplete registered recording")
    return root, frames


def frame_image(directory, frame):
    path = directory / frame["png"]
    if preview.digest(path) != frame["png_sha256"]:
        raise ValueError("frame file differs from capture")
    with Image.open(path) as image:
        pixels = image.convert("RGB")
    if pixels.size != (1280,720) or preview.digest_bytes(pixels.tobytes()) != frame["rgb_sha256"]:
        raise ValueError("frame RGB differs from native capture")
    return pixels


def compare(original, filtered, output, font):
    before, after = dataset(original,False), dataset(filtered,True)
    if output.exists():
        raise FileExistsError(output)
    label = ImageFont.truetype(str(font), 24)
    detail = ImageFont.truetype(str(font), 19)
    ffmpeg = ["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pixel_format","rgb24",
              "-video_size","1280x720","-framerate","30","-i","-","-an","-vf",
              "scale=in_range=pc:out_range=tv:out_color_matrix=bt709","-c:v","libx264","-preset","fast",
              "-crf","16","-pix_fmt","yuv420p","-color_primaries","bt709","-color_trc","iec61966-2-1",
              "-colorspace","bt709","-movflags","+faststart",str(output)]
    with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as process:
        try:
            for a,b in zip(before[1],after[1],strict=True):
                preview_assess.check_identity(a,b)
                if a["sensors"] != b["sensors"]:
                    raise ValueError("different geometric visibility")
                board=Image.new("RGB",(1280,720),"#172127")
                draw=ImageDraw.Draw(board)
                draw.text((20,12),"同一段运动里的白壳和阴影",font=label,fill="#faf5e6")
                draw.text((1008,18),f"第 {a['pose_index']:03d} 帧   /   299",font=detail,fill="#bbccd0")
                for x,root,frame,title in ((16,before[0],a,"原八路一跳"),(648,after[0],b,"八路一跳＋间接光预滤波")):
                    native=frame_image(root,frame)
                    draw.text((x,63),title,font=detail,fill="#fbdfb0")
                    board.paste(native.resize((616,346),Image.Resampling.LANCZOS),(x,99))
                    draw.text((x,463),"左臂与夹爪下方   ·   原像素局部",font=detail,fill="#d0dcdb")
                    board.paste(native.crop((0,330,616,536)),(x,500))
                process.stdin.write(board.tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("comparison encoder failed")
    args=["ffprobe","-v","error","-select_streams","v:0","-show_frames","-show_entries",
          "frame=best_effort_timestamp_time","-of","json",str(output)]
    probe=json.loads(subprocess.check_output(args))
    pts=[float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
    if len(pts)!=300 or any(abs(t-i/30)>1e-5 for i,t in enumerate(pts)):
        raise ValueError("comparison timeline differs from the 300 paired captures")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("original","filtered","output","font"):
        parser.add_argument("--"+key,type=Path,required=True)
    compare(**vars(parser.parse_args()))
