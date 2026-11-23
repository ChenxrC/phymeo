"""Show the same native Panda exposure at full-frame and 1:1 detail scales."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_pose(directory, reference):
    path = directory / ("summary.json" if reference else "receipt.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    records = receipt["records"] if reference else receipt["frames"]
    row, = (r for r in records if r["pose_index"] == 60)
    filename = directory / row["png"]
    if sha256(filename.read_bytes()).hexdigest() != row["png_sha256"]:
        raise ValueError("source PNG digest differs")
    with Image.open(filename) as original:
        image = original.convert("RGB")
    if image.size != (1280, 720) or sha256(image.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError("decoded native RGB differs")
    if reference and receipt["batches"] * 16 != 8192:
        raise ValueError("this plate requires the 8192-sample reference")
    return image, {k: row[k] for k in ("frame_key","time_s","camera_sha256","state_sha256")}


def compare(args):
    inputs = [read_pose(args.baseline, False), read_pose(args.candidate, False), read_pose(args.reference, True)]
    if any(key != inputs[0][1] for image, key in inputs):
        raise ValueError("different pose, time, state or camera")
    board = Image.new("RGB", (1280, 694), "#192026")
    draw = ImageDraw.Draw(board)
    bold = ImageFont.truetype(str(args.font), 26)
    normal = ImageFont.truetype(str(args.font), 20)
    small = ImageFont.truetype(str(args.font), 17)
    draw.text((22, 9), "同一时刻的三种光照结果", font=bold, fill="#faf9f4")
    draw.text((22, 49), "Panda 抓放   ·   第 60 帧   ·   相同机位、姿态、灯光和曝光", font=small, fill="#bbc9ca")
    for i, ((image, _), title) in enumerate(zip(inputs, ("原完整一跳   /   16 路", "近场运动预览   /   4 路", "完整一跳参考   /   8192 样本"), strict=True)):
        x = 15 + 424 * i
        draw.text((x, 88), title, font=normal, fill="#f7ebcb")
        board.paste(image.resize((400,225),Image.Resampling.LANCZOS),(x,123))
        draw.text((x,359),"白壳与凹槽   ·   原像素",font=small,fill="#e0e6df")
        board.paste(image.crop((420,157,820,313)),(x,387))
        draw.text((x,553),"夹指与方块   ·   原像素",font=small,fill="#e0e6df")
        board.paste(image.crop((422,321,822,427)),(x,582))
    if args.output.exists():
        raise FileExistsError(args.output)
    board.save(args.output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","reference","output","font"):
        parser.add_argument("--"+name,type=Path,required=True)
    compare(parser.parse_args())
