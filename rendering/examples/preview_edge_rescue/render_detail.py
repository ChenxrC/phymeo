"""Publish a registered thin-edge crop with its actual 69-pixel disocclusion mask."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_image(path, expected):
    with Image.open(path) as im:
        image = im.convert("RGB")
    if sha256(image.tobytes()).hexdigest() != expected:
        raise ValueError("decoded RGB differs from its capture")
    return image


def render(reference, near, font, output, rescue=None):
    def summary(directory):
        return json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    b, n = summary(reference), summary(near)
    if not (b["status"] == n["status"] == "complete" and b["fixture_sha256"] == n["fixture_sha256"]):
        raise ValueError("incomplete or unrelated runs")
    if any(b["frames"]["15"][k] != n["frames"]["15"][k] for k in ("sensors","key","time")):
        raise ValueError("frame binding differs")
    with Image.open(reference/"exposed-right-finger.png") as source:
        mask=source.convert("L")
    if mask.histogram()[255] != 69 or sum(mask.histogram()[1:-1]) != 0:
        raise ValueError("wrong exposed-pixel mask")
    box=mask.getbbox()
    images=[
        load_image(reference/"pose-15.png",b["frames"]["15"]["rgb_sha256"]),
        load_image(near/"pose-15.png",n["frames"]["15"]["rgb_sha256"]),
        load_image(reference/"seed-1-pose-15.png",b["references"]["1-15"]["rgb_sha256"]),
    ]
    mid_x=(box[0]+box[2])//2
    mid_y=(box[1]+box[3])//2
    radius=max(32, (max(box[2]-box[0],box[3]-box[1])+9)//2)
    region=(mid_x-radius,mid_y-radius,mid_x+radius,mid_y+radius)
    marked=images[-1].copy()
    marked.paste((23,238,169),mask=mask)
    images.append(marked)
    board=Image.new("RGB",(1160,385),"#192026")
    ink=ImageDraw.Draw(board)
    header=ImageFont.truetype(str(font),23)
    small=ImageFont.truetype(str(font),16)
    ink.text((17,8),"遮挡板移开后，刚露出的右夹指",font=header,fill="#f9f6ef")
    scale=224/(2*radius)
    ink.text((17,44),f"第 15 帧  ·  右侧绿色为真实 69 个像素  ·  同一裁区，最近邻放大 {scale:.2f} 倍",font=small,fill="#bbc9ca")
    names=("完整一跳 16 路","12 cm 近场预览","1024 样本参考","显露区域位置")
    if rescue is not None:
        evidence=summary(rescue)
        row=evidence["frames"]["15"]
        detail=evidence["actual_rescue"]
        if (evidence["status"]!="actual_readbacks_bound" or
                detail["registered_pixels"]!=69 or detail["valid_results"]!=69 or
                any(row[k]!=b["frames"]["15"][k] for k in ("sensors","key","time"))):
            raise ValueError("rescue evidence does not match the 69 registered pixels")
        with Image.open(rescue/"exposed-right-finger.png") as rescue_mask:
            if rescue_mask.convert("L").tobytes()!=mask.tobytes():
                raise ValueError("registered rescue mask differs")
        images=[images[1],load_image(rescue/"pose-15.png",row["rgb_sha256"]),images[2],marked]
        names=("12 cm 近场预览","低支撑像素补 16 路","1024 样本参考","显露区域位置")
    for i,(im,name) in enumerate(zip(images,names,strict=True)):
        x=24+i*284
        ink.text((x,86),name,font=header,fill="#f7ebcb")
        board.paste(im.crop(region).resize((224,224),Image.Resampling.NEAREST),(x,130))
    if output.exists():
        raise FileExistsError(output)
    board.save(output)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for name in ("reference","near","font","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--rescue",type=Path,help="optional measured edge-rescue exposure from the same thin fixture")
    render(**vars(parser.parse_args()))
