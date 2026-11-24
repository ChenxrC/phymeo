"""Verify a registered 240-frame R1 finish comparison and present the original PNGs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageChops, ImageDraw, ImageFont


SIZE = (1280, 720)
FRAMES = 240
CROPS = {"Stereo camera": (506, 41, 668, 153), "Wrist and gripper": (525, 260, 755, 387)}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def same(a, b, name):
    if a != b:
        raise ValueError(f"R1 comparison mismatch in {name}")


def load(directory):
    report = json.loads((directory/"render.json").read_bytes())
    path = directory/"trajectory.json"
    same(digest(path), report["trajectory_sha256"], f"{directory.name} trajectory hash")
    same(digest(directory/report["video"]), report["movie_sha256"], f"{directory.name} movie hash")
    same((report["width"], report["height"], report["captures"], report["frame_rate"]),
         (*SIZE, FRAMES, 30), "720p / 240 frames / 30 fps")
    stream = report["stream"]
    same((stream["width"], stream["height"], int(stream["nb_read_frames"]), stream["avg_frame_rate"]),
         (*SIZE, FRAMES, "30/1"), "encoded stream")
    images = {}
    for i in (0, 143):
        with Image.open(directory/f"frame-{i:03}.png") as im:
            same((im.format, im.mode, im.size), ("PNG", "RGB", SIZE), "source PNG")
            images[i] = im.copy()
    return report, json.loads(path.read_bytes()), images


def verify_trajectory(a, b, before, after):
    same({k:v for k,v in a.items() if k != "frames"}, {k:v for k,v in b.items() if k != "frames"}, "motion source")
    same((len(a["frames"]), len(b["frames"])), (FRAMES, FRAMES), "trajectory lengths")
    expected_links = set(before["model"]["links"])
    for i, (old, new) in enumerate(zip(a["frames"], b["frames"])):
        same((old["frame"], new["frame"]), (i,i), f"frame indices {i}")
        same(old["time_s"], i/30, f"frame time {i}")
        same(set(old["link_poses"]), expected_links, f"full FK link coverage {i}")
        if not old["joints"] or old["root"] != old["link_poses"]["base_link"]:
            raise ValueError(f"incomplete root or joint record at frame {i}")
        same({k:v for k,v in old.items() if k != "camera"},
             {k:v for k,v in new.items() if k != "camera"}, f"root, joints, link poses and clock {i}")
        for row, report in ((old,before),(new,after)):
            cam = row["camera"]
            same((cam["width"],cam["height"]), SIZE, f"camera dimensions {i}")
            same((cam["frame_key"]["scene_key"],cam["frame_key"]["tick"],cam["frame_key"]["revision"]),
                 (report["scene"],i,i), f"camera bound to scene and frame {i}")
        cam_a, cam_b = old["camera"], new["camera"]
        same({k:v for k,v in cam_a.items() if k != "frame_key"},
             {k:v for k,v in cam_b.items() if k != "frame_key"}, f"lens and camera pose {i}")
        same({k:v for k,v in cam_a["frame_key"].items() if k != "scene_key"},
             {k:v for k,v in cam_b["frame_key"].items() if k != "scene_key"}, f"camera frame clock {i}")


def verify_materials(before, after, directory):
    model = before["model"]
    # The original film predates the finish switch. Its scene identity and
    # source-material metadata identify how it was shaded.
    original_key = f"galaxea-r1-original-{model['revision'][:12]}:studio-arc-v1"
    same((before["scene"], before.get("finish","urdf"),model["source_rgba"],model["metallic"],model["roughness"]),
         (original_key, "urdf", "official per-visual URDF", 0., .65), "original URDF materials")
    same(after["finish"], "studio", "studio finish")
    displayed = after["displayed_materials"]
    materials = {m["id"]:m for m in displayed}
    expected = {"r1-material-"+link for link in model["links"]}
    same((set(materials),len(displayed)), (expected,len(expected)), "one material for every R1 link")
    changed = sum((m["metallic"],m["roughness"]) != (model["metallic"],model["roughness"])
                  for m in displayed)
    if not changed or len({tuple(m["base_color"]) for m in displayed}) < 2:
        raise ValueError("expected distinct studio finishes in displayed materials")

    assets = directory/"finish-assets"
    detail_file = assets/"details.json"
    detail = json.loads(detail_file.read_bytes())
    prefix = original_key+":studio-finish-v1:"
    signature = detail["source_scene"].removeprefix(prefix)
    if (not detail["source_scene"].startswith(prefix) or len(signature) != 64
            or any(c not in "0123456789abcdef" for c in signature)):
        raise ValueError("studio palette binding missing before UV mapping")
    same(after["scene"], detail["source_scene"]+":r1-details-"+digest(detail_file)[:12], "UV detail identity")
    wanted = {"zed_link", *(f"wheel_motor_link{i}" for i in (1,2,3))}
    same({r["link"] for r in detail["links"]}, wanted, "textured link set")
    for row in detail["links"]:
        link = row["link"]
        mat = materials["r1-material-"+link]
        geometry = model["geometries"][model["links"][link]]
        same({p["id"]:p["source_sha256"] for p in row["parts"]},
             {p["id"]:p["sha256"] for p in geometry["parts"]}, link+" original input meshes")
        for part in row["parts"]:
            actual = assets/f"{part['id']}-uv-{part['derived_sha256'][:18]}.gltf"
            same(digest(actual), part["derived_sha256"], link+" UV mesh")
        for texture, expected_hash in ((mat["base_color_map"], row["base_sha256"]),
                                       (mat["roughness_map"]["image"], row["roughness_sha256"])):
            name = Path(unquote(urlsplit(texture["uri"]).path)).name
            same((texture["sha256"],digest(assets/name)), (expected_hash,expected_hash), link+" texture")
    return dict(displayed_links=len(displayed), links_with_changed_metallic_or_roughness=changed,
                textured_links=sorted(wanted), studio_details_sha256=digest(detail_file))


def board(path, old, new):
    gap, width, heading = 16, 720, 44
    full_height = 405
    canvas = Image.new("RGB",(2*width+3*gap,715),"#172028")
    draw = ImageDraw.Draw(canvas)
    for family in ("DejaVuSans.ttf","arial.ttf"):
        try:
            title = ImageFont.truetype(family,22)
            label = ImageFont.truetype(family,17)
            break
        except OSError:
            continue
    else:
        title, label = ImageFont.load_default(), ImageFont.load_default()
    for index, (images, name) in enumerate(((old,"Original URDF"),(new,"Studio finish"))):
        x = gap + index*(width+gap)
        draw.text((x,10),name+"  |  Frame 0",fill="white",font=title)
        canvas.paste(images[0].resize((width,full_height),Image.Resampling.LANCZOS),(x,heading))
        y = heading+full_height+gap
        for col, (part, crop) in enumerate(CROPS.items()):
            left = x + (width//2)*col
            draw.text((left,y),f"{part}  |  Frame 143",fill="white",font=label)
            im = images[143].crop(crop)
            scale = min((width//2-8)/im.width,210/im.height)
            picture = im.resize((round(scale*im.width),round(scale*im.height)),Image.Resampling.LANCZOS)
            canvas.paste(picture,(left,y+28))
    canvas.save(path)


def compare(before_dir, after_dir, output):
    before, trajectory_a, old = load(before_dir)
    after, trajectory_b, new = load(after_dir)
    shared = ("engine","device","film_aa","exposure","width","height","camera_cuts","frame_rate","captures","studio","model","stream")
    for key in shared:
        same(before[key],after[key],key)
    hdr = before["studio"]["environment_map"]
    hdr_file = Path(__file__).resolve().parents[1]/"materials"/Path(unquote(urlsplit(hdr["uri"]).path)).name
    same(digest(hdr_file), hdr["sha256"], "HDR panorama bytes")
    for asset in ("r1-stage-cyclorama.gltf","r1-stage-floor-marks.gltf"):
        same(digest(before_dir/"stage"/asset),digest(after_dir/"stage"/asset),asset)
    verify_trajectory(trajectory_a,trajectory_b,before,after)
    finishes = verify_materials(before,after,after_dir)
    sources = {}
    for i in old:
        if ImageChops.difference(old[i],new[i]).getbbox() is None:
            raise ValueError(f"identical displayed RGB at frame {i}")
        sources[i] = {label:digest(root/f"frame-{i:03}.png") for label,root in (("urdf",before_dir),("studio",after_dir))}
    output.mkdir(parents=True, exist_ok=False)
    board(output/"comparison.png",old,new)
    evidence = dict(frames_checked=FRAMES, size=SIZE, frame_rate=30, links_per_frame=len(before["model"]["links"]),
                    matching=["URDF and original converted meshes", "stage geometry", "all joints, link poses and base poses",
                              "all camera intrinsics, extrinsics and frame clocks", "studio lighting, HDR panorama and exposure",
                              "device, film sampling and stream format"],
                    film_aa=before["film_aa"], exposure=before["exposure"], hdr_sha256=hdr["sha256"],
                    urdf_sha256=before["model"]["urdf_sha256"],
                    movie_sha256={"urdf":before["movie_sha256"],"studio":after["movie_sha256"]},
                    materials=finishes, png_sha256=sources, frame_143_crop_boxes=CROPS)
    (output/"comparison.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("before","after","output"):
        cli.add_argument("--"+arg,type=Path,required=True)
    args = cli.parse_args()
    compare(args.before,args.after,args.output)
