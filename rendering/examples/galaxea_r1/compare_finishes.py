"""Verify a registered 240-frame R1 finish comparison and present the original PNGs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


SIZE = (1280, 720)
FRAMES = 240
CROPS = {"Stereo camera": (506, 41, 668, 153), "Wrist and gripper": (525, 260, 755, 387)}
POLISHED_CROPS = {"Camera housing": CROPS["Stereo camera"], "Near forearm": (394, 233, 704, 371)}
PAINTED_ARMS = {f"{side}_{link}" for side in ("left","right")
                for link in ("arm_link1","arm_link2","arm_link3","gripper_link")}


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


def verify_normals(report, directory, rows, source_scene):
    root = directory/"polished-normals"
    receipt = root/"polished-normals.json"
    record = json.loads(receipt.read_bytes())
    same(record["source_scene"], source_scene, "normal refinement input scene")
    code = digest(receipt)
    same(report["scene"], source_scene+":polished-normals-"+code[:12], "normal refinement rendered scene")
    targets = PAINTED_ARMS | {"zed_link"}
    same(sorted(record["targets"]), sorted(targets), "normal refinement targets")
    model = report["model"]
    geometry = {link:{p["id"]:p["sha256"] for p in model["geometries"][model["links"][link]]["parts"]}
                for link in targets}
    for part in rows["zed_link"]["parts"]:
        geometry["zed_link"][part["id"]] = part["derived_sha256"]
    covered, changed = set(), 0
    for group in record["groups"]:
        links = group["links"]
        if not links or not set(links) <= targets or covered.intersection(links):
            raise ValueError("unexpected or repeated normal refinement group")
        parts = group["meshes"]
        for link in links:
            same({p["mesh_id"]:p["source_sha256"] for p in parts},geometry[link],link+" normal inputs")
        covered.update(links)
        for part in parts:
            key, old, new = part["mesh_id"], part["source_sha256"], part["derived_sha256"]
            count = part["changed_normals"]
            same(count > 0, old != new, key+" changed normal count and SHA")
            if old != new:
                same(digest(root/f"{key}-normals-{new[:18]}.gltf"),new,key+" normal output")
            changed += count
    for group in record["skipped"]:
        covered.update(targets.intersection(group["links"]))
    same(covered,targets,"normal refinement accounted targets")
    same(changed,record["changed_vertices"],"normal refinement vertex total")
    return dict(targets=sorted(targets), changed_normals=changed, receipt_sha256=code)


def verified_finish(report, directory):
    model = report["model"]
    displayed = report["displayed_materials"]
    materials = {m["id"]:m for m in displayed}
    expected = {"r1-material-"+link for link in model["links"]}
    same((set(materials),len(displayed)), (expected,len(expected)), "one material for every R1 link")
    assets = directory/"finish-assets"
    detail_file = assets/"details.json"
    detail = json.loads(detail_file.read_bytes())
    prefix = f"galaxea-r1-original-{model['revision'][:12]}:studio-arc-v1:{report['finish']}-finish-v1:"
    signature = detail["source_scene"].removeprefix(prefix)
    if (not detail["source_scene"].startswith(prefix) or len(signature) != 64
            or any(c not in "0123456789abcdef" for c in signature)):
        raise ValueError("palette binding missing before UV mapping")
    details_hash = digest(detail_file)
    source_scene = detail["source_scene"]+":r1-details-"+details_hash[:12]
    wanted = {"zed_link", *(f"wheel_motor_link{i}" for i in (1,2,3))}
    rows = {r["link"]:r for r in detail["links"]}
    same((set(rows),len(detail["links"])), (wanted,len(wanted)), "textured link set")
    for row in rows.values():
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
    normals = None
    if report["finish"] == "polished" and (directory/"polished-normals"/"polished-normals.json").is_file():
        normals = verify_normals(report,directory,rows,source_scene)
    else:
        same(report["scene"],source_scene,"UV detail identity")
    return materials, rows, details_hash, normals


def verify_materials(before, after, directory):
    model = before["model"]
    original_key = f"galaxea-r1-original-{model['revision'][:12]}:studio-arc-v1"
    same((before["scene"], before.get("finish","urdf"),model["source_rgba"],model["metallic"],model["roughness"]),
         (original_key, "urdf", "official per-visual URDF", 0., .65), "original URDF materials")
    same(after["finish"], "studio", "studio finish")
    mats, rows, details_hash, _ = verified_finish(after,directory)
    changed = sum((m["metallic"],m["roughness"]) != (model["metallic"],model["roughness"]) for m in mats.values())
    if not changed or len({tuple(m["base_color"]) for m in mats.values()}) < 2:
        raise ValueError("expected distinct studio finishes in displayed materials")
    return dict(displayed_links=len(mats), links_with_changed_metallic_or_roughness=changed,
                textured_links=sorted(rows), studio_details_sha256=details_hash)


def comparable_material(value):
    # Both runs publish into their own directories. Texture identity is the
    # digest of the file, verified against the actual bytes in verified_finish.
    if isinstance(value,dict):
        return {k:comparable_material(v) for k,v in value.items() if k != "uri"}
    return value


def checked_head_atlas(before_dir, after_dir, old, new, old_row, new_row):
    palette, updated = old_row["color_srgb_and_linear_roughness"], new_row["color_srgb_and_linear_roughness"]
    same(set(updated),{"window","rim","shell"},"head palette regions")
    for name in ("window","rim"):
        same(palette[name],updated[name],name+" swatch")
    same(palette["shell"][0],updated["shell"][0],"head shell base color")
    if not 0. <= updated["shell"][1] < palette["shell"][1] <= 1.:
        raise ValueError("camera housing roughness must decrease")

    def roughness(directory, material):
        uri = material["roughness_map"]["image"]["uri"]
        name = Path(unquote(urlsplit(uri).path)).name
        with Image.open(directory/"finish-assets"/name) as image:
            same((image.size,image.mode),((512,384),"RGB"),"registered stereo atlas")
            values = np.asarray(image).copy()
        if not (np.array_equal(values[:,:,0],values[:,:,1]) and np.array_equal(values[:,:,0],values[:,:,2])):
            raise ValueError("stereo roughness texture is not grayscale")
        return values[:,:,0]

    a,b = roughness(before_dir,old),roughness(after_dir,new)
    # The unchanged UV atlas covers a 256 x 192 mm rectangle, starting at
    # (-64,-64) mm. Pixel centres inside either 11.7 mm rim receive zero shell
    # weight; centres outside both 12.5 mm rims receive pure shell roughness.
    y,x = np.indices(a.shape)
    x = (x+.5)*.0005-.064
    y = (y+.5)*.0005-.064
    radius = np.minimum(np.hypot(x,y),np.hypot(x-.120,y))
    shell = radius >= .0125
    protected = radius <= .0117
    if not np.array_equal(a[protected],b[protected]) or np.any(b>a):
        raise ValueError("roughness changed inside the lens/rim, or increased outside it")
    for name,mask in (("window",radius <= .0093),("rim",(radius >= .0098) & protected),("shell",shell)):
        for texels,swatches in ((a,palette),(b,updated)):
            if not np.all(texels[mask] == round(255*swatches[name][1])):
                raise ValueError(f"stereo {name} texels differ from declared roughness")
    return dict(before=palette["shell"][1],after=updated["shell"][1],color_sha256=old_row["base_sha256"],
                roughness_sha256=[old_row["roughness_sha256"],new_row["roughness_sha256"]],
                unchanged_window_and_rim_texels=int(protected.sum()),changed_roughness_texels=int((a!=b).sum()))


def verify_polish(before, after, before_dir, after_dir):
    old, old_rows, old_hash, _ = verified_finish(before,before_dir)
    new, new_rows, new_hash, normals = verified_finish(after,after_dir)
    roughness = {}
    for material_id in old:
        a = comparable_material(old[material_id])
        b = comparable_material(new[material_id])
        link = material_id.removeprefix("r1-material-")
        if link in PAINTED_ARMS:
            r0,r1 = a.pop("roughness"),b.pop("roughness")
            if not 0. <= r1 < r0 <= 1.:
                raise ValueError(f"painted link must have lower roughness: {link}")
            roughness[link] = (r0,r1)
        if link == "zed_link":
            a["roughness_map"]["image"].pop("sha256")
            b["roughness_map"]["image"].pop("sha256")
        same(a,b,link+" remaining material parameters and texture contents")
    same(set(roughness),PAINTED_ARMS,"registered arm and gripper shells")
    for link in old_rows:
        if link != "zed_link":
            same(old_rows[link],new_rows[link],link+" wheel textures and UV mapping")
    h0,h1 = old_rows["zed_link"],new_rows["zed_link"]
    for k in ("parts","mapping","components","base_sha256"):
        same(h0[k],h1[k],"stereo "+k)
    head = checked_head_atlas(before_dir,after_dir,old["r1-material-zed_link"],new["r1-material-zed_link"],h0,h1)
    return dict(roughness_by_link=roughness, camera_shell=head, normal_refinement=normals,
                studio_details_sha256=old_hash,polished_details_sha256=new_hash)


def fonts():
    for family in ("DejaVuSans.ttf","arial.ttf"):
        try:
            return ImageFont.truetype(family,22), ImageFont.truetype(family,17)
        except OSError:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def polish_board(path, old, new):
    gap, width = 14, 510
    canvas = Image.new("RGB",(2*width+3*gap,507),"#172028")
    draw = ImageDraw.Draw(canvas)
    title,label = fonts()
    for col,(image,name) in enumerate(((old[143],"Studio"),(new[143],"Polished"))):
        x = gap+col*(width+gap)
        draw.text((x,9),name+"  |  Frame 143",fill="white",font=title)
        y = 50
        for name,crop in POLISHED_CROPS.items():
            draw.text((x,y),name,fill="white",font=label)
            part = image.crop(crop)
            factor = min(width/part.width,190/part.height)
            size = (round(part.width*factor),round(part.height*factor))
            canvas.paste(part.resize(size,Image.Resampling.LANCZOS),(x,y+24))
            y += 224
    canvas.save(path)


def board(path, old, new):
    gap, width, heading = 16, 720, 44
    full_height = 405
    canvas = Image.new("RGB",(2*width+3*gap,715),"#172028")
    draw = ImageDraw.Draw(canvas)
    title,label = fonts()
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
    pair = (before.get("finish","urdf"),after["finish"])
    if pair == ("urdf","studio"):
        finishes = verify_materials(before,after,after_dir)
        draw_board,crops = board,CROPS
    elif pair == ("studio","polished"):
        finishes = verify_polish(before,after,before_dir,after_dir)
        draw_board,crops = polish_board,POLISHED_CROPS
    else:
        raise ValueError(f"unsupported R1 finish comparison: {pair}")
    sources = {}
    for i in old:
        if ImageChops.difference(old[i],new[i]).getbbox() is None:
            raise ValueError(f"identical displayed RGB at frame {i}")
        sources[i] = {label:digest(root/f"frame-{i:03}.png") for label,root in zip(pair,(before_dir,after_dir))}
    region_changes = {}
    for label,box in crops.items():
        difference = ImageChops.difference(old[143].crop(box),new[143].crop(box))
        per_pixel = np.asarray(difference)
        count = int(np.any(per_pixel!=0,axis=2).sum())
        if not count:
            raise ValueError(f"unchanged displayed RGB in frame 143 {label}")
        region_changes[label] = dict(changed_pixels=count,mean_absolute_code_difference=float(per_pixel.mean()))
    output.mkdir(parents=True, exist_ok=False)
    draw_board(output/"comparison.png",old,new)
    evidence = dict(frames_checked=FRAMES, size=SIZE, frame_rate=30, links_per_frame=len(before["model"]["links"]),
                    matching=["URDF and original converted meshes", "stage geometry", "all joints, link poses and base poses",
                              "all camera intrinsics, extrinsics and frame clocks", "studio lighting, HDR panorama and exposure",
                              "device, film sampling and stream format"],
                    film_aa=before["film_aa"], exposure=before["exposure"], hdr_sha256=hdr["sha256"],
                    urdf_sha256=before["model"]["urdf_sha256"],
                    movie_sha256=dict(zip(pair,(before["movie_sha256"],after["movie_sha256"]))),
                    materials=finishes,png_sha256=sources,frame_143_crop_boxes=crops,frame_143_crop_changes=region_changes)
    (output/"comparison.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("before","after","output"):
        cli.add_argument("--"+arg,type=Path,required=True)
    args = cli.parse_args()
    compare(args.before,args.after,args.output)
