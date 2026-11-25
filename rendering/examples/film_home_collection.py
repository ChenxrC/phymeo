"""Film the first thirty numbered GLB categories in one deep-focus Vulkan room."""
import argparse
from contextlib import closing, nullcontext
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from _home_collection import (
    COLLECTION_CHAPTERS, build_collection_room, collection_camera, collection_lighting,
    collection_photo_checks, collection_placements,
)
from _home_film_quality import capture_film_frame
from _home_showcase_assets import load_home_assets
from _home_showcase_room import ROOM_EXPOSURE, with_home_backplate
from _panda_movie_encode import _verify
from _panda_scene import pose_matrix
from film_home_showcase import FPS, FRAMES, SIZE, assemble_sources, encoder
from phymeo_rendering import StudioRenderer
from phymeo_rendering.scene import FrameKey, SceneFrame


def record_visible(labels, ids, image, frame, best):
    """Keep each object's most legible actual view and its full camera-frame bounds."""
    counts = {}
    for name, parts in ids.items():
        ys, xs = np.nonzero(np.isin(labels, parts))
        count = int(xs.size)
        counts[name] = count
        if not count:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        # A tighter camera can magnify a sliver of an object cut off by the
        # screen. Prefer a view with all visible parts inside the frame.
        inside = box[0] > 0 and box[1] > 0 and box[2] < image.width and box[3] < image.height
        previous = best.get(name)
        if previous and (inside, count) <= (previous[4], previous[0]):
            continue
        pad = max(6, round(.10 * max(box[2]-box[0], box[3]-box[1])))
        crop = (max(0, box[0]-pad), max(0, box[1]-pad),
                min(image.width, box[2]+pad), min(image.height, box[3]+pad))
        photo = image.crop(crop)
        photo.thumbnail((236, 135), Image.Resampling.LANCZOS)
        best[name] = count, frame, box, photo, inside
    return counts


def save_index(best, path):
    cell_w, cell_h = 260, 183
    sheet = Image.new("RGB", (cell_w*5, cell_h*6), "#f1f0ea")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    for index, (name, (_, frame, _, photo, _)) in enumerate(sorted(best.items())):
        x, y = cell_w*(index % 5), cell_h*(index // 5)
        framed = ImageOps.expand(photo, border=1, fill="#d3d1ca")
        sheet.paste(framed, (x+(cell_w-framed.width)//2, y+6+(136-framed.height)//2))
        number, title = name.split("_", 1)
        draw.text((x+10,y+145), number + "   " + title.replace("-", " "), fill="#172826", font=font)
        draw.text((x+10,y+165), f"{frame/FPS:.1f} s", fill="#506660", font=small)
    sheet.save(path, quality=94, subsampling=0)


def chair_detail(labels, camera, scene):
    """Check five authored oak spindles and the four gaps between them in image space."""
    oak_id, = (part.id for part in scene.instances if part.mesh_id == 'home-oak')
    world_to_eye = np.linalg.inv(pose_matrix(camera.world_pose))
    rows = []
    # Sample interiors at three heights, below the horizontal rail and above the seat.
    for fraction in (.56, .66, .76):
        hits = []
        for offset in np.arange(-2., 2.25, .5):
            world = (-.76 + offset*(.045 + .004*fraction), .702 + .058*fraction,
                     .452 + .365*fraction, 1.)
            x, y, z, _ = world_to_eye @ world
            if z >= -camera.near_m:
                raise RuntimeError('the authored chair is behind the lens')
            u = int(np.floor(camera.cx_px - camera.fx_px*x/z))
            v = int(np.floor(camera.cy_px + camera.fy_px*y/z))
            if not (1 <= u < camera.width-1 and 1 <= v < camera.height-1):
                raise RuntimeError('the chair detail shot cuts off the selected backrest')
            hits.append(bool(np.count_nonzero(labels[v-1:v+2, u-1:u+2] == oak_id) >= 5))
        rows.append(hits)
    pattern = [True, False]*4 + [True]
    return dict(fractions=[.56, .66, .76], oak_id=oak_id, samples=rows,
                five_bars_and_four_gaps=any(row == pattern for row in rows))


def run(args):
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    movie = output / "home-collection-30.mp4"
    if movie.exists() and not args.preview:
        raise FileExistsError(movie)
    root = args.assets.resolve()
    print("Importing 30 source GLBs into one room", flush=True)
    assets = load_home_assets(root, cache / "objects", count=30)
    room = build_collection_room(cache / "room")
    scene, poses, ids, sources = assemble_sources(assets, room, collection_placements(assets), root, "home-collection")
    library = args.library.resolve()
    renderer = StudioRenderer(library_path=library, shader_directory=library.parent, device_index=args.device_index,
                              validation=True, studio=collection_lighting(), exposure=ROOM_EXPOSURE, panel_samples=64)
    checks = collection_photo_checks()
    preparation_views = []
    for i in checks:
        key = FrameKey(scene.key, "whole-room", 0, i, i, "authored")
        frame = SceneFrame(key, i/FPS, poses)
        camera = collection_camera(key, frame.time_s, SIZE[0]*2, SIZE[1]*2)
        preparation_views.append((frame, camera))
    prepared = renderer.prepare_scene(scene, usage='film', cameras=preparation_views,
                                      preserve_meshes=('home-oak',), cache_dir=cache/'scene-preparation')
    if prepared.scene.instances != scene.instances:
        raise RuntimeError('scene preparation changed instance identities, node bindings or local poses')
    scene = prepared.scene
    indices = checks if args.preview else range(FRAMES)
    best = {}
    report = dict(sources=sources, preparation=prepared.report, chapters=COLLECTION_CHAPTERS, scene_key=scene.key,
                  focus="pinhole, full-depth indoor subjects", resolution=SIZE, native_size=[v*2 for v in SIZE],
                  panel_samples=64, visibility={})
    with TemporaryDirectory(dir=output) as staging:
        staged = Path(staging) / movie.name
        print("Opening the complete native scene", flush=True)
        with closing(renderer.open_scene(prepared)) as native:
            session = with_home_backplate(native, scene)
            with nullcontext() if args.preview else encoder(staged, *SIZE) as pipe:
                for i in indices:
                    key = FrameKey(prepared.source_scene_key, "whole-room", 0, i, i, "authored")
                    frame = SceneFrame(key, i/FPS, poses)
                    camera = collection_camera(key, frame.time_s, *SIZE)
                    frame, camera = prepared.bind_view(frame, camera)
                    inspect = i in checks
                    # Zero aperture preserves thin interior bars and the new small objects
                    # at every depth. Spatial integration still antialiases silhouettes.
                    rgb, labels, _ = capture_film_frame(session, frame, camera, focus_distance_m=1.,
                                                        aperture_radius_m=0., scale=2, inspect=inspect)
                    if pipe is not None:
                        pipe.write(rgb)
                    if inspect:
                        image = Image.frombytes("RGB", SIZE, rgb)
                        image.save(output / f"frame-{i:04d}.png")
                        report["visibility"][str(i)] = record_visible(labels, ids, image, i, best)
                        if i == 420:
                            report['chair_detail'] = chair_detail(labels, camera, scene)
                    if inspect or i % 90 == 0:
                        print(f"Captured frame {i+1}/{FRAMES}", flush=True)
            info = session.diagnostics()
            dev = info["device"]
            ray = dev["resident_ray_query_info"]
            report["vulkan"] = dict(device=dev["device_name"], triangles=info["triangles"],
                                    msaa=info["msaa"]["actual_color_samples"], validation_errors=dev["validation_errors"],
                                    ray_query=ray["enabled"], shadow_captures=ray["shadow_captures"])
            if (dev["validation_errors"] or not ray["enabled"] or ray["shadow_captures"] != len(indices)
                    or info["msaa"]["actual_color_samples"] != 4 or info["area_light_samples_per_panel"] != 64
                    or not dev["resident_environment_info"]["loaded"]):
                raise RuntimeError("Vulkan capture differs from the requested optical or lighting settings")
        report["best_views"] = {k: dict(visible_pixels=v[0], frame=v[1], bounds_px=v[2],
                                       inside_frame=v[4]) for k,v in best.items()}
        save_index(best, output / "model-index.jpg")
        missing = [name for name in ids if name not in best or best[name][0] < 256 or not best[name][4]]
        report["insufficient_coverage"] = missing
        manifest = output / ("preview.json" if args.preview else "film.json")
        if missing or not report['chair_detail']['five_bars_and_four_gaps']:
            manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"tour coverage failed: models={missing}, chair={report['chair_detail']}")
        if not args.preview:
            report["video"] = _verify(staged, shutil.which("ffmpeg"), shutil.which("ffprobe"), FPS, FRAMES, *SIZE)
            staged.rename(movie)
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "cache", "output", "library"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    run(parser.parse_args())
