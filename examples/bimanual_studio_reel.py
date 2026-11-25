"""Replace only the overview of a verified dual-wrist film with a studio view.

The wrist images, clock, outcome and letterbox come from the original video.
This is presentation geometry; it is absent from the policy and the simulator.
"""

import argparse
from contextlib import closing, contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import mujoco as mj
import numpy as np
from PIL import Image

from bimanual_wrist_reel import _verify_movie
from bimanual_wrist_render import _box_faces, _digest, _ffmpeg, _pose, _round_faces, _scene
from phymeo_rendering._prepare_cache import publish_bytes
from phymeo_rendering._prepare_geometry import _mesh_bytes
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.outputs import Channel
from phymeo_rendering.photography import StudioRenderer
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
from phymeo_rendering.studio import StudioLighting


SIZE = (960, 720)
IDENTITY = (0., 0., 0., 1.)


def _unit(vector):
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def _axis_pose(axis):
    z = _unit(axis)
    x = _unit(np.cross((0., 1., 0.) if abs(z[1]) < .9 else (1., 0., 0.), z))
    return _pose((0., 0., 0.), np.column_stack((x, np.cross(z, x), z))).orientation_xyzw


def _studio_scene(model, cache, marker_version):
    source, bindings = _scene(model, cache)
    if marker_version is not None:
        from bimanual_wrist_render import _attach_markers

        source, _, _ = _attach_markers(model, cache, source, marker_version)
    table_id = model.geom("table").id + 1
    instances = [instance for instance in source.instances if instance.id != table_id]
    next_id = max(i.id for i in source.instances) + 1
    meshes = {}
    swatches = {
        "worktop": ((.43, .39, .31), .03, .56), "laminate": ((.22, .16, .10), 0., .64),
        "enamel": ((.095, .16, .17), .30, .39), "granite": ((.105, .135, .145), .05, .68),
        "brushed": ((.53, .57, .59), .72, .32), "graphite": ((.040, .056, .062), .48, .39),
        "wall": ((.66, .71, .70), .0, .93), "floor": ((.31, .34, .33), .0, .77),
        "blue": ((.18, .35, .49), .40, .34), "silver": ((.62, .66, .67), .51, .33),
        "rubber": ((.018, .025, .027), .0, .81), "edge": ((.72, .44, .105), .21, .46),
    }
    materials = {k: Material("studio-"+k, *p) for k, p in swatches.items()}
    static = "body-0"

    def add(kind, position, shape, material, *, node=static, orientation=IDENTITY):
        nonlocal next_id
        key = kind, tuple(shape)
        if key not in meshes:
            if kind == "box":
                pts, normals, faces = _box_faces(np.array(shape))
            else:
                pts, normals, faces = _round_faces(mj.mjtGeom.mjGEOM_CYLINDER, (*shape, 0.))
            blob = _mesh_bytes(pts, normals, None, faces)
            digest = hashlib.sha256(blob).hexdigest()
            path = cache / (digest + ".gltf")
            publish_bytes(path, blob)
            meshes[key] = Mesh(digest, path.as_uri())
        instances.append(Instance(next_id, meshes[key].id, materials[material].id, node,
                                  Pose(tuple(map(float, position)), orientation)))
        next_id += 1

    def box(center, half, material):
        add("box", center, half, material)

    def cylinder(center, radius, half, material, *, node=static, axis=(0., 0., 1.)):
        add("cylinder", center, (radius, half), material, node=node, orientation=_axis_pose(axis))

    # The bench surface is exactly on the physical contact plane. The orange
    # T and turquoise goal graphics retain the source colour and height.
    box((-.16, 0., -.012), (.74, .48, .012), "worktop")
    box((-.16, 0., -.030), (.741, .481, .006), "laminate")
    box((-.16, 0., -.055), (.733, .473, .019), "enamel")
    for x in (-.84, .52):
        for y in (-.42, .42):
            box((x, y, -.415), (.026, .026, .341), "brushed")
            box((x, y, -.742), (.032, .032, .026), "rubber")
    for y in (-.42, .42):
        box((-.16, y, -.56), (.676, .016, .021), "enamel")
    box((-.16, -.475, -.057), (.699, .002, .0016), "edge")
    # A neutral workshop with architectural depth, a service cabinet, reveals
    # and narrow pull handles. The props stay behind the experimental area.
    box((0., 0., -.78), (8., 8., .012), "floor")
    box((0., 1.56, 1.1), (5., .045, 1.88), "wall")
    box((-2.35, -.1, 1.1), (.045, 1.66, 1.88), "wall")
    box((-.20, 1.30, -.25), (.97, .205, .51), "enamel")
    box((-.20, 1.30, .275), (1., .23, .014), "worktop")
    for x in (-.83, -.20, .43):
        box((x, 1.088, -.18), (.302, .006, .40), "granite")
        box((x+.20, 1.07, .075), (.061, .011, .007), "brushed")

    d = mj.MjData(model)
    mj.mj_kinematics(model, d)
    bases = []
    for side, shell in (("left", "blue"), ("right", "silver")):
        first = model.joint(f"{side}_j0").id
        bx, by, h = map(float, d.xanchor[first])
        if h <= .028:
            raise ValueError("root joint too close to worktop for the pedestal")
        bases.append((bx, by, h))
        cylinder((bx, by, .006), .079, .006, "rubber")
        cylinder((bx, by, .016), .075, .004, "brushed")
        cylinder((bx, by, (h+.020)/2), .046, (h-.020)/2, "enamel")
        cylinder((bx, by, h-.006), .050, .006, "brushed")
        for angle in np.arange(4) * (.5*np.pi) + .25*np.pi:
            cylinder((bx+.063*math.cos(angle), by+.063*math.sin(angle), .023),
                     .006, .003, "graphite")
        for j in range(model.njnt):
            if not model.joint(j).name.startswith(side + "_j"):
                continue
            body = int(model.jnt_bodyid[j])
            axis = model.jnt_axis[j]
            center = model.jnt_pos[j]
            index = int(model.joint(j).name.split("_j")[-1])
            radius = (.039 if index < 3 else .024)
            half = (.025 if index < 3 else .017)
            cylinder(center, radius, half, shell, node=f"body-{body}", axis=axis)
            for sign in (-1., 1.):
                cap = center + sign*(half+.001)*axis
                cylinder(cap, radius*.81, .001, "graphite", node=f"body-{body}", axis=axis)
                disk = cap + sign*.002*axis
                cylinder(disk, radius*.53, .001, "brushed", node=f"body-{body}", axis=axis)
    available = {m.id: m for m in (*source.meshes, *meshes.values())}
    all_materials = source.materials + tuple(materials.values())
    digest = hashlib.sha256(repr((instances, all_materials)).encode()).hexdigest()
    result = VisualScene("working-cell-"+digest, tuple(available.values()), all_materials,
                         tuple(instances), source.lights)
    return result, bindings, bases


def _camera(model, data, q, bindings, bases):
    # Fit the physical arms, original contours and mounting feet for every
    # retained state. Do not include the surrounding architecture in this fit.
    ids = np.array([g for g, _, _ in bindings if g != model.geom("table").id])
    half = model.geom_size[ids].copy()
    for i, g in enumerate(ids):
        kind = model.geom_type[g]
        if kind == mj.mjtGeom.mjGEOM_CAPSULE:
            half[i] = (half[i, 0], half[i, 0], half[i, 0]+half[i, 1])
        elif kind == mj.mjtGeom.mjGEOM_SPHERE:
            half[i] = half[i, 0]
    signs = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    corners = half[:,None,:]*signs
    points = [np.array([(x+dx, y+dy, z) for x,y,_ in bases
                        for dx in (-.079,.079) for dy in (-.079,.079) for z in (-.025,.02)])]
    for state in q:
        data.qpos[:] = state
        mj.mj_kinematics(model, data)
        rotated = corners @ data.geom_xmat[ids].reshape(-1,3,3).transpose(0,2,1)
        points.append((rotated + data.geom_xpos[ids,None,:]).reshape(-1,3))
    cloud = np.concatenate(points)
    back = _unit((.53, -1., .67))
    right = _unit(np.cross((0.,0.,1.), back))
    up = np.cross(back, right)
    axes = np.column_stack((right, up, back))
    projected = cloud @ axes
    center = .5*(projected.max(axis=0)+projected.min(axis=0))
    offsets = projected - center
    tan_v = math.tan(math.radians(36.)/2)
    required = np.maximum(abs(offsets[:,0])/(tan_v*SIZE[0]/SIZE[1]), abs(offsets[:,1])/tan_v)
    distance = float(np.max(offsets[:,2] + 1.16*required)) + .021
    return _pose(axes @ center + distance*back, axes)


@contextmanager
def _session(args, model, q, marker_version):
    cache = args.cache_dir.resolve()
    scene, bindings, bases = _studio_scene(model, cache, marker_version)
    data = mj.MjData(model)
    pose = _camera(model, data, q, bindings, bases)
    rig = StudioLighting(center_m=(-.13, 0., .06), scale_m=.98, panel_size_scale=1.08,
                         panel_strength=.58, environment_radiance=.15)
    renderer = StudioRenderer(size=SIZE, library_path=args.library, device_index=args.device_index,
                              studio=rig, panel_samples=16, exposure=3.0, profile="auto")
    bodies = sorted({b for _, b, _ in bindings})
    with closing(renderer.open_scene(scene)) as session:
        def capture(state, tick, stamp):
            data.qpos[:] = state
            mj.mj_kinematics(model, data)
            key = FrameKey(scene.key, "bimanual", 0, int(tick), int(tick), "accepted_endpoint")
            frame = SceneFrame(key, float(stamp), tuple(NodePose(f"body-{b}", _pose(data.xpos[b], data.xmat[b]))
                                                        for b in bodies))
            cam = renderer.camera("overview", key, pose, size=SIZE, vertical_fov_degrees=36., near_m=.01, far_m=12.)
            shot = session.capture(frame, CaptureRequest(cam, RenderPath.RASTER, (Channel.RGB,)))
            if shot.frame_key != key or shot.time_s != frame.time_s or shot.camera != cam:
                raise RuntimeError("studio exposure did not use the retained physical endpoint")
            return shot.planes[0].data
        yield capture, dict(device=renderer.probe_device(), scene_key=scene.key,
                            bases_world_m=bases, camera_position_m=pose.position_m,
                            camera_orientation_xyzw=pose.orientation_xyzw, panel_samples=16)


def _chapter(args, clips_dir, chapter):
    name = Path(chapter["clip"])
    if name.name != chapter["clip"]:
        raise ValueError("unexpected clip file name")
    proof = (clips_dir/name).with_suffix(".frames.json")
    if _digest(proof) != chapter["clip_manifest_sha256"]:
        raise ValueError(f"original exposures do not match the accepted reel: {proof}")
    film = json.loads(proof.read_text(encoding="utf-8"))
    receipt = Path(chapter["receipt"])
    if args.episodes_dir:
        receipt = args.episodes_dir/receipt.name
    model, trace = receipt.with_suffix(".xml"), receipt.with_suffix(".npz")
    if (_digest(trace) != film["trace_sha256"] or _digest(model) != film["model_sha256"]
            or _digest(receipt) != film["episode_outcome"]["receipt_sha256"]
            or film["episode_outcome"]["success"] is not True
            or film["movie_frames"] != chapter["frames"]):
        raise ValueError("original XML, NPZ, success receipt or episode clock mismatch")
    with np.load(trace, allow_pickle=False) as saved:
        q, ticks, t = (saved[k] for k in ("qpos", "ticks", "time_s"))
    for frame in film["captures"]:
        i = frame["sample_index"]
        if (frame["qpos_sha256"] != hashlib.sha256(q[i].tobytes()).hexdigest()
                or len(frame["views"]) != 3
                or any((view["tick"], view["time_s"]) != (ticks[i], t[i])
                       for view in frame["views"].values())):
            raise ValueError("archived sensor plane and retained physical state disagree")
    return mj.MjModel.from_xml_path(str(model)), q, ticks, t, film


def _poster(movie, index, studio_rgb):
    command = [_ffmpeg(), "-v", "error", "-nostdin", "-i", str(movie),
               "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
               "-c:v", "png", "-f", "image2pipe", "pipe:1"]
    data = subprocess.check_output(command)
    with Image.open(io.BytesIO(data)) as original:
        image = original.convert("RGB")
    if image.size != (1352, 828):
        raise ValueError("unexpected original three-camera layout")
    image.paste(Image.frombytes("RGB", SIZE, studio_rgb), (8, 72))
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-movie", type=Path, required=True, help="original verified three-view MP4")
    p.add_argument("--source-manifest", type=Path, required=True, help="original reel JSON")
    p.add_argument("--episodes-dir", type=Path, help="local folder of original XML/NPZ/JSON triplets when relocated")
    p.add_argument("--clips-dir", type=Path, help="folder of the original clip-*.frames.json exposure manifests")
    p.add_argument("--output", type=Path, required=True, help="fresh MP4, or PNG with --poster-only")
    p.add_argument("--poster-only", action="store_true")
    p.add_argument("--poster-frame", type=int, help="absolute movie frame, default middle of first episode")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--device-index", type=int)
    args = p.parse_args()
    source = args.source_movie.resolve(strict=True)
    meta = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if _digest(source) != meta["sha256"]:
        raise ValueError("source MP4 SHA does not match the recorded film")
    chapters = meta["chapters"]
    clips_dir = args.clips_dir or args.source_manifest.parent/meta["segments_directory"]
    fps, n = meta["fps"], meta["frame_count"]
    poster_i = chapters[0]["frames"]//2 if args.poster_frame is None else args.poster_frame
    if not 0 <= poster_i < n:
        raise ValueError("requested poster frame lies outside the film")
    out = args.output.resolve()
    if out.suffix.lower() != (".png" if args.poster_only else ".mp4"):
        p.error("use .png for --poster-only and .mp4 for a movie")
    paths = {out, out.with_suffix(".json"), out.with_suffix(".png")}
    if any(path.exists() for path in paths):
        raise FileExistsError("choose a fresh output and accompanying report/poster")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = dict(source_mp4_sha256=meta["sha256"], source_manifest_sha256=_digest(args.source_manifest),
               fps=fps, poster_frame=poster_i, chapters=[], poster_only=args.poster_only,
               overview="PBR industrial bench and actual joint-axis housings",
               wrists="from the verified input movie, identical crops and timestamps before final H.264 encoding")
    with tempfile.TemporaryDirectory(dir=out.parent) as directory:
        tmp = Path(directory)
        proc = None
        written = 0
        try:
            if not args.poster_only:
                # Even offsets and dimensions preserve the original 4:2:0
                # boundaries. The wrists and labels never enter the renderer.
                filter_graph = ("[1:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[w];"
                                "[0:v][w]overlay=x=8:y=72:shortest=1:format=yuv420[out]")
                cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-n", "-i", str(source),
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x720", "-framerate",
                       str(fps), "-i", "pipe:0", "-filter_complex_threads", "1", "-filter_complex", filter_graph,
                       "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
                       "-threads", "3", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                       "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
                       "-movflags", "+faststart", str(tmp/"studio.mp4")]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for chapter in chapters:
                first, count = chapter["first_frame"], chapter["frames"]
                if args.poster_only and not first <= poster_i < first + count:
                    continue
                if not args.poster_only and first != written:
                    raise ValueError("gap in recorded chapter schedule")
                model, q, ticks, times, film = _chapter(args, clips_dir, chapter)
                schedule = film["presentation_source_indices"]
                if len(schedule) != count:
                    raise ValueError("incorrect number of scheduled exposures")
                marker_version = film.get("marker_layout_version", "split-floor-v1") if film["fiducials"] else None
                with _session(args, model, q, marker_version) as (capture, settings):
                    log["chapters"].append(dict(seed=chapter["seed"], first_frame=first, frames=count,
                                                original_timing_sha256=chapter["clip_manifest_sha256"], **settings))
                    previous = None
                    indices = [poster_i-first] if args.poster_only else range(count)
                    for i in indices:
                        state = schedule[i]
                        if state != previous:
                            image = capture(q[state], ticks[state], times[state])
                            previous = state
                        if first+i == poster_i:
                            _poster(source, poster_i, image).save(tmp/"poster.png")
                        if proc is not None:
                            proc.stdin.write(image)
                            written += 1
            if proc is not None:
                proc.stdin.close()
                if proc.wait() or written != n:
                    raise RuntimeError("studio compositor failed to preserve the full reel clock")
                log["verification"] = _verify_movie(tmp/"studio.mp4", fps, written)
                log["video_sha256"] = _digest(tmp/"studio.mp4")
                log["movie_frames"] = written
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                if not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait()
        log["poster_sha256"] = _digest(tmp/"poster.png")
        (tmp/"report.json").write_text(json.dumps(log, indent=2, allow_nan=False), encoding="utf-8")
        if not args.poster_only:
            os.replace(tmp/"studio.mp4", out)
        os.replace(tmp/"poster.png", out.with_suffix(".png"))
        os.replace(tmp/"report.json", out.with_suffix(".json"))
    print(out)


if __name__ == "__main__":
    main()
