"""Dual wrist RGB observations and a synchronized film from recorded joint states.

Training supplies the current step input. Film replay supplies the saved qpos at
each recorded timestamp. A private MuJoCo data object performs forward kinematics
only; Phymeo's resident Vulkan renderer owns the image formation in both paths.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo_rendering._prepare_cache import publish_bytes
from phymeo_rendering._prepare_geometry import _mesh_bytes
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.outputs import Channel
from phymeo_rendering.photography import StudioRenderer
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene, compose_pose
from phymeo_rendering.studio import StudioLighting


WRIST_SITES = ("left_wrist_camera", "right_wrist_camera")
WRIST_FOV_DEG = 78.0


def marker_layout(task):
    """Printed sticker centres in metres in the payload body or task_goal site frame.

    rgb is linear material reflectance. hue_deg describes the intended display
    colour. Only the coloured disc has radius_m; the white rim adds 3 mm.
    The same manufactured part geometry is used for every seed of its task.
    """
    palette = (
        ("obj-magenta", (.88, .003, .82), 302),
        ("obj-yellow", (.93, .82, .003), 57),
        ("obj-blue", (.007, .018, .84), 237),
        ("obj-violet", (.20, .004, .85), 270),
        ("obj-red", (.90, .003, .003), 0),
        ("goal-green", (.018, .82, .004), 115),
        ("goal-cyan", (.003, .62, .94), 190),
        ("goal-rose", (.90, .004, .205), 330),
    )
    if task == "carry":
        centers = ((-.035, .078, .0503), (-.035, -.078, .0503),
                   (.034, -.055, .0503), (.034, .055, .0503), (-.0603, 0., .004),
                   (-.026, .139, 0.), (-.026, -.139, 0.), (-.085, 0., 0.))
    elif task == "push_t":
        centers = ((0., .092, .0283), (0., -.092, .0283),
                   (.145, -.011, .0283), (.061, .012, .0283), (-.0283, .007, .003),
                   (-.053, .095, 0.), (-.053, -.095, 0.), (.057, .064, 0.))
    else:
        raise ValueError(f"no fiducial installation for {task!r}")
    return tuple(dict(id=name, role="object" if i < 5 else "goal", xyz=xyz,
                      normal=(-1., 0., 0.) if i == 4 else (0., 0., 1.),
                      radius_m=.013 if i < 5 else .014, rgb=rgb, hue_deg=hue)
                 for i, ((name, rgb, hue), xyz) in enumerate(zip(palette, centers, strict=True)))


def _pose(position, rotation):
    quat = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=np.float64).reshape(9))
    return Pose(tuple(map(float, position)), tuple(map(float, quat[[1, 2, 3, 0]])))


def _box_faces(half, *, plane=False):
    points, normals, triangles = [], [], []
    axes = np.eye(3)
    for axis, sign in ([(2, 1)] if plane else [(a, s) for a in range(3) for s in (-1, 1)]):
        normal, u = axes[axis] * sign, axes[(axis + 1) % 3]
        v = np.cross(normal, u)
        start = len(points)
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            points.append(((np.zeros(3) if plane else normal) + x * u + y * v) * half)
            normals.append(normal)
        triangles.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
    return np.array(points), np.array(normals), triangles


def _round_faces(kind, size):
    # Ring normals are analytic. A pole has a single vertex; the cylinder caps
    # have independent normals so the highlights stay faithful to the primitive.
    radius, half = float(size[0]), float(size[1])
    sectors, hemi = 32, 8
    angle = np.arange(sectors) * (2 * np.pi / sectors)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    points, normals, triangles = [], [], []

    def ring(r, z, nr, nz):
        start = len(points)
        if r == 0:
            points.append((0., 0., z))
            normals.append((0., 0., nz))
            return [start]
        points.extend((r * x, r * y, z) for x, y in circle)
        normals.extend((nr * x, nr * y, nz) for x, y in circle)
        return list(range(start, start + sectors))

    def bridge(lower, upper):
        for i in range(sectors):
            j = (i + 1) % sectors
            if len(lower) == 1:
                triangles.append((lower[0], upper[j], upper[i]))
            elif len(upper) == 1:
                triangles.append((lower[i], lower[j], upper[0]))
            else:
                triangles.extend(((lower[i], lower[j], upper[j]), (lower[i], upper[j], upper[i])))

    if kind == mj.mjtGeom.mjGEOM_CYLINDER:
        for low, high in [
            ((0., -half, 0., -1.), (radius, -half, 0., -1.)),
            ((radius, -half, 1., 0.), (radius, half, 1., 0.)),
            ((radius, half, 0., 1.), (0., half, 0., 1.)),
        ]:
            bridge(ring(*low), ring(*high))
    else:
        half = half if kind == mj.mjtGeom.mjGEOM_CAPSULE else 0.
        rings = [ring(0., -half - radius, 0., -1.)]
        for i in range(1, 2 * hemi):
            latitude = (i - hemi) * (math.pi / (2 * hemi))
            nr, nz = math.cos(latitude), math.sin(latitude)
            zs = (-half, half) if i == hemi and half else ((-half if i < hemi else half),)
            for z in zs:
                rings.append(ring(radius * nr, radius * nz + z, nr, nz))
        rings.append(ring(0., half + radius, 0., 1.))
        for lower, upper in zip(rings, rings[1:]):
            bridge(lower, upper)
    points, normals = np.array(points), np.array(normals)
    if kind == mj.mjtGeom.mjGEOM_ELLIPSOID:
        scale = np.asarray(size) / radius
        points *= scale
        normals /= scale
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    return points, normals, triangles


def _scene(model, cache):
    cache.mkdir(parents=True, exist_ok=True)
    meshes, materials, instances = {}, {}, []
    bindings = []
    for i in range(model.ngeom):
        kind = int(model.geom_type[i])
        size = np.array(model.geom_size[i])
        mat_id = int(model.geom_matid[i])
        rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[i]
        if rgba[3] == 0:
            continue
        if rgba[3] != 1.:
            raise ValueError(f"geom {i} uses partial transparency outside the opaque sensor path")
        if kind == mj.mjtGeom.mjGEOM_BOX:
            pts, normals, faces = _box_faces(size)
        elif kind == mj.mjtGeom.mjGEOM_PLANE:
            # MuJoCo's infinite contact plane has a finite photography extent.
            xy = np.maximum(size[:2], 3.)
            pts, normals, faces = _box_faces(np.array([*xy, 0.]), plane=True)
        elif kind in (mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_CAPSULE,
                      mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_ELLIPSOID):
            pts, normals, faces = _round_faces(kind, size)
        else:
            raise ValueError(f"unsupported visual geom {model.geom(i).name!r} of type {kind}")
        blob = _mesh_bytes(pts, normals, None, faces)
        mesh_id = hashlib.sha256(blob).hexdigest()
        if mesh_id not in meshes:
            path = cache / (mesh_id + ".gltf")
            publish_bytes(path, blob)
            meshes[mesh_id] = Mesh(mesh_id, path.as_uri())
        body = int(model.geom_bodyid[i])
        node = f"body-{body}"
        color = tuple(map(float, rgba[:3]))
        mat_key = (color, body == 0)
        if mat_key not in materials:
            materials[mat_key] = Material(f"surface-{len(materials)}", color,
                                          .0 if body == 0 else .08, .72 if body == 0 else .4)
        w, x, y, z = map(float, model.geom_quat[i])
        local = Pose(tuple(map(float, model.geom_pos[i])), (x, y, z, w))
        instances.append(Instance(i + 1, mesh_id, materials[mat_key].id, node, local))
        bindings.append((i, body, node))
    identity = hashlib.sha256(repr((tuple(instances), tuple(materials.values()))).encode()).hexdigest()
    scene = VisualScene("bimanual-wrists-" + identity, tuple(meshes.values()),
                        tuple(materials.values()), tuple(instances))
    return scene, bindings


def _sticker_mesh(cache, radius, rim):
    # The annulus and disc tessellate the same plane without overlap. Their
    # normal is +Z; the attachment pose turns them onto the supporting face.
    sectors = 24
    angle = np.arange(sectors) * (2. * np.pi / sectors)
    circle = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(sectors)))
    if rim:
        points = np.vstack((radius * circle, (radius + .003) * circle))
        faces = []
        for i in range(sectors):
            j = (i + 1) % sectors
            faces.extend(((i, i + sectors, j + sectors), (i, j + sectors, j)))
    else:
        points = np.vstack((np.zeros((1, 3)), radius * circle))
        faces = [(0, 1 + i, 1 + (i + 1) % sectors) for i in range(sectors)]
    normals = np.broadcast_to((0., 0., 1.), points.shape)
    blob = _mesh_bytes(points, normals, None, faces)
    digest = hashlib.sha256(blob).hexdigest()
    path = cache / (digest + ".gltf")
    publish_bytes(path, blob)
    return Mesh(digest, path.as_uri())


def _attach_markers(model, cache, scene):
    model_name = bytes(model.names).split(b"\0", 1)[0].decode("utf-8")
    task = model_name.removeprefix("bimanual_").rsplit("_", 1)[0]
    markers = marker_layout(task)
    payload = int(model.body("payload").id)
    goal = int(model.site("task_goal").id)
    goal_body = int(model.site_bodyid[goal])
    w, x, y, z = map(float, model.site_quat[goal])
    goal_pose = Pose(tuple(map(float, model.site_pos[goal])), (x, y, z, w))
    meshes, materials, instances, names = {}, [], [], {}
    white = Material("sticker-rim", (.80, .80, .76), 0., .84)
    materials.append(white)
    for spec in markers:
        is_goal = spec["role"] == "goal"
        node = f"body-{goal_body if is_goal else payload}"
        # The red disc faces forward (payload -X); the others face upwards.
        orientation = ((0., -math.sqrt(.5), 0., math.sqrt(.5))
                       if spec["normal"][0] < 0 else (0., 0., 0., 1.))
        local = Pose(spec["xyz"], orientation)
        if is_goal:
            local = compose_pose(goal_pose, local)
        ink = Material("sticker-" + spec["id"], spec["rgb"], 0., .84)
        materials.append(ink)
        for rim, mat in ((False, ink), (True, white)):
            key = (spec["radius_m"], rim)
            if key not in meshes:
                meshes[key] = _sticker_mesh(cache, *key)
            instance_id = model.ngeom + 1 + len(instances)
            instances.append(Instance(instance_id, meshes[key].id, mat.id, node, local))
            names[instance_id] = spec["id"] + ("-rim" if rim else "")
    digest = hashlib.sha256(repr((scene.key, instances, materials)).encode()).hexdigest()
    result = VisualScene("bimanual-marked-" + digest, scene.meshes + tuple(meshes.values()),
                         scene.materials + tuple(materials), scene.instances + tuple(instances),
                         scene.lights)
    return result, names, markers


class BimanualWristRenderer:
    """Resident RGB cameras. qpos is the visual input only, never a policy feature."""

    def __init__(self, model, cache_dir, *, library_path=None, size=(96, 96),
                 device_index=None, panel_samples=4, fiducials=False):
        self.model = model
        self.data = mj.MjData(model)
        self.site_ids = tuple(int(model.site(name).id) for name in WRIST_SITES)
        cache = Path(cache_dir).resolve()
        self.scene, self.bindings = _scene(model, cache)
        self.visual_names = {i + 1: model.geom(i).name or str(i) for i, _, _ in self.bindings}
        self.fiducials = fiducials
        self.marker_specs = ()
        if fiducials:
            self.scene, extra_names, self.marker_specs = _attach_markers(model, cache, self.scene)
            self.visual_names.update(extra_names)
        self.bodies = tuple(sorted({b for _, b, _ in self.bindings}))
        self.subject = tuple(i for i, _, _ in self.bindings if model.geom_type[i] != mj.mjtGeom.mjGEOM_PLANE)
        if not self.subject:
            raise ValueError("expected visible robot and payload geometry")
        mj.mj_kinematics(model, self.data)
        center = tuple(map(float, self.data.site_xpos[list(self.site_ids)].mean(axis=0)))
        lighting = StudioLighting(center_m=center, scale_m=.85, panel_strength=.28,
                                  panel_size_scale=.7, environment_radiance=.10)
        self.renderer = StudioRenderer(size=size, library_path=library_path, device_index=device_index,
                                       studio=lighting, panel_samples=panel_samples, profile="auto",
                                       exposure=3.0)
        self.session = self.renderer.open_scene(self.scene)
        self.device_report = self.renderer.probe_device()
        self.last_capture = {}
        self.wide_pose = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def _set_q(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError("qpos must be one finite state matching the compiled model")
        self.data.qpos[:] = q
        mj.mj_kinematics(self.model, self.data)

    def _frame(self, q, tick, time_s, epoch, phase):
        self._set_q(q)
        key = FrameKey(self.scene.key, "bimanual", epoch, tick, tick, phase)
        nodes = tuple(NodePose(f"body-{b}", _pose(self.data.xpos[b], self.data.xmat[b]))
                      for b in self.bodies)
        return SceneFrame(key, float(time_s), nodes)

    def fit_overview(self, qpos):
        """Fit one stationary overview to the physical states retained for the entire film."""
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        ids = list(self.subject)
        radius = self.model.geom_rbound[ids, None]
        wrist_axis = None
        for q in qpos:
            self._set_q(q)
            if wrist_axis is None:
                wrist_axis = (self.data.site_xpos[self.site_ids[1]] - self.data.site_xpos[self.site_ids[0]])[:2]
            centers = self.data.geom_xpos[ids]
            low = np.minimum(low, np.min(centers - radius, axis=0))
            high = np.maximum(high, np.max(centers + radius, axis=0))
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("empty trajectory")
        aim = .5 * (low + high)
        # Keep the observation axis oblique to the line between the wrists.
        flat = wrist_axis / max(np.linalg.norm(wrist_axis), .001)
        if np.linalg.norm(flat) < .01:
            flat = np.array((0., -1.))
        # A front three-quarter view keeps the workpiece in front of the shoulders.
        back = np.array([flat[0] - .55 * flat[1], flat[1] + .55 * flat[0], .72])
        back /= np.linalg.norm(back)
        right = np.cross((0., 0., 1.), back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        half = .5 * (high - low)
        # The 4:3 overview uses a fixed 43 degree vertical lens.
        tangent = math.tan(math.radians(43.) * .5)
        distance = abs(back) @ half + 1.08 * max(abs(right) @ half / (4 / 3), abs(up) @ half) / tangent
        self.wide_pose = _pose(aim + distance * back, np.column_stack((right, up, back)))

    def _rgb(self, frame, name, pose, size, vfov, inspect=False):
        cam = self.renderer.camera(name, frame.key, pose, size=size, vertical_fov_degrees=vfov,
                                   near_m=.004, far_m=12.)
        channels = (Channel.RGB, Channel.INSTANCE_ID) if inspect else (Channel.RGB,)
        shot = self.session.capture(frame, CaptureRequest(cam, RenderPath.RASTER, channels))
        if shot.frame_key != frame.key or shot.camera != cam or shot.time_s != frame.time_s:
            raise RuntimeError("camera exposure differs from the requested state")
        planes = {p.channel: p.data for p in shot.planes}
        pixels = planes[Channel.RGB]
        self.last_capture[name] = {
            "phase": frame.key.phase, "time_s": frame.time_s, "tick": frame.key.tick,
            "epoch": frame.key.epoch, "position_m": pose.position_m,
            "orientation_xyzw": pose.orientation_xyzw, "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
            "size": [cam.width, cam.height], "fx_fy_px": [cam.fx_px, cam.fy_px],
        }
        if inspect:
            ids, counts = np.unique(np.frombuffer(planes[Channel.INSTANCE_ID], "<u4"), return_counts=True)
            self.last_capture[name]["visible_geom_pixels"] = {
                self.visual_names[int(i)]: int(n)
                for i, n in zip(ids, counts) if i
            }
        return np.frombuffer(pixels, np.uint8).reshape(cam.height, cam.width, 3)

    def _wrists(self, frame, size, inspect=False):
        return np.stack([
            self._rgb(frame, name, _pose(self.data.site_xpos[site], self.data.site_xmat[site]),
                      size, WRIST_FOV_DEG, inspect)
            for site, name in zip(self.site_ids, WRIST_SITES, strict=True)
        ])

    def observe(self, qpos, tick, time_s, *, epoch=0):
        """Return [left, right] × H × W × RGB uint8 pixels at this exact step input."""
        frame = self._frame(qpos, tick, time_s, epoch, "step_input")
        self.last_capture = {}
        return self._wrists(frame, self.renderer.size)

    def film_views(self, qpos, tick, time_s, *, wrist_size=(360, 360), inspect=False):
        if self.wide_pose is None:
            raise ValueError("fit the overview to the recorded trajectory before filming")
        frame = self._frame(qpos, tick, time_s, 0, "accepted_endpoint")
        self.last_capture = {}
        wide = self._rgb(frame, "overview", self.wide_pose, (960, 720), 43., inspect)
        return wide, self._wrists(frame, wrist_size, inspect)


def _canvas(wide, wrists, title, t, tick, playback_speed=1., outcome=None, *, fiducials=False):
    from PIL import Image, ImageDraw, ImageFont

    # All lettering is outside the sensor planes. The wrists retain the same
    # square field of view as the policy at a higher sampling resolution.
    board = Image.new("RGB", (1352, 828), "#1c282c")
    board.paste(Image.fromarray(wide), (8, 72))
    board.paste(Image.fromarray(wrists[0]), (984, 51))
    board.paste(Image.fromarray(wrists[1]), (984, 447))
    painter = ImageDraw.Draw(board)
    try:
        heading = ImageFont.truetype("DejaVuSans.ttf", 23)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        heading = small = ImageFont.load_default()
    painter.text((12, 9), title, fill="#f0eee5", font=heading)
    painter.text((12, 46), "OVERVIEW", fill="#9bc5c4", font=small)
    if fiducials:
        painter.text((696, 46), "PRINTED FIDUCIALS", fill="#b6c7c9", font=small)
    painter.text((985, 26), "LEFT WRIST", fill="#9bc5c4", font=small)
    painter.text((985, 420), "RIGHT WRIST", fill="#9bc5c4", font=small)
    painter.text((12, 799), f"Simulation  {t:7.3f} s     Tick  {tick:d}     Playback  {playback_speed:g}x",
                 fill="#d2d9d4", font=small)
    if outcome is not None:
        label = {"carry": "Carry", "push_t": "Push T"}[outcome["task"]]
        image_state = "RGB masked" if outcome["policy_rgb_masked"] else "wrist RGB"
        painter.text((120, 46), f"{label}   /   seed {outcome['seed']}   /   "
                     f"{outcome['active_arms']}   /   {image_state}", fill="#b6c7c9", font=small)
        status = "Final outcome: GOAL MET" if outcome["success"] else "Final outcome: GOAL NOT MET"
        painter.text((676, 799), status, fill="#abd8b6" if outcome["success"] else "#f4bc8a", font=small)
    return board


def _ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _digest(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_result(path, trace, model, model_sha, digest, ticks, times, q, rewards):
    """Join authored episode outcomes to their own model, decision clock and state history."""
    row = json.loads(path.read_text(encoding="utf-8"))
    metrics = row["final_metrics"]
    task = row["task"]
    seed = row["seed"]
    archived_task = bytes(model.names).split(b"\0", 1)[0].decode("utf-8")
    same_run = (
        task in ("carry", "push_t") and metrics["task"] == task and metrics["seed"] == seed
        and archived_task == f"bimanual_{task}_{seed}"
        and metrics["xml_sha256"] == row["scene_sha256"] == model_sha
        and row["physics_ticks"] == metrics["tick"] == int(ticks[-1])
        and metrics["time_s"] == times[-1] and row["decisions"] == len(ticks) - 1
        and type(row["success"]) is bool and metrics["success"] is row["success"]
        and type(row["terminated"]) is bool and type(row["truncated"]) is bool
        and row["terminated"] != row["truncated"] and row["terminated"] == row["success"]
        and metrics["active_arms"] == row["job"]["arms"]
        and metrics["mode"] == row["physics"] and type(row["job"]["black"]) is bool
        and np.array_equal(row["rewards"], rewards)
    )
    if not same_run:
        raise ValueError("episode receipt is inconsistent with the model, final state, task or rewards")
    archive_sha = row.get("trajectory_sha256")
    if archive_sha is not None:
        if archive_sha != digest:
            raise ValueError("receipt SHA-256 belongs to a different trajectory")
        verification = "episode trajectory SHA-256"
    else:
        # Early evaluator receipts predate the NPZ hash. They contain a
        # physical payload pose and both endpoint ticks for *every* decision.
        if path.stem != trace.stem:
            raise ValueError("a legacy receipt must have the same basename as the trajectory")
        measured = row.get("decision_diagnostics", [])
        if len(measured) != len(ticks) - 1:
            raise ValueError("legacy receipt lacks a complete decision history")
        jid = int(model.joint("payload_free").id)
        offset = int(model.jnt_qposadr[jid])
        for i, step in enumerate(measured):
            if (step["input_tick"] != ticks[i] or step["output_tick"] != ticks[i + 1]
                    or not np.array_equal(step["measured"]["payload_xyz_m"], q[i + 1, offset:offset+3])):
                raise ValueError(f"receipt and recorded physical payload disagree at decision {i}")
            quat = q[i + 1, offset+3:offset+7].astype(float)
            w, x, y, z = quat / np.linalg.norm(quat)
            yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            recorded_yaw = float(step["measured"]["payload_yaw_rad"])
            if not np.isfinite(recorded_yaw) or abs(np.arctan2(np.sin(yaw - recorded_yaw),
                                                               np.cos(yaw - recorded_yaw))) > 1e-7:
                raise ValueError(f"receipt and recorded payload yaw disagree at decision {i}")
        verification = "legacy matching basename, original model, all rewards, ticks, payload translations and yaws"
    marked = row.get("fiducials", False)
    if type(marked) is not bool:
        raise ValueError("receipt fiducials must identify whether the wrist images used printed markers")
    return dict(task=task, seed=seed, success=row["success"], physics=row["physics"], fiducials=marked,
                active_arms=row["job"]["arms"], policy_rgb_masked=row["job"]["black"],
                receipt_sha256=_digest(path), trajectory_binding=verification)


def record(args):
    """Display the most recent retained state at each presentation timestamp, plus the endpoint."""
    model_path = args.model.resolve(strict=True)
    trajectory_path = args.trajectory.resolve(strict=True)
    output = args.output.resolve()
    report = output.with_suffix(".frames.json")
    poster = output.with_suffix(".png")
    if any(path.exists() for path in (output, poster, report)):
        raise FileExistsError("choose a fresh output name for the MP4, PNG and exposure manifest")
    if output.suffix.lower() != ".mp4" or not 1 <= args.fps <= 120:
        raise ValueError("supply an .mp4 output and an integer fps from 1 to 120")
    speed = getattr(args, "playback_speed", 1.)
    if not math.isfinite(speed) or speed <= 0.:
        raise ValueError("playback speed must be finite and positive")
    receipt = getattr(args, "episode_receipt", None)
    fiducials = getattr(args, "fiducials", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        q = np.asarray(trajectory["qpos"])
        times = np.asarray(trajectory["time_s"], dtype=np.float64)
        ticks = np.asarray(trajectory["ticks"])
        binding = str(trajectory["model_sha256"].item()) if "model_sha256" in trajectory else None
        rewards = trajectory["rewards"] if receipt is not None else None
    model_sha = _digest(model_path)
    trace_sha = _digest(trajectory_path)
    if binding is not None and binding != model_sha:
        raise ValueError("the recorded episode belongs to a different physical model")
    model = (mj.MjModel.from_binary_path(str(model_path)) if model_path.suffix.lower() == ".mjb"
             else mj.MjModel.from_xml_path(str(model_path)))
    if (q.ndim != 2 or q.shape[1] != model.nq or len(q) < 2 or times.shape != (len(q),)
            or ticks.shape != times.shape or ticks.dtype.kind not in "iu" or ticks[0] < 0
            or not np.isfinite(q).all() or not np.isfinite(times).all() or times[0] < 0
            or not np.all(ticks[1:] > ticks[:-1]) or not np.all(times[1:] > times[:-1])):
        raise ValueError("expected complete increasing saved ticks/times and a finite qpos for each")
    outcome = None
    if receipt is not None:
        if binding is None:
            raise ValueError("episode result display requires the trajectory's physical model SHA-256")
        outcome = _episode_result(receipt.resolve(strict=True), trajectory_path, model, model_sha,
                                  trace_sha, ticks, times, q, rewards)
        if fiducials is not None and fiducials != outcome["fiducials"]:
            raise ValueError("requested marker layout differs from the recorded policy's wrist scene")
        fiducials = outcome["fiducials"]
    fiducials = bool(fiducials)
    elapsed = times - times[0]
    last_presentation = max(1, math.ceil(float(elapsed[-1]) * args.fps / speed - 1e-9))
    clock = speed * np.arange(last_presentation + 1) / args.fps
    # Accommodate timestamps that differ only by binary rounding at the exposure boundary.
    chosen = np.searchsorted(elapsed, clock + 1e-9, side="right") - 1
    chosen[-1] = len(times) - 1
    fresh = np.unique(chosen)
    inspected = set(fresh[np.linspace(0, len(fresh) - 1, min(7, len(fresh)), dtype=int)].tolist())
    frames = []
    tmp = output.with_name(output.stem + ".partial.mp4")
    encoder = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1352x828",
               "-framerate", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "17", "-threads", "3",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
               "-movflags", "+faststart", str(tmp)]
    with BimanualWristRenderer(model, args.cache_dir, library_path=args.library,
                               device_index=args.device_index, panel_samples=16, fiducials=fiducials) as renderer:
        renderer.fit_overview(q)
        device = renderer.device_report
        # stderr inherits the terminal, so a full diagnostic pipe cannot deadlock the writer.
        proc = subprocess.Popen(encoder, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        try:
            last = -1
            for index in chosen:
                index = int(index)
                if index != last:
                    stamp, tick = float(times[index]), int(ticks[index])
                    wide, wrists = renderer.film_views(q[index], tick, stamp, inspect=index in inspected)
                    board = _canvas(wide, wrists, args.title, stamp, tick, speed, outcome, fiducials=fiducials)
                    pixels = board.tobytes()
                    frames.append(dict(sample_index=index, tick=tick, time_s=stamp,
                                       qpos_sha256=hashlib.sha256(q[index].tobytes()).hexdigest(),
                                       views=renderer.last_capture.copy()))
                    if index == fresh[len(fresh) // 2]:
                        poster_bytes = board.copy()
                    last = index
                proc.stdin.write(pixels)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("H.264 encoder failed")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            if not proc.stdin.closed:
                proc.stdin.close()
    os.replace(tmp, output)
    poster_bytes.save(poster)
    manifest = dict(
        model_sha256=model_sha, model_digest_in_trace=binding, trace_sha256=trace_sha,
        video_sha256=_digest(output), source_samples=len(q), movie_frames=len(chosen),
        fps=args.fps, playback_speed=speed, simulated_duration_s=float(elapsed[-1]),
        video_duration_s=len(chosen)/args.fps, episode_outcome=outcome, frame_size=[1352, 828],
        exposure_rule="latest retained sample at the movie clock, with final endpoint included",
        presentation_source_indices=chosen.tolist(), captures=frames, device=device,
        lighting="two area sources, neutral environment, metallic-roughness",
        area_samples_per_light=16, display="pbr-neutral", wrist_vertical_fov_deg=WRIST_FOV_DEG,
        fiducials=fiducials, marker_layout=renderer.marker_specs,
    )
    report.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return output, poster, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="the rollout's exact XML or compiled MJB")
    parser.add_argument("--trajectory", required=True, type=Path, help="NPZ with qpos, time_s and ticks")
    parser.add_argument("--output", required=True, type=Path, help="new H.264 MP4 filename")
    parser.add_argument("--cache-dir", type=Path, required=True, help="generated visual mesh cache")
    parser.add_argument("--library", type=Path, help="native Vulkan library beside its SPIR-V shaders")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--fiducials", action="store_true", default=None,
                        help="film the printed-marker scene; defaults to the matched receipt or an unmarked scene")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--playback-speed", type=float, default=1.,
                        help="simulated seconds per video second (1 is real time, .25 is quarter speed)")
    parser.add_argument("--episode-receipt", type=Path,
                        help="matched evaluation episode JSON for displaying its measured final outcome")
    parser.add_argument("--title", default="Bimanual manipulation")
    for path in record(parser.parse_args()):
        print(path)


if __name__ == "__main__":
    main()
