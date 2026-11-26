"""One indexed, instanced 3-D shop floor built from source MJCF and recorded qpos.

No render contexts or offscreen station images are created here. Coordinates are
right-handed metres, Z up. Matrices are mathematical world-from-mesh transforms
in C-contiguous row-major arrays; transpose once when packing a GLSL mat4.
"""

from collections import defaultdict
from dataclasses import dataclass
import json

import mujoco as mj
import numpy as np

from bimanual_wrist_render import LEGACY_MARKER_LAYOUT, _box_faces, marker_layout


def _transform(xyz=(0., 0., 0.), rotation=None, scale=(1., 1., 1.)):
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = (np.eye(3) if rotation is None else rotation) * np.asarray(scale)
    m[:3, 3] = xyz
    return m


def _rotation(quat):
    matrix = np.empty(9)
    mj.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def _along(axis):
    z = np.asarray(axis, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross((1, 0, 0) if abs(z[0]) < .8 else (0, 1, 0), z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def _revolve(profile, sectors=16):
    """Revolve (radius, z, radial-normal, z-normal) rings, with single poles."""
    angle = np.arange(sectors)*2.*np.pi/sectors
    xy = np.column_stack((np.cos(angle), np.sin(angle)))
    pos, norm, faces = [], [], []
    previous = None
    for radius, z, nr, nz in profile:
        if radius == 0:
            ring = [len(pos)]
            pos.append((0., 0., z))
            norm.append((0., 0., nz))
        else:
            ring = list(range(len(pos), len(pos)+sectors))
            pos.extend((radius*x, radius*y, z) for x, y in xy)
            norm.extend((nr*x, nr*y, nz) for x, y in xy)
        if previous is not None and previous[:2] != ring[:2]:
            # Coplanar duplicated cap/side rings have different normals. They
            # meet at a crease, so never emit triangles between those rings.
            if tuple(pos[previous[0]]) != tuple(pos[ring[0]]):
                for i in range(sectors):
                    j = (i+1) % sectors
                    if len(previous) == 1:
                        faces.append((previous[0], ring[j], ring[i]))
                    elif len(ring) == 1:
                        faces.append((previous[i], previous[j], ring[0]))
                    else:
                        faces.extend(((previous[i], previous[j], ring[j]), (previous[i], ring[j], ring[i])))
        previous = ring
    return pos, norm, faces


class _Atlas:
    def __init__(self):
        self.slots = {}
        self.vertices = []
        self.indices = []
        self.ranges = []
        self.nv = self.ni = 0

    def mesh(self, key):
        if key in self.slots:
            return self.slots[key]
        kind = key[0]
        if kind == "box":
            pos, normal, faces = _box_faces(np.ones(3))
        elif kind == "cylinder":
            pos, normal, faces = _revolve(((0,-1,0,-1),(1,-1,0,-1),(1,-1,1,0),
                                            (1,1,1,0),(1,1,0,1),(0,1,0,1)))
        elif kind in ("disc", "rim"):
            # Order outer -> inner gives the upward normal for a flat top.
            inner = 0. if kind == "disc" else key[1]
            pos, normal, faces = _revolve(((1,0,0,1),(inner,0,0,1)), sectors=24)
        else:
            half = key[1] if kind == "capsule" else 0.
            profile = [(0., -half-1., 0., -1.)]
            for i in range(-2, 3):
                angle = i*np.pi/6
                r, z = np.cos(angle), np.sin(angle)
                offsets = (-half, half) if i == 0 and half else ((-half if i < 0 else half),)
                profile.extend((r, z+h, r, z) for h in offsets)
            profile.append((0., half+1., 0., 1.))
            pos, normal, faces = _revolve(profile)
        vertices = np.column_stack((pos, normal)).astype("f4")
        index = np.asarray(faces, dtype="u4").reshape(-1)
        slot = len(self.ranges)
        self.ranges.append((self.nv, len(vertices), self.ni, len(index)))
        self.nv += len(vertices)
        self.ni += len(index)
        self.slots[key] = slot
        self.vertices.append(vertices)
        self.indices.append(index)
        return slot

    def shape(self, kind, dims):
        if kind == mj.mjtGeom.mjGEOM_BOX:
            return self.mesh(("box",)), dims
        if kind in (mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_ELLIPSOID):
            return self.mesh(("sphere",)), (np.repeat(dims[0], 3) if kind == mj.mjtGeom.mjGEOM_SPHERE else dims)
        r, half = dims[:2]
        if kind == mj.mjtGeom.mjGEOM_CYLINDER:
            return self.mesh(("cylinder",)), (r,r,half)
        if kind == mj.mjtGeom.mjGEOM_CAPSULE:
            # Tessellation LOD: group ratios to 0.01 radius while retaining
            # both exact pole positions and the original cross-sectional radius.
            ratio = round(float(half/r), 2)
            return self.mesh(("capsule", ratio)), (r, r, (half+r)/(ratio+1.))
        raise ValueError(f"unsupported visible MuJoCo geometry type {kind}")


def _template(model, atlas, task, record):
    parts = []
    box, cylinder = atlas.mesh(("box",)), atlas.mesh(("cylinder",))
    metal = (.080,.125,.146,1.)
    gray = (.36,.42,.42,1.)
    arm_dark = (.070,.105,.13,1.)
    trim = (.020,.39,.34,1.) if task == "carry" else (.69,.32,.065,1.)

    def add(mesh, pos, scale, color, body=0, rotation=None):
        parts.append((mesh, body, _transform(pos, rotation, scale), np.asarray(color, dtype="f4")))

    # Bench top is exactly the experimental support plane z=0. Leave open
    # space under the worktop so the close travelling shot reads as a room.
    add(box, (-.16,0,-.021), (.70,.442,.021), (.43,.45,.42,1.))
    add(box, (-.16,0,-.060), (.69,.435,.018), metal)
    add(box, (-.16,-.437,-.052), (.69,.003,.004), trim)
    for x in (-.795,.475):
        for y in (-.39,.39):
            add(box, (x,y,-.411), (.021,.021,.334), gray)
            add(box, (x,y,-.751), (.032,.032,.008), arm_dark)
        add(box, (x,0,-.56), (.013,.40,.014), metal)
    add(box, (-.16,.39,-.56), (.638,.013,.015), metal)
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    for side in ("left", "right"):
        j = model.joint(side+"_j0").id
        x,y,h = data.xanchor[j]
        if h <= .020:
            raise ValueError(f"{side} base cannot be mounted above the worktop")
        add(cylinder, (x,y,.006), (.070,.070,.006), arm_dark)
        add(cylinder, (x,y,(h+.012)/2), (.044,.044,(h-.012)/2), metal)
        add(cylinder, (x,y,h-.005), (.051,.051,.005), gray)
    for g in range(model.ngeom):
        if model.geom(g).name == "table" and model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE:
            continue
        mat_id = model.geom_matid[g]
        rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[g]
        if rgba[3] == 0:
            continue
        if rgba[3] != 1:
            raise ValueError(f"unsupported partial alpha on {model.geom(g).name}")
        mesh, scale = atlas.shape(model.geom_type[g], model.geom_size[g])
        add(mesh, model.geom_pos[g], scale, rgba, model.geom_bodyid[g], _rotation(model.geom_quat[g]))
    # Covers surround the actual hinge axes. Distal segments keep their source
    # collision/visual shapes, with no invented links or tool extension.
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        body = model.jnt_bodyid[j]
        pos, axis = model.jnt_pos[j], model.jnt_axis[j]
        rad = .033 if int(model.joint(j).name.rsplit("j",1)[-1]) < 3 else .021
        rot = _along(axis)
        add(cylinder, pos, (rad,rad,.68*rad), arm_dark, body, rot)
        add(cylinder, pos+.69*rad*axis, (.69*rad,.69*rad,.047*rad), gray, body, rot)
    if record.get("fiducials"):
        goal = model.site("task_goal").id
        goal_frame = _transform(model.site_pos[goal], _rotation(model.site_quat[goal]))
        for mark in marker_layout(task, record.get("marker_layout_version") or LEGACY_MARKER_LAYOUT):
            goal_mark = mark["role"] == "goal"
            body = model.site_bodyid[goal] if goal_mark else model.body("payload").id
            local = _transform(mark["xyz"], _along(mark["normal"]))
            if goal_mark:
                local = goal_frame @ local
            radius = mark["radius_m"]
            for key, size, rgb in ((("disc",),radius,mark["rgb"]),
                                   (("rim",radius/(radius+.003)),radius+.003,(.80,.80,.76))):
                add(atlas.mesh(key),local[:3,3],(size,size,1),(*rgb,1.),body,local[:3,:3])
    return parts, data


@dataclass
class _Motion:
    stations: np.ndarray
    times: np.ndarray
    phases: np.ndarray
    rates: np.ndarray
    poses: np.ndarray
    slots: np.ndarray
    lane: np.ndarray
    body: np.ndarray


class TrainingFloor:
    """Resident geometry, paint and object IDs with one streamed world matrix per piece."""

    def __init__(self, stations):
        stations = tuple(stations)
        if not stations or not all(s.sampled for s in stations):
            raise ValueError("training floor requires recorded endpoints: build_stations(..., sampled=True)")
        rows = max(s.row for s in stations)+1
        cols = max(s.col for s in stations)+1
        if [(s.row,s.col) for s in stations] != list(np.ndindex(rows,cols)):
            raise ValueError("expected one station per cell in row-major order")
        # Group aisles give scale cues without hiding arm links in overhead shots.
        xx = np.arange(cols)*1.52 + (np.arange(cols)//4)*.28
        yy = np.arange(rows)*1.00 + (np.arange(rows)//4)*.28
        xx -= .5*(xx[-1]+xx[0])
        yy -= .5*(yy[-1]+yy[0])
        self.station_centers = np.zeros((rows,cols,3), dtype="f4")
        self.station_centers[:,:,0] = xx
        self.station_centers[:,:,1] = yy[:,None]
        centers = self.station_centers.reshape(-1,3)
        # The source worktop is centered at x=-.16. Report table centers as
        # station_centers and move its whole MJCF world by the same rigid shift.
        self.offsets = centers + np.array((.16,0,0), dtype="f4")
        self.bounds = np.array(((xx[0]-.73, yy[0]-.46, -.765),(xx[-1]+.73,yy[-1]+.46,.72)))
        self.focus = np.array((.16,yy[0],.105))
        atlas = _Atlas()
        meshes, owners, body_ids, locals_, colors = [], [], [], [], []
        self.motions = []
        def append(index, station_id, body, local, rgba):
            meshes.append(index)
            owners.append(station_id)
            body_ids.append(body)
            locals_.append(local)
            colors.append(rgba)
        box = atlas.mesh(("box",))
        add_global = lambda p, dims, rgba: append(box,-1,0,_transform(p,scale=dims),rgba)
        # Keep the factory floor under the entire moving camera frustum.
        add_global((0,0,-.785),(100.,100.,.024),(.28,.32,.32,1.))
        back = yy[-1] + 3.4
        add_global((0,back,1.32),(48.,.09,2.10),(.62,.67,.65,1.))
        for x in np.arange(-42.,43.,3.):
            add_global((x,back-.102,2.07),(1.44,.014,.83),(.60,.77,.80,1.))
            add_global((x,back-.125,1.23),(1.49,.035,.021),(.21,.29,.30,1.))
        # Subtle painted strip beside each group aisle, shared across the floor.
        for column in range(4,cols,4):
            x = .5*(xx[column-1]+xx[column])
            for off in (-.100,.100):
                add_global((x+off,0,-.760),( .009,yy[-1]+.56,.0005),(.63,.55,.35,1.))
        by_model = defaultdict(list)
        for index, station in enumerate(stations):
            by_model[station.clip.provenance["model_sha256"]].append(index)
        for ids in by_model.values():
            first = stations[ids[0]].clip
            model = mj.MjModel.from_xml_path(str(first.xml))
            by_clip = defaultdict(list)
            for i in ids:
                by_clip[stations[i].clip.motion_id].append(i)
            for lane_ids in by_clip.values():
                clip = stations[lane_ids[0]].clip
                rec = json.loads(clip.receipt.read_bytes())
                parts, data = _template(model, atlas, clip.task, rec)
                moving = sorted({int(p[1]) for p in parts if p[1]})
                body_slot = {b:i for i,b in enumerate(moving)}
                poses = np.empty((len(clip.qpos),len(moving),3,4), dtype="f4")
                for tick, qpos in enumerate(clip.qpos):
                    data.qpos[:] = qpos
                    mj.mj_kinematics(model, data)
                    poses[tick,:,:,:3] = data.xmat[moving].reshape(-1,3,3)
                    poses[tick,:,:,3] = data.xpos[moving]
                del data
                dynamic, lanes, bodies = [], [], []
                for lane, i in enumerate(lane_ids):
                    for slot, body, local, rgba in parts:
                        if body:
                            dynamic.append(len(meshes))
                            lanes.append(lane)
                            bodies.append(body_slot[body])
                        append(slot,i,body,local,rgba)
                self.motions.append(_Motion(np.array(lane_ids),clip.times,
                    np.array([stations[i].phase_s for i in lane_ids]),
                    np.array([stations[i].playback_rate for i in lane_ids]),poses,
                    np.asarray(dynamic),np.asarray(lanes),np.asarray(bodies)))
            del model
        self.vertices = np.ascontiguousarray(np.vstack(atlas.vertices),dtype="f4")
        self.indices = np.ascontiguousarray(np.concatenate(atlas.indices),dtype="u4")
        self.meshes = np.asarray(atlas.ranges,dtype="u4")
        order = np.lexsort((owners,meshes))
        inverse = np.argsort(order)
        self.instance_mesh = np.asarray(meshes,dtype="u4")[order]
        self.instance_id = np.arange(1,len(meshes)+1,dtype="u4")[order]
        self.instance_color = np.asarray(colors,dtype="f4")[order]
        self.instance_station = np.asarray(owners,dtype="i4")[order]
        self._local = np.asarray(locals_,dtype="f4")[order]
        self._world = self._local.copy()
        local_station = self.instance_station >= 0
        self._world[local_station,:3,3] += self.offsets[self.instance_station[local_station]]
        for motion in self.motions:
            motion.slots = inverse[motion.slots]
        counts = np.bincount(self.instance_mesh.astype(int),minlength=len(self.meshes))
        self.draws = np.asarray(np.column_stack((np.cumsum(counts)-counts,counts)),dtype="u4")

    def update(self, seconds):
        """Evaluate exactly the nearest logged qpos as specified by each Station."""
        if not np.isfinite(seconds):
            raise ValueError("non-finite presentation time")
        for group in self.motions:
            times = group.times
            t = (seconds*group.rates+group.phases) % (times[-1]-times[0]) + times[0]
            k = np.clip(np.searchsorted(times,t,side="right")-1,0,len(times)-2)
            u = (t-times[k])/(times[k+1]-times[k])
            sample = k+(u >= .5)
            fk = group.poses[sample[group.lane],group.body]
            local = self._local[group.slots]
            self._world[group.slots,:3,:3] = fk[:,:,:3] @ local[:,:3,:3]
            center = (fk[:,:,:3] @ local[:,:3,3,None])[:,:,0] + fk[:,:,3]
            self._world[group.slots,:3,3] = center + self.offsets[group.stations[group.lane]]
        return self._world


def build_floor(stations):
    return TrainingFloor(stations)
