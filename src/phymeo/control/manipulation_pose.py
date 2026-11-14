"""Finite-seed contact-pose geometry audits, without physical integration.

Pad-plane clearance is exact for the declared BOX corners. Object projections
are exact along each pad's three local axes; overlap is only a screening result,
not a grasp, contact, whole-scene collision test, or global reachability proof.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product

import numpy as np

from .manipulation_diagnostic import (
    NativePointIKResult,
    _configuration,
    _joints,
    _readonly,
    _require,
    _vector,
    solve_native_point_ik,
)

MAX_SEEDS = 3
MAX_OBJECT_GEOMS = 32
MAX_MODEL_BYTES = 8 * 1024 * 1024
_IK_PARAMETERS = dict(damping=1e-3, max_step=0.15, tolerance_m=5e-4, max_iterations=100)


@dataclass(frozen=True)
class NativePadObjectProjection:
    geom_name: str
    kind: str
    size: np.ndarray
    world_position: np.ndarray
    world_rotation: np.ndarray
    pad_local_interval_m: np.ndarray
    signed_overlap_m: np.ndarray


@dataclass(frozen=True)
class NativePadPose:
    geom_name: str
    world_corners: np.ndarray
    plane_clearances_m: np.ndarray
    minimum_plane_clearance_m: float
    object_projections: tuple


@dataclass(frozen=True)
class NativeContactPoseCandidate:
    input_binding: str
    model_sha256: str
    seed_index: int
    q_seed: np.ndarray
    joint_names: tuple
    body_name: str
    local_point: np.ndarray
    target: np.ndarray
    support_geom_name: str
    support_origin: np.ndarray
    support_normal: np.ndarray
    ik: NativePointIKResult
    pads: tuple


def _names(values, name, minimum, maximum):
    _require(type(values) in (tuple, list) and minimum <= len(values) <= maximum,
             name + " requires an explicit bounded tuple/list")
    result = tuple(values)
    _require(all(type(value) is str and value for value in result) and len(set(result)) == len(result),
             name + " requires distinct nonempty names")
    return result


def _ancestors(model, body):
    result = set()
    while body:
        _require(0 < body < model.nbody, "invalid body address")
        result.add(body)
        parent = int(model.body_parentid[body])
        _require(0 <= parent < body, "native body parents must precede children")
        body = parent
    return result


def _model_digest(mj, model):
    size = mj.mj_sizeModel(model)
    _require(0 < size <= MAX_MODEL_BYTES, "model binding exceeds audit byte budget")
    binary = np.empty(size, dtype=np.uint8)
    mj.mj_saveModel(model, None, binary)
    return sha256(binary.tobytes()).hexdigest()


def audit_native_contact_poses(
    model, target, q_seeds, joint_names, *, body_name, local_point,
    pad_geom_names, support_geom_name, object_geom_names=(),
):
    """Audit 1..3 explicit robot seeds at one fixed world-space TCP target.

    Seeds are complete native configurations and must share every unselected
    coordinate. Only selected scalar ancestors of body_name can change. Each
    original IK call keeps its fixed 100-iteration budget and existing gates.
    Pads must be two BOX geoms below the TCP body; declared object geoms must
    be independent of all selected joints. Supported projections are sphere,
    capsule (native local Z axis) and box, including original geom transforms.

    A plane's positive side is its transformed native local +Z. Corner order
    follows product((-1, 1), repeat=3). Positive signed_overlap_m means interval
    overlap; negative means a separating gap along that pad axis. These three
    axes are not a complete oriented-box separating-axis or contact test.
    No candidate is selected or applied. Model, caller seeds and FREE states
    remain unchanged; returned arrays own immutable byte-backed copies.
    """
    import mujoco as mj

    _require(isinstance(model, mj.MjModel), "an original MuJoCo model is required")
    _require(0 < model.nv <= 35 and 0 < model.nq <= 45 and model.nbody <= 128
             and model.ngeom <= 128, "model exceeds finite pose audit dimensions")
    joint_names = _names(joint_names, "joint_names", 1, 35)
    pads = _names(pad_geom_names, "pad_geom_names", 2, 2)
    objects = _names(object_geom_names, "object_geom_names", 0, MAX_OBJECT_GEOMS)
    _require(type(body_name) is str and body_name and type(support_geom_name) is str
             and support_geom_name, "body and support names must be nonempty")
    _require(type(q_seeds) in (tuple, list) and 1 <= len(q_seeds) <= MAX_SEEDS,
             "q_seeds requires 1..3 explicit seeds; no random restarts")
    _, joints, qa, _, _, _ = _joints(mj, model, joint_names)
    target, point = _vector(target, 3, "target"), _vector(local_point, 3, "local_point")
    seeds = tuple(_configuration(mj, model, seed) for seed in q_seeds)
    unselected = np.ones(model.nq, dtype=np.bool_)
    unselected[list(qa)] = False
    reference = seeds[0][unselected].tobytes()
    _require(all(seed[unselected].tobytes() == reference for seed in seeds),
             "seeds may differ only at selected robot coordinates")
    body = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name))
    _require(body > 0, "unknown or world TCP body")
    robot_bodies = {int(model.jnt_bodyid[joint]) for joint in joints}
    _require(robot_bodies <= _ancestors(model, body), "IK joints must be TCP-body ancestors")

    def geom_id(name):
        index = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name))
        _require(index >= 0, "unknown geometry " + name)
        return index

    pad_ids, object_ids = tuple(map(geom_id, pads)), tuple(map(geom_id, objects))
    support = geom_id(support_geom_name)
    _require(len(set((*pad_ids, *object_ids, support))) == 3 + len(objects),
             "pad, object and support geometries must be distinct")
    _require(int(model.geom_type[support]) == int(mj.mjtGeom.mjGEOM_PLANE),
             "declared support must be a native PLANE")
    for pad in pad_ids:
        _require(int(model.geom_type[pad]) == int(mj.mjtGeom.mjGEOM_BOX), "pads must be native BOX geoms")
        _require(body in _ancestors(model, int(model.geom_bodyid[pad])), "pads must descend from TCP body")
        _require(np.all(model.geom_size[pad] > 0), "pad half sizes must be positive")
    kinds = {int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
             int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mj.mjtGeom.mjGEOM_BOX): "box"}
    for obj in object_ids:
        _require(int(model.geom_type[obj]) in kinds, "object projection supports only sphere/capsule/box")
        _require(not robot_bodies.intersection(_ancestors(model, int(model.geom_bodyid[obj]))),
                 "declared objects cannot move with selected robot joints")
        kind, size = kinds[int(model.geom_type[obj])], model.geom_size[obj]
        _require(size[0] > 0 and (kind != "capsule" or size[1] >= 0)
                 and (kind != "box" or np.all(size > 0)), "invalid object dimensions")
    for name in ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
                 "jnt_pos", "jnt_axis", "geom_pos", "geom_quat", "geom_size"):
        _require(np.isfinite(getattr(model, name)).all(), "nonfinite native geometry " + name)
    # Validate the whole bounded tree before any native scratch allocation or IK.
    for index in range(1, model.nbody):
        _ancestors(model, index)
    model_hash = _model_digest(mj, model)
    binding = dict(schema="native-contact-pose-audit-v1", model_sha256=model_hash,
                   joint_names=joint_names, body_name=body_name, local_point=point.tolist(),
                   target=target.tolist(), pad_geom_names=pads, support_geom_name=support_geom_name,
                   object_geom_names=objects, seeds=[seed.tobytes().hex() for seed in seeds],
                   ik_parameters=_IK_PARAMETERS)
    request = json.dumps(binding, sort_keys=True, allow_nan=False).encode()
    signs = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
    data, results = mj.MjData(model), []
    for index, seed in enumerate(seeds):
        ik = solve_native_point_ik(model, target, seed, joint_names, body_name=body_name,
                                   local_point=point, **_IK_PARAMETERS)
        _require(ik.configuration[unselected].tobytes() == reference,
                 "IK changed FREE or other unselected coordinates")
        data.qpos[:] = ik.configuration
        mj.mj_kinematics(model, data)
        _require(data.qpos.tobytes() == ik.configuration.tobytes(), "FK changed candidate q")
        _require(np.isfinite(data.geom_xpos).all() and np.isfinite(data.geom_xmat).all(),
                 "nonfinite native FK geometry")
        origin = data.geom_xpos[support].copy()
        normal = data.geom_xmat[support].reshape(3, 3)[:, 2].copy()
        pad_results = []
        for name, pad in zip(pads, pad_ids, strict=True):
            center, rotation, size = data.geom_xpos[pad], data.geom_xmat[pad].reshape(3, 3), model.geom_size[pad]
            corners = center + (signs * size) @ rotation.T
            clearance = (corners - origin) @ normal
            projections = []
            for object_name, obj in zip(objects, object_ids, strict=True):
                kind = kinds[int(model.geom_type[obj])]
                obj_size, obj_rotation = model.geom_size[obj], data.geom_xmat[obj].reshape(3, 3)
                obj_center = rotation.T @ (data.geom_xpos[obj] - center)
                relative = rotation.T @ obj_rotation
                if kind == "sphere":
                    half = np.full(3, obj_size[0])
                elif kind == "capsule":
                    half = obj_size[0] + obj_size[1] * np.abs(relative[:, 2])
                else:
                    half = np.abs(relative) @ obj_size
                interval = np.stack((obj_center - half, obj_center + half), axis=1)
                overlap = np.minimum(size, interval[:, 1]) - np.maximum(-size, interval[:, 0])
                projections.append(NativePadObjectProjection(
                    object_name, kind, _readonly(obj_size), _readonly(data.geom_xpos[obj]),
                    _readonly(obj_rotation), _readonly(interval), _readonly(overlap),
                ))
            pad_results.append(NativePadPose(name, _readonly(corners), _readonly(clearance),
                                             float(np.min(clearance)), tuple(projections)))
        results.append(NativeContactPoseCandidate(
            sha256(request + b"\0" + str(index).encode()).hexdigest(), model_hash, index,
            _readonly(seed), joint_names, body_name, _readonly(point), _readonly(target),
            support_geom_name, _readonly(origin), _readonly(normal), ik, tuple(pad_results),
        ))
    _require(_model_digest(mj, model) == model_hash, "native model changed during pose audit")
    return tuple(results)
