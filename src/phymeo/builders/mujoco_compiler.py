"""Optional CPU MuJoCo compilation and explicit articulation asset lowering.

MuJoCo is imported lazily. Compilation is not a physics backend selection. The
asset is articulation-only: world environment and contact law must be assembled
explicitly. Native MjSpec/MjModel are mutable owned objects, never advertised as
immutable. A self-contained compiled MJB snapshot permits independent runtimes.
"""

import importlib
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..core.rigid import RigidBodyModel
from ..core.robot import RobotModel
from ..dynamics.floating_robot import FloatingRobotModel
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ._robot_asset import RobotAsset, make_collider
from .floating_urdf import FloatingRobotAsset


def _owned(value):
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


def _rotation(wxyz):
    return quaternion_matrix(np.asarray(wxyz)[None, [1, 2, 3, 0]])[0]


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _mujoco():
    try:
        module = importlib.import_module("mujoco")
    except ImportError as exc:
        raise ImportError("optional MuJoCo compiler requires the mujoco extra/environment") from exc
    if not hasattr(module, "MjSpec"):
        raise ImportError("optional MuJoCo compiler requires MjSpec support (MuJoCo 3.3.7+)")
    return module


class MuJoCoCompilationError(ValueError):
    """Native XML/spec compilation or source snapshot creation failed."""


@dataclass(frozen=True)
class LoweringIssue:
    code: str
    detail: str


class MuJoCoLoweringError(ValueError):
    """Native model is valid, but its semantics cannot become the current asset."""

    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{row.code}: {row.detail}" for row in self.issues))


@dataclass(frozen=True)
class MuJoCoSource:
    """Owned source/compiled bytes; binary is the complete native semantic snapshot.

    primary_xml is the original top-level file only. archive_bytes is normalized
    MjSpec XML plus its VFS assets; it is not claimed to retain every original
    disk include/mesh byte. model_binary owns all compiled resources and remains
    loadable after source paths disappear. No keyframe is selected implicitly.
    """

    origin: str
    primary_xml: bytes
    primary_sha256: str
    compiled_xml: str
    archive_bytes: bytes
    archive_sha256: str
    model_binary: bytes
    model_sha256: str
    mujoco_version: str


@dataclass(frozen=True)
class BodyMapping:
    mj_id: int
    name: str
    authored_name: str | None
    link_index: int | None
    authored_origin_in_link: tuple[float, ...]
    principal_to_authored_xyzw: tuple[float, ...]


@dataclass(frozen=True)
class JointMapping:
    mj_id: int
    name: str
    authored_name: str | None
    link_index: int
    qpos_address: int
    dof_address: int
    reference: float


@dataclass(frozen=True)
class GeometryMapping:
    mj_id: int
    name: str
    authored_name: str | None
    body_id: int
    shape_key: CollisionKey | None
    contype: int
    conaffinity: int
    world_static: bool
    visual_only: bool


@dataclass(frozen=True)
class MuJoCoMapping:
    bodies: tuple[BodyMapping, ...]
    joints: tuple[JointMapping, ...]
    geoms: tuple[GeometryMapping, ...]
    root_body_id: int
    free_joint_id: int | None
    native_nq: int
    native_nv: int
    root_com: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class MuJoCoCompilation:
    """Native mutable compiler products plus independent immutable asset metadata.

    Native objects belong to this record. External mutation is possible; never
    share them as writable runtime state. fresh_native_model() reconstructs an
    independent native model from owned bytes. Phymeo lowering and coordinate
    mapping do not borrow native arrays. require_asset=False permits native-only
    records with explicit lowering_issues; mapping may be None for non-tree q.
    """

    mj_model: Any
    mj_spec: Any
    asset: RobotAsset | FloatingRobotAsset | None
    mapping: MuJoCoMapping | None
    source: MuJoCoSource
    initial_q: np.ndarray | None
    initial_v: np.ndarray | None
    native_initial_qpos: np.ndarray
    native_initial_qvel: np.ndarray
    lowering_issues: tuple[LoweringIssue, ...]
    unlowered_world_geom_ids: tuple[int, ...]
    asset_scope: str = "articulation-only; environment and contact law require explicit scene assembly"

    def fresh_native_model(self):
        """Fresh MjModel, including geometry/assets/options/keys, without source IO."""
        mj = _mujoco()
        return mj.MjModel.from_binary_path("compiled.mjb", {"compiled.mjb": self.source.model_binary})

    def _mapping(self):
        if self.mapping is None:
            raise MuJoCoLoweringError(self.lowering_issues)
        return self.mapping

    def qpos_to_q(self, qpos):
        mapping = self._mapping()
        native = _vector(qpos, mapping.native_nq, "native qpos")
        child = [native[row.qpos_address] - row.reference for row in mapping.joints]
        if mapping.free_joint_id is None:
            return _owned(child)
        # The admitted free root is the first native joint and owns addresses 0/0.
        quaternion = native[[4, 5, 6, 3]]
        if abs(np.linalg.norm(quaternion) - 1) > 1e-10:
            raise ValueError("native root quaternion must be unit WXYZ")
        return _owned(np.r_[native[:3], quaternion, child])

    def q_to_qpos(self, q):
        mapping = self._mapping()
        floating = mapping.free_joint_id is not None
        public = _vector(q, len(mapping.joints) + (7 if floating else 0), "Phymeo q")
        result = self.native_initial_qpos.copy()
        offset = 0
        if floating:
            if abs(np.linalg.norm(public[3:7]) - 1) > 1e-10:
                raise ValueError("Phymeo root quaternion must be unit XYZW")
            result[:3], result[3:7] = public[:3], public[[6, 3, 4, 5]]
            offset = 7
        for index, row in enumerate(mapping.joints):
            result[row.qpos_address] = public[offset + index] + row.reference
        return _owned(result)

    def velocity_map(self, q):
        """Return T(q): native qvel = T @ public v (not an acceleration map).

        Native free translation is origin velocity in world coordinates; native
        angular velocity is root-local. Public translation is root COM velocity
        and angular velocity is world. Scalar coordinates subtract qpos0 refs.
        """
        mapping = self._mapping()
        floating = mapping.free_joint_id is not None
        public = _vector(q, len(mapping.joints) + (7 if floating else 0), "Phymeo q")
        result = np.zeros((mapping.native_nv, len(mapping.joints) + (6 if floating else 0)))
        offset = 0
        if floating:
            if abs(np.linalg.norm(public[3:7]) - 1) > 1e-10:
                raise ValueError("Phymeo root quaternion must be unit XYZW")
            rotation = quaternion_matrix(public[None, 3:7])[0]
            result[:3, :3] = np.eye(3)
            result[:3, 3:6] = _skew(rotation @ mapping.root_com)
            result[3:6, 3:6] = rotation.T
            offset = 6
        for index, row in enumerate(mapping.joints):
            result[row.dof_address, offset + index] = 1
        return _owned(result)

    def v_to_qvel(self, q, v):
        transform = self.velocity_map(q)
        return _owned(transform @ _vector(v, transform.shape[1], "Phymeo v"))

    def qvel_to_v(self, q, qvel):
        transform = self.velocity_map(q)
        return _owned(np.linalg.solve(transform, _vector(qvel, transform.shape[0], "native qvel")))


def _name(mj, model, category, index):
    authored = mj.mj_id2name(model, category, index)
    return authored, authored or f"{str(category).split('.')[-1].lower()}-{index}"


def _source(mj, spec, model, primary_xml, origin):
    archive = BytesIO()
    spec.to_zip(archive)
    binary = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, None, binary)
    archive_bytes, model_bytes = archive.getvalue(), binary.tobytes()
    return MuJoCoSource(
        origin, primary_xml, sha256(primary_xml).hexdigest(), spec.to_xml(),
        archive_bytes, sha256(archive_bytes).hexdigest(), model_bytes,
        sha256(model_bytes).hexdigest(), str(mj.__version__),
    )


def compile_mujoco(path, *, require_asset=True):
    """Compile a local MJCF/URDF using native MjSpec, then lower its admitted tree.

    Relative includes/resources use MuJoCo's own path semantics. The native MJB
    result is snapshotted before returning; no later runtime needs source files.
    """
    mj = _mujoco()
    if not isinstance(require_asset, bool):
        raise TypeError("require_asset must be bool")
    path = Path(path).resolve()
    try:
        primary = path.read_bytes()
        spec = mj.MjSpec.from_file(str(path))
        model = spec.compile()
        source = _source(mj, spec, model, primary, str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise MuJoCoCompilationError(f"MuJoCo compilation failed for {path}: {exc}") from exc
    return _compilation(mj, spec, model, source, require_asset)


def compile_mujoco_xml(xml, *, assets=None, includes=None, require_asset=True):
    """Compile XML with owned explicit virtual resources (no fallback physics).

    assets maps native resource names to bytes; includes is the MjSpec include
    mapping. MuJoCo resolves defaults, include expansion, body inertia and names.
    """
    mj = _mujoco()
    if not isinstance(xml, str) or not isinstance(require_asset, bool):
        raise TypeError("xml must be str and require_asset must be bool")
    resources = {str(name): bytes(value) for name, value in (assets or {}).items()}
    include = {str(name): bytes(value) for name, value in (includes or {}).items()}
    try:
        spec = mj.MjSpec.from_string(xml, assets=resources, include=include)
        model = spec.compile()
        source = _source(mj, spec, model, xml.encode("utf-8"), "<xml>")
    except (OSError, ValueError, RuntimeError) as exc:
        raise MuJoCoCompilationError(f"MuJoCo XML compilation failed: {exc}") from exc
    return _compilation(mj, spec, model, source, require_asset)


def _tree_layout(mj, model):
    issues = []
    kinds = np.asarray(model.jnt_type)
    free = np.flatnonzero(kinds == int(mj.mjtJoint.mjJNT_FREE))
    scalar = np.flatnonzero((kinds == int(mj.mjtJoint.mjJNT_HINGE)) | (kinds == int(mj.mjtJoint.mjJNT_SLIDE)))
    if len(free) > 1 or len(free) + len(scalar) != model.njnt:
        return None, [LoweringIssue("joint-type", "one free root and scalar hinge/slide joints only")]
    if model.njnt == 0:
        return None, [LoweringIssue("fixed-empty", "fixed assets require at least one scalar joint")]
    if len(free):
        free_id = int(free[0])
        root = int(model.jnt_bodyid[free_id])
        if int(model.body_parentid[root]) != 0 or int(model.body_jntnum[root]) != 1 or int(model.jnt_qposadr[free_id]) != 0 or int(model.jnt_dofadr[free_id]) != 0:
            return None, [LoweringIssue("free-root", "free root must directly attach to world and own native addresses 0/0")]
        top = root
    else:
        free_id = None
        tops = set()
        for joint in scalar:
            body = int(model.jnt_bodyid[joint])
            while int(model.body_parentid[body]) != 0:
                body = int(model.body_parentid[body])
            tops.add(body)
        if len(tops) != 1:
            return None, [LoweringIssue("multiple-articulations", "asset lowering admits one connected scalar tree")]
        top = tops.pop()
        root = top if int(model.body_jntnum[top]) == 0 else 0
    selected = {root}
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != top:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == top:
            selected.add(body)
    moving = [int(model.jnt_bodyid[j]) for j in scalar]
    if any(body not in selected for body in moving):
        issues.append(LoweringIssue("multiple-articulations", "all scalar joints must belong to the one root"))
    for body in sorted(selected - {root}):
        if int(model.body_jntnum[body]) != 1:
            issues.append(LoweringIssue("fixed-child", f"body {body} requires fixed-body folding or multi-joint decomposition"))
    if issues:
        return None, issues
    return (root, free_id, tuple(int(j) for j in scalar), selected), []


def _semantic_issues(mj, model, floating):
    issues = []
    for field, code in (("ntendon", "tendon"), ("neq", "equality"), ("nflex", "flex"), ("nplugin", "plugin"), ("nmocap", "mocap")):
        if int(getattr(model, field, 0)):
            issues.append(LoweringIssue(code, f"native {field} is not represented by current articulation assets"))
    for field, code in (("dof_frictionloss", "joint-frictionloss"), ("jnt_stiffness", "joint-stiffness"), ("body_gravcomp", "body-gravcomp")):
        if np.any(getattr(model, field)):
            issues.append(LoweringIssue(code, f"nonzero {field} requires its own physical adapter"))
    if np.any(model.dof_armature < 0) or np.any(model.dof_damping < 0):
        issues.append(LoweringIssue("joint-actuators", "armature/damping must be nonnegative"))
    if not floating and (np.any(model.dof_armature) or np.any(model.dof_damping)):
        issues.append(LoweringIssue("fixed-actuators", "current RobotModel has no armature/damping physical fields"))
    if floating and (np.any(model.dof_armature[:6]) or np.any(model.dof_damping[:6])):
        issues.append(LoweringIssue("free-actuators", "free-root armature/damping is not a scalar-joint field"))
    if np.any(model.opt.wind) or float(model.opt.viscosity) or float(model.opt.density):
        issues.append(LoweringIssue("fluid-forces", "ambient fluid forces need their own adapter"))
    if int(model.opt.disableflags) or int(model.opt.enableflags):
        issues.append(LoweringIssue("native-option-flags", "nondefault native enable/disable flags require explicit backend semantics"))
    return issues


def _motor_limits(mj, model, scalar):
    limits, issues, seen = {}, [], set()
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        expected_gear = np.array([1., 0., 0., 0., 0., 0.])
        valid = (
            int(model.actuator_trntype[actuator]) == int(mj.mjtTrn.mjTRN_JOINT)
            and joint in scalar and joint not in seen
            and int(model.actuator_dyntype[actuator]) == int(mj.mjtDyn.mjDYN_NONE)
            and int(model.actuator_gaintype[actuator]) == int(mj.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[actuator]) == int(mj.mjtBias.mjBIAS_NONE)
            and np.array_equal(model.actuator_gear[actuator], expected_gear)
            and model.actuator_gainprm[actuator, 0] == 1
            and not np.any(model.actuator_gainprm[actuator, 1:])
            and not np.any(model.actuator_biasprm[actuator])
        )
        if not valid:
            issues.append(LoweringIssue("actuator-transmission", f"actuator {actuator} requires a unique unit-gear direct motor"))
            continue
        seen.add(joint)
        bounds = []
        for flag, field in (("actuator_ctrllimited", "actuator_ctrlrange"), ("actuator_forcelimited", "actuator_forcerange")):
            if bool(getattr(model, flag)[actuator]):
                low, high = getattr(model, field)[actuator]
                if low != -high or high <= 0:
                    issues.append(LoweringIssue("actuator-limits", f"actuator {actuator} has a non-symmetric effort range"))
                else:
                    bounds.append(float(high))
        limits[joint] = min(bounds) if bounds else None
    # Joint-wide effort clamps also affect the actual generalized force.
    if hasattr(model, "jnt_actfrclimited"):
        for joint in scalar:
            if bool(model.jnt_actfrclimited[joint]):
                low, high = model.jnt_actfrcrange[joint]
                if low != -high or high <= 0:
                    issues.append(LoweringIssue("actuator-limits", f"joint {joint} has non-symmetric force clamping"))
                else:
                    previous = limits.get(joint)
                    limits[joint] = float(high) if previous is None else min(previous, float(high))
    return limits, issues


def _decode(mj, model):
    layout, issues = _tree_layout(mj, model)
    if layout is None:
        return None, None, tuple(issues), tuple(range(model.ngeom))
    root, free_id, scalar, selected = layout
    floating = free_id is not None
    issues.extend(_semantic_issues(mj, model, floating))
    effort, motor_issues = _motor_limits(mj, model, scalar)
    issues.extend(motor_issues)
    body_ids = [int(model.jnt_bodyid[j]) for j in scalar]
    index_by_body = {body: index for index, body in enumerate(body_ids)}
    index_by_body[root] = -1
    frames = {body: _rotation(model.body_iquat[body]) for body in body_ids}
    anchors = {int(model.jnt_bodyid[j]): np.asarray(model.jnt_pos[j]).copy() for j in scalar}
    frames[root], anchors[root] = np.eye(3), np.zeros(3)
    root_rotation = np.eye(3) if root == 0 else _rotation(model.body_quat[root])
    root_position = np.zeros(3) if root == 0 else np.asarray(model.body_pos[root])
    bodies, joint_rows, geom_rows, colliders = [], [], [], []
    link_names, joint_names = [], []
    for body in range(model.nbody):
        authored, name = _name(mj, model, mj.mjtObj.mjOBJ_BODY, body)
        link_index = index_by_body.get(body)
        frame = frames.get(body, np.eye(3))
        anchor = anchors.get(body, np.zeros(3))
        bodies.append(BodyMapping(body, name, authored, link_index, tuple(-frame.T @ anchor), tuple(matrix_quaternion(frame))))
    for index, joint in enumerate(scalar):
        authored, name = _name(mj, model, mj.mjtObj.mjOBJ_JOINT, joint)
        qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        joint_rows.append(JointMapping(joint, name, authored, index, qa, va, float(model.qpos0[qa])))
        joint_names.append(name)
        link_names.append(bodies[body_ids[index]].name)
    world_geoms = []
    shape_counts = {}
    for geom in range(model.ngeom):
        authored, name = _name(mj, model, mj.mjtObj.mjOBJ_GEOM, geom)
        body = int(model.geom_bodyid[geom])
        contype, affinity = int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])
        visual = contype == 0 and affinity == 0
        world_static = body not in selected or body == 0
        key = None
        if world_static:
            if not visual:
                world_geoms.append(geom)
        elif not visual:
            kinds = {int(mj.mjtGeom.mjGEOM_SPHERE): ("sphere", 1), int(mj.mjtGeom.mjGEOM_CAPSULE): ("capsule", 2), int(mj.mjtGeom.mjGEOM_BOX): ("box", 3)}
            kind_size = kinds.get(int(model.geom_type[geom]))
            if kind_size is None:
                issues.append(LoweringIssue("geometry-type", f"physical geom {name!r} is not sphere/capsule/box"))
            elif contype < 0 or affinity < 0 or max(contype, affinity) > 2**31 - 1:
                issues.append(LoweringIssue("collision-filter", f"geom {name!r} masks exceed the current nonnegative i32 format"))
            else:
                frame, anchor = frames[body], anchors[body]
                position = frame.T @ (model.geom_pos[geom] - anchor)
                rotation = frame.T @ _rotation(model.geom_quat[geom])
                index = index_by_body[body]
                ordinal = shape_counts.get(body, 0)
                shape_counts[body] = ordinal + 1
                kind, count = kind_size
                collider = make_collider(name, bodies[body].name, index, ordinal, position, matrix_quaternion(rotation), kind, np.asarray(model.geom_size[geom, :count]), name_is_authored=authored is not None)
                collider = replace(collider, shape=replace(collider.shape, group=contype, mask=affinity))
                colliders.append(collider)
                key = collider.shape.key
        geom_rows.append(GeometryMapping(geom, name, authored, body, key, contype, affinity, world_static, visual))
    # MuJoCo masks use OR; current Phymeo pair filters use AND. Refuse asymmetric
    # cases where direct bit preservation would actually alter source eligibility.
    physical = [row for row in geom_rows if not row.visual_only]
    for index, left in enumerate(physical):
        for right in physical[index + 1:]:
            if left.shape_key is None and right.shape_key is None:
                continue
            first = bool(left.contype & right.conaffinity)
            second = bool(right.contype & left.conaffinity)
            if (first or second) != (first and second):
                issues.append(LoweringIssue("collision-filter", f"source OR masks for geoms {left.mj_id}/{right.mj_id} differ from current AND filtering"))
                break
        else:
            continue
        break
    mapping = MuJoCoMapping(tuple(bodies), tuple(joint_rows), tuple(geom_rows), root, free_id, int(model.nq), int(model.nv), tuple(model.body_ipos[root]) if floating else (0., 0., 0.))
    if issues:
        return None, mapping, tuple(issues), tuple(world_geoms)
    parents, kinds, positions, rotations, axes, masses, centers, moments = [], [], [], [], [], [], [], []
    lowers, uppers, forces = [], [], []
    for joint, body in zip(scalar, body_ids, strict=True):
        parent = int(model.body_parentid[body])
        if parent not in index_by_body:
            return None, mapping, (LoweringIssue("fixed-child", "unfolded fixed ancestor is not an asset body"),), tuple(world_geoms)
        parent_index = index_by_body[parent]
        frame, parent_frame = frames[body], frames[parent]
        local_rotation = _rotation(model.body_quat[body])
        joint_position = np.asarray(model.body_pos[body]) + local_rotation @ anchors[body]
        if parent == root and not floating:
            joint_position = root_position + root_rotation @ joint_position
            joint_rotation = root_rotation @ local_rotation @ frame
        else:
            joint_position = parent_frame.T @ (joint_position - anchors[parent])
            joint_rotation = parent_frame.T @ local_rotation @ frame
        parents.append(parent_index)
        kinds.append("revolute" if int(model.jnt_type[joint]) == int(mj.mjtJoint.mjJNT_HINGE) else "prismatic")
        positions.append(joint_position)
        rotations.append(matrix_quaternion(joint_rotation))
        axes.append(frame.T @ model.jnt_axis[joint])
        masses.append(float(model.body_mass[body]))
        centers.append(frame.T @ (model.body_ipos[body] - anchors[body]))
        moments.append(model.body_inertia[body])
        ref = model.qpos0[int(model.jnt_qposadr[joint])]
        low, high = model.jnt_range[joint] if bool(model.jnt_limited[joint]) else (-np.inf, np.inf)
        lowers.append(float(low - ref))
        uppers.append(float(high - ref))
        forces.append(effort.get(joint))
    children = None
    if scalar:
        children = RobotModel(np.asarray(parents, dtype=np.int64), tuple(kinds), np.asarray(positions), np.asarray(rotations), np.asarray(axes), np.asarray(masses), np.asarray(centers), np.asarray(moments), np.array(model.opt.gravity, dtype=np.float64))
    common = (
        str(getattr(model, "name", "compiled-mujoco")), bodies[root].name,
        tuple(link_names), tuple(joint_names), tuple(lowers), tuple(uppers),
        tuple(forces), tuple(None for _ in scalar), tuple(colliders),
        CollisionModel(tuple(collider.shape for collider in colliders)),
    )
    name, root_name, links, joints, lower, upper, force, speed, collisions, collision_model = common
    if floating:
        root_model = RigidBodyModel(np.array([1 / model.body_mass[root]]), np.array(model.body_inertia[root][None]), np.array(model.opt.gravity))
        va = [int(model.jnt_dofadr[j]) for j in scalar]
        physical = FloatingRobotModel(root_model, np.array(model.body_ipos[root]), children, root_inertia_frame_xyzw=np.array(model.body_iquat[root])[[1, 2, 3, 0]], joint_armature=np.array(model.dof_armature[va]), joint_damping=np.array(model.dof_damping[va]))
        asset = FloatingRobotAsset(name, root_name, physical, links, joints, lower, upper, force, speed, collisions, collision_model)
    else:
        asset = RobotAsset(name, root_name, children, links, joints, lower, upper, force, speed, collisions, collision_model, tuple(root_position), tuple(matrix_quaternion(root_rotation)))
    return asset, mapping, (), tuple(world_geoms)


def _compilation(mj, spec, model, source, require_asset):
    try:
        asset, mapping, issues, world_geoms = _decode(mj, model)
    except (ValueError, FloatingPointError) as exc:
        asset, mapping, issues, world_geoms = None, None, (LoweringIssue("asset-domain", str(exc)),), tuple(range(model.ngeom))
    if require_asset and issues:
        raise MuJoCoLoweringError(issues)
    if asset is not None:
        asset = replace(asset, name=str(spec.modelname))
    native_q, native_v = _owned(model.qpos0), _owned(np.zeros(model.nv))
    record = MuJoCoCompilation(model, spec, asset, mapping, source, None, None, native_q, native_v, tuple(issues), tuple(world_geoms))
    if mapping is not None:
        initial_q = record.qpos_to_q(native_q)
        record = replace(record, initial_q=initial_q, initial_v=_owned(np.zeros(mapping.native_nv)))
    return record
