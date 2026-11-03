"""Import a static OpenUSD articulation as a fixed-base SI robot. OpenUSD is optional."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..dynamics.robot_kinematics import forward_kinematics
from ..dynamics.rotation import quaternion_matrix
from ._robot_asset import RobotAsset, assemble_robot, make_collider, principal_inertia


@dataclass(frozen=True, eq=False, kw_only=True)
class USDAsset(RobotAsset):
    """Named collider vertices stay in authored link coordinates, expressed in metres.

    RobotModel places a moving link's origin at its inbound joint anchor. The
    native and authored axes are parallel. Use authored_to_model_points before
    passing an authored local point to RobotPointPort or the native Jacobian.
    """

    child_anchors_m: tuple[tuple[float, ...], ...]
    meters_per_unit: float
    kilograms_per_unit: float
    up_axis: str

    def authored_to_model_points(self, link_name, points_m):
        points = np.asarray(points_m, dtype=float)
        if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("authored points must be finite 3-vectors in metres")
        if link_name == self.root_link:
            return points.copy()
        return points - self.child_anchors_m[self.link_names.index(link_name)]

    def _authored_pose(self, index, geometry):
        if index == -1:
            return np.array(self.root_xyz), quaternion_matrix(np.array([self.root_xyzw]))[0]
        rotation = geometry.rotation[index]
        return geometry.position[index] - rotation @ self.child_anchors_m[index], rotation

    def authored_link_pose(self, link_name, q):
        """Return the original USD link origin and local-to-world rotation in SI, Z-up."""
        index = -1 if link_name == self.root_link else self.link_names.index(link_name)
        return self._authored_pose(index, forward_kinematics(self.model, q))

    def collision_motion(self, q):
        geometry = forward_kinematics(self.model, q)
        paths = {}
        for collider in self.collisions:
            position, rotation = self._authored_pose(collider.link_index, geometry)
            points = position + np.asarray(collider.local_vertices) @ rotation.T
            paths[collider.shape.key] = CollisionMotion(points)
        return paths


def _quat(value, where):
    quaternion = np.array([*value.GetImaginary(), value.GetReal()], dtype=float)
    norm = np.hypot.reduce(quaternion)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"{where}: invalid quaternion")
    return quaternion / norm


def _to_quat(Gf, rotation):
    # Gf matrices act on row vectors; RobotModel uses column vectors.
    matrix = Gf.Matrix3d(*rotation.T.ravel().tolist())
    return _quat(matrix.ExtractRotation().GetQuat(), "rotation matrix")


def _frame(position, rotation, length, where):
    frame = np.eye(4)
    frame[:3, :3] = quaternion_matrix(_quat(rotation, where)[None, :])[0]
    frame[:3, 3] = np.asarray(position, dtype=float) * length
    if not np.isfinite(frame).all():
        raise ValueError(f"{where}: nonfinite joint frame")
    return frame


def _matrix(prim, cache, length):
    matrix = np.array(cache.GetLocalToWorldTransform(prim)).T.copy()
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12
    ):
        raise ValueError(f"{prim.GetPath()}: unsupported nonaffine transform")
    matrix[:3, 3] *= length
    if not np.isfinite(matrix[:3, 3]).all():
        raise ValueError(f"{prim.GetPath()}: position is not representable in metres")
    return matrix


def _rigid_axes(matrix, where, *, allow_scale=False):
    linear = matrix[:3, :3]
    scale = np.hypot.reduce(linear, axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError(f"{where}: singular scale")
    rotation = linear / scale
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-6)
        or np.linalg.det(rotation) <= 0
    ):
        raise ValueError(f"{where}: shear or reflected scale is unsupported")
    if not allow_scale and not np.allclose(scale, 1, rtol=0, atol=2e-6):
        raise ValueError(f"{where}: rigid-body transform cannot include scale")
    return rotation, scale


def _stage_units(UsdGeom, UsdPhysics, stage):
    length = UsdGeom.GetStageMetersPerUnit(stage)
    mass = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("USD metersPerUnit must be positive and finite")
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("USD kilogramsPerUnit must be positive and finite")
    up = str(UsdGeom.GetStageUpAxis(stage))
    basis = np.eye(4)
    if up == "Y":
        basis[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    elif up != "Z":
        raise ValueError(f"unsupported USD upAxis {up!r}")
    return length, mass, up, basis


def _check_semantics(prim):
    path = prim.GetPath()
    schemas = {
        "PhysicsArticulationRootAPI",
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsCollisionAPI",
    }
    for schema in prim.GetAppliedSchemas():
        if (
            schema.lower().startswith(("physics", "physx", "newton", "mjc"))
            and schema not in schemas
        ):
            raise ValueError(f"{path}: unsupported API {schema}")
    properties = {
        "axis",
        "body0",
        "body1",
        "centerOfMass",
        "collisionEnabled",
        "diagonalInertia",
        "mass",
        "localPos0",
        "localPos1",
        "localRot0",
        "localRot1",
        "lowerLimit",
        "upperLimit",
        "principalAxes",
        "rigidBodyEnabled",
        "kinematicEnabled",
        "jointEnabled",
        "gravityDirection",
        "gravityMagnitude",
    }
    for prop in prim.GetAuthoredProperties():
        name = prop.GetName()
        if name.startswith("physics:") and name[8:] not in properties:
            raise ValueError(f"{path}: unsupported physics property {name}")
        if name.startswith(
            ("physx", "newton:", "mjc:", "drive:", "state:", "material:binding:physics")
        ):
            raise ValueError(f"{path}: unsupported physics property {name}")
    for attr in prim.GetAuthoredAttributes():
        if attr.GetNumTimeSamples():
            raise ValueError(f"{path}: time-sampled attribute {attr.GetName()} is unsupported")


def _body_inertia(UsdPhysics, prim, mass_unit, length, anchor, *, fixed=False):
    path = str(prim.GetPath())
    if not prim.HasAPI(UsdPhysics.MassAPI):
        if fixed:
            return None
        raise ValueError(f"{path}: explicit PhysicsMassAPI inertia is required")
    props = UsdPhysics.MassAPI(prim)
    attrs = (props.GetMassAttr(), props.GetCenterOfMassAttr(), props.GetDiagonalInertiaAttr())
    if not all(a.HasAuthoredValueOpinion() for a in attrs):
        raise ValueError(f"{path}: explicitly author mass, centerOfMass and diagonalInertia")
    mass = float(attrs[0].Get()) * mass_unit
    com = np.array(attrs[1].Get(), dtype=float) * length - anchor
    diagonal = np.array(attrs[2].Get(), dtype=float) * mass_unit * length * length
    if not np.isfinite(mass) or not np.isfinite(com).all() or not np.isfinite(diagonal).all():
        raise ValueError(f"{path}: nonfinite mass, COM or inertia")
    axes = props.GetPrincipalAxesAttr()
    if axes.HasAuthoredValueOpinion():
        rotation = _quat(axes.Get(), path + " principalAxes")
    else:
        if not np.all(diagonal == diagonal[0]):
            raise ValueError(f"{path}: anisotropic inertia requires explicit principalAxes")
        rotation = np.array([0.0, 0.0, 0.0, 1.0])
    return principal_inertia(mass, com, rotation, np.diag(diagonal), path + " inertia", root=fixed)


def _connections(UsdPhysics, bodies, joints, length):
    edges = {path: [] for path in bodies}
    mounts = []
    names = set()
    for prim in joints:
        name, path = prim.GetName(), str(prim.GetPath())
        if name in names:
            raise ValueError(f"duplicate joint name {name!r}")
        names.add(name)
        joint = UsdPhysics.Joint(prim)
        if not joint.GetJointEnabledAttr().Get():
            raise ValueError(f"{path}: disabled joint")
        endpoints = []
        for relation, p, r in (
            (joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
            (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
        ):
            targets = relation.GetForwardedTargets()
            if len(targets) > 1:
                raise ValueError(f"{path}: each joint side needs one body or world")
            body = str(targets[0]) if targets else None
            if body is not None and body not in bodies:
                raise ValueError(f"{path}: unknown rigid body target {body}")
            endpoints.append((body, _frame(p.Get(), r.Get(), length, path)))
        (first, f0), (second, f1) = endpoints
        if first == second:
            raise ValueError(f"{path}: equal body endpoints")
        if prim.IsA(UsdPhysics.FixedJoint):
            if first is not None and second is not None:
                raise ValueError(f"{path}: internal fixed joint is unsupported")
            mounts.append(
                (second, f0 @ np.linalg.inv(f1), prim)
                if first is None
                else (first, f1 @ np.linalg.inv(f0), prim)
            )
            continue
        if first is None or second is None:
            raise ValueError(f"{path}: expected fixed world mount and two moving-joint targets")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed, kind, unit = UsdPhysics.RevoluteJoint(prim), "revolute", np.pi / 180
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed, kind, unit = UsdPhysics.PrismaticJoint(prim), "prismatic", length
        else:
            raise ValueError(f"{path}: unsupported joint type {prim.GetTypeName()}")
        axis = str(typed.GetAxisAttr().Get())
        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"{path}: invalid joint axis {axis!r}")
        axis = np.eye(3)["XYZ".index(axis)]
        lo, hi = (
            float(typed.GetLowerLimitAttr().Get()) * unit,
            float(typed.GetUpperLimitAttr().Get()) * unit,
        )
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"{path}: finite increasing scalar joint limits required")
        limits = (lo, hi, None, None)
        collide = bool(joint.GetCollisionEnabledAttr().Get())
        edges[first].append((second, name, f0, f1, axis, kind, limits, collide))
        edges[second].append((first, name, f1, f0, -axis, kind, limits, collide))
    if len(mounts) != 1 or len(joints) != len(bodies):
        raise ValueError(
            "USD tree requires one fixed world mount and one scalar joint per remaining body"
        )
    return mounts[0], edges


def _shape_bindings(Gf, UsdGeom, UsdPhysics, shapes, bodies, cache, length):
    by_owner = {path: [] for path in bodies}
    for prim in shapes:
        path = str(prim.GetPath())
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        owner = prim
        while owner and str(owner.GetPath()) not in bodies:
            owner = owner.GetParent()
        if not owner:
            raise ValueError(f"{path}: collider lacks an owning rigid body")
        key = str(owner.GetPath())
        local = np.linalg.solve(bodies[key][1], _matrix(prim, cache, length))
        rotation, scale = _rigid_axes(local, path, allow_scale=True)
        if prim.IsA(UsdGeom.Sphere):
            if not np.allclose(scale / scale[0], 1, rtol=0, atol=2e-6):
                raise ValueError(f"{path}: sphere has nonuniform scale")
            size = np.array([UsdGeom.Sphere(prim).GetRadiusAttr().Get() * length * scale[0]])
            kind = "sphere"
        elif prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get() * length * scale / 2
            kind = "box"
        else:
            raise ValueError(f"{path}: unsupported collider {prim.GetTypeName()}")
        name = prim.GetName()
        if any(rec[0] == name for rec in by_owner[key]):
            raise ValueError(f"{path}: duplicate collider name on {key}")
        by_owner[key].append((name, local[:3, 3], _to_quat(Gf, rotation), kind, size))
    return by_owner


def _gravity(UsdPhysics, scenes, length, basis, override):
    if len(scenes) > 1:
        raise ValueError("USD robot supports at most one physics scene")
    result = np.array([0.0, 0.0, -9.81])
    if scenes:
        scene = UsdPhysics.Scene(scenes[0])
        d, mag = scene.GetGravityDirectionAttr(), scene.GetGravityMagnitudeAttr()
        if d.HasAuthoredValueOpinion():
            vec = basis[:3, :3] @ np.array(d.Get(), dtype=float)
            norm = np.hypot.reduce(vec)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("invalid USD gravityDirection")
            result = 9.81 * vec / norm
        if mag.HasAuthoredValueOpinion():
            acceleration = float(mag.Get()) * length
            if not np.isfinite(acceleration) or acceleration < 0:
                raise ValueError("invalid USD gravityMagnitude")
            result = (result / 9.81) * acceleration
    return result if override is None else override


def load_usd(source, *, gravity=None) -> USDAsset:
    """Read an OpenUSD path or Usd.Stage into SI, with engine world +Z up.

    One rooted static articulation, one world weld, finite scalar limits, explicit moving-body
    masses and link-frame diagonal inertias are required. Anisotropic inertias need explicit
    principal axes. Instancing, drives, material bindings and time-sampled physics are outside
    this geometry and inertia importer. The authored rigid transforms must
    agree with the zero-coordinate joint pose. Both joint anchor frames and relation directions
    are honored. Spheres and rigidly oriented boxes are supported; display prims contribute no
    collision. Shapes may carry scale; body frames must be proper rigid transforms.
    """
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as error:
        raise ImportError(
            "load_usd requires the optional usd-core package (phymeo[usd])"
        ) from error
    stage = source if isinstance(source, Usd.Stage) else Usd.Stage.Open(str(Path(source)))
    if not stage or stage.GetCompositionErrors():
        raise ValueError("could not compose the USD stage")
    length, mass_unit, up, basis = _stage_units(UsdGeom, UsdPhysics, stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    roots, joints, shapes, scenes = [], [], [], []
    bodies = {}
    checked = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsInstance() or prim.IsA(UsdGeom.PointInstancer):
            raise ValueError(f"{path}: instanced geometry is outside the robot asset domain")
        is_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        recognized = is_body or is_collider or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim)
        if is_body:
            body = UsdPhysics.RigidBodyAPI(prim)
            if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
                raise ValueError(f"{path}: disabled or kinematic rigid body")
            pose = _matrix(prim, cache, length)
            _rigid_axes(pose, path)
            bodies[path] = (prim, pose)
        elif prim.HasAPI(UsdPhysics.MassAPI):
            raise ValueError(f"{path}: child mass aggregation is unsupported")
        if is_collider:
            shapes.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
            recognized = True
        if prim.IsA(UsdPhysics.Scene):
            scenes.append(prim)
            recognized = True
        if not recognized:
            if prim.GetTypeName().startswith("Physics") or any(
                prop.GetName().startswith("physics:") for prop in prim.GetAuthoredProperties()
            ):
                raise ValueError(f"{path}: unsupported physics prim {prim.GetTypeName()}")
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot() and ancestor.GetPath() not in checked:
            _check_semantics(ancestor)
            checked.add(ancestor.GetPath())
            ancestor = ancestor.GetParent()
    if len(roots) != 1 or not bodies:
        raise ValueError("stage requires one articulation root and rigid bodies")
    if len({prim.GetName() for prim, _ in bodies.values()}) != len(bodies):
        raise ValueError("rigid body names must be unique")
    (root, stage_root_pose, mount), edges = _connections(UsdPhysics, bodies, joints, length)
    scope = roots[0]
    if scope not in (bodies[root][0], mount) and any(
        not prim.GetPath().HasPrefix(scope.GetPath()) for prim, _ in bodies.values()
    ):
        raise ValueError("rigid body lies outside the USD articulation root")
    if not np.allclose(stage_root_pose, bodies[root][1], rtol=0, atol=2e-6):
        raise ValueError(f"{root}: zero pose disagrees with the fixed world mount")
    root_pose = basis @ stage_root_pose
    registrations = _shape_bindings(Gf, UsdGeom, UsdPhysics, shapes, bodies, cache, length)
    indices, anchors = {root: -1}, {root: np.zeros(3)}
    order, parents, names, joint_data, inertias = [], [], [], [], []
    root_name = bodies[root][0].GetName()
    _body_inertia(UsdPhysics, bodies[root][0], mass_unit, length, anchors[root], fixed=True)
    queue, disabled_pairs = deque([(root, None)]), []
    while queue:
        parent, previous = queue.popleft()
        for child, name, pframe, cframe, axis, kind, limits, collide in edges[parent]:
            if name == previous:
                continue
            if child in indices:
                raise ValueError(f"{name}: USD joint cycle")
            rest = bodies[parent][1] @ pframe @ np.linalg.inv(cframe)
            if not np.allclose(rest, bodies[child][1], rtol=0, atol=2e-6):
                raise ValueError(f"{child}: authored zero pose disagrees with joint {name}")
            anchor = pframe.copy()
            anchor[:3, :3] = pframe[:3, :3] @ cframe[:3, :3].T
            anchor[:3, 3] -= anchors[parent]
            if parent == root:
                anchor = root_pose @ anchor
            indices[child] = len(order)
            anchors[child] = cframe[:3, 3]
            order.append(bodies[child][0].GetName())
            parents.append(indices[parent])
            names.append(name)
            joint_data.append(
                (kind, anchor[:3, 3], _to_quat(Gf, anchor[:3, :3]), cframe[:3, :3] @ axis, limits)
            )
            inertias.append(
                _body_inertia(UsdPhysics, bodies[child][0], mass_unit, length, anchors[child])
            )
            if not collide:
                disabled_pairs.append((parent, child))
            queue.append((child, name))
    if len(indices) != len(bodies) or not order:
        raise ValueError("USD articulation must be a connected, nonempty fixed-base moving tree")
    colliders, by_link = [], {}
    for path, index in indices.items():
        by_link[path] = [
            make_collider(name, bodies[path][0].GetName(), index, ordinal, pos, quat, kind, size)
            for ordinal, (name, pos, quat, kind, size) in enumerate(registrations[path])
        ]
        colliders.extend(by_link[path])
    excluded = [
        (a.shape.key, b.shape.key)
        for parent, child in disabled_pairs
        for a in by_link[parent]
        for b in by_link[child]
    ]
    base = assemble_robot(
        scope.GetName(),
        root_name,
        order,
        parents,
        names,
        joint_data,
        inertias,
        colliders,
        _gravity(UsdPhysics, scenes, length, basis, gravity),
        root_xyz=root_pose[:3, 3],
        root_xyzw=_to_quat(Gf, root_pose[:3, :3]),
        excluded=excluded,
    )
    return USDAsset(
        **vars(base),
        child_anchors_m=tuple(tuple(anchors[p]) for p in indices if p != root),
        meters_per_unit=length,
        kilograms_per_unit=mass_unit,
        up_axis=up,
    )
