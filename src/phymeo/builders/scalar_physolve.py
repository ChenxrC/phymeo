"""Lower a URDF scalar tree into the autonomous PhySolve profile."""

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.robot import RobotModel
from ..dynamics._mimic_coordinates import compile_mimic_coordinates
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from ..dynamics.robot_dynamics import _mass_from_terms, _world_inertia, robot_com_kinematics
from ..dynamics.rotation import quaternion_matrix
from ..geometry.visual import VisualInstance, VisualScene
from ..solvers.floating_tree import _inertial_geometry
from ._physolve_fixed import FixedArticulation, read_fixed_articulation
from ._robot_asset import _NonDiagonalLinkInertia
from ._robot_xml import _attributes, _child, _children, _number
from .collision_sources import CollisionSourceMapping, SourceGeometry, _digest
from .floating_urdf import load_floating_urdf
from .native_assembly import NativeAssetIdentity, NativeJointInput, NativeTemplate
from .native_frames import NativeFrameMetadata, NativeJointFrame
from .primitive_physolve import (
    _free_body_input,
    _mass_and_inertia,
    _wxyz,
    require_rounded_primitives,
)
from .urdf import load_urdf
from .urdf_import_options import authored_visuals, take_compiler_options


@dataclass(frozen=True)
class ScalarCompilation:
    """Immutable authored and derived inputs for one native scalar articulation."""

    model: RobotModel | FixedArticulation
    robot_pose: tuple
    joint_names: tuple[str, ...]
    joints: tuple[NativeJointInput, ...]
    self_collision: bool
    limits: tuple
    geom_rows: tuple
    collision_source: CollisionSourceMapping
    timestep: float
    gravity: tuple
    integrator: str
    floating: FloatingRobotModel | None = None
    root_body: str | None = None
    free_objects: tuple = ()
    mimic: tuple = ()
    static_geom_frames: tuple = ()

    @property
    def extended(self):
        return isinstance(self.model, FixedArticulation)

    @property
    def has_free_root(self):
        return self.model.floating if self.extended else self.floating is not None

    @property
    def free_start(self):
        return len(self.model.bodies) + 1 if self.extended else len(self.joints) + 2

    @property
    def scalar_offsets(self):
        return (1, 7, 6) if self.has_free_root else (0, 0, 0)

    @property
    def free_bindings(self):
        root = ()
        if self.has_free_root:
            com = self.model.bodies[0].com if self.extended else self.floating.root_com_m
            root = ((self.root_body, 1, 0, 0, tuple(com)),)
        _, q0, v0 = self.scalar_offsets
        n = len(self.joints)
        return root + tuple((shape.name, self.free_start + i, q0 + n + i * 7, v0 + n + i * 6, (0., 0., 0.))
                            for i, (shape, _, _) in enumerate(self.free_objects))

    @property
    def principal_inertia(self):
        if self.extended:
            robot_inertia = tuple(body.inertia for body in self.model.bodies)
        else:
            root = (0., 0., 0.) if self.floating is None else self.floating.root.inertia_diagonal[0]
            robot_inertia = (tuple(root), *map(tuple, self.model.inertia_diagonal))
        return robot_inertia + tuple(inertia for _, _, inertia in self.free_objects)

    @property
    def inertia_frames(self):
        identity = (1., 0., 0., 0.)
        if self.extended:
            return (tuple(_wxyz(body.inertia_frame) for body in self.model.bodies)
                    + (identity,) * len(self.free_objects))
        root = identity if self.floating is None else _wxyz(self.floating.root_inertia_frame_xyzw)
        return (root, *((identity,) * (len(self.joints) + len(self.free_objects))))

    def reference_inertia(self, armatures):
        """Recompute articulated qpos0 weights for one ordered scalar armature vector."""
        if self.extended:
            return self.model.reference(armatures)
        return _reference_inertia(self.model, armatures, floating=self.floating, pose=self.robot_pose)

    def upload_model(self, frontend):
        if getattr(frontend, "CYLINDER_INPUT_CONTRACT", None) != 1 and (
            any(row[2] == "cylinder" for row in self.geom_rows)
            or any(shape.kind == "cylinder" for shape, _, _ in self.free_objects)
        ):
            raise ValueError("PhySolve cylinder input contract v1 is required")
        require_rounded_primitives(frontend, (row[2] for row in self.geom_rows))
        require_rounded_primitives(frontend, (shape.kind for shape, _, _ in self.free_objects))
        extra = {}
        if self.extended:
            from physolve_tyx._native_mesh import Mesh

            if self.has_free_root and getattr(frontend, "FLOATING_FIXED_INPUT_CONTRACT", None) != 1:
                raise ValueError("PhySolve floating fixed-body input contract v1 is required")
            meshes = tuple(Mesh(item.visual.vertices, item.visual.triangles, item.polygons, item.normals)
                           for item in self.model.meshes)
            extra.update(
                fixed_bodies=tuple(
                    (b.parent, b.dof, b.position, _wxyz(b.rotation), b.mass, b.com,
                     b.inertia, _wxyz(b.inertia_frame)) for b in self.model.bodies
                ),
                mimic=self.mimic,
            )
            geom = tuple(frontend.Geometry(kind, size, p, _wxyz(q), friction, dim,
                                           None if index is None else meshes[index])
                         for _, _, kind, size, p, q, friction, dim, index in self.geom_rows)
        else:
            geom = tuple(frontend.Geometry(kind, size, p, _wxyz(q), friction, dim)
                         for _, _, kind, size, p, q, friction, dim in self.geom_rows)
        if ("continuous" in self.model.joint_types
                and getattr(frontend, "CONTINUOUS_INPUT_CONTRACT", None) != 1):
            raise ValueError("PhySolve continuous joint input contract v1 is required")
        joints = tuple(
            frontend.ScalarJoint(kind, tuple(axis), limits[:2], row.armature, row.damping,
                                 row.frictionloss, row.kp, row.kd, row.effort_limit)
            for kind, axis, limits, row in zip(self.model.joint_types, self.model.joint_axis,
                                               self.limits, self.joints, strict=True)
        )
        base, orientation = self.robot_pose
        if self.extended:
            moving = self.model.moving
            links = tuple((body.position, _wxyz(body.rotation)) for body in moving)
            masses, com, inertia = (tuple(getattr(body, name) for body in moving)
                                   for name in ("mass", "com", "inertia"))
        else:
            links = tuple((position, _wxyz(rotation)) for position, rotation in zip(
                self.model.joint_position, self.model.joint_orientation, strict=True
            ))
            masses, com, inertia = self.model.mass, self.model.com, self.model.inertia_diagonal
        mean, body_weights, dof_weights, actuator_acc = self.reference_inertia(
            [row.armature for row in self.joints]
        )
        root = self.floating
        free_root = None if root is None else (
            1. / root.root.inverse_mass[0], root.root_com_m,
            root.root.inertia_diagonal[0], self.inertia_frames[0],
        )
        if self.extended and self.has_free_root:
            root_body = self.model.bodies[0]
            free_root = root_body.mass, root_body.com, root_body.inertia, self.inertia_frames[0]
        free_objects = tuple(
            _free_body_input(frontend, shape, mass, inertia)
            for shape, mass, inertia in self.free_objects
        )
        return frontend.put_articulated_model(
            geom, tuple(row[1] for row in self.geom_rows),
            base_pose=(base, _wxyz(orientation)), link_poses=links, parents=self.model.parents,
            masses=masses, com=com, inertia=inertia,
            joints=joints, mean_inertia=mean, body_weights=body_weights, dof_weights=dof_weights,
            actuator_acc=actuator_acc, self_collision=self.self_collision,
            timestep=self.timestep, gravity=self.gravity, free_root=free_root,
            **({"integrator": self.integrator} if self.integrator != "euler" else {}),
            **({"free_objects": free_objects} if free_objects else {}),
            **extra,
        )

    def create_initial_bank(self, frontend, model, **capacity):
        _, _, root_dofs = self.scalar_offsets
        position, rotation = self.robot_pose
        root_q = () if not root_dofs else (*position, *_wxyz(rotation))
        root_v = (0.,) * root_dofs
        free_q = tuple(value for shape, _, _ in self.free_objects
                       for value in (*shape.position, *_wxyz(shape.orientation_xyzw)))
        free_v = (0.,) * (6 * len(self.free_objects))
        ids = tuple(row[0] for row in self.static_geom_frames)
        poses = tuple(row[1] for row in self.static_geom_frames)
        rotations = np.asarray([row[2] for row in self.static_geom_frames]).reshape(-1, 3, 3)
        use_indices = ids != tuple(range(len(ids)))
        if use_indices and getattr(frontend, "STATIC_GEOM_INPUT_CONTRACT", None) != 1:
            raise RuntimeError("autonomous static geometry indices require frontend contract v1")
        return frontend.make_primitive_data(
            model, (*root_q, *(row.position for row in self.joints), *free_q),
            poses, rotations, **capacity,
            **({"static_geom_ids": ids} if use_indices else {}),
            qvel=(*root_v, *(row.velocity for row in self.joints), *free_v),
            ctrl=tuple(row.kp * row.target_position + row.kd * row.target_velocity for row in self.joints),
            generalized_force=(*root_v, *(row.applied_force for row in self.joints), *free_v),
        )


def _reference_inertia(model, armatures, *, floating, pose):
    """Reference weights from the same PHYMEO scalar mechanics used by its importers."""
    n = model.dof_count
    q = np.zeros(n)
    base_dofs = 0
    if floating is None:
        geometry, linear, angular = robot_com_kinematics(model, q)
        masses, inertia = model.mass, _world_inertia(model, geometry)
    else:
        base_dofs = 6
        state = FloatingRobotState(np.r_[pose[0], pose[1], q], np.zeros(base_dofs + n))
        geometry, _, masses, linear, angular, inertia = _inertial_geometry(floating, state)
        # PHYMEO uses world COM velocity and world angular velocity. The Host
        # stores world origin velocity and body-local angular velocity.
        rotation = geometry.rotation[0]
        to_public = np.eye(base_dofs + n)
        to_public[3:6, 3:6] = rotation
        to_public[:3, 3:6] = np.cross(rotation.T, rotation @ floating.root_com_m).T
        linear, angular = linear @ to_public, angular @ to_public
    matrix = _mass_from_terms(masses, linear, angular, inertia)
    scalar = np.arange(base_dofs, base_dofs + n)
    matrix[scalar, scalar] += armatures
    inverse = np.linalg.solve(matrix, np.eye(base_dofs + n))
    dof_weights = inverse.diagonal().copy()
    actuator_acc = np.linalg.norm(inverse[:, scalar], axis=0)
    body_weights = np.column_stack(tuple(
        np.sum((jacobian @ inverse) * jacobian, axis=(1, 2)) / 3
        for jacobian in (linear, angular)
    ))
    # Match the existing aligned-slide reference convention for an independent leaf.
    if floating is None:
        has_children = np.bincount(model.parents + 1, minlength=n + 1)[1:] > 0
        aligned = np.count_nonzero(np.abs(model.joint_axis) > 1e-14, axis=1) == 1
        at_origin = np.all(np.abs(model.com) < 1e-6, axis=1)
        simple_slides = ((model.parents == -1) & ~has_children & aligned & at_origin
                         & (np.asarray(model.joint_types) == "prismatic"))
        dof_weights[simple_slides] = 1 / np.maximum(1e-15, model.mass[simple_slides])
        body_weights[simple_slides, 0] = dof_weights[simple_slides]
    else:
        # Native constraint reference weights group the FREE translations and rotations.
        dof_weights[:3] = dof_weights[:3].mean()
        dof_weights[3:6] = dof_weights[3:6].mean()
    for row in body_weights:
        smaller, larger = np.argsort(row)
        if row[smaller] < 1e-15 < row[larger]:
            row[smaller] = row[larger]
    mean = float(np.trace(matrix) / (base_dofs + n))
    with np.errstate(over="ignore", under="ignore"):
        moments = model.inertia_diagonal if floating is None else np.vstack(
            (floating.root.inertia_diagonal, model.inertia_diagonal)
        )
        positive = np.asarray([*masses, *moments.flat, mean,
                               *dof_weights, *actuator_acc], dtype=np.float32)
        weights = body_weights.astype(np.float32)
    if not (np.isfinite(positive).all() and np.all(positive > 0) and np.isfinite(weights).all()):
        raise ValueError("articulated mass, inertia and reference weights require representable FP32")
    return mean, body_weights, dof_weights, actuator_acc


def _resolved_joint(asset, robot, index, row, *, follower=False, passive=(None, None)):
    if row is None:
        row = NativeJointInput(robot.joint_names[index])
    defaults = dict(asset.joint_defaults)
    updates = {}
    full = isinstance(robot, FixedArticulation)
    damping, friction = (robot.damping[index], robot.friction[index]) if full else passive
    damping = 0. if damping is None else damping
    friction = 1e-5 if friction is None else friction
    for key, default in (
        ("kp", 500.), ("kd", 50.), ("armature", .01), ("damping", damping), ("frictionloss", friction),
        ("position", 0.), ("velocity", 0.), ("target_velocity", 0.),
        ("effort_limit", robot.effort_limit[index]), ("velocity_limit", robot.velocity_limit[index]),
        ("lower_limit", robot.lower[index]), ("upper_limit", robot.upper[index]),
    ):
        if getattr(row, key) is None:
            updates[key] = 0. if follower and key in {"kp", "kd"} else defaults.get(key, default)
    row = replace(row, **updates)
    if row.target_position is None:
        row = replace(row, target_position=row.position)
    if row.effort_limit is None:
        row = replace(row, effort_limit=0.)
    with np.errstate(over="ignore", under="ignore"):
        position_limits = np.asarray([row.lower_limit, row.upper_limit], np.float32)
        ctrl = np.asarray(row.kp * row.target_position + row.kd * row.target_velocity, np.float32)
    continuous = full and robot.joint_types[index] == "continuous"
    if continuous and (row.lower_limit is not None or row.upper_limit is not None):
        raise ValueError(f"continuous joint {row.name!r} cannot have position limits")
    if not np.isfinite(ctrl) or (not continuous and not position_limits[0] < position_limits[1]):
        raise ValueError("autonomous position limits and initial ctrl must be valid in FP32")
    return row


def _bind_mimic(robot, authored, joints):
    """Retain the mechanical constraint and compose the follower's servo command."""
    count = len(joints)
    if not isinstance(robot, FixedArticulation) or not robot.relations:
        return joints, tuple(range(count)), tuple((i, 1., 0.) for i in range(count)), ()
    graph = compile_mimic_coordinates(
        robot.joint_names, robot.joint_types, robot.lower, robot.upper,
        robot.velocity_limit, robot.effort_limit, robot.relations,
    )
    independent = graph.independent_indices
    initial = tuple(joints[i] for i in independent)
    joints = list(joints)
    target_bindings = []
    for i, row in enumerate(joints):
        column, = np.flatnonzero(graph.matrix[i])
        ratio, offset = float(graph.matrix[i, column]), float(graph.offset_vector[i])
        if not np.isfinite(np.float32([ratio, offset])).all() or np.float32(ratio) == 0:
            raise ValueError("autonomous mimic target map must be representable in FP32")
        target_bindings.append((int(column), ratio, offset))
        driver = initial[column]
        if i == independent[column]:
            continue
        source = authored.get(row.name)
        q, v = ratio * driver.position + offset, ratio * driver.velocity
        if source is not None:
            if source.target_position is not None or source.target_velocity is not None:
                raise ValueError("mimic follower cannot own an independent target command")
            if any(x is not None and not np.isclose(x, y, atol=1e-12, rtol=0)
                   for x, y in ((source.position, q), (source.velocity, v))):
                raise ValueError("conflicting initial mimic position/velocity")
        joints[i] = replace(row, position=q, velocity=v, target_position=ratio * driver.target_position + offset,
                            target_velocity=ratio * driver.target_velocity)
    return tuple(joints), independent, tuple(target_bindings), graph.indices


def _static_geom_frames(geoms, model, asset):
    """Seed world-welded geometry skipped by native kinematics in both banks."""
    fixed = {0: (np.zeros(3), np.eye(3))}
    if isinstance(model, FixedArticulation) and not asset.floating:
        for index, body in enumerate(model.bodies, 1):
            if body.dof < 0 and body.parent in fixed:
                parent_position, parent_rotation = fixed[body.parent]
                rotation = quaternion_matrix(np.asarray([body.rotation]))[0]
                fixed[index] = (parent_position + parent_rotation @ body.position,
                                parent_rotation @ rotation)
    elif not asset.floating:
        fixed[1] = (asset.position, quaternion_matrix(np.asarray([asset.orientation_xyzw]))[0])
    frames = []
    for index, row in enumerate(geoms):
        if row[1] in fixed:
            position, rotation = fixed[row[1]]
            geometry_rotation = quaternion_matrix(np.asarray([row[5]]))[0]
            frames.append((index, tuple(position + rotation @ row[4]),
                           tuple(map(tuple, rotation @ geometry_rotation))))
    return tuple(frames)


def _robot_geometries(robot, asset):
    friction = dict(asset.body_friction)
    if friction.keys() - {collider.link_name for collider in robot.collisions}:
        raise ValueError("autonomous body friction requires a colliding source link")
    for collider in robot.collisions:
        shape = collider.shape
        if shape.kind == "sphere":
            kind, size = "sphere", (shape.radius,)
        elif shape.kind == "cylinder":
            kind, size = "cylinder", shape.curved.dimensions_m
        elif shape.kind == "hull":
            # The URDF importer emits a hull only for its box primitive.
            local_rotation = quaternion_matrix(np.asarray([collider.origin_xyzw]))[0]
            vertices = (np.asarray(collider.local_vertices) - collider.origin_xyz) @ local_rotation
            kind, size = "box", tuple(np.ptp(vertices, axis=0) / 2)
        else:
            raise NotImplementedError("autonomous URDF collisions require sphere, box or cylinder")
        name = f"{asset.name}::{collider.link_name}::{collider.name}"
        yield (name, collider.link_index + 2, kind, size, collider.origin_xyz, collider.origin_xyzw,
               (friction.get(collider.link_name, 1.), .005, .0001), 3)


def _take_floating_passive(root):
    """Bind authored passive loads by joint name before topology reorders DOFs."""
    passive = {}
    for joint in root.findall("joint"):
        dynamics = _child(joint, "dynamics", "floating URDF joint", required=False)
        if dynamics is None or joint.get("type") == "fixed":
            continue
        name = joint.attrib["name"]
        _attributes(dynamics, {"damping", "friction"}, name)
        _children(dynamics, set(), name)
        row = tuple(_number(dynamics, k, name) if k in dynamics.attrib else None
                    for k in ("damping", "friction"))
        if any(value is not None and value < 0 for value in row):
            raise ValueError(f"joint {name!r} requires nonnegative passive damping and friction")
        passive[name] = row
        joint.remove(dynamics)
    return passive


def compile_scalar_template(asset, *, timestep, gravity, statics, integrator,
                            capture_visuals, capture_frames, free_objects=()):
    """Use the PHYMEO fixed-base importer and keep its source frames and SI values."""
    source = asset.urdf_xml.encode("utf8") if asset.urdf_xml is not None else Path(asset.path).read_bytes()
    xml = ET.fromstring(source)
    if xml.tag != "robot":
        raise NotImplementedError("autonomous PhySolve requires a scalar URDF tree")
    import_options = take_compiler_options(xml)
    display = (authored_visuals(xml, asset, import_options)
               if capture_visuals and not import_options.discard_visual else None)
    extended = bool(
        xml.findall("joint[@type='fixed']")
        or xml.findall("joint[@type='continuous']")
        or xml.findall(".//collision/geometry/mesh")
        or (not asset.floating and (xml.findall("joint/mimic") or xml.findall("joint/dynamics")))
    )
    passive = {}
    floating = None
    if not extended:
        if any(geom.tag not in {"sphere", "box", "cylinder"}
               for geom in xml.findall(".//collision/geometry/*")):
            raise NotImplementedError("autonomous scalar URDF collisions require sphere, box or cylinder")
        compact_xml = deepcopy(xml) if asset.floating else xml
        if asset.floating:
            passive = _take_floating_passive(compact_xml)
        loader = load_floating_urdf if asset.floating else load_urdf
        try:
            robot = loader(asset.path, gravity=gravity, collapse_fixed_joints=False,
                           source=ET.tostring(compact_xml), depth_first=True)
        except _NonDiagonalLinkInertia:
            if not asset.floating:
                raise
            # PhySolve already accepts principal moments and an independent
            # inertia frame. Keep the untouched authored poses and passives.
            extended = True
            passive = {}
        else:
            floating = robot.model if asset.floating else None
            model = robot.model if floating is None else floating.children
    if extended:
        if asset.floating:
            _attributes(xml, {"name", "version"}, "floating URDF")
            _children(xml, {"link", "joint", "material"}, "floating URDF")
            if xml.get("version", "1.0") != "1.0":
                raise ValueError("floating URDF requires version 1.0")
        robot = model = read_fixed_articulation(asset, xml, import_options=import_options)
    if model is None:
        raise NotImplementedError("autonomous floating URDF requires scalar child joints")
    n = model.dof_count
    # The frozen dense MathDx payload holds exact tiles for 1 through 60 DOFs.
    if n + (6 if asset.floating else 0) > 60:
        raise NotImplementedError("autonomous PhySolve exceeds the frozen 60-DOF tile range")
    supplied = {row.name: row for row in asset.joints}
    if supplied.keys() - set(robot.joint_names):
        raise ValueError(f"{asset.name}: unknown autonomous joint inputs")
    followers = {row[1] for row in robot.relations} if extended else set()
    if followers and not asset.prepare_follower_drives:
        raise NotImplementedError("autonomous mimic requires prepared follower drives")
    joints = tuple(_resolved_joint(asset, robot, i, supplied.get(name), follower=name in followers,
                                  passive=passive.get(name, (None, None)))
                   for i, name in enumerate(robot.joint_names))
    joints, independent, target_bindings, mimic = _bind_mimic(robot, supplied, joints)
    limits = tuple((j.lower_limit, j.upper_limit, j.velocity_limit) for j in joints)
    joint_names = tuple(asset.name + "::" + joint.name for joint in joints)
    source_hash = sha256(source).hexdigest()
    if extended:
        bodies = tuple(body.name for body in model.bodies)
        links = tuple(body.split("::", 1)[1] for body in bodies)
        robot_geoms = model.geometries
    else:
        links = (robot.root_link, *robot.link_names)
        bodies = tuple(f"{asset.name}::{link}" for link in links)
        robot_geoms = tuple(sorted(_robot_geometries(robot, asset), key=lambda item: item[1]))
    geoms = tuple((shape.name, 0, shape.kind, shape.size, shape.position, shape.orientation_xyzw,
                   shape.friction, shape.contact_dim, *((None,) if extended else ()))
                  for shape in statics) + robot_geoms
    scene_bodies = bodies + tuple(shape.name for shape in free_objects)
    free_geoms = tuple((shape.name, len(bodies) + 1 + i, shape.kind, shape.size, (0., 0., 0.),
                        shape.geom_orientation_xyzw, shape.friction, shape.contact_dim,
                        *((None,) if extended else ()))
                       for i, shape in enumerate(free_objects))
    scene_geoms = geoms + free_geoms
    if (len(set(scene_bodies)) != len(scene_bodies)
            or len({g[0] for g in scene_geoms}) != len(scene_geoms)):
        raise ValueError("duplicate autonomous geometry name")
    rows = []
    counts = [0] * (len(scene_bodies) + 1)
    names = (None, *scene_bodies)
    for index, geom in enumerate(scene_geoms):
        geom_name, body = geom[:2]
        rows.append(SourceGeometry(CollisionKey(0, body, counts[body]), index,
                                   names[body], geom_name, body))
        counts[body] += 1
    mesh_identities = tuple(sha256(item.visual.vertices.tobytes() + item.visual.triangles.tobytes()).hexdigest()
                            for item in model.meshes) if extended else ()
    identity = _digest(("phymeo-scalar-v1", source_hash, asset.name, scene_geoms, mesh_identities))
    collision = CollisionSourceMapping(identity, asset.name, 0, tuple(rows), ())
    compilation = ScalarCompilation(
        model, (asset.position, asset.orientation_xyzw), joint_names, joints,
        asset.self_collision, limits, geoms, collision, float(timestep), tuple(gravity), integrator,
        floating, bodies[0] if asset.floating else None,
        tuple((shape, *_mass_and_inertia(shape)) for shape in free_objects),
        mimic,
        _static_geom_frames(geoms, model, asset),
    )
    j0, q0, v0 = compilation.scalar_offsets
    root_bindings = ((asset.name + "::free", 0, 0, 0, None),) if asset.floating else ()
    root_dimensions = ((7, 6),) if asset.floating else ()
    root_com = (0., 0., 0.) if floating is None else tuple(floating.root_com_m)
    if extended:
        com = tuple(body.com for body in model.bodies)
        source_joints = {row.find("child").attrib["link"]: row for row in xml.findall("joint")}
        joint_frames = tuple(
            NativeJointFrame(asset.name, source_joints[link].attrib["name"],
                             ("revolute" if source_joints[link].attrib["type"] == "continuous"
                              else source_joints[link].attrib["type"]), names[body.parent], body.name,
                             (*body.position, *body.rotation), (0., 0., 0., 0., 0., 0., 1.),
                             None if body.dof < 0 else model.joint_axis[body.dof])
            for link, body in zip(links[1:], model.bodies[1:], strict=True)
        )
    else:
        com = (root_com, *map(tuple, model.com))
        joint_frames = tuple(NativeJointFrame(
            asset.name, row.name, model.joint_types[i], bodies[model.parents[i] + 1], bodies[i + 1],
            (*model.joint_position[i], *model.joint_orientation[i]),
            (0., 0., 0., 0., 0., 0., 1.), tuple(model.joint_axis[i]),
        ) for i, row in enumerate(joints))
    frames = NativeFrameMetadata(
        tuple(zip(scene_bodies, (*com, *((0., 0., 0.),) * len(free_objects)), strict=True)),
        joint_frames,
    )
    authored_bodies = set() if display is None else {item.body for item in display.instances}
    visuals = VisualScene(tuple(
        VisualInstance(g[0], names[g[1]], g[2], () if g[2] == "mesh" else g[3], g[4], g[5],
                       (.5, .5, .5, 1.), mesh=g[0] if g[2] == "mesh" else None) for g in scene_geoms
        if names[g[1]] not in authored_bodies
    ) + (() if display is None else display.instances),
        (tuple(item.visual for item in model.meshes) if extended else ())
        + (() if display is None else display.meshes)) if capture_visuals else None
    diagnostics = (model.diagnostics if extended else ()) + tuple(
        f"initial_position_outside_limit: {name}: {row.position} outside {bound[:2]}; unchanged"
        for name, row, bound in zip(joint_names, joints, limits, strict=True)
        if bound[0] is not None and not bound[0] <= row.position <= bound[1]
    )
    return NativeTemplate(
        compilation, None,
        root_bindings + tuple((name, j0+i, q0+i, v0+i, i) for i, name in enumerate(joint_names))
        + tuple((shape.name + "::free", j0 + n + i, q0 + n + 7*i, v0 + n + 6*i, None)
                for i, shape in enumerate(free_objects)),
        tuple((name, i + 1) for i, name in enumerate(scene_bodies)),
        tuple((geom[0], i) for i, geom in enumerate(scene_geoms)), ((asset.name, asset.path, source_hash),),
        qualification=f"native {'floating' if asset.floating else 'fixed-base'} PhySolve {integrator}",
        joint_dimensions=root_dimensions + ((1, 1),) * n + ((7, 6),) * len(free_objects),
        scalar_controls=tuple((joint_names[i], i, joints[i].kp, joints[i].kd,
                               joints[i].target_position, joints[i].target_velocity) for i in independent),
        target_bindings=tuple((name, *binding) for name, binding in zip(joint_names, target_bindings, strict=True)),
        follower_drives=tuple((name, i, *binding, joints[i].kp, joints[i].kd)
                              for i, (name, binding) in enumerate(zip(joint_names, target_bindings, strict=True))
                              if i not in independent),
        joint_limits=tuple((name, *bound) for name, bound in zip(joint_names, limits, strict=True)),
        import_diagnostics=diagnostics,
        asset_identities=(NativeAssetIdentity(asset.name, "urdf", robot.name, tuple(
            (source_name, name, i + 1) for i, (source_name, name) in enumerate(zip(links, bodies, strict=True))
        )),),
        frame_metadata=frames if capture_frames else None, visual_scene=visuals,
    )
