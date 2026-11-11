"""Cold lowering of an owned articulation world to an explicit MuJoCo profile.

This is a solver adapter, not a second physical model. Resident compilation
continues to consume SceneDeclaration without MuJoCo. Native PD, soft limits
and contact law are explicitly selected, never inferred from resident gains.
"""

from dataclasses import dataclass, field, replace
from xml.etree import ElementTree as ET

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.validation import validate_dt
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ._robot_asset import RobotAsset
from ._urdf_frames import FoldedURDFAsset
from .floating_urdf import FloatingRobotAsset
from .mujoco_compiler import MuJoCoCompilation, compile_mujoco_xml
from .scene_compiler import ArticulationWorld, SceneDeclaration


def _owned(value):
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _text(value):
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).ravel())


def _quat(xyzw):
    return _text(np.asarray(xyzw)[[3, 0, 1, 2]])


@dataclass(frozen=True)
class MuJoCoWorldProfile:
    """Explicit alternate laws: affine capped PD, native soft stops/contact.

    Resident material stiffness/damping is NOT converted to solref/solimp.
    Friction is native sliding/torsional/rolling friction; SI timestep seconds.
    Limits can be disabled explicitly to retain metadata-only asset bounds.
    """

    timestep: float
    friction: tuple[float, float, float]
    joint_limits: bool
    contact_dim: int = 6

    def __post_init__(self):
        object.__setattr__(self, "timestep", float(validate_dt(self.timestep, np.float64)))
        f = np.asarray(self.friction, dtype=np.float64)
        if f.shape != (3,) or not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError("friction requires three finite nonnegative coefficients")
        if not isinstance(self.joint_limits, bool):
            raise TypeError("joint_limits must be explicitly boolean")
        if type(self.contact_dim) is not int or self.contact_dim not in (1, 3, 4, 6):
            raise ValueError("contact_dim must be 1, 3, 4 or 6")
        active = {1: 0, 3: 1, 4: 2, 6: 3}[self.contact_dim]
        if np.any(f[active:] != 0):
            raise ValueError("contact_dim would silently ignore requested friction")
        object.__setattr__(self, "friction", tuple(float(x) for x in f))


@dataclass(frozen=True)
class NativeJointBinding:
    name: str
    joint_id: int
    qpos_address: int
    dof_address: int
    actuator_id: int


@dataclass(frozen=True)
class NativeShapeBinding:
    """Domain-qualified source identity; geom_id is only a compilation address.

    Asset-local keys can equal environment keys. Use source_identity together
    with the compiled world's instance/world, never source_key alone.
    """

    source_key: CollisionKey
    name: str
    geom_id: int
    environment: bool

    @property
    def source_identity(self):
        return ("environment" if self.environment else "asset", self.source_key)


@dataclass(frozen=True)
class NativeFrameBinding:
    """Authored frame relative to a native physical body, including fixed fusion."""

    name: str
    body_id: int
    local_position: tuple[float, ...]
    local_orientation_xyzw: tuple[float, ...]


@dataclass(frozen=True)
class CompiledMuJoCoWorld:
    compilation: MuJoCoCompilation
    world: int
    instance: str
    declaration_identity: str
    body_ids: tuple[tuple[str, int], ...]
    joints: tuple[NativeJointBinding, ...]
    shapes: tuple[NativeShapeBinding, ...]
    frames: tuple[NativeFrameBinding, ...]
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    initial_ctrl: np.ndarray
    binding_version: int = field(default=1, init=False)
    contact_law: str = "mujoco-pyramidal-default-solref-solimp"
    drive_law: str = "mujoco-affine-pd-feedforward-inside-cap"

    def fresh_native_data(self):
        """Owned CPU initialization only; no simulation or forward is performed."""
        import mujoco

        model = self.compilation.fresh_native_model()
        data = mujoco.MjData(model)
        data.qpos[:] = self.initial_qpos
        data.qvel[:] = self.initial_qvel
        data.ctrl[:] = self.initial_ctrl
        return model, data


def _geom(parent, assets, name, shape, points, profile):
    """Preserve finite material geometry; no visual/collision approximation."""
    points = np.asarray(points)
    attrs = dict(
        name=name, contype="0", conaffinity="0", mass="0", friction=_text(profile.friction)
    )
    if shape.kind == "sphere":
        attrs.update(type="sphere", pos=_text(points[0]), size=_text([shape.radius]))
    elif shape.kind == "capsule":
        attrs.update(type="capsule", fromto=_text(points), size=_text([shape.radius]))
    elif shape.kind == "hull" and shape.radius == 0:
        ET.SubElement(assets, "mesh", name=name, vertex=_text(points))
        attrs.update(type="mesh", mesh=name)
    elif shape.kind in ("cylinder", "ellipsoid"):
        attrs.update(
            type=shape.kind,
            pos=_text(points[0]),
            size=_text(shape.curved.dimensions_m),
            quat=_quat(shape.curved.orientation_xyzw),
        )
    else:
        raise NotImplementedError(f"native scene geometry not admitted: {shape.kind}")
    ET.SubElement(parent, "geom", **attrs)


def compile_mujoco_world(world, *, profile, instance):
    """Compile one declared world, retaining original public coordinates/identity.

    A cold single-world compiler is not a batched runtime implementation.
    Fixed/floating ordinary assets are admitted; mimic and same-world multiple
    articulations need their own conversion. Self contact remains unqualified.
    The result must be consumed with its initial arrays, not just its MJB.
    """
    if not isinstance(world, ArticulationWorld) or not isinstance(profile, MuJoCoWorldProfile):
        raise TypeError("expected ArticulationWorld and explicit MuJoCoWorldProfile")
    declaration = SceneDeclaration((world,))
    world = declaration.worlds[0]
    asset = world.asset
    if type(asset) not in (RobotAsset, FloatingRobotAsset) and not isinstance(
        asset, FoldedURDFAsset
    ):
        raise NotImplementedError("native scene requires an ordinary fixed/floating asset")
    if world.self_collision:
        raise NotImplementedError("native scene self contact is not yet admitted")
    floating = isinstance(asset, FloatingRobotAsset)
    model = asset.model.children if floating else asset.model
    names = (asset.root_link, *asset.link_names)
    if len(set(names)) != len(names) or len(set(asset.joint_names)) != len(asset.joint_names):
        raise ValueError("public body/joint names must be unique within their index space")
    count = 0 if model is None else model.dof_count
    if len(asset.link_names) != count or len(asset.joint_names) != count:
        raise ValueError("asset names do not match its physical tree")
    if len(asset.collisions) != len(asset.collision_model.shapes) or {
        row.shape.key for row in asset.collisions
    } != {s.key for s in asset.collision_model.shapes}:
        raise ValueError("asset collider bindings do not cover the physical shape table")
    joint_control = world.initial_control.joints if floating else world.initial_control
    if floating and world.initial_control.external is not None:
        raise NotImplementedError(
            "native scene initial external body loads require explicit binding"
        )

    xml = ET.Element("mujoco", model=asset.name)
    ET.SubElement(xml, "compiler", angle="radian", fusestatic="false", inertiafromgeom="false")
    gravity = asset.model.root.gravity if floating else model.gravity
    ET.SubElement(
        xml,
        "option",
        timestep=_text([profile.timestep]),
        gravity=_text(gravity),
        integrator="Euler",
        solver="Newton",
        jacobian="dense",
        cone="pyramidal",
    )
    assets = ET.SubElement(xml, "asset")
    worldbody = ET.SubElement(xml, "worldbody")
    root_pos = world.initial_state.q[:3] if floating else np.asarray(asset.root_xyz)
    root_quat = world.initial_state.q[3:7] if floating else np.asarray(asset.root_xyzw)
    root = ET.SubElement(worldbody, "body", name="root", pos=_text(root_pos), quat=_quat(root_quat))
    if floating:
        ET.SubElement(root, "freejoint", name="free_root")
        physical = asset.model
        ET.SubElement(
            root,
            "inertial",
            mass=_text([1 / physical.root.inverse_mass[0]]),
            pos=_text(physical.root_com_m),
            quat=_quat(physical.root_inertia_frame_xyzw),
            diaginertia=_text(physical.root.inertia_diagonal[0]),
        )
    bodies = {-1: root}
    root_rotation = quaternion_matrix(np.asarray(root_quat)[None])[0]
    for i in range(count):
        parent = int(model.parents[i])
        position, quat = model.joint_position[i], model.joint_orientation[i]
        # Fixed RobotModel root children already use world coordinates.
        if parent == -1 and not floating:
            position = root_rotation.T @ (position - root_pos)
            quat = matrix_quaternion(root_rotation.T @ quaternion_matrix(quat[None])[0])
        body = ET.SubElement(
            bodies[parent], "body", name=f"body_{i}", pos=_text(position), quat=_quat(quat)
        )
        bodies[i] = body
        attrs = dict(
            name=f"joint_{i}",
            type="hinge" if model.joint_types[i] == "revolute" else "slide",
            axis=_text(model.joint_axis[i]),
            limited="false",
        )
        lo, hi = asset.lower[i], asset.upper[i]
        if profile.joint_limits and (np.isfinite(lo) or np.isfinite(hi)):
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise NotImplementedError("native soft limits require two distinct finite bounds")
            attrs.update(limited="true", range=_text([lo, hi]))
        if floating:
            attrs.update(
                armature=_text([asset.model.joint_armature[i]]),
                damping=_text([asset.model.joint_damping[i]]),
            )
        ET.SubElement(body, "joint", **attrs)
        ET.SubElement(
            body,
            "inertial",
            mass=_text([model.mass[i]]),
            pos=_text(model.com[i]),
            diaginertia=_text(model.inertia_diagonal[i]),
        )

    shape_rows = []
    for i, row in enumerate(asset.collisions):
        if row.link_index not in bodies:
            raise ValueError("collider source link does not match its physical binding")
        shape, points = row.shape, np.asarray(row.local_vertices)
        if isinstance(asset, FoldedURDFAsset):
            site = asset.sites[row.link_name]
            if site.link_index != (None if row.link_index < 0 else row.link_index):
                raise ValueError("folded collider frame differs from physical owner")
            rotation = quaternion_matrix(np.asarray(site.local_orientation_xyzw)[None])[0]
            position = np.asarray(site.local_position_m)
            if site.link_index is None:
                position = root_rotation.T @ (position - root_pos)
                rotation = root_rotation.T @ rotation
            points = position + points @ rotation.T
            if shape.curved is not None:
                orientation = (
                    rotation @ quaternion_matrix(np.asarray(shape.curved.orientation_xyzw)[None])[0]
                )
                shape = replace(
                    shape,
                    curved=replace(
                        shape.curved, orientation_xyzw=tuple(matrix_quaternion(orientation))
                    ),
                )
        elif row.link_name != names[row.link_index + 1]:
            raise ValueError("collider source link does not match its physical binding")
        name = f"robot_geom_{i}"
        _geom(bodies[row.link_index], assets, name, shape, points, profile)
        shape_rows.append((row.shape.key, row.name, name, False))
    for i, row in enumerate(world.static_shapes):
        name = f"environment_geom_{i}"
        _geom(worldbody, assets, name, row.shape, row.points_m, profile)
        shape_rows.append((row.shape.key, name, name, True))

    # SceneDeclaration currently enables only explicitly material-bound robot /
    # static pairs. Explicit native pairs also avoid implicit parent filtering.
    contact = ET.SubElement(xml, "contact")
    for i, collider in enumerate(asset.collisions):
        if collider.link_index < 0 and not floating:
            continue
        for j, static in enumerate(world.static_shapes):
            if static.material is not None and (
                collider.shape.group & static.shape.mask
                and static.shape.group & collider.shape.mask
            ):
                f = profile.friction
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=f"robot_geom_{i}",
                    geom2=f"environment_geom_{j}",
                    condim=str(profile.contact_dim),
                    friction=_text([f[0], f[0], f[1], f[2], f[2]]),
                )

    actuators = ET.SubElement(xml, "actuator")
    for i in range(count):
        cap = world.effort_limit[i]
        # A zero cap in PHYMEO means zero drive, unlike P2PE's unlimited sentinel.
        attrs = dict(
            name=f"drive_{i}",
            joint=f"joint_{i}",
            gainprm="1",
            biastype="affine",
            biasprm=_text([0, -world.kp[i], -world.kd[i]]),
        )
        if cap == 0:
            attrs.update(gainprm="0", biasprm="0 0 0")
        else:
            attrs.update(forcelimited="true", forcerange=_text([-cap, cap]))
        ET.SubElement(actuators, "general", **attrs)

    compilation = compile_mujoco_xml(
        ET.tostring(xml, encoding="unicode"),
        require_asset=False,
        instance=instance,
        world=world.world,
    )
    native = compilation.mj_model
    qpos, qvel, ctrl = native.qpos0.copy(), np.zeros(native.nv), np.zeros(native.nu)
    if floating:
        qpos[:7] = np.r_[world.initial_state.q[:3], world.initial_state.q[[6, 3, 4, 5]]]
        v = world.initial_state.v
        qvel[:3] = v[:3] - np.cross(v[3:6], root_rotation @ asset.model.root_com_m)
        qvel[3:6] = root_rotation.T @ v[3:6]
    joints = []
    for i in range(count):
        joint, actuator = native.joint(f"joint_{i}"), native.actuator(f"drive_{i}")
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        joints.append(NativeJointBinding(asset.joint_names[i], joint.id, qa, va, actuator.id))
        qpos[qa] = world.initial_state.q[i + (7 if floating else 0)]
        qvel[va] = world.initial_state.v[i + (6 if floating else 0)]
        velocity = 0 if joint_control.target_velocity is None else joint_control.target_velocity[i]
        effort = 0 if joint_control.feedforward is None else joint_control.feedforward[i]
        ctrl[actuator.id] = (
            world.kp[i] * joint_control.target_q[i] + world.kd[i] * velocity + effort
        )
    if not all(np.isfinite(value).all() for value in (qpos, qvel, ctrl)):
        raise ValueError("native initialization is not representable as finite FP64")
    body_ids = ((asset.root_link, native.body("root").id),) + tuple(
        (name, native.body(f"body_{i}").id) for i, name in enumerate(asset.link_names)
    )
    shapes = tuple(
        NativeShapeBinding(key, name, native.geom(native_name).id, environment)
        for key, name, native_name, environment in shape_rows
    )
    frames = []
    if isinstance(asset, FoldedURDFAsset):
        for name, site in asset.sites.items():
            position = np.asarray(site.local_position_m)
            quat = np.asarray(site.local_orientation_xyzw)
            if site.link_index is None:
                position = root_rotation.T @ (position - root_pos)
                quat = matrix_quaternion(root_rotation.T @ quaternion_matrix(quat[None])[0])
            body_name = "root" if site.link_index is None else f"body_{site.link_index}"
            frames.append(
                NativeFrameBinding(name, native.body(body_name).id, tuple(position), tuple(quat))
            )
    else:
        frames = [
            NativeFrameBinding(name, body, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            for name, body in body_ids
        ]
    return CompiledMuJoCoWorld(
        compilation,
        world.world,
        instance,
        declaration.identity,
        body_ids,
        tuple(joints),
        shapes,
        tuple(frames),
        _owned(qpos),
        _owned(qvel),
        _owned(ctrl),
    )
