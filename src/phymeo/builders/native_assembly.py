"""PhySolve-specific asset assembly; native-only, never a resident declaration.

Uses the existing native compiler and runtime contracts. No P2PE objects or
Newton compatibility model are introduced. All participants share one model.
"""

from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from ..backends.physolve_initial import NativeInitialState
from .mujoco_compiler import compile_mujoco_spec
from .native_resources import bind_attached_resources, load_asset_source, own_spec_resources


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} requires finite shape ({size},)")
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class NativeJointInput:
    name: str
    position: float | None = None
    velocity: float | None = None
    target_position: float | None = None
    target_velocity: float | None = None
    applied_force: float = 0.0
    kp: float | None = None
    kd: float | None = None
    armature: float | None = None
    damping: float | None = None
    frictionloss: float | None = None
    # None keeps URDF motor rating; explicit zero means unlimited in this
    # *native profile*. ArticulationWorld's zero-cap semantics do not change.
    effort_limit: float | None = None
    lower_limit: float | None = None
    upper_limit: float | None = None
    velocity_limit: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or "::" in self.name:
            raise ValueError("joint input requires an asset-local name")
        for key, value in vars(self).items():
            if key == "name" or value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"joint {self.name}.{key} requires a finite scalar")
            # Freeze accepted real scalars before retaining them; never keep caller storage.
            try:
                value = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError(f"joint {self.name}.{key} requires a finite scalar") from error
            if not np.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
                raise ValueError(f"joint {self.name}.{key} requires a finite scalar")
            if (
                key
                in {
                    "kp",
                    "kd",
                    "armature",
                    "damping",
                    "frictionloss",
                    "effort_limit",
                    "velocity_limit",
                }
                and value < 0
            ):
                raise ValueError(f"joint {self.name}.{key} must be nonnegative")
            object.__setattr__(self, key, value)
        if self.lower_limit is not None and self.upper_limit is not None:
            if self.lower_limit >= self.upper_limit:
                raise ValueError("native position limits require lower < upper")


@dataclass(frozen=True)
class NativeAssetInstance:
    name: str
    path: str
    position: tuple = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    joints: tuple[NativeJointInput, ...] = ()
    drive_mode: str = "preserve"
    self_collision: bool = True
    joint_defaults: tuple = ()
    floating: bool = False
    urdf_xml: str | None = None
    resource_roots: tuple[str, ...] = ()
    # Asset-local body name and sliding friction; other contact parameters stay native.
    body_friction: tuple[tuple[str, float], ...] = ()
    # Opt-in zero-gain servo slots for cold runtime follower gain edits.
    prepare_follower_drives: bool = False

    def __post_init__(self):
        if type(self.prepare_follower_drives) is not bool:
            raise TypeError("prepare_follower_drives requires bool")
        if self.prepare_follower_drives and self.drive_mode != "scalar_pd":
            raise ValueError("prepare_follower_drives requires scalar_pd")
        if not isinstance(self.name, str) or not self.name or "::" in self.name:
            raise ValueError("instance requires a unique unscoped name")
        object.__setattr__(self, "path", str(Path(self.path).resolve()))
        if self.urdf_xml is not None and (
            not isinstance(self.urdf_xml, str) or not self.urdf_xml.strip()
        ):
            raise ValueError("urdf_xml requires nonempty XML text")
        if isinstance(self.resource_roots, (str, bytes)):
            raise TypeError("resource_roots requires a sequence of paths")
        roots = tuple(self.resource_roots)
        if any(not isinstance(p, (str, Path)) or not str(p).strip() for p in roots):
            raise ValueError("resource_roots requires nonempty paths")
        object.__setattr__(self, "resource_roots", tuple(str(Path(p).resolve()) for p in roots))
        friction = tuple(tuple(row) for row in self.body_friction)
        names = set()
        for row in friction:
            if len(row) != 2:
                raise ValueError("body_friction requires (source body name, sliding friction)")
            name, value = row
            if not isinstance(name, str) or not name or "::" in name or name in names:
                raise ValueError("body_friction requires unique asset-local body names")
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= float(np.finfo(np.float32).max)):
                raise ValueError("body friction requires a nonnegative finite FP32 scalar")
            names.add(name)
        object.__setattr__(self, "body_friction", tuple((name, float(v)) for name, v in friction))
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        q = _vector(self.orientation_xyzw, 4, "orientation")
        if abs(np.linalg.norm(q) - 1) > 1e-10:
            raise ValueError("orientation requires unit XYZW")
        object.__setattr__(self, "orientation_xyzw", q)
        rows = tuple(self.joints)
        if any(not isinstance(row, NativeJointInput) for row in rows):
            raise TypeError("joints requires NativeJointInput")
        if len({r.name for r in rows}) != len(rows):
            raise ValueError("duplicate joint input")
        object.__setattr__(self, "joints", rows)
        defaults = dict(self.joint_defaults)
        if len(defaults) != len(self.joint_defaults) or set(defaults) - {
            "kp",
            "kd",
            "armature",
            "damping",
            "frictionloss",
        }:
            raise ValueError("invalid scalar joint defaults")
        if any(value is None for value in defaults.values()):
            raise ValueError("scalar joint defaults must contain explicit finite values")
        validated = NativeJointInput("validation", **defaults)
        object.__setattr__(self, "joint_defaults",
                           tuple((key, getattr(validated, key)) for key in defaults))
        if (
            self.drive_mode not in {"preserve", "scalar_pd"}
            or type(self.self_collision) is not bool
        ):
            raise ValueError("explicit drive_mode/self_collision required")
        if self.drive_mode == "preserve" and (rows or defaults):
            raise ValueError("preserve mode cannot silently ignore joint overrides")
        if type(self.floating) is not bool:
            raise TypeError("floating must be bool")
        if self.floating and self.drive_mode != "scalar_pd":
            raise NotImplementedError("floating assembly currently requires scalar_pd URDF")


@dataclass(frozen=True)
class NativeStaticShape:
    name: str
    kind: str
    size: tuple
    position: tuple = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    friction: tuple = (1.0, 0.005, 0.0001)
    contact_dim: int = 6

    def __post_init__(self):
        widths = {"sphere": 1, "box": 3, "capsule": 2, "cylinder": 2, "plane": 3, "ellipsoid": 3}
        if self.kind not in widths or not self.name or "::" in self.name:
            raise ValueError("native static requires a named primitive or plane")
        object.__setattr__(self, "size", _vector(self.size, widths[self.kind], "size"))
        if self.kind == "plane":
            # MuJoCo planes are infinite collision surfaces. The first two
            # dimensions only bound display; the third is display grid spacing.
            if min(self.size[:2]) < 0 or self.size[2] <= 0:
                raise ValueError("plane display extents must be nonnegative and spacing positive")
            if (self.size[0] == 0) != (self.size[1] == 0):
                raise ValueError("partially infinite plane display is not qualified")
        elif min(self.size) <= 0:
            raise ValueError("shape dimensions must be positive")
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        q = _vector(self.orientation_xyzw, 4, "orientation")
        if abs(np.linalg.norm(q) - 1) > 1e-10:
            raise ValueError("orientation requires unit XYZW")
        object.__setattr__(self, "orientation_xyzw", q)
        object.__setattr__(self, "friction", _vector(self.friction, 3, "friction"))
        if min(self.friction) < 0:
            raise ValueError("friction must be nonnegative")
        if type(self.contact_dim) is not int or self.contact_dim not in (1, 3, 4, 6):
            raise ValueError("contact_dim requires 1, 3, 4 or 6")
        # Native contact dimension selects active friction components. Preserve
        # inactive coefficients as authored; do not silently rewrite materials.


@dataclass(frozen=True)
class NativeRigidShape(NativeStaticShape):
    """One native solid primitive with a genuine FREE root; explicit mass wins."""

    mass: float | None = None
    density: float = 1000.0

    def __post_init__(self):
        super().__post_init__()
        if self.kind == "plane":
            raise ValueError("plane requires static world ownership")
        for name, value in (("mass", self.mass), ("density", self.density)):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or value <= 0
                or value > np.finfo(np.float32).max
            ):
                raise ValueError(f"rigid {name} requires positive finite FP32 value")
        if self.density is None:
            raise ValueError("rigid density requires a positive value")


@dataclass(frozen=True)
class NativeAssetIdentity:
    """Source names bound to one compiled instance, not global entity IDs."""

    instance: str
    format: str
    model_name: str
    # (original link name, scoped compiled body name, world-local native index).
    bodies: tuple


@dataclass(frozen=True)
class NativeTemplate:
    primitive_contact_contract_version = 1
    mjcf_scalar_contract_version = 1

    compilation: object
    initial_state: NativeInitialState
    # Names are public scope; native indices are this compilation only.
    joint_bindings: tuple  # (name, joint id, qpos address, DOF address, actuator id)
    body_bindings: tuple
    shape_bindings: tuple
    asset_sources: tuple  # instance, path, primary SHA256
    qualification: str = "native-only/physolve; resident-unqualified"
    contract_version: int = 1
    collision_plans: tuple = ()
    # (public joint, native actuator, kp, kd, initial target q, initial target qd).
    scalar_controls: tuple = ()
    # name, native actuator, independent command index, ratio, offset, kp, kd
    follower_drives: tuple = ()
    follower_drive_contract_version: int = 1
    # (public joint, independent command index, multiplier, position offset).
    target_bindings: tuple = ()
    target_contract_version: int = 1
    import_diagnostics: tuple = ()
    # Immutable declarations, not another solver state. Velocity is not clamped.
    # (public scalar joint, lower or None, upper or None, velocity rating or None).
    joint_limits: tuple = ()
    # Widths aligned with joint_bindings; native FREE coordinates are not scalar joints.
    joint_dimensions: tuple = ()
    coordinate_contract_version: int = 1
    visual_scene: object = None
    visual_diagnostics: tuple = ()
    asset_identities: tuple[NativeAssetIdentity, ...] = ()
    asset_identity_contract_version: int = 1
    # Private cold compiler owner. XML serialization rounds native authoring
    # values; the public compilation.mj_spec is caller-mutable. Never expose or
    # edit this snapshot, only copy it for a newly requested rule compilation.
    _source_spec: object = field(default=None, repr=False, compare=False)
    # None means not requested; an empty contract means no logical URDF joints.
    frame_metadata: object = None
    complete_scene_contract_version: int = 0
    actuator_bindings: tuple = ()
    native_identity: tuple = ()

    @property
    def scalar_joint_bindings(self):
        if self.coordinate_contract_version != 1 or len(self.joint_dimensions) != len(
            self.joint_bindings
        ):
            raise ValueError("native coordinate binding contract is missing or unsupported")
        if any(width not in ((1, 1), (7, 6)) for width in self.joint_dimensions):
            raise NotImplementedError("scalar consumer does not admit ball or multi-DOF joints")
        return tuple(
            row for row, width in zip(self.joint_bindings, self.joint_dimensions) if width == (1, 1)
        )

    def with_collision_exclusions(self, plan):
        """Return a separately compiled template with an exact explicit rule.

        Unsupported shape/body combinations fail closed. No default adjacency
        policy is inferred; an empty plan returns this unchanged template.
        """
        from .native_exclusions import exclude_template_pairs

        return exclude_template_pairs(self, plan)

    def bind(self, nworld):
        if isinstance(nworld, bool) or not isinstance(nworld, Integral) or nworld <= 0:
            raise ValueError("homogeneous binding requires positive integer nworld")
        return HomogeneousNativeBinding(self, int(nworld))

    def bind_body_sites(self, sites):
        """Resolve named BodySite values against this compilation only."""
        from .native_sites import bind_body_sites

        return bind_body_sites(self, sites)

    def compile_ray_scene(self):
        """Convert native geometry to named body-local public ray surfaces once."""
        from .native_rays import compile_ray_scene

        return compile_ray_scene(self)

    def compile_visual_scene(self):
        """Return the owned cold display snapshot; never reparse source files."""
        if self.visual_scene is None:
            if self.visual_diagnostics:
                raise NotImplementedError("; ".join(self.visual_diagnostics))
            raise RuntimeError("compile_native_template requires capture_visuals=True")
        return self.visual_scene


@dataclass(frozen=True)
class HomogeneousNativeBinding:
    """Reuse one compiled template; world-local native indices, no env replicas.

    Runtime data remains [nworld, width]. Public identity is (world, name), not
    a flattened fictitious body index. bind performs no file IO/compilation.
    """

    template: NativeTemplate
    nworld: int

    def __post_init__(self):
        if not isinstance(self.template, NativeTemplate):
            raise TypeError("homogeneous binding requires NativeTemplate")
        if type(self.nworld) is not int or self.nworld <= 0:
            raise ValueError("homogeneous binding requires positive integer nworld")

    def create_runtime(self, *, payload, **options):
        from ..backends.physolve_runtime import PhySolveRuntime

        if self.template.contract_version != 1:
            raise ValueError("unsupported native integration contract")
        if type(self.template.complete_scene_contract_version) is not int or self.template.complete_scene_contract_version not in (0, 1):
            raise ValueError("unsupported complete scene contract")
        if self.template.follower_drive_contract_version != 1:
            raise ValueError("unsupported follower drive contract")
        return PhySolveRuntime(
            self.template.compilation,
            payload=payload,
            initial_state=self.template.initial_state,
            scalar_controls=self.template.scalar_controls,
            follower_drives=self.template.follower_drives,
            nworld=self.nworld,
            complete_scene=self.template.complete_scene_contract_version == 1,
            **options,
        )


def compile_native_template(
    assets,
    *,
    timestep,
    gravity,
    static_shapes=(),
    rigid_shapes=(),
    integrator="euler",
    capture_visuals=False,
    capture_frames=False,
):
    """Compile a same-world assembly once; retain admitted asset-local semantics.

    Asset timestep/gravity are replaced by explicit assembly options. A source
    default Euler integrator can adopt the assembly selection; a nondefault
    source integrator must match it. Other
    non-default global options are rejected until represented by this profile.
    Source actuator/equality/contact/resource references are attached together.
    Native MuJoCo collision rules apply; explicit pair policies remain separate.
    """
    import mujoco as mj

    if type(capture_visuals) is not bool:
        raise TypeError("capture_visuals must be bool")
    if type(capture_frames) is not bool:
        raise TypeError("capture_frames must be bool")

    assets, static_shapes, rigid_shapes = tuple(assets), tuple(static_shapes), tuple(rigid_shapes)
    if any(not isinstance(a, NativeAssetInstance) for a in assets):
        raise TypeError("native template requires asset instances")
    if any(type(s) is not NativeStaticShape for s in static_shapes):
        raise TypeError("static_shapes requires NativeStaticShape")
    if any(not isinstance(s, NativeRigidShape) for s in rigid_shapes):
        raise TypeError("rigid_shapes requires NativeRigidShape")
    names = [a.name for a in (*assets, *static_shapes, *rigid_shapes)]
    if len(set(names)) != len(names):
        raise ValueError("duplicate assembly instance name")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be positive finite seconds")
    if not isinstance(integrator, str) or integrator.lower() not in {"euler", "implicitfast"}:
        raise ValueError("native template integrator requires euler or implicitfast")
    native_integrator = {
        "euler": mj.mjtIntegrator.mjINT_EULER,
        "implicitfast": mj.mjtIntegrator.mjINT_IMPLICITFAST,
    }[integrator.lower()]
    gravity = _vector(gravity, 3, "gravity")
    spec = mj.MjSpec()
    spec.compiler.fusestatic = False
    spec.option.timestep, spec.option.gravity = timestep, gravity
    spec.option.integrator = native_integrator
    spec.option.solver = mj.mjtSolver.mjSOL_NEWTON
    spec.option.jacobian = mj.mjtJacobian.mjJAC_DENSE
    pending, sources, target_maps, diagnostics = [], [], {}, []
    source_bodies = []
    frame_sources = []
    captured_visuals = {}
    visual_diagnostics = []
    for asset in assets:
        child, source_root, source_digest = load_asset_source(asset)
        if capture_frames:
            frame_sources.append((asset.name, source_root))
        if source_root.tag == "robot":
            source_bodies.append((asset.name, "urdf", source_root.get("name", ""),
                                  tuple(link.attrib["name"] for link in source_root.findall("link"))))
        if capture_visuals:
            from .native_visuals import capture_urdf_visuals

            try:
                visual = capture_urdf_visuals(asset, source_root)
                if visual is not None:
                    captured_visuals[asset.name] = visual
            except (ValueError, NotImplementedError, FileNotFoundError) as error:
                visual_diagnostics.append(f"{asset.name}: visual export unavailable: {error}")
        default = mj.MjSpec().option
        for name in dir(default):
            if name.startswith("_") or name in {"gravity", "timestep"}:
                continue
            value, expected = getattr(child.option, name), getattr(default, name)
            if name == "integrator" and value == native_integrator:
                continue
            if isinstance(expected, (int, float, np.ndarray)) and not np.array_equal(
                value, expected
            ):
                raise NotImplementedError(
                    f"{asset.name}: source option {name} needs an assembly profile"
                )
        child.compiler.fusestatic = False
        # Apply while source body ownership still exists, before namespaces and
        # attachment. Do not broadcast to descendants or another asset instance.
        for name, sliding in asset.body_friction:
            body = child.body(name)
            if body is None or name == "world":
                raise ValueError(f"{asset.name}: unknown source body for friction: {name}")
            geoms = tuple(body.geoms)
            if not geoms:
                raise ValueError(f"{asset.name}: body friction has no shapes: {name}")
            for geom in geoms:
                geom.friction[0] = sliding
        # MjSpec.to_zip only stores VFS assets, not files loaded implicitly from
        # disk. Own them before attachment (which namespaces their paths too),
        # so a later explicit rule compile never rereads the source directory.
        resources = own_spec_resources(child)
        # Namespace unnamed elements deterministically before native attachment.
        for kind in ("bodies", "joints", "geoms"):
            for i, element in enumerate(getattr(child, kind)):
                if not element.name:
                    element.name = f"__{kind}_{i}"
        if source_root.tag == "mujoco":
            source_bodies.append((asset.name, "mjcf", source_root.get("model", asset.name),
                                  tuple(body.name for body in child.bodies if body.name != "world")))
        if not asset.self_collision:
            if child.pairs:
                raise NotImplementedError(
                    "self_collision=False conflicts with native explicit pairs"
                )
            bodies = list(child.bodies)[1:]
            for i, a in enumerate(bodies):
                for b in bodies[i + 1 :]:
                    child.add_exclude(bodyname1=a.name, bodyname2=b.name)
        rows, targets = (), {}
        if asset.drive_mode == "scalar_pd":
            from ._native_drives import prepare_scalar_drives

            rows, targets, notes = prepare_scalar_drives(
                mj, child, asset, NativeJointInput, source_root
            )
            diagnostics.extend(notes)
        if asset.floating:
            if source_root.tag != "robot":
                raise NotImplementedError("floating assembly requires URDF")
            roots = [
                body
                for body in child.bodies
                if body is not child.worldbody and body.parent == child.worldbody
            ]
            if len(roots) != 1 or list(roots[0].joints):
                raise NotImplementedError("floating URDF requires one unconstrained retained root")
            roots[0].add_freejoint(name="__assembly_free")
        prefix = asset.name + "::"
        q = asset.orientation_xyzw
        frame = spec.worldbody.add_frame(pos=asset.position, quat=(q[3], *q[:3]))
        spec.attach(child, prefix=prefix, frame=frame)
        bind_attached_resources(spec, resources, prefix)
        pending.extend((prefix, row) for row in rows)
        target_maps.update(
            {
                prefix + name: (prefix + leader, ratio, offset)
                for name, (leader, ratio, offset) in targets.items()
            }
        )
        sources.append((asset.name, asset.path, source_digest))
    for shape in static_shapes:
        q = shape.orientation_xyzw
        spec.worldbody.add_geom(
            name=shape.name,
            type=getattr(mj.mjtGeom, f"mjGEOM_{shape.kind.upper()}"),
            size=(*shape.size, *(0.0,) * (3 - len(shape.size))),
            pos=shape.position,
            quat=(q[3], *q[:3]),
            friction=shape.friction,
            condim=shape.contact_dim,
        )
    for shape in rigid_shapes:
        q = shape.orientation_xyzw
        body = spec.worldbody.add_body(name=shape.name, pos=shape.position, quat=(q[3], *q[:3]))
        body.add_freejoint(name=shape.name + "::__assembly_free")
        physical = dict(density=shape.density) if shape.mass is None else dict(mass=shape.mass)
        body.add_geom(
            name=shape.name,
            type=getattr(mj.mjtGeom, f"mjGEOM_{shape.kind.upper()}"),
            size=(*shape.size, *(0.0,) * (3 - len(shape.size))),
            friction=shape.friction,
            condim=shape.contact_dim,
            **physical,
        )
    compilation = compile_mujoco_spec(spec, instance="assembly")
    model = compilation.fresh_native_model()
    frame_metadata = None
    if capture_frames:
        from .native_frames import compile_urdf_frames

        frame_metadata = compile_urdf_frames(model, frame_sources)
    identities = []
    for instance, source_format, model_name, links in source_bodies:
        rows = []
        for source_name in links:
            compiled_name = instance + "::" + source_name
            body = model.body(compiled_name)
            rows.append((source_name, compiled_name, int(body.id)))
        identities.append(NativeAssetIdentity(instance, source_format, model_name, tuple(rows)))
    if model.nsensor or model.nplugin or model.nflex or model.nuserdata:
        raise NotImplementedError("PhySolve template excludes native sensors/plugins/flex/userdata")
    data = mj.MjData(model)
    actuator_ids = {}
    scalar_controls = []
    joint_limits = []
    for prefix, row in pending:
        joint = model.joint(prefix + row.name)
        joint_limits.append(
            (
                prefix + row.name,
                float(joint.range[0]) if joint.limited else None,
                float(joint.range[1]) if joint.limited else None,
                row.velocity_limit,
            )
        )
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        if row.position is not None:
            data.qpos[qa] = row.position
        data.qvel[va] = 0.0 if row.velocity is None else row.velocity
        data.qfrc_applied[va] = row.applied_force
        if target_maps[prefix + row.name][0] != prefix + row.name:
            continue
        target = data.qpos[qa] if row.target_position is None else row.target_position
        actuator = model.actuator(prefix + "__drive_" + row.name)
        target_velocity = 0.0 if row.target_velocity is None else row.target_velocity
        data.ctrl[actuator.id] = row.kp * target + row.kd * target_velocity
        actuator_ids[joint.id] = actuator.id
        scalar_controls.append(
            (prefix + row.name, actuator.id, row.kp, row.kd, float(target), target_velocity)
        )
    control_ids = {row[0]: i for i, row in enumerate(scalar_controls)}
    follower_drives = []
    actuator_names = {model.actuator(i).name: i for i in range(model.nu)}
    for prefix, row in pending:
        name = prefix + row.name
        leader, ratio, offset = target_maps[name]
        actuator = actuator_names.get(prefix + "__drive_" + row.name)
        if name == leader or actuator is None:
            continue
        index = control_ids[leader]
        q, v = scalar_controls[index][4:6]
        data.ctrl[actuator] = row.kp * (ratio * q + offset) + row.kd * ratio * v
        follower_drives.append((name, actuator, index, ratio, offset, row.kp, row.kd))
        actuator_ids[model.joint(name).id] = actuator
    joints = tuple(
        (
            model.joint(i).name,
            i,
            int(model.jnt_qposadr[i]),
            int(model.jnt_dofadr[i]),
            actuator_ids.get(i),
        )
        for i in range(model.njnt)
    )
    for i in range(model.njnt):
        if model.jnt_limited[i] and model.jnt_type[i] in (
            mj.mjtJoint.mjJNT_HINGE,
            mj.mjtJoint.mjJNT_SLIDE,
        ):
            q = float(data.qpos[model.jnt_qposadr[i]])
            lo, hi = model.jnt_range[i]
            if not lo <= q <= hi:
                diagnostics.append(
                    f"initial_position_outside_limit: {model.joint(i).name}: {q} outside [{lo}, {hi}]; unchanged"
                )
    target_bindings = tuple(
        (name, control_ids[root], ratio, offset)
        for name, (root, ratio, offset) in target_maps.items()
    )
    for name, index, ratio, offset in target_bindings:
        with np.errstate(over="ignore", invalid="ignore"):
            q = np.float32(scalar_controls[index][4]) * np.float32(ratio) + np.float32(offset)
            v = np.float32(scalar_controls[index][5]) * np.float32(ratio)
        if not np.isfinite((q, v)).all():
            raise ValueError(f"{name}: initial derived target must be finite in FP32")
    visual_scene = None
    if capture_visuals and not visual_diagnostics:
        from .native_visuals import assemble_visual_scene

        try:
            visual_scene = assemble_visual_scene(model, captured_visuals, assets)
        except (ValueError, NotImplementedError) as error:
            visual_diagnostics.append(f"visual export unavailable: {error}")
    return NativeTemplate(
        compilation,
        NativeInitialState.from_data(compilation, data),
        joints,
        tuple((model.body(i).name, i) for i in range(1, model.nbody)),
        tuple((model.geom(i).name, i) for i in range(model.ngeom)),
        tuple(sources),
        asset_identities=tuple(identities),
        frame_metadata=frame_metadata,
        scalar_controls=tuple(scalar_controls),
        follower_drives=tuple(follower_drives),
        target_bindings=target_bindings,
        import_diagnostics=tuple(diagnostics),
        joint_limits=tuple(joint_limits),
        joint_dimensions=tuple(
            (
                int(
                    (model.jnt_qposadr[i + 1] if i + 1 < model.njnt else model.nq)
                    - model.jnt_qposadr[i]
                ),
                int(
                    (model.jnt_dofadr[i + 1] if i + 1 < model.njnt else model.nv)
                    - model.jnt_dofadr[i]
                ),
            )
            for i in range(model.njnt)
        ),
        _source_spec=compilation.mj_spec.copy(),
        visual_scene=visual_scene,
        visual_diagnostics=tuple(visual_diagnostics),
    )
