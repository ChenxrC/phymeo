"""PhySolve-specific asset assembly; native-only, never a resident declaration.

Uses the existing native compiler and runtime contracts. No P2PE objects or
Newton compatibility model are introduced. All participants share one model.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral
from pathlib import Path

import numpy as np

from ..backends.physolve_initial import NativeInitialState
from .mujoco_compiler import compile_mujoco_spec
from .native_resources import bind_attached_resources, own_spec_resources


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

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or "::" in self.name:
            raise ValueError("joint input requires an asset-local name")
        for key, value in vars(self).items():
            if key == "name" or value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or abs(value) > np.finfo(np.float32).max
            ):
                raise ValueError(f"joint {self.name}.{key} requires a finite scalar")
            if (
                key in {"kp", "kd", "armature", "damping", "frictionloss", "effort_limit"}
                and value < 0
            ):
                raise ValueError(f"joint {self.name}.{key} must be nonnegative")


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

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or "::" in self.name:
            raise ValueError("instance requires a unique unscoped name")
        object.__setattr__(self, "path", str(Path(self.path).resolve()))
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
        NativeJointInput("validation", **defaults)
        object.__setattr__(self, "joint_defaults", tuple(defaults.items()))
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
        widths = {"sphere": 1, "box": 3, "capsule": 2, "cylinder": 2}
        if self.kind not in widths or not self.name or "::" in self.name:
            raise ValueError("native static requires a named finite primitive")
        object.__setattr__(self, "size", _vector(self.size, widths[self.kind], "size"))
        if min(self.size) <= 0:
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
        active = {1: 0, 3: 1, 4: 2, 6: 3}[self.contact_dim]
        if any(self.friction[active:]):
            raise ValueError("contact_dim would ignore nonzero friction")


@dataclass(frozen=True)
class NativeRigidShape(NativeStaticShape):
    """One native solid primitive with a genuine FREE root; explicit mass wins."""

    mass: float | None = None
    density: float = 1000.0

    def __post_init__(self):
        super().__post_init__()
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
class NativeTemplate:
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
    # (public joint, independent command index, multiplier, position offset).
    target_bindings: tuple = ()
    target_contract_version: int = 1
    import_diagnostics: tuple = ()
    # Widths aligned with joint_bindings; native FREE coordinates are not scalar joints.
    joint_dimensions: tuple = ()
    coordinate_contract_version: int = 1
    # Private cold compiler owner. XML serialization rounds native authoring
    # values; the public compilation.mj_spec is caller-mutable. Never expose or
    # edit this snapshot, only copy it for a newly requested rule compilation.
    _source_spec: object = field(default=None, repr=False, compare=False)

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
        return PhySolveRuntime(
            self.template.compilation,
            payload=payload,
            initial_state=self.template.initial_state,
            scalar_controls=self.template.scalar_controls,
            nworld=self.nworld,
            **options,
        )


def compile_native_template(
    assets, *, timestep, gravity, static_shapes=(), rigid_shapes=(), integrator="euler"
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
    for asset in assets:
        child = mj.MjSpec.from_file(asset.path)
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
        # MjSpec.to_zip only stores VFS assets, not files loaded implicitly from
        # disk. Own them before attachment (which namespaces their paths too),
        # so a later explicit rule compile never rereads the source directory.
        resources = own_spec_resources(child)
        # Namespace unnamed elements deterministically before native attachment.
        for kind in ("bodies", "joints", "geoms"):
            for i, element in enumerate(getattr(child, kind)):
                if not element.name:
                    element.name = f"__{kind}_{i}"
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

            rows, targets, notes = prepare_scalar_drives(mj, child, asset, NativeJointInput)
            diagnostics.extend(notes)
        if asset.floating:
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
        sources.append((asset.name, asset.path, sha256(Path(asset.path).read_bytes()).hexdigest()))
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
    if model.nsensor or model.nplugin or model.nflex or model.nuserdata:
        raise NotImplementedError("PhySolve template excludes native sensors/plugins/flex/userdata")
    data = mj.MjData(model)
    actuator_ids = {}
    scalar_controls = []
    for prefix, row in pending:
        joint = model.joint(prefix + row.name)
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
    control_ids = {row[0]: i for i, row in enumerate(scalar_controls)}
    for i in range(model.njnt):
        if model.jnt_limited[i] and model.jnt_type[i] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            q = float(data.qpos[model.jnt_qposadr[i]])
            lo, hi = model.jnt_range[i]
            if not lo <= q <= hi:
                diagnostics.append(f"initial_position_outside_limit: {model.joint(i).name}: {q} outside [{lo}, {hi}]; unchanged")
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
    return NativeTemplate(
        compilation,
        NativeInitialState.from_data(compilation, data),
        joints,
        tuple((model.body(i).name, i) for i in range(1, model.nbody)),
        tuple((model.geom(i).name, i) for i in range(model.ngeom)),
        tuple(sources),
        scalar_controls=tuple(scalar_controls),
        target_bindings=target_bindings,
        import_diagnostics=tuple(diagnostics),
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
    )
