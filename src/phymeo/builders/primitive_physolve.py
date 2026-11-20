"""Online PHYMEO lowering for centred free primitives and static surfaces."""

from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..dynamics.rotation import quaternion_matrix
from ..geometry.visual import VisualInstance, VisualScene
from .collision_sources import CollisionSourceMapping, SourceGeometry, _digest
from .native_assembly import NativeRigidShape, NativeStaticShape, NativeTemplate
from .native_frames import NativeFrameMetadata

_IDENTITY = (0., 0., 0., 1.)


def _wxyz(q):
    return q[3], q[0], q[1], q[2]


def _cylinder_mass_properties(radius, half_height):
    """Volume and principal moments per unit mass for a solid Z-axis cylinder."""
    volume = 2 * np.pi * radius ** 2 * half_height
    transverse = radius ** 2 / 4 + half_height ** 2 / 3
    return volume, np.array((transverse, transverse, radius ** 2 / 2))


def _mass_and_inertia(shape):
    """Resolve the authored mass/density and principal moments in SI."""
    if shape.kind == "sphere":
        radius = shape.size[0]
        volume = 4 / 3 * np.pi * radius ** 3
        per_mass = np.full(3, .4 * radius ** 2)
    elif shape.kind == "box":
        half = np.asarray(shape.size)
        volume = 8 * np.prod(half)
        per_mass = (np.roll(half, 1) ** 2 + np.roll(half, -1) ** 2) / 3
    elif shape.kind == "cylinder":
        volume, per_mass = _cylinder_mass_properties(*shape.size)
    elif shape.kind == "ellipsoid":
        radii = np.asarray(shape.size)
        volume = 4 * np.pi / 3 * np.prod(radii)
        per_mass = (np.roll(radii, 1) ** 2 + np.roll(radii, -1) ** 2) / 5
    elif shape.kind == "capsule":
        radius, half_height = shape.size
        barrel, barrel_moments = _cylinder_mass_properties(radius, half_height)
        caps = 4 * np.pi / 3 * radius ** 3
        # Each hemisphere COM is 3r/8 from its equatorial plane.
        polar = .4 * radius ** 2
        transverse = polar + half_height ** 2 + .75 * half_height * radius
        volume = barrel + caps
        per_mass = (barrel * barrel_moments + caps * np.array((transverse, transverse, polar))) / volume
    else:
        raise NotImplementedError(f"primitive PhySolve does not support dynamic {shape.kind}")
    mass = shape.mass if shape.mass is not None else shape.density * volume
    inertia = per_mass * mass
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        exported = np.asarray([mass, *inertia, 1 / mass, *(1 / inertia)], np.float32)
    if not np.isfinite(exported).all() or not (exported > 0.).all():
        raise ValueError("primitive mass, inertia and inverse values require finite positive FP32")
    return float(mass), tuple(map(float, inertia))


@dataclass(frozen=True)
class PrimitiveCompilation:
    """Immutable solvers inputs derived from Stage, including world-local identities."""

    bodies: tuple[NativeRigidShape, ...]
    statics: tuple[NativeStaticShape, ...]
    mass: tuple
    principal_inertia: tuple
    timestep: float
    gravity: tuple
    integrator: str
    collision_source: CollisionSourceMapping

    @property
    def free_bindings(self):
        return tuple((body.name, i + 1, i * 7, i * 6, (0., 0., 0.))
                     for i, body in enumerate(self.bodies))

    def upload_model(self, frontend):
        """Send compiled mass properties and shape options to the native frontend."""
        if (getattr(frontend, "CYLINDER_INPUT_CONTRACT", None) != 1
                and any(shape.kind == "cylinder" for shape in (*self.bodies, *self.statics))):
            raise ValueError("PhySolve cylinder input contract v1 is required")
        require_rounded_primitives(frontend, (shape.kind for shape in (*self.bodies, *self.statics)))

        def geometry(shape):
            x, y, z, w = shape.orientation_xyzw
            return frontend.Geometry(shape.kind, shape.size, shape.position, (w, x, y, z),
                                     shape.friction, shape.contact_dim)

        return frontend.put_primitive_model(
            tuple(frontend.FreeBody(geometry(shape), mass, inertia)
                  for shape, mass, inertia in zip(self.bodies, self.mass, self.principal_inertia)),
            tuple(map(geometry, self.statics)), timestep=self.timestep,
            gravity=self.gravity, integrator=self.integrator,
        )

    def create_initial_bank(self, frontend, model, **capacity):
        """Seed authored pose plus static frames; the Host computes derived observations."""
        qpos = []
        for body in self.bodies:
            x, y, z, w = body.orientation_xyzw
            qpos.extend((*body.position, w, x, y, z))
        static_q = np.asarray([shape.orientation_xyzw for shape in self.statics]).reshape(-1, 4)
        return frontend.make_primitive_data(
            model, qpos, tuple(shape.position for shape in self.statics),
            quaternion_matrix(static_q), **capacity,
        )


def require_rounded_primitives(frontend, kinds):
    if (getattr(frontend, "ROUNDED_PRIMITIVE_INPUT_CONTRACT", None) != 1
            and any(kind in ("capsule", "ellipsoid") for kind in kinds)):
        raise ValueError("PhySolve capsule and ellipsoid input contract v1 is required")


def compile_primitive_template(*, timestep, gravity, static_shapes, rigid_shapes,
                               integrator, capture_visuals=False, capture_frames=False, assets=()):
    """Compile an independent primitive scene into the existing homogeneous template ABI."""
    statics, bodies = tuple(static_shapes), tuple(rigid_shapes)
    if integrator not in ("euler", "implicitfast"):
        raise NotImplementedError("autonomous PhySolve requires Euler or implicitfast")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("primitive timestep requires a positive finite value")
    if len(gravity) != 3 or not np.isfinite(gravity).all():
        raise ValueError("primitive gravity requires three finite components")
    if assets:
        from .scalar_physolve import compile_scalar_template

        if len(assets) != 1:
            raise NotImplementedError("autonomous PhySolve requires one robot")
        return compile_scalar_template(
            assets[0], timestep=timestep, gravity=gravity, statics=statics, integrator=integrator,
            capture_visuals=capture_visuals, capture_frames=capture_frames, free_objects=bodies,
        )
    if not bodies:
        raise NotImplementedError("primitive PhySolve requires at least one dynamic shape")
    names = [shape.name for shape in (*statics, *bodies)]
    if len(set(names)) != len(names):
        raise ValueError("duplicate primitive identity")

    physical = tuple(map(_mass_and_inertia, bodies))
    source_rows = tuple(
        SourceGeometry(CollisionKey(0, 0, i), i, None, shape.name, 0)
        for i, shape in enumerate(statics)
    ) + tuple(
        SourceGeometry(CollisionKey(0, i + 1, 0), i + len(statics), shape.name, shape.name, i + 1)
        for i, shape in enumerate(bodies)
    )
    identity = _digest(("phymeo-primitive-v1", tuple(repr(shape) for shape in (*statics, *bodies))))
    sources = CollisionSourceMapping(identity, "stage", 0, source_rows, ())
    compilation = PrimitiveCompilation(
        bodies, statics, tuple(row[0] for row in physical), tuple(row[1] for row in physical),
        float(timestep), tuple(map(float, gravity)), integrator, sources,
    )
    frames = NativeFrameMetadata(tuple((body.name, (0., 0., 0.)) for body in bodies), ())
    visual = None
    if capture_visuals:
        entries = [
            VisualInstance(shape.name, None, shape.kind, shape.size, shape.position,
                           shape.orientation_xyzw, (.5, .5, .5, 1.))
            for shape in statics
        ]
        entries.extend(
            VisualInstance(body.name, body.name, body.kind, body.size, (0., 0., 0.),
                           _IDENTITY, (.5, .5, .5, 1.))
            for body in bodies
        )
        visual = VisualScene(tuple(entries))
    return NativeTemplate(
        compilation=compilation, initial_state=None,
        joint_bindings=tuple((body.name + "::free", i, i * 7, i * 6, None)
                             for i, body in enumerate(bodies)),
        joint_dimensions=((7, 6),) * len(bodies),
        body_bindings=tuple((body.name, i + 1) for i, body in enumerate(bodies)),
        shape_bindings=tuple((name, i) for i, name in enumerate(names)),
        asset_sources=(), qualification=f"native primitive PhySolve {integrator}",
        frame_metadata=frames if capture_frames else None, visual_scene=visual,
    )
