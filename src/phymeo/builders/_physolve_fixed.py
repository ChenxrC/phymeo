"""URDF lowering that keeps every authored body frame, including welded links."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..dynamics.robot_dynamics import _mass_from_terms
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
from ._mimic_xml import extract_mimic_relations
from ._physolve_stl import compile_stl
from ._robot_xml import _attributes, _child, _children, _number, _numbers
from ._urdf_fixed import _mass_component, _principal
from .primitive_physolve import _cylinder_mass_properties
from .urdf import _joint_data, _origin, _tree
from .urdf_import_options import resolve_mesh


@dataclass(frozen=True)
class FixedBody:
    name: str
    parent: int
    dof: int
    position: tuple
    rotation: tuple
    mass: float
    com: tuple
    inertia: tuple
    inertia_frame: tuple


@dataclass(frozen=True)
class FixedArticulation:
    name: str
    bodies: tuple
    joint_names: tuple
    joint_types: tuple
    joint_axis: tuple
    parents: tuple
    lower: tuple
    upper: tuple
    effort_limit: tuple
    velocity_limit: tuple
    damping: tuple
    friction: tuple
    joint_bodies: tuple
    relations: tuple
    geometries: tuple
    meshes: tuple
    diagnostics: tuple
    floating: bool

    @property
    def dof_count(self):
        return len(self.joint_names)

    @property
    def moving(self):
        return tuple(self.bodies[body - 1] for body in self.joint_bodies)

    def reference(self, armatures):
        """Reference state at model zero in authored body and actuator coordinates."""
        root_width = 6 if self.floating else 0
        count = len(self.joint_names) + root_width
        positions = [np.zeros(3)]
        rotations = [np.eye(3)]
        chains = [()]
        axes, pivots = np.empty((len(armatures), 3)), np.empty((len(armatures), 3))
        masses, inertias = [], []
        linear = np.zeros((len(self.bodies), 3, count))
        angular = np.zeros_like(linear)
        for index, body in enumerate(self.bodies):
            parent = body.parent
            quat, inertia_quat = quaternion_matrix(
                np.asarray((body.rotation, body.inertia_frame))
            )
            position = positions[parent] + rotations[parent] @ body.position
            rotation = rotations[parent] @ quat
            positions.append(position)
            rotations.append(rotation)
            chain = chains[parent]
            if body.dof >= 0:
                axes[body.dof] = rotation @ self.joint_axis[body.dof]
                pivots[body.dof] = position
                chain = (*chain, body.dof)
            chains.append(chain)
            center = position + rotation @ body.com
            if self.floating:
                root_rotation = rotations[1]
                linear[index, :, :3] = np.eye(3)
                linear[index, :, 3:6] = np.cross(root_rotation.T, center - positions[1]).T
                angular[index, :, 3:6] = root_rotation
            basis = rotation @ inertia_quat
            masses.append(body.mass)
            inertias.append((basis * body.inertia) @ basis.T)
            for joint in chain:
                column = root_width + joint
                if self.joint_types[joint] == "prismatic":
                    linear[index, :, column] = axes[joint]
                else:
                    angular[index, :, column] = axes[joint]
                    linear[index, :, column] = np.cross(axes[joint], center - pivots[joint])
        mass = _mass_from_terms(masses, linear, angular, inertias)
        diagonal = np.arange(root_width, count)
        mass[diagonal, diagonal] += armatures
        inverse = np.linalg.solve(mass, np.eye(count))
        weights = np.column_stack([
            np.sum((jac @ inverse) * jac, axis=(1, 2)) / 3 for jac in (linear, angular)
        ])
        for row in weights:
            if row.min() < 1e-15 < row.max():
                row[:] = row.max()
        dof_weights = inverse.diagonal().copy()
        if self.floating:
            dof_weights[:3] = dof_weights[:3].mean()
            dof_weights[3:6] = dof_weights[3:6].mean()
        actuator_acc = np.linalg.norm(inverse[:, diagonal], axis=0)
        mean = float(np.trace(mass) / count)
        if self.floating:
            with np.errstate(over="ignore", under="ignore"):
                positive = np.asarray([mean, *dof_weights, *actuator_acc], dtype=np.float32)
                finite_weights = weights.astype(np.float32)
            if not (np.isfinite(positive).all() and (positive > 0.).all()
                    and np.isfinite(finite_weights).all()):
                raise ValueError("floating articulated reference weights require representable FP32")
        return mean, weights, dof_weights, actuator_acc


def _cylinder_size(shape, label):
    size = (_number(shape, "radius", label), _number(shape, "length", label) / 2)
    if min(size) <= 0:
        raise ValueError(f"{label}: cylinder radius and length must be positive")
    return size


def _inferred_inertia(link, label):
    """Infer un-authored inertia from the physical URDF primitives in SI."""
    components = []
    for geom in link.findall("collision"):
        center, rotation = _origin(geom, label)
        rotation = quaternion_matrix(np.asarray([rotation]))[0]
        node = _child(geom, "geometry", label)
        if len(node) != 1:
            raise ValueError(f"{label}: collision needs one geometry")
        shape = node[0]
        if shape.tag == "box":
            size = _numbers(shape, "size", 3, label)
            if np.any(size <= 0):
                raise ValueError(f"{label}: collision box side lengths must be positive")
            mass = 1000. * float(np.prod(size))
            principal = mass / 12 * (np.roll(size, -1) ** 2 + np.roll(size, 1) ** 2)
        elif shape.tag == "sphere":
            radius = _number(shape, "radius", label)
            if radius <= 0:
                raise ValueError(f"{label}: collision sphere radius must be positive")
            mass = (4000. * np.pi / 3) * radius ** 3
            principal = np.full(3, .4 * mass * radius ** 2)
        elif shape.tag == "cylinder":
            volume, per_mass = _cylinder_mass_properties(*_cylinder_size(shape, label))
            mass = 1000. * volume
            principal = mass * per_mass
        else:
            raise NotImplementedError(f"{label}: mesh inertia requires an authored inertial element")
        components.append((mass, center, (rotation * principal) @ rotation.T))
    return _principal(components, label) if components else None


def read_fixed_articulation(asset, xml, *, import_options):
    """Parse scalar actuators and fixed bodies from the original resource tree."""
    root, links, order, inbound = _tree(xml, depth_first=True)
    relations = extract_mimic_relations(order, inbound) if xml.findall("joint/mimic") else ()
    if asset.floating and relations:
        raise NotImplementedError("autonomous floating articulation does not support mimic")
    names = (root, *order)
    body_ids = {name: i + 1 for i, name in enumerate(names)}
    bodies, joints, parents, measures, joint_bodies, diagnostics = [], [], [], [], [], []
    nearest = [-1]
    zero, identity = (0., 0., 0.), (0., 0., 0., 1.)
    for name in names:
        pose, parent, dof = (asset.position, asset.orientation_xyzw), 0, -1
        if name in inbound:
            parent_name, joint_name, element = inbound[name]
            parent = body_ids[parent_name]
            if element.get("type") == "fixed":
                _children(element, {"origin", "parent", "child"}, joint_name)
                pose = _origin(element, joint_name)
            else:
                dynamics = _child(element, "dynamics", joint_name, required=False)
                damping = friction = None
                if dynamics is not None:
                    _children(dynamics, set(), joint_name)
                    extra = sorted(set(dynamics.attrib) - {"damping", "friction"})
                    if extra:
                        diagnostics.append(f"{asset.name}::{joint_name}: ignored URDF dynamics attributes {extra}")
                    if "damping" in dynamics.attrib:
                        damping = _number(dynamics, "damping", joint_name)
                    if "friction" in dynamics.attrib:
                        friction = _number(dynamics, "friction", joint_name)
                    element.remove(dynamics)
                safety = _child(element, "safety_controller", joint_name, required=False)
                if safety is not None:
                    element.remove(safety)  # Controller metadata, as on the reference route.
                # Application controller metadata is read by the consumer.
                application = _child(element, "phyzoic_case", joint_name, required=False)
                if application is not None:
                    _attributes(application, {"armature", "velocity_gain"}, joint_name)
                    _children(application, set(), joint_name)
                    element.remove(application)
                kind, position, rotation, axis, bounds = _joint_data(
                    element, joint_name, allow_continuous=True)
                dof = len(joints)
                joints.append((joint_name, kind, tuple(axis), *bounds))
                parents.append(nearest[parent])
                measures.append((damping, friction))
                joint_bodies.append(body_ids[name])
                pose = position, rotation
        nearest.append(dof if dof >= 0 else nearest[parent])
        link = links[name]
        component = _mass_component(link, name, np.zeros(3), np.eye(3), allow_massless=True)
        if component is None and link.find("inertial") is None:
            inferred = _inferred_inertia(link, name)
        else:
            inferred = None
        if inferred is not None:
            axes, (mass, com_principal, inertia) = inferred
            com, inertia_frame = axes @ com_principal, matrix_quaternion(axes)
        elif component is None:
            # A massless fixed frame uses its authored parent-relative origin
            # as the reference COM metadata, as in the installed legacy route.
            mass, com, inertia, inertia_frame = 0., zero if asset.floating else pose[0], zero, identity
        else:
            mass, com, tensor = component
            inertia, axes = np.linalg.eigh(tensor)
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            inertia_frame = matrix_quaternion(axes)
        bodies.append(FixedBody(f"{asset.name}::{name}", parent, dof, *map(tuple, pose),
                                mass, tuple(com), tuple(inertia), tuple(inertia_frame)))
    if asset.floating:
        root_body = bodies[0]
        if root_body.mass <= 0. or min(root_body.inertia) <= 0.:
            raise ValueError("autonomous floating articulation requires a massive root with positive inertia")
        with np.errstate(over="ignore", under="ignore"):
            inertial_values = np.asarray([(b.mass, *b.inertia) for b in bodies], dtype=np.float32)
        if (not np.isfinite(inertial_values).all() or np.any(inertial_values < 0.)
                or np.any(inertial_values[0] <= 0.)):
            raise ValueError("floating articulation body inertia must be finite in FP32")
    geometries, meshes = [], []
    friction = dict(asset.body_friction)
    collision_links = {link.get("name") for link in xml.findall("link") if link.findall("collision")}
    if friction.keys() - collision_links:
        raise ValueError("autonomous body friction requires a colliding source link")
    collision_names = Counter(
        element.get("name") for link in links.values() for element in link.findall("collision")
    )
    for name in names:
        for element in links[name].findall("collision"):
            authored_name = element.get("name")
            label = f"{asset.name}::{element.get('name') or f'__geoms_{len(geometries)}'}"
            if authored_name and collision_names[authored_name] > 1:
                label = f"{asset.name}::{name}::{authored_name}"
            origin, rotation = _origin(element, label)
            geometry = _child(element, "geometry", label)
            if len(geometry) != 1:
                raise ValueError(f"{label}: collision needs one geometry")
            shape = geometry[0]
            mesh_id = None
            if shape.tag == "mesh":
                path = resolve_mesh(asset, import_options, shape.attrib["filename"])
                if path.suffix.lower() != ".stl":
                    raise NotImplementedError(f"{label}: autonomous mesh requires STL")
                scale = _numbers(shape, "scale", 3, label) if "scale" in shape.attrib else (1., 1., 1.)
                mesh = compile_stl(label, path.read_bytes(), scale)
                size = tuple(map(float, np.ptp(mesh.visual.vertices, axis=0) / 2))
                mesh_id = len(meshes)
                meshes.append(mesh)
            elif shape.tag == "sphere":
                size = (_number(shape, "radius", label),)
            elif shape.tag == "box":
                size = tuple(_numbers(shape, "size", 3, label) / 2)
            elif shape.tag == "cylinder":
                size = _cylinder_size(shape, label)
            else:
                raise NotImplementedError(f"{label}: unsupported collision geometry {shape.tag}")
            if min(size) <= 0:
                raise ValueError(f"{label}: collision size must be positive")
            geometries.append((label, body_ids[name], shape.tag, size, tuple(origin), tuple(rotation),
                               (friction.get(name, 1.), .005, .0001), 3, mesh_id))
    cols = tuple(zip(*joints, strict=True))
    if not cols:
        raise ValueError("authored articulation requires at least one scalar joint")
    damping, joint_friction = zip(*measures, strict=True)
    jnames, kinds, axes, lower, upper, effort, velocity = cols
    return FixedArticulation(
        xml.get("name", ""), tuple(bodies), jnames, kinds, axes, tuple(parents),
        lower, upper, effort, velocity, damping, joint_friction, tuple(joint_bodies), relations,
        tuple(geometries), tuple(meshes), tuple(diagnostics), asset.floating,
    )
