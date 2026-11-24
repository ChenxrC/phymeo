"""Pinned Galaxea R1 visuals and forward kinematics in the official URDF frame."""
from dataclasses import dataclass
from hashlib import sha1, sha256
import json
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import Instance, NodePose, Pose, VisualScene, compose_pose


ORIGIN = Path(__file__).with_name('origin.json')
STL_RECORD = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])


def _axis_rotation(axis, angle):
    s = math.sin(angle / 2)
    return Pose(orientation_xyzw=tuple(v*s for v in axis) + (math.cos(angle/2),))


def _origin(element):
    if element is None:
        return Pose()
    xyz = tuple(map(float, element.get('xyz', '0 0 0').split()))
    roll, pitch, yaw = map(float, element.get('rpy', '0 0 0').split())
    rotation = compose_pose(_axis_rotation((0, 0, 1), yaw), _axis_rotation((0, 1, 0), pitch))
    return Pose(xyz, compose_pose(rotation, _axis_rotation((1, 0, 0), roll)).orientation_xyzw)


@dataclass(frozen=True)
class Joint:
    kind: str
    parent: str
    child: str
    origin: Pose
    axis: tuple
    lower: float
    upper: float


def _joints(robot):
    joints = {}
    for element in robot.findall('joint'):
        kind = element.attrib['type']
        if kind not in ('continuous', 'revolute', 'prismatic', 'fixed'):
            raise ValueError('unsupported joint type: ' + kind)
        axis = tuple(map(float, element.find('axis').get('xyz').split()))
        lower, upper = -math.inf, math.inf
        if kind != 'fixed':
            axis = tuple(x/math.hypot(*axis) for x in axis)
        if kind in ('prismatic', 'revolute'):
            limits = element.find('limit')
            lower, upper = float(limits.attrib['lower']), float(limits.attrib['upper'])
        joints[element.attrib['name']] = Joint(kind, element.find('parent').attrib['link'],
            element.find('child').attrib['link'], _origin(element.find('origin')), axis, lower, upper)
    return joints


@dataclass
class Model:
    scene: VisualScene
    links: tuple
    joints: dict
    floor_offset_m: float
    wheel_radius_m: float
    diagnostics: dict

    def pose_nodes(self, joints, base_pose=Pose()):
        unknown = set(joints) - self.joints.keys()
        if unknown:
            raise ValueError('unknown R1 joints: ' + ', '.join(sorted(unknown)))
        world = {'base_link': base_pose}
        # The canonical URDF declares parents before their children.
        for name, joint in self.joints.items():
            value = float(joints.get(name, 0.))
            if (not math.isfinite(value) or not joint.lower-1e-8 <= value <= joint.upper+1e-8
                    or (joint.kind == 'fixed' and value != 0)):
                raise ValueError(f'invalid joint coordinate {name}={value}')
            local = joint.origin
            if value:
                delta = (Pose(tuple(value*v for v in joint.axis)) if joint.kind == 'prismatic'
                         else _axis_rotation(joint.axis, value))
                local = compose_pose(local, delta)
            world[joint.child] = compose_pose(world[joint.parent], local)
        return tuple(NodePose(name, world[name]) for name in self.links)


def _read_pinned(package, relative, blobs):
    if relative not in blobs:
        raise ValueError('unregistered Galaxea source: ' + relative)
    data = (package / relative).read_bytes()
    if sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != blobs[relative]:
        raise ValueError('Galaxea source differs from the pinned Git blob: ' + relative)
    return data


def _nondegenerate(p, t):
    corners = p[t].astype(float)
    cross = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    return np.any(cross != 0., axis=1)


def _convert(raw, folder, stem):
    import fast_simplification
    from _panda_scene import write_mesh
    from preview_stable_lighting.shell_normals import _smoothed

    count, = struct.unpack_from('<I', raw, 80)
    if not count or len(raw) != 84 + count * STL_RECORD.itemsize:
        raise ValueError('invalid official binary STL')
    triangles = np.frombuffer(raw, offset=84, dtype=STL_RECORD)['vertices']
    if not np.isfinite(triangles).all():
        raise ValueError('non-finite R1 positions')
    p, ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    t = ids.reshape(-1, 3).astype(np.int32)
    bounds = np.array([p.min(axis=0), p.max(axis=0)], dtype=float)
    radius_xz = float(np.linalg.norm(p[:, (0, 2)].astype(float), axis=1).max())
    t = t[_nondegenerate(p, t)]
    usable = len(t)
    if usable > 35_000:
        p, t = fast_simplification.simplify(p.astype(float), t, target_count=int(.58*usable),
                                            preserve_border=True)
        p = p.astype('<f4')
        t = t[_nondegenerate(p, t)]
    if len(t) == 0:
        raise ValueError('empty Galaxea visual')
    corners = p[t].astype(float)
    face = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    face /= np.linalg.norm(face, axis=1)[:, None]
    # STL provides facet normals. Reconstruct smooth CAD shading only across
    # shallow manifold edges, preserving splits, open rims and sharp corners.
    soup = corners.reshape(-1, 3)
    normals, normal_stats = _smoothed(soup, np.repeat(face, 3, axis=0), np.arange(len(soup)).reshape(-1, 3))
    entries = []
    for start in range(0, len(t), 21_000):
        stop = min(start+21_000, len(t))
        block = np.column_stack((soup[3*start:3*stop], normals[3*start:3*stop])).astype('<f4')
        unique, inverse = np.unique(block, axis=0, return_inverse=True)
        name = f'{stem}-{len(entries):02d}'
        write_mesh(folder, name, unique[:, :3], unique[:, 3:], inverse.reshape(-1, 3))
        path = folder / (name+'.gltf')
        entries.append(dict(id=name, file=path.name, vertices=len(unique), triangles=stop-start,
                            sha256=sha256(path.read_bytes()).hexdigest()))
    final = np.array([soup.min(axis=0), soup.max(axis=0)])
    return dict(source_triangles=count, source_nondegenerate=usable, triangles=len(t), parts=entries,
                source_bounds_m=bounds.tolist(), derived_bounds_m=final.tolist(),
                max_bounds_change_m=float(np.abs(bounds-final).max()), source_radius_xz_m=radius_xz,
                smooth_normal_vertices=normal_stats['changed_vertices'])


def _cached(raw, blob, cache, generator):
    folder = cache / (blob[:12]+'-'+generator[:12])
    receipt = folder / 'geometry.json'
    if receipt.is_file():
        row = json.loads(receipt.read_bytes())
        if all(sha256((folder / part['file']).read_bytes()).hexdigest() == part['sha256'] for part in row['parts']):
            return folder, row
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as tmp:
        stage = Path(tmp)
        row = _convert(raw, stage, 'r1-'+blob[:12])
        for part in row['parts']:
            os.replace(stage / part['file'], folder / part['file'])
        (stage / 'geometry.json').write_text(json.dumps(row, allow_nan=False), encoding='utf-8')
        os.replace(stage / 'geometry.json', receipt)
    return folder, row


def load_model(asset_dir, cache_dir):
    """Load real visual links. Base height and wheel radius use original STL metres."""
    from preview_stable_lighting import shell_normals
    package = Path(asset_dir).resolve()
    if (package/'r1_description').is_dir():
        package = package/'r1_description'
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = json.loads(ORIGIN.read_bytes())
    blobs = source['blobs']
    raw = _read_pinned(package, source['urdf'], blobs)
    robot = ET.fromstring(raw)
    joints = _joints(robot)
    links = tuple(link.attrib['name'] for link in robot.findall('link'))
    generator = sha256(Path(__file__).read_bytes() + Path(shell_normals.__file__).read_bytes()).hexdigest()
    geometries, meshes, materials, instances, uses = {}, [], [], [], {}
    wheel_min, wheel_radius = [], []
    for link in robot.findall('link'):
        name = link.attrib['name']
        visual, = link.findall('visual')
        src_mesh = visual.find('geometry/mesh')
        uri = src_mesh.attrib['filename']
        if not uri.startswith('package://r1_description/') or src_mesh.get('scale', '1 1 1') != '1 1 1':
            raise ValueError('unexpected mesh source or scale: ' + uri)
        path = uri.removeprefix('package://r1_description/')
        data = _read_pinned(package, path, blobs)
        blob = blobs[path]
        if blob not in geometries:
            directory, row = _cached(data, blob, cache, generator)
            meshes.extend(Mesh(part['id'], (directory/part['file']).as_uri()) for part in row['parts'])
            geometries[blob] = row
        uses[name] = blob
        material = Material('r1-material-'+name, tuple(map(float, visual.find('material/color').attrib['rgba'].split()))[:3], 0., .65)
        materials.append(material)
        local = _origin(visual.find('origin'))
        for part in geometries[blob]['parts']:
            instances.append(Instance(100 + len(instances), part['id'], material.id, name, local))
        if name.startswith('wheel_motor_link'):
            row = geometries[blob]
            wheel_min.append(row['source_bounds_m'][0][2] + joints['wheel_motor_joint'+name[-1]].origin.position_m[2]
                             + joints['steer_motor_joint'+name[-1]].origin.position_m[2])
            wheel_radius.append(row['source_radius_xz_m'])
    scene = VisualScene('galaxea-r1-original-'+source['revision'][:12], tuple(meshes), tuple(materials), tuple(instances))
    total = sum(row['triangles'] for row in geometries.values())
    vertices = sum(part['vertices'] for row in geometries.values() for part in row['parts'])
    if total > 520_000 or vertices > 1_040_000:
        raise ValueError(f'Galaxea geometry exceeds resident budget: {total} faces, {vertices} vertices')
    report = dict(revision=source['revision'], package=source['package'], urdf_sha256=sha256(raw).hexdigest(),
                  unique_triangles=total, unique_vertices=vertices, links=uses, geometries=geometries,
                  max_bounds_change_m=max(r['max_bounds_change_m'] for r in geometries.values()),
                  metallic=0., roughness=.65, source_rgba='official per-visual URDF',
                  units='metres', reduction='quadric error decimation at 58% for meshes over 35000 nondegenerate triangles')
    return Model(scene, links, joints, -min(wheel_min), max(wheel_radius), report)
