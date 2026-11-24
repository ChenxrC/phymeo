"""Derive corner-angle-weighted shading normals for approved Panda CAD shells."""
import base64
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


# Main painted housings, plus the registered left-arm, wrist and hand shells.
# Small labels, fingers, connector inserts, and scene furnishings are excluded.
SHELL_IDS = frozenset('panda_visual_' + name for name in (
    'link0_03', 'link1_00', 'link2_00', 'link3_01', 'link3_03', 'link4_02',
    'link5_00', 'link6_00', 'link7_00', 'link7_07', 'hand_00', 'hand_01', 'hand_04'))
CREASE_DEGREES = 24.0
MANIFEST = Path(__file__).resolve().parents[2] / 'assets/panda/manifest.json'


def _smoothed(positions, normals, triangles, texcoords=None):
    """Join corner fans only across two oppositely wound, smooth common edges."""
    p, n = np.asarray(positions, dtype=float), np.asarray(normals, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int32)
    tri = p[triangles]
    edges = np.roll(tri, -1, axis=1) - tri
    cross = np.cross(edges[:, 0], -edges[:, 2])
    faces = cross / np.linalg.norm(cross, axis=1)[:, None]
    angles = np.arctan2(np.linalg.norm(np.cross(edges, -np.roll(edges, 1, axis=1)), axis=2),
                        np.sum(edges * -np.roll(edges, 1, axis=1), axis=2)).ravel()
    limit = math.cos(math.radians(CREASE_DEGREES))

    # The manufacturer splits vertices by their authored normals. Exact
    # coordinate equality identifies potential neighbours without moving them.
    _, vertex_pos = np.unique(p, axis=0, return_inverse=True)
    vertex_pos = vertex_pos.reshape(-1)
    vertex = triangles.ravel()
    pos = vertex_pos[vertex]
    corner = np.arange(vertex.size, dtype=np.int32)
    following = np.roll(corner.reshape(-1, 3), -1, axis=1).ravel()
    dest = pos[following]
    lo, hi = np.minimum(pos, dest), np.maximum(pos, dest)
    order = np.lexsort((hi, lo))
    starts = np.r_[0, np.flatnonzero((np.diff(lo[order]) != 0) | (np.diff(hi[order]) != 0)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    pairs = starts[counts == 2]
    a, b = order[pairs], order[pairs + 1]
    an, bn = following[a], following[b]
    smooth = ((pos[a] == dest[b]) & (pos[b] == dest[a]) &
              ((faces[a//3] * faces[b//3]).sum(axis=1) >= limit))
    # An authored hard shading seam can also occur on coplanar geometry.
    smooth &= ((n[vertex[a]] * n[vertex[bn]]).sum(axis=1) >= limit)
    smooth &= ((n[vertex[an]] * n[vertex[b]]).sum(axis=1) >= limit)
    if texcoords is not None:
        uv = np.asarray(texcoords)
        smooth &= ((uv[vertex[a]] == uv[vertex[bn]]).all(axis=1) &
                   (uv[vertex[an]] == uv[vertex[b]]).all(axis=1))
    unsafe = np.r_[order[np.repeat(counts != 2, counts)], a[~smooth]]
    locked_pos = np.zeros(vertex_pos.max() + 1, dtype=bool)
    locked_pos[pos[unsafe]] = True
    locked_pos[dest[unsafe]] = True

    # A shared point alone does not connect a fan. Each union joins the same
    # endpoint on opposite sides of a verified manifold edge.
    parent = list(range(vertex.size))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for left, right in zip(np.r_[a[smooth], an[smooth]].tolist(),
                           np.r_[bn[smooth], b[smooth]].tolist()):
        parent[find(left)] = find(right)
    fan = np.fromiter((find(i) for i in range(vertex.size)), dtype=np.int32, count=vertex.size)
    acc = np.zeros((vertex.size, 3))
    np.add.at(acc, fan, n[vertex] * angles[:, None])
    # If an existing index belongs to multiple disconnected fans, pin it.
    # Splitting that index would change the authored mesh topology.
    first = np.full(len(p), vertex.size, dtype=np.int32)
    last = np.full(len(p), -1, dtype=np.int32)
    np.minimum.at(first, vertex, fan)
    np.maximum.at(last, vertex, fan)
    chosen = np.flatnonzero((first == last) & ~locked_pos[vertex_pos])
    sums = acc[first[chosen]]
    length = np.linalg.norm(sums, axis=1)
    usable = length > 1e-12
    candidate = sums[usable] / length[usable, None]
    chosen = chosen[usable]
    cosine = np.einsum('ij,ij->i', candidate, n[chosen] / np.linalg.norm(n[chosen], axis=1)[:, None])
    # Keep already-flat faces byte-identical and bound the deviation from CAD.
    use = (cosine >= limit) & (cosine < math.cos(math.radians(.001)))
    result = np.array(n, dtype='<f4')
    result[chosen[use]] = candidate[use]
    # Measure jumps between shading normals at identical geometric endpoints.
    # The geometry is fixed, so a zero jump here means interpolation is
    # continuous across that edge. Locked corners are reported separately.
    left = np.r_[a[smooth], an[smooth]]
    right = np.r_[bn[smooth], b[smooth]]
    interior = ~locked_pos[pos[left]]
    va, vb = vertex[left[interior]], vertex[right[interior]]
    def jump_rms(values):
        if not len(va):
            return None
        unit = np.asarray(values, dtype=float)
        unit = unit / np.linalg.norm(unit, axis=1)[:, None]
        cosine = (unit[va] * unit[vb]).sum(axis=1)
        return float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(cosine, -1, 1)))**2)))

    return result, dict(vertices=len(p), triangles=len(triangles), smooth_manifold_edges=int(smooth.sum()),
                        locked_vertices=int(locked_pos[vertex_pos].sum()), changed_vertices=int(use.sum()),
                        unlocked_edge_endpoints=len(va), source_jump_rms_degrees=jump_rms(n),
                        derived_jump_rms_degrees=jump_rms(result),
                        max_change_degrees=float(np.degrees(np.arccos(np.clip(cosine[use], -1, 1))).max())
                        if use.any() else 0.)


def _replace_normals(raw, authored, derived):
    doc = json.loads(raw)
    primitive = doc['meshes'][0]['primitives'][0]
    accessor = doc['accessors'][primitive['attributes']['NORMAL']]
    view = doc['bufferViews'][accessor['bufferView']]
    buffer = doc['buffers'][view['buffer']]
    prefix, data = buffer['uri'].split(',', 1)
    binary = bytearray(base64.b64decode(data, validate=True))
    normal_view = np.ndarray((accessor['count'], 3), dtype='<f4', buffer=binary,
                             offset=view.get('byteOffset', 0) + accessor.get('byteOffset', 0),
                             strides=(view.get('byteStride', 12), 4))
    if normal_view.tobytes() != np.asarray(authored, dtype='<f4').tobytes():
        raise ValueError('NORMAL accessor does not match decoded source')
    normal_view[:] = derived
    buffer['uri'] = prefix + ',' + base64.b64encode(binary).decode('ascii')
    return json.dumps(doc, separators=(',', ':')).encode()


def _publish(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        try:
            tmp.write(data)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def scene_with_smooth_shells(scene, output_dir):
    """Return a lighting variant; mesh/instance/material IDs and positions persist."""
    from phymeo_rendering.mesh_data import _local_path, load_mesh
    from phymeo_rendering.panda_visuals import _MANIFEST_SHA256

    manifest_bytes = MANIFEST.read_bytes()
    if sha256(manifest_bytes).hexdigest() != _MANIFEST_SHA256:
        raise ValueError('Panda asset manifest identity differs')
    manifest = json.loads(manifest_bytes)
    source = {r['id']: r for r in manifest['meshes']}
    declared = {r['id']: r for r in manifest['materials']}
    bindings = {r['mesh_id']: r['material_id'] for r in manifest['instances']}
    materials = {m.id: m for m in scene.materials}
    target = Path(output_dir).resolve()
    target.mkdir(exist_ok=True, parents=True)
    code_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    meshes, rows, skipped = [], [], {}
    for mesh in scene.meshes:
        meshes.append(mesh)
        if mesh.id not in SHELL_IDS:
            continue
        identity = bindings[mesh.id]
        material = materials.get(identity)
        used = {i.material_id for i in scene.instances if i.mesh_id == mesh.id}
        if (used != {identity} or material is None or material.metallic != 0 or material.roughness != 1
                or tuple(material.base_color) != tuple(declared[identity]['base_color'])
                or min(material.base_color) < .85 or any(getattr(material, m, None) is not None for m in
                                                        ('normal_map', 'roughness_map', 'base_color_map'))):
            skipped[mesh.id] = 'different material binding'
            continue
        raw = _local_path(mesh.uri).read_bytes()
        sha = sha256(raw).hexdigest()
        if sha != source[mesh.id]['sha256']:
            skipped[mesh.id] = 'different source mesh'
            continue
        path = target / f'{mesh.id}-{sha[:12]}-{code_hash[:12]}.gltf'
        receipt = path.with_suffix('.json')
        row = None
        if path.is_file() and receipt.is_file():
            cached = json.loads(receipt.read_bytes())
            if (cached['source_sha256'] == sha and cached['generator_sha256'] == code_hash
                    and cached['derived_sha256'] == sha256(path.read_bytes()).hexdigest()):
                row = cached
        if row is None:
            original = load_mesh(mesh)
            derived, stats = _smoothed(original.positions, original.normals, original.triangles, original.texcoords)
            data = _replace_normals(raw, original.normals, derived)
            row = dict(mesh_id=mesh.id, material_id=identity, source_sha256=sha,
                       generator_sha256=code_hash, derived_sha256=sha256(data).hexdigest(), **stats)
            _publish(path, data)
            _publish(receipt, json.dumps(row, allow_nan=False).encode())
        if row['changed_vertices']:
            meshes[-1] = replace(mesh, uri=path.as_uri())
        rows.append(row)
    return replace(scene, meshes=tuple(meshes)), dict(crease_degrees=CREASE_DEGREES, meshes=rows,
        changed_vertices=sum(row['changed_vertices'] for row in rows), skipped=skipped)
