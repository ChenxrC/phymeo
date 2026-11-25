"""Budget indexed resident geometry without changing mesh, material or instance identities."""

import base64
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path

from ._prepare_cache import publish_bytes
from .backends import UnsupportedRequestError
from .mesh_data import load_mesh


def _mesh_bytes(positions, normals, uv, triangles):
    import numpy as np

    arrays = [(positions.astype('<f4', copy=False), 'VEC3', 5126),
              (normals.astype('<f4', copy=False), 'VEC3', 5126)]
    attributes = {'POSITION': 0, 'NORMAL': 1}
    if uv is not None:
        attributes['TEXCOORD_0'] = len(arrays)
        arrays.append((uv.astype('<f4', copy=False), 'VEC2', 5126))
    index = len(arrays)
    arrays.append((np.asarray(triangles, dtype='<u2').ravel(), 'SCALAR', 5123))
    accessors, views, binary = [], [], bytearray()
    for slot, (array, kind, component) in enumerate(arrays):
        views.append(dict(buffer=0, byteOffset=len(binary), byteLength=array.nbytes))
        binary.extend(array.tobytes())
        accessors.append(dict(bufferView=slot, componentType=component, type=kind, count=len(array)))
    accessors[0].update(min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist())
    doc = dict(asset={'version': '2.0'}, scenes=[{'nodes': [0]}], nodes=[{'mesh': 0}],
               meshes=[{'primitives': [{'attributes': attributes, 'indices': index, 'mode': 4}]}],
               buffers=[{'byteLength': len(binary),
                         'uri': 'data:application/octet-stream;base64,' + base64.b64encode(binary).decode()}],
               bufferViews=views, accessors=accessors)
    return json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()


def _edges(triangles, uv):
    """Pin open borders and shared edges where authored UV handedness changes."""
    import numpy as np

    start = triangles.ravel()
    end = triangles[:, (1, 2, 0)].ravel()
    packed = (np.minimum(start, end).astype('int64') << 32) | np.maximum(start, end)
    order = np.argsort(packed)
    sorted_edges = packed[order]
    unique, offset, count = np.unique(sorted_edges, return_index=True, return_counts=True)
    protected = count != 2
    if uv is not None:
        tex = uv[triangles].astype('float64')
        a, b = tex[:, 1] - tex[:, 0], tex[:, 2] - tex[:, 0]
        orientation = np.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        pair = np.flatnonzero(count == 2)
        at = offset[pair]
        protected[pair] = orientation[order[at] // 3] != orientation[order[at + 1] // 3]
    return unique[protected]


def _invalid_faces(p, uv, t, mapped):
    import numpy as np

    points = p[t].astype('float64')
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    invalid = np.all(cross == 0, axis=1)
    if mapped:
        tex = uv[t].astype('float64')
        a, b = tex[:, 1] - tex[:, 0], tex[:, 2] - tex[:, 0]
        first, second = a[:, 0]*b[:, 1], a[:, 1]*b[:, 0]
        invalid |= abs(first - second) <= 2**-18 * (abs(first) + abs(second))
    return invalid


def _simplify(p, uv, original, target, mapped):
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkDecimatePro
    from vtkmodules.vtkFiltersGeneral import vtkDistancePolyDataFilter

    def surface(points, faces):
        obj = vtkPolyData()
        coordinates = vtkPoints()
        coordinates.SetData(numpy_to_vtk(points, deep=True))
        obj.SetPoints(coordinates)
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(3 * np.arange(len(faces) + 1, dtype='int64'), deep=True),
                      numpy_to_vtkIdTypeArray(faces.ravel().astype('int64'), deep=True))
        obj.SetPolys(cells)
        return obj

    seam = _edges(original, uv)
    locked = np.unique(np.r_[seam >> 32, seam & 0xffffffff])
    step = float(np.max(np.ptp(p, axis=0))) * .001
    for attempt in range(5):
        # A temporary fan makes each locked source vertex non-manifold in VTK's
        # working topology. Its original vertex tuple and seam edges must survive.
        extra = len(p) + np.arange(len(locked))
        guard = np.column_stack((locked, extra, extra + len(locked)))
        working_p = np.vstack((p, p[locked] + [0., 0., step], p[locked] + [0., step, 0.]))
        working = surface(working_p, np.vstack((original, guard)))
        ids = numpy_to_vtk(np.arange(len(working_p), dtype='int32'), deep=True)
        ids.SetName('source_vertex')
        working.GetPointData().AddArray(ids)
        reducer = vtkDecimatePro()
        reducer.SetInputData(working)
        reducer.SetTargetReduction((len(original) - target) / (len(original) + len(guard)))
        reducer.PreserveTopologyOn()
        reducer.BoundaryVertexDeletionOff()
        reducer.SplittingOff()
        reducer.Update()
        reduced = reducer.GetOutput()
        source_ids = vtk_to_numpy(reduced.GetPointData().GetArray('source_vertex'))
        t = source_ids[vtk_to_numpy(reduced.GetPolys().GetConnectivityArray()).reshape(-1, 3)]
        t = t[np.all(t < len(p), axis=1)]
        if not len(t) or not np.array_equal(working_p[source_ids], vtk_to_numpy(reduced.GetPoints().GetData())):
            return original, None, 'reducer could not retain original vertex tuples'
        bad = _invalid_faces(p, uv, t, mapped)
        actual_seam = _edges(t, uv)
        if not bad.any() and np.array_equal(seam, actual_seam):
            break
        changed = np.setxor1d(seam, actual_seam)
        affected = np.unique(np.r_[changed >> 32, changed & 0xffffffff, t[bad].ravel()])
        # Pin the original incident triangles, not only the final three corners.
        # Otherwise a neighbouring collapse can recreate the same invalid face.
        adjacent = original[np.isin(original, affected).any(axis=1)]
        next_locks = np.union1d(locked, adjacent.ravel())
        if len(next_locks) == len(locked):
            return original, None, 'candidate would damage a UV seam or create a degenerate triangle'
        locked = next_locks
    else:
        return original, None, 'could not preserve every UV seam and valid triangle within five lock refinements'
    if len(t) >= len(original):
        return original, None, 'protected seams and topology prevented an edge collapse'

    used, inverse = np.unique(t, return_inverse=True)
    distance = vtkDistancePolyDataFilter()
    source_verts = np.unique(original)
    source_indices = np.searchsorted(source_verts, original)
    distance.SetInputData(0, surface(p[source_verts], source_indices))
    distance.SetInputData(1, surface(p[used], inverse.reshape(-1, 3)))
    distance.SignedDistanceOff()
    distance.ComputeCellCenterDistanceOn()
    distance.ComputeSecondDistanceOn()
    distance.Update()
    audit = {}
    for side, direction in enumerate(('source_to_lod', 'lod_to_source')):
        geometry = distance.GetOutput(side)
        values = np.concatenate([vtk_to_numpy(data.GetArray('Distance'))
                                 for data in (geometry.GetPointData(), geometry.GetCellData())])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite distance between original and display meshes')
        audit[direction] = dict(samples=len(values), p95_m=float(np.percentile(values, 95)), max_m=float(values.max()))
    return t, audit, None

@dataclass
class _Entry:
    asset: object
    source_sha: str
    source_v: int
    source_t: int
    uses: int
    weight: float
    protected: bool
    mapped: bool
    arrays: tuple | None
    output: object
    vertices: int
    triangles: int
    output_sha: str
    distance: dict | None = None
    stop_reason: str | None = None
    force_floor: bool = False


def _write_derived(entry, cache, triangles, distance):
    import numpy as np

    p, n, uv, _ = entry.arrays
    used, local = np.unique(triangles, return_inverse=True)
    if len(triangles) > entry.triangles or len(used) > entry.vertices:
        return
    if len(triangles) == entry.triangles and len(used) == entry.vertices:
        return
    raw = _mesh_bytes(p[used], n[used], uv[used] if uv is not None else None, local.reshape(-1, 3))
    digest = sha256(raw).hexdigest()
    path = cache / (digest + '.gltf')
    publish_bytes(path, raw)
    entry.output = replace(entry.asset, uri=path.as_uri())
    entry.output_sha = digest
    entry.triangles = len(triangles)
    entry.vertices = len(used)
    entry.distance = distance


def _adapt(entry, target, cache, version):
    # A source hash includes all attributes, face ordering and the exact local frame.
    key = sha256(f'geometry-v2/{version}/{entry.source_sha}/{target}/{int(entry.mapped)}'.encode()).hexdigest()
    receipt = cache / (key + '.json')
    if receipt.is_file():
        record = json.loads(receipt.read_bytes())
        payload = record['payload']
        if record['checksum'] != sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() or payload['key'] != key:
            raise ValueError(f'geometry cache record integrity failure: {receipt}')
        digest = payload['output_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'invalid geometry cache identity: {receipt}')
        path = cache / (digest + '.gltf')
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == digest:
            if payload['triangles'] <= entry.triangles and payload['vertices'] <= entry.vertices:
                entry.output = replace(entry.asset, uri=path.as_uri())
                entry.output_sha = digest
                entry.triangles, entry.vertices = payload['triangles'], payload['vertices']
                entry.distance = payload['distance']
            return None

    p, _, uv, original = entry.arrays
    result, distance, reason = _simplify(p, uv, original, target, entry.mapped)
    if reason:
        return reason
    before = entry.triangles
    _write_derived(entry, cache, result, distance)
    if entry.triangles < before:
        payload = dict(key=key, triangles=entry.triangles, vertices=entry.vertices,
                       output_sha256=entry.output_sha, distance=entry.distance)
        publish_bytes(receipt, json.dumps(dict(payload=payload,
                checksum=sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())).encode())
    return None


def _costs(entries, targets=None):
    vertex_count = index_count = instanced_triangles = 0
    for entry in entries:
        count = targets[entry.asset.id] if targets else entry.triangles
        # Used only when choosing targets. Acceptance always uses actual vertices.
        vertex_count += math.ceil(entry.vertices * count / entry.triangles) if targets else entry.vertices
        index_count += 3 * count
        instanced_triangles += count * entry.uses
    return dict(max_vertices=vertex_count, max_indices=index_count, max_meshes=len(entries),
                max_instances=sum(e.uses for e in entries), max_instanced_triangles=instanced_triangles,
                unique_triangles=index_count // 3)


def _excess(cost, limits):
    return {key: value for key, value in limits.items() if cost[key] > value}


def _describe(cost, limits):
    return ', '.join(f'{k} {cost[k]} > {v}' for k, v in _excess(cost, limits).items())


def _targets(entries, limits):
    # Square-root scaling retains proportionally more of a small mesh. An
    # importance multiplier keeps more triangles in prominent objects.
    # Work in log space so tiny or large finite user weights cannot overflow.
    weight = {e.asset.id: math.log(e.weight) + .5 * math.log(e.source_t) for e in entries}
    eligible = [e for e in entries if not e.protected and not e.stop_reason]
    def counts(scale):
        result = {}
        for e in entries:
            value = scale + weight[e.asset.id]
            if e.protected or e.stop_reason:
                count = e.triangles
            elif e.force_floor or value <= 0:
                count = 1
            elif value >= math.log(e.triangles):
                count = e.triangles
            else:
                count = int(math.exp(value))
            result[e.asset.id] = count
        return result

    low = min((-weight[e.asset.id] for e in eligible), default=0.)
    high = max((math.log(e.triangles) - weight[e.asset.id] for e in eligible), default=0.)
    for _ in range(48):
        middle = (low + high) * .5
        if _excess(_costs(entries, counts(middle)), limits):
            high = middle
        else:
            low = middle
    return counts(low)


def _mesh_cost(vertices, triangles, uses):
    return dict(vertices=vertices, triangles=triangles, instances=uses, instanced_triangles=triangles*uses)


def fit_geometry(scene, *, limits, cache, protected_meshes, importance, allow_adapt):
    """Prepare resident triangle geometry for measured ABI capacities.

    Sizes are measured on unique mesh IDs and all instance occurrences. A
    higher positive importance retains more detail when a limit binds. A
    callable importance(mesh, decoded_mesh) can use the decoded local bounds.
    Geometry error is sampled in both directions at vertices and triangle
    centres, in metres; these samples do not bound the continuous surface.
    """
    keys = ('max_vertices', 'max_indices', 'max_meshes', 'max_instances', 'max_instanced_triangles')
    limits = {k: limits[k] for k in keys}
    if type(allow_adapt) is not bool:
        raise ValueError('allow_adapt must be a boolean')
    if any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError('resident geometry capacities must be nonnegative integer counts')
    ids = {m.id for m in scene.meshes}
    if set(protected_meshes) - ids or (not callable(importance) and set(importance) - ids):
        raise ValueError('geometry priority refers to a mesh outside the scene')
    for key, count in [('max_meshes', len(scene.meshes)), ('max_instances', len(scene.instances))]:
        if count > limits[key]:
            raise UnsupportedRequestError(f'immutable scene count exceeds {key}: {count} > {limits[key]}')
    uses = Counter(i.mesh_id for i in scene.instances)
    materials = {m.id: m for m in scene.materials}
    mapped = {i.mesh_id for i in scene.instances if getattr(materials[i.material_id], 'normal_map', None)}
    # Only hold compact arrays across mesh decoding; a single Python object per
    # coordinate for the entire scene would greatly exceed the native budget.
    np = None
    if allow_adapt:
        try:
            import numpy as np
        except ImportError:
            pass
    entries = []
    for asset in scene.meshes:
        mesh = load_mesh(asset)
        priority = importance(asset, mesh) if callable(importance) else importance.get(asset.id, 1.)
        if isinstance(priority, bool) or not isinstance(priority, Real) or not math.isfinite(priority) or priority <= 0:
            raise ValueError(f'{asset.id!r}: mesh importance must be finite and positive')
        v, t = len(mesh.positions), len(mesh.triangles)
        protected = asset.id in protected_meshes
        arrays = None
        if np is not None and not protected:
            arrays = (np.asarray(mesh.positions, dtype='<f4'), np.asarray(mesh.normals, dtype='<f4'),
                      np.asarray(mesh.texcoords, dtype='<f4') if mesh.texcoords is not None else None,
                      np.asarray(mesh.triangles, dtype=np.int32))
        entries.append(_Entry(asset, mesh.source_sha256, v, t, uses[asset.id], float(priority),
                              protected, asset.id in mapped, arrays, asset, v, t, mesh.source_sha256))
        del mesh

    before = _costs(entries)
    current = before
    if _excess(current, limits):
        if not allow_adapt:
            raise UnsupportedRequestError('exact resident geometry exceeds capacity: ' + _describe(before, limits))
        fixed = _costs([e for e in entries if e.protected])
        if _excess(fixed, limits):
            raise UnsupportedRequestError('protected geometry alone exceeds capacity: ' + _describe(fixed, limits))
        if np is None:
            raise UnsupportedRequestError('geometry adaptation requires NumPy')
        cache = Path(cache).resolve()
        for e in entries:
            if e.protected:
                continue
            p, _, uv, faces = e.arrays
            if e.mapped and (uv is None or (abs(uv) > 65536).any() or _invalid_faces(p, uv, faces, True).any()):
                raise UnsupportedRequestError(f'{e.asset.id}: source has missing, out-of-range or singular normal-map UVs')
            if before['max_vertices'] > limits['max_vertices'] and len(np.unique(faces)) < e.vertices:
                # Exact removal of unreferenced tuples can be sufficient by itself.
                _write_derived(e, cache, faces, None)
        current = _costs(entries)
    version = None
    while _excess(current, limits):
        targets = _targets(entries, limits)
        pending = [e for e in entries if targets[e.asset.id] < e.triangles]
        if not pending:
            reasons = sorted({e.stop_reason for e in entries if e.stop_reason})
            if any(e.protected for e in entries):
                reasons.append('protected meshes must be retained exactly')
            reasons.append('each mesh must retain its surface and instance bindings')
            raise UnsupportedRequestError('geometry cannot fit with this seam-preserving reducer: ' +
                                          _describe(current, limits) + '; ' + '; '.join(reasons))
        if version is None:
            try:
                from vtkmodules.vtkCommonCore import vtkVersion
                version = vtkVersion.GetVTKVersion()
            except ImportError as exc:
                raise UnsupportedRequestError('geometry simplification requires optional VTK') from exc
        for e in pending:
            target = targets[e.asset.id]
            previous = e.triangles
            reason = _adapt(e, target, cache, version)
            if reason and target + 1 < previous:
                # A less ambitious reduction can preserve a surface that an
                # aggressive collapse would invalidate.
                reason = _adapt(e, (target + previous) // 2, cache, version)
            if e.triangles == previous:
                if target == 1 or reason:
                    e.stop_reason = reason or 'topology and attribute seams prevent further reduction'
                else:
                    e.force_floor = True
        current = _costs(entries)

    meshes = {}
    for e in entries:
        action = 'unchanged' if e.output is e.asset else 'compact' if e.triangles == e.source_t else 'simplify'
        meshes[e.asset.id] = dict(action=action, protected=e.protected, importance=e.weight,
                                  source_uri=e.asset.uri, source_sha256=e.source_sha,
                                  output_uri=e.output.uri, output_sha256=e.output_sha,
                                  before=_mesh_cost(e.source_v, e.source_t, e.uses),
                                  after=_mesh_cost(e.vertices, e.triangles, e.uses), distance=e.distance)
    result = (scene if all(e.output is e.asset for e in entries)
              else replace(scene, meshes=tuple(e.output for e in entries)))
    return result, dict(before=before, after=current, limits=limits, meshes=meshes,
                        distance_method='Bidirectional vertex and triangle-centre distances in metres')
