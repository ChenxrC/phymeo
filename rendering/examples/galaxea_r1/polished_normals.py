"""Angle-weighted shading-normal fans for the painted R1 CAD housings."""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from preview_stable_lighting.shell_normals import CREASE_DEGREES, _publish, _replace_normals


PAINTED_SHELLS = frozenset({"zed_link"} | {
    f"{side}_{link}" for side in ("left", "right")
    for link in ("arm_link1", "arm_link2", "arm_link3", "gripper_link")
})


def _fans(corners, left, right):
    """Join corresponding corners across individually admitted common edges."""
    parent = list(range(corners))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a,b in zip(left.tolist(), right.tolist()):
        parent[find(a)] = find(b)
    return np.fromiter((find(i) for i in range(corners)), count=corners, dtype=np.int32)


def _smooth(chunks):
    positions = np.concatenate([np.asarray(m.positions, dtype=float) for m in chunks])
    source = np.concatenate([np.asarray(m.normals, dtype=float) for m in chunks])
    counts = [len(m.positions) for m in chunks]
    offsets = np.cumsum([0, *counts[:-1]])
    triangles = np.concatenate([np.asarray(m.triangles, dtype=np.int32) + offset
                                for m, offset in zip(chunks, offsets)])
    if any((m.texcoords is None) != (chunks[0].texcoords is None) for m in chunks):
        raise ValueError("inconsistent UV declarations across R1 mesh chunks")
    uv = None if chunks[0].texcoords is None else np.concatenate([np.asarray(m.texcoords) for m in chunks])
    points = positions[triangles]
    outgoing = np.roll(points, -1, axis=1) - points
    incoming = np.roll(points, 1, axis=1) - points
    cross = np.cross(outgoing, incoming)
    angles = np.arctan2(np.linalg.norm(cross, axis=2), np.sum(outgoing*incoming, axis=2)).ravel()
    face = cross[:, 0] / np.linalg.norm(cross[:, 0], axis=1)[:, None]
    unit_source = source / np.linalg.norm(source, axis=1)[:, None]
    limit = math.cos(math.radians(CREASE_DEGREES))

    # Only bit-identical CAD positions are candidates; nearby shells and a
    # shared point alone do not establish common faces. UVs are compared at
    # both endpoints after matching opposite half-edges.
    _, position_ids = np.unique(positions, axis=0, return_inverse=True)
    vertex = triangles.ravel()
    origin = position_ids.reshape(-1)[vertex]
    corner = np.arange(len(vertex), dtype=np.int32)
    following = np.roll(corner.reshape(-1, 3), -1, axis=1).ravel()
    dest = origin[following]
    lo, hi = np.minimum(origin, dest), np.maximum(origin, dest)
    order = np.lexsort((hi, lo))
    starts = np.r_[0, 1 + np.flatnonzero((np.diff(lo[order]) != 0) | (np.diff(hi[order]) != 0))]
    incidence = np.diff(np.r_[starts, len(order)])
    pair = starts[incidence == 2]
    a, b = order[pair], order[pair+1]
    an, bn = following[a], following[b]
    smooth = ((origin[a] == dest[b]) & (dest[a] == origin[b]) &
              (np.sum(face[a//3] * face[b//3], axis=1) >= limit))
    for left, right in ((a,bn), (an,b)):
        smooth &= np.sum(unit_source[vertex[left]] * unit_source[vertex[right]], axis=1) >= limit
        if uv is not None:
            smooth &= np.all(uv[vertex[left]] == uv[vertex[right]], axis=1)
    left = np.r_[a[smooth], an[smooth]]
    right = np.r_[bn[smooth], b[smooth]]
    fan = _fans(len(corner), left, right)

    # A glTF index only has one normal. Detect identical source attributes
    # across chunk splits too, so partitioning never changes what is writable.
    attributes = (positions,source) if uv is None else (positions,source,uv)
    keys, aliases = np.unique(np.column_stack(attributes), axis=0, return_inverse=True)
    alias = aliases.reshape(-1)[vertex]
    first = np.full(len(keys), len(corner), dtype=np.int32)
    last = np.full(len(keys), -1, dtype=np.int32)
    np.minimum.at(first, alias, fan)
    np.maximum.at(last, alias, fan)
    blocked = np.zeros(len(corner), dtype=bool)
    blocked[fan[first[alias] != last[alias]]] = True
    # Preserve a fan if it loops around to reach both sides of a disallowed
    # edge, or if its edge has more than two incidences. An ordinary open
    # boundary can still use the average of the faces on its own side.
    bad = order[np.repeat(incidence > 2, incidence)]
    blocked[fan[np.r_[bad, following[bad]]]] = True
    alternate = np.where(origin[a] == origin[b], b, bn)
    alternate_end = np.where(origin[a] == origin[b], bn, b)
    for i,j in ((a, alternate), (an, alternate_end)):
        blocked[fan[i[(~smooth) & (fan[i] == fan[j])]]] = True

    sums = np.zeros((len(corner), 3))
    np.add.at(sums, fan, np.repeat(face, 3, axis=0) * angles[:, None])
    lengths = np.linalg.norm(sums, axis=1)
    candidate = sums / np.maximum(lengths[:, None], 1e-20)
    deviation = np.sum(candidate[fan] * unit_source[vertex], axis=1)
    blocked[fan[deviation < limit]] = True
    # Isolated faces and unused vertices retain exactly their source bytes.
    sizes = np.bincount(fan, minlength=len(corner))
    eligible = (~blocked[fan]) & (sizes[fan] >= 2) & (lengths[fan] > 1e-12)
    result = source.astype("<f4")
    result[vertex[eligible]] = candidate[fan[eligible]]
    # Leave already-smooth registered normals bit-identical within 0.001°.
    changed = np.linalg.norm(result.astype(float) - unit_source, axis=1) > math.radians(.001)
    result[~changed] = source[~changed]

    def endpoint_rms(values):
        if not len(left):
            return None
        unit = values.astype(float)
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        cosine = np.sum(unit[vertex[left]] * unit[vertex[right]], axis=1)
        return float(np.sqrt(np.mean(np.square(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))))))
    return np.split(result, offsets[1:]), dict(
        triangles=len(triangles), vertices=len(positions), changed_vertices=int(changed.sum()),
        admitted_manifold_edges=int(smooth.sum()), open_edges=int((incidence == 1).sum()),
        nonmanifold_edges=int((incidence > 2).sum()), crease_or_seam_edges=int((~smooth).sum()),
        source_admitted_edge_jump_rms_deg=endpoint_rms(source),
        polished_admitted_edge_jump_rms_deg=endpoint_rms(result))


def smooth_painted_shells(scene, directory, *, targets):
    """Derive only the NORMAL accessor on selected, rigidly registered shells."""
    from phymeo_rendering.mesh_data import _local_path, load_mesh
    from .finish_details import _store
    targets = set(targets)
    if targets - PAINTED_SHELLS:
        raise ValueError(f"unsupported polished shell links: {sorted(targets - PAINTED_SHELLS)}")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Chunks can share topology only within one rigid visual and material.
    bindings = defaultdict(set)
    for item in scene.instances:
        bindings[(item.node_id, item.local_pose, item.material_id)].add(item.mesh_id)
    groups = defaultdict(set)
    used_by = defaultdict(set)
    for (link, _, _), ids in bindings.items():
        group = tuple(sorted(ids))
        groups[group].add(link)
        for mesh_id in ids:
            used_by[mesh_id].add(group)
    meshes = {mesh.id: mesh for mesh in scene.meshes}
    rows, skipped = [], []
    for group, users in groups.items():
        if not users & targets:
            continue
        if users - targets or any(len(used_by[key]) != 1 for key in group):
            skipped.append(dict(mesh_ids=group, links=sorted(users), reason="mesh also bound outside selected visual"))
            continue
        chunks = [load_mesh(meshes[key]) for key in group]
        derived, stats = _smooth(chunks)
        parts = []
        for key, mesh, normals in zip(group, chunks, derived):
            source_hash = mesh.source_sha256
            derived_hash = source_hash
            updated = int(np.any(normals != np.asarray(mesh.normals, dtype="<f4"), axis=1).sum())
            if updated:
                original = _local_path(meshes[key].uri).read_bytes()
                if sha256(original).hexdigest() != source_hash:
                    raise ValueError("R1 mesh changed during normal derivation")
                data = _replace_normals(original, mesh.normals, normals)
                path, derived_hash = _store(directory, key+"-normals", data, "gltf")
                meshes[key] = replace(meshes[key], uri=path.as_uri())
            parts.append(dict(mesh_id=key, source_sha256=source_hash, derived_sha256=derived_hash,
                              changed_normals=updated))
        rows.append(dict(links=sorted(users), meshes=parts, **stats))
    evidence = dict(source_scene=scene.key, crease_degrees=CREASE_DEGREES, targets=sorted(targets),
                    groups=rows, skipped=skipped, changed_vertices=sum(r["changed_vertices"] for r in rows))
    raw = json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
    _publish(directory/"polished-normals.json", raw)
    return replace(scene, key=scene.key+":polished-normals-"+sha256(raw).hexdigest()[:12],
                   meshes=tuple(meshes.values()))
