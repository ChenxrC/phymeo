"""Owned convex support data from compiled MuJoCo boxes/meshes.

This is geometry admission only. It neither enables the dynamics compiler nor
changes the source contact law. MuJoCo mesh collision uses the vertex convex
hull; retaining every compiled vertex preserves that support function.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from ._constraint_model import constraint_pairs


def _owned(array, dtype):
    value = np.asarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class MeshQueryCatalog:
    geom_ids: np.ndarray
    body_ids: np.ndarray
    offsets: np.ndarray
    points: np.ndarray
    pairs: np.ndarray
    margins: np.ndarray
    source_ngeom: int

    @property
    def ngeom(self):
        return len(self.geom_ids)


def compile_mesh_query_catalog(mj, model, *, geom_ids=None):
    """Compile original geometry IDs, with native implicit pair filtering.

    A subset is an explicit diagnostic selection, never full-scene admission.
    Explicit contact pairs/custom filtering are rejected, not silently lost.
    """
    override = int(model.opt.enableflags) & int(mj.mjtEnableBit.mjENBL_OVERRIDE)
    if model.npair or int(model.opt.disableflags) or override:
        raise ValueError("mesh query requires implicit pairs and default collision flags")
    ids = np.arange(model.ngeom, dtype=np.int32) if geom_ids is None else np.asarray(geom_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
        raise ValueError("geom_ids must be a nonempty integer vector")
    if np.any(ids < 0) or np.any(ids >= model.ngeom) or len(set(map(int, ids))) != len(ids):
        raise ValueError("invalid or duplicate geometry ID")
    ids = ids.astype(np.int32)
    offsets, points = [0], []
    for geom in ids:
        kind = int(model.geom_type[geom])
        if kind == int(mj.mjtGeom.mjGEOM_BOX):
            local = np.asarray(tuple(product((-1.0, 1.0), repeat=3))) * model.geom_size[geom]
        elif kind == int(mj.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        else:
            raise ValueError(f"geometry {geom} is not a box or convex mesh collision")
        if len(local) < 4 or not np.isfinite(local).all():
            raise ValueError(f"geometry {geom} has invalid support vertices")
        # Translation-invariant rank check; no hull simplification or vertex dropping.
        if np.linalg.matrix_rank(local - local[0]) < 3:
            raise ValueError(f"geometry {geom} has no three-dimensional convex core")
        stored = np.asarray(local, dtype=np.float32)
        if not np.isfinite(stored).all() or np.any((local != 0) & (stored == 0)):
            raise ValueError("support vertices cannot be represented by finite FP32")
        points.extend(stored)
        offsets.append(len(points))
    remap = {int(geom): row for row, geom in enumerate(ids)}
    pairs = [(remap[a], remap[b]) for a, b in constraint_pairs(model) if a in remap and b in remap]
    margins = np.asarray(model.geom_margin[ids], dtype=np.float64)
    if not np.isfinite(margins).all() or np.any(margins < 0):
        raise ValueError("finite nonnegative margins required")
    return MeshQueryCatalog(
        _owned(ids, np.int32),
        _owned(model.geom_bodyid[ids], np.int32),
        _owned(offsets, np.int32),
        _owned(points, np.float32),
        _owned(np.asarray(pairs).reshape(-1, 2), np.int32),
        _owned(margins, np.float32),
        int(model.ngeom),
    )
