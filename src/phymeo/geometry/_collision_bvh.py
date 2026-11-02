"""Balanced AABB forests. Node escape links allow bounded stackless traversal."""

import numpy as np

from ..core.capabilities import SolverFailure
from ._collision_spatial import WORD_BITS


class BoundsTree:
    def __init__(self, boxes, groups, planes):
        centers = (boxes[:, :3].astype(np.float64) + boxes[:, 3:].astype(np.float64)) / 2
        nodes, ordered = [], []
        self.roots = np.empty(len(boxes), dtype=np.int32)

        def build(ids):
            node = len(nodes)
            begin = len(ordered)
            nodes.append(None)
            if len(ids) == 1:
                ordered.append(int(ids[0]))
            else:
                axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
                ordered_ids = ids[np.lexsort((ids, centers[ids, axis]))]
                split = len(ids) // 2
                build(ordered_ids[:split])
                build(ordered_ids[split:])
            largest = int(np.max(ids))
            nodes[node] = (
                begin,
                len(ordered),
                len(nodes),
                largest,
                int(np.any(planes[ids])),
                1 << (largest % WORD_BITS) if len(ids) == 1 else 0,
            )
            return node

        for group in groups:
            self.roots[group] = build(group)
        self.nodes = np.asarray(nodes, dtype=np.int32).reshape(-1, 6)
        self.ordered = np.asarray(ordered, dtype=np.int32)

    def host_hits(self, leaves, queries, params, allowed):
        nodes = self.nodes
        bounds = np.empty((len(nodes), 6), dtype=np.float32)
        for i in range(len(nodes) - 1, -1, -1):
            begin, end, _, _, _, _ = nodes[i]
            if end == begin + 1:
                bounds[i] = leaves[self.ordered[begin]]
            else:
                left = i + 1
                right = nodes[left, 2]
                bounds[i, :3] = np.minimum(bounds[left, :3], bounds[right, :3])
                bounds[i, 3:] = np.maximum(bounds[left, 3:], bounds[right, 3:])
        words = (len(leaves) + WORD_BITS - 1) // WORD_BITS
        out = np.zeros((len(queries), words + 2), np.int32)
        for row, (box, (root, minimum, plane)) in enumerate(zip(queries, params, strict=True)):
            node = int(root)
            stop = nodes[node, 2]
            while node < stop:
                begin, end, escape, largest, unbounded, bit = nodes[node]
                out[row, -2] += 1
                leaf = end == begin + 1
                if largest <= minimum or (
                    leaf and allowed is not None and not allowed[row, largest // WORD_BITS] & bit
                ):
                    node = escape
                    continue
                if not (plane or unbounded):
                    out[row, -1] += int(leaf)
                    if np.any(bounds[node, :3] > box[3:]) or np.any(box[:3] > bounds[node, 3:]):
                        node = escape
                        continue
                if leaf:
                    out[row, largest // WORD_BITS] |= bit
                    node = escape
                else:
                    node += 1
        return out


class CollisionTrees:
    """Only immutable layouts survive publication. Every query refits fresh bounds."""

    def __init__(self):
        self.layouts = {}
        self.counts = {
            f"{role}_bvh_{stat}": 0
            for role in ("shape", "mesh")
            for stat in ("builds", "refits", "node_visits", "leaf_tests")
        }

    def hits(self, pending, key, leaves, queries, space, driver):
        role = key[0]
        shape = space is not None
        tree = pending.get(key)
        if tree is None:
            groups = space.worlds if shape else (np.arange(len(leaves), dtype=np.int32),)
            planes = space.tags[:, 5] if shape else np.zeros(len(leaves), dtype=bool)
            tree = pending[key] = BoundsTree(leaves, groups, planes)
            self.counts[role + "_bvh_builds"] += 1
        params = np.zeros((len(queries), 3), dtype=np.int32)
        params[:, 0] = tree.roots if shape else 0
        params[:, 1] = np.arange(len(leaves), dtype=np.int32) if shape else -1
        if shape:
            params[:, 2] = space.tags[:, 5]
        if driver is None:
            raw = tree.host_hits(leaves, queries, params, space.allowed if shape else None)
        else:
            raw = driver.tree_hits(key, tree, leaves, queries, params, shape)
        words = (len(leaves) + WORD_BITS - 1) // WORD_BITS
        if raw.shape != (len(queries), words + 2) or raw.dtype != np.int32 or np.any(raw < 0):
            raise SolverFailure("invalid BVH output")
        self.counts[role + "_bvh_refits"] += 1
        self.counts[role + "_bvh_node_visits"] += int(raw[:, -2].sum())
        self.counts[role + "_bvh_leaf_tests"] += int(raw[:, -1].sum())
        return raw

    def facets(self, pending, key, leaves, queries, driver):
        raw = self.hits(pending, key, leaves, queries, None, driver)
        indices = []
        for packed in raw[:, :-2]:
            row = []
            for w in np.flatnonzero(packed):
                bits = int(packed[w])
                while bits:
                    bit = bits & -bits
                    i = int(w) * WORD_BITS + bit.bit_length() - 1
                    if i >= len(leaves):
                        raise SolverFailure("invalid BVH facet bit")
                    row.append(i)
                    bits -= bit
            indices.append(row)
        return indices
