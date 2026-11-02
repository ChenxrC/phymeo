"""Reusable tyx buffers for node refit and stackless AABB traversal."""

import numpy as np


class TreeBuffers:
    def __init__(self, engine, tree, max_queries, shape):
        self.engine = engine
        self.words = (len(tree.ordered) + 29) // 30
        self.nodes = engine._scratch(tree.nodes.shape, np.int32)
        self.order = engine._scratch(tree.ordered.shape, np.int32)
        self.node_boxes = engine._scratch((len(tree.nodes), 6), np.float32)
        self.params = engine._scratch((max_queries, 3), np.int32)
        self.output = engine._scratch((max_queries, self.words + 3), np.int32)
        self.host_params = np.zeros((max_queries, 3), np.int32)
        if not shape:
            self.leaves = engine._scratch((len(tree.ordered), 6), np.float32)
            self.queries = engine._scratch((max_queries, 6), np.float32)
            self.host_queries = np.zeros((max_queries, 6), np.float32)
        self.layout = None

    def hits(self, tree, leaves, queries, params, shape):
        from ._tyx_spatial_kernels import bvh_candidates, refit_boxes

        engine = self.engine
        if self.layout is not tree:
            engine._spatial_upload(self.nodes, tree.nodes)
            engine._spatial_upload(self.order, tree.ordered)
            self.layout = tree
        count = len(queries)
        self.host_params[:count] = params
        engine._spatial_upload(self.params, self.host_params)
        if shape:
            input_boxes = query_boxes = engine._bounds
            allowed = engine._allowed
        else:
            engine._spatial_upload(self.leaves, leaves)
            self.host_queries[:count] = queries
            engine._spatial_upload(self.queries, self.host_queries)
            input_boxes, query_boxes = self.leaves, self.queries
            allowed = self.params  # not read by unmasked traversal
        engine._counts["launches"] += 2
        refit_boxes(
            len(tree.nodes), self.nodes, self.order, input_boxes, self.node_boxes, mode=engine._mode
        )
        bvh_candidates(
            count,
            self.words,
            int(shape),
            self.nodes,
            self.node_boxes,
            query_boxes,
            self.params,
            allowed,
            self.output,
            mode=engine._mode,
        )
        raw = engine._spatial_output(self.output)[:count]
        engine._counts["completed_launches"] += 2
        engine._spatial_counts["spatial_completed_launches"] += 2
        stop = tree.nodes[params[:, 0], 2]
        if not np.array_equal(raw[:, -1], stop):
            raise RuntimeError("incomplete device BVH traversal")
        return raw[:, :-1]
