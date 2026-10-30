"""Explicit native bending adapter; host audits may reject, never replace output."""

import numpy as np

from ..constraints.bending import BendingConstraints
from ..constraints.material_trace import MaterialBatch
from ..core.capabilities import SolverFailure
from .tyx_math import as_compute, as_index
from .tyx_node_math import TyxNodeMath


class TyxBendingMath(TyxNodeMath):
    def project_bending(self, position, weights, constraints, alpha, multipliers):
        if not isinstance(constraints, BendingConstraints):
            raise TypeError("native bending requires BendingConstraints")
        count = len(constraints.indices)
        x, w = self._nodes(position, weights)
        if np.any(w <= 0):
            raise ValueError("native composite requires all dynamic nodes")
        ids = as_index(constraints.indices, "hinges")
        if np.any(ids >= len(x)):
            raise ValueError("hinge outside node model")
        rest = as_compute(constraints.rest_angle, "rest angles")
        a, old_lambda = as_compute(alpha, "alpha"), as_compute(multipliers, "multipliers")
        if a.shape != (count,) or old_lambda.shape != (count,) or np.any(a <= 0):
            raise ValueError("invalid native bending constitutive arrays")
        start, lambda_start = x.copy(), old_lambda.copy()
        before = self.stats()
        if count:
            from ._tyx_bending_kernels import project_bending_traced

            x, result, rows, status, failed = self.call(
                project_bending_traced,
                count,
                ids,
                rest,
                a,
                w,
                x,
                old_lambda,
                np.zeros((count, 45), np.float32),
                np.zeros(count, np.int32),
                np.zeros(1, np.int32),
                np.zeros((6, 3), np.float32),
                outputs=(5, 6, 7, 8, 9),
            )
            if failed[0] or np.any(status):
                raise SolverFailure(f"native bending row rejected: {int(failed[0])}")
        else:
            x, start = np.array(position, copy=True), np.array(position, copy=True)
            result, lambda_start = (
                np.array(multipliers, copy=True),
                np.array(multipliers, copy=True),
            )
            w, rest, a = (
                np.array(weights, copy=True),
                constraints.rest_angle,
                np.array(alpha, copy=True),
            )
            rows, status = np.empty((0, 45)), np.zeros(0, np.int32)
        after = self.stats()

        def promoted(value):
            return np.asarray(value, dtype=np.float64)

        batch = MaterialBatch(
            "bending",
            constraints.indices,
            position,
            promoted(start),
            promoted(x),
            promoted(x),
            multipliers,
            promoted(lambda_start),
            promoted(result),
            promoted(result),
            promoted(w),
            promoted(rest),
            promoted(a),
            promoted(rows),
            status.astype(np.int64),
            after["launches"] - before["launches"],
            after["completed_launches"] - before["completed_launches"],
        )
        return promoted(x), promoted(result), batch
