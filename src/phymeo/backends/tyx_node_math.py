"""Explicit FP32 node mathematics; host FP64 is storage and domain validation."""

import numpy as np

from ..constraints.distance import DistanceConstraints
from ..core.capabilities import SolverFailure
from ..core.tetrahedral import TetrahedralModel
from ..materials.stvk import deformation_gradient
from .tyx_math import TyxMath, as_compute, as_index


class TyxNodeMath(TyxMath):
    def _nodes(self, x, weights):
        self._check_open()
        x, weights = as_compute(x, "node array"), as_compute(weights, "inverse mass")
        if x.ndim != 2 or x.shape[1] != 3 or weights.shape != (len(x),):
            raise ValueError("nodes require (n,3) values and (n,) inverse mass")
        if np.any(weights < 0):
            raise ValueError("inverse mass must be non-negative")
        return x, weights

    @staticmethod
    def _h(h):
        value = as_compute(h, "substep")
        if value.ndim != 0 or value <= 0:
            raise ValueError("substep must be a positive scalar")
        return float(value)

    @staticmethod
    def _vectors(value, shape, name):
        value = as_compute(value, name)
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
        return value

    @staticmethod
    def _restore_fixed(output, original, weights):
        result = output.astype(np.float64)
        result[weights == 0] = np.asarray(original)[weights == 0]
        return result

    def predict(self, x, v, force, weights, gravity, h):
        from ._tyx_kernels import integrate_particles

        position, w = self._nodes(x, weights)
        velocity = self._vectors(v, position.shape, "velocity")
        force = self._vectors(force, position.shape, "force")
        gravity = self._vectors(gravity, (3,), "gravity")
        h = self._h(h)
        if not len(position):
            return np.array(x, dtype=np.float64, copy=True), velocity.astype(np.float64)
        px, pv = self.call(
            integrate_particles,
            len(position),
            h,
            position,
            velocity,
            force,
            w,
            gravity,
            np.zeros_like(position),
            np.zeros_like(position),
            outputs=(7, 8),
        )
        return self._restore_fixed(px, x, w), pv.astype(np.float64)

    def reconstruct(self, pred_x, pred_v, x, weights, h):
        from ._tyx_node_kernels import reconstruct_nodes

        position, w = self._nodes(x, weights)
        px = self._vectors(pred_x, position.shape, "predicted position")
        pv = self._vectors(pred_v, position.shape, "predicted velocity")
        h = self._h(h)
        if not len(position):
            return np.zeros_like(position, dtype=np.float64)
        (result,) = self.call(
            reconstruct_nodes,
            len(position),
            h,
            px,
            pv,
            position,
            w,
            np.zeros_like(position),
            outputs=(6,),
        )
        return result.astype(np.float64)

    def apply_load(self, state, indices, loads, weights):
        from ._tyx_node_kernels import apply_node_loads

        x, w = self._nodes(state, weights)
        indices = as_index(indices, "node indices")
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(x)):
            raise ValueError("node indices outside model")
        loads = self._vectors(loads, (len(indices), 3), "node loads")
        if not len(indices):
            return np.array(state, dtype=np.float64, copy=True)
        (result,) = self.call(apply_node_loads, len(indices), indices, loads, w, x, outputs=(4,))
        return self._restore_fixed(result, state, w)

    def project_distances(self, x, weights, constraints, alpha, multipliers):
        from ._tyx_node_kernels import project_node_distances

        position, w = self._nodes(x, weights)
        if not isinstance(constraints, DistanceConstraints):
            raise TypeError("distance projection requires DistanceConstraints")
        edges = as_index(constraints.indices, "distance edges")
        if np.any(edges >= len(position)):
            raise ValueError("distance edge outside model")
        count = len(edges)
        rest = self._vectors(constraints.rest_length, (count,), "rest length")
        alpha = self._vectors(alpha, (count,), "alpha")
        multipliers = self._vectors(multipliers, (count,), "multipliers")
        if np.any(alpha < 0):
            raise ValueError("alpha must be non-negative")
        if not count:
            return np.array(x, dtype=np.float64, copy=True), multipliers.astype(np.float64)
        position, result = self.call(
            project_node_distances,
            count,
            edges,
            rest,
            alpha,
            w,
            position,
            multipliers,
            outputs=(5, 6),
        )
        return self._restore_fixed(position, x, w), result.astype(np.float64)

    def material(self, model, x):
        from ..validation.material_device import validate_device_material
        from ._tyx_node_kernels import stvk_material

        self._check_open()
        if not isinstance(model, TetrahedralModel):
            raise TypeError("material requires TetrahedralModel")
        position = self._vectors(x, model.rest_position.shape, "material position")
        # Independent host domain checks reject invalid candidates, but provide
        # no strain, stress, energy or internal force to the device operator.
        deformation_gradient(model, np.asarray(x, dtype=np.float64))
        deformation_gradient(model, position.astype(np.float64))
        cells = as_index(model.tetrahedra, "tetrahedra")
        inverse = as_compute(model.rest_inverse, "reference inverse")
        volume = as_compute(model.volume, "reference volume")
        bulk = model.young_modulus / (3 * (1 - 2 * model.poisson_ratio))
        # In the bounded F domain, entries are <=1.4. This conservative static
        # bound covers quadratic omission propagated through stiffness. Direct
        # Piola and final-gradient omissions also need stiffness-independent
        # terms: 3 omitted products/entry, then 3 origin-node force columns.
        # It is smaller than the frozen material absolute error gate (2e-4).
        # Reject unsafe stiffness/scale combinations; never correct a force here.
        tiny = float(np.finfo(np.float32).tiny)
        volume_sum = float(np.sum(model.volume))
        inverse_scale = float(np.max(np.abs(model.rest_inverse)))
        loss_bound = (
            1000.0
            * tiny
            * (model.lame_mu + abs(model.lame_lambda) + bulk)
            * volume_sum
            * (1.0 + 9.0 * inverse_scale)
            + 27.0 * tiny * volume_sum * inverse_scale
            + 9.0 * tiny * volume_sum
        )
        if not np.isfinite(loss_bound) or loss_bound > 1e-6:
            raise SolverFailure("material quadratic underflow error budget is not representable")
        force, energy = self.call(
            stvk_material,
            len(cells),
            model.lame_mu,
            model.lame_lambda,
            bulk,
            cells,
            inverse,
            volume,
            position,
            *(np.zeros((len(cells), 3, 3), np.float32) for _ in range(4)),
            np.zeros_like(position),
            np.zeros(1, np.float32),
            outputs=(12, 13),
        )
        if energy[0] < 0:
            raise SolverFailure("device material energy lost positivity")
        force, energy = force.astype(np.float64), float(energy[0])
        validate_device_material(model, np.asarray(x, dtype=np.float64), force, energy)
        return force, energy
