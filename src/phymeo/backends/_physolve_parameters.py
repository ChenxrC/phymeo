"""Read-only homogeneous model fields; no per-world physical parameter mirror."""

import numpy as np

from ..dynamics.rotation import quaternion_matrix


class PreparedBodyParameters:
    def __init__(self, model, cpu, device):
        import tyx as tx

        self.arrays = {
            'model_body_mass': model.body_mass.view(tx.f32),
            'model_body_com': model.body_ipos.view(tx.f32),
        }
        # The solver stores principal moments and an inertia frame. Consumers
        # need the full COM tensor in body axes, compiled once for this model.
        rotation = quaternion_matrix(cpu.body_iquat[:, [1, 2, 3, 0]])
        self._initial = np.asarray(
            (rotation * cpu.body_inertia[:, None, :]) @ rotation.transpose(0, 2, 1),
            np.float32,
        ).reshape(1, cpu.nbody, 9)
        self.arrays['model_body_inertia'] = tx.empty(self._initial.shape, tx.f32, device=device)

    def prepare(self, stream):
        self.arrays['model_body_inertia'].assign(self._initial, stream=stream)

    @property
    def allocated_bytes(self):
        return self.arrays['model_body_inertia'].size * 4
