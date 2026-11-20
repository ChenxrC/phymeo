"""Preparation and readback of homogeneous PhySolve model parameters."""

import numpy as np

from ..dynamics.rotation import quaternion_matrix


def prepare_contact_inverse_weights(cpu, model, stream):
    """Apply the GPU reference-weight regularization to imported CPU constants."""
    import mujoco
    import tyx as tx

    weights = np.asarray(cpu.body_invweight0, dtype=np.float32)
    moving = cpu.body_weldid != 0
    translation, rotation = weights.T
    min_value = mujoco.mjMINVAL
    use_rotation = moving & (translation < min_value) & (rotation > min_value)
    use_translation = moving & (rotation < min_value) & (translation > min_value)
    if use_rotation.any() or use_translation.any():
        # Match the frontend set_const_0 fallback for a vanishing component
        # without overwriting unrelated constants (e.g. actuator damping).
        weights[use_rotation, 0] = rotation[use_rotation]
        weights[use_translation, 1] = translation[use_translation]
        target = model.body_invweight0.view(tx.f32)
        target.assign(weights.reshape(target.shape), stream=stream)


class PreparedBodyParameters:
    def __init__(self, model, principal_inertia, inertia_rotation_wxyz, device):
        import tyx as tx

        self.arrays = {
            'model_body_mass': model.body_mass.view(tx.f32),
            'model_body_com': model.body_ipos.view(tx.f32),
        }
        # The solver stores principal moments and an inertia frame. Consumers
        # need the full COM tensor in body axes, compiled once for this model.
        rotation = quaternion_matrix(inertia_rotation_wxyz[:, [1, 2, 3, 0]])
        self._initial = np.asarray(
            (rotation * principal_inertia[:, None, :]) @ rotation.transpose(0, 2, 1),
            np.float32,
        ).reshape(1, len(principal_inertia), 9)
        self.arrays['model_body_inertia'] = tx.empty(self._initial.shape, tx.f32, device=device)

    def prepare(self, stream):
        self.arrays['model_body_inertia'].assign(self._initial, stream=stream)

    @property
    def allocated_bytes(self):
        return self.arrays['model_body_inertia'].size * 4
