"""Rotation algebra for validated body-to-world xyzw quaternions.

Kernels use FP64 intermediate arithmetic and preserve the input storage dtype.
"""

import numpy as np


def quaternion_matrix(orientation: np.ndarray) -> np.ndarray:
    """Return (n,3,3) body-to-world matrices from validated unit quaternions."""
    q = orientation.astype(np.float64)
    # The storage boundary permits unit-norm roundoff; frames must remain orthogonal.
    q /= np.hypot.reduce(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def advance_world_orientation(orientation, angular_velocity, dt):
    """Left-compose exp(dt*omega_world) with the current body-to-world pose."""
    rotation_vector = float(dt) * angular_velocity.astype(np.float64)
    angle = np.hypot.reduce(rotation_vector, axis=1)
    delta_vector = rotation_vector * (0.5 * np.sinc(angle / (2 * np.pi)))[:, None]
    delta_scalar = np.cos(0.5 * angle)
    q = orientation.astype(np.float64)
    result = np.empty_like(q)
    result[:, :3] = (
        delta_scalar[:, None] * q[:, :3]
        + q[:, 3, None] * delta_vector
        + np.cross(delta_vector, q[:, :3])
    )
    result[:, 3] = delta_scalar * q[:, 3] - np.sum(delta_vector * q[:, :3], axis=1)
    result /= np.hypot.reduce(result, axis=1, keepdims=True)
    return result.astype(orientation.dtype)
