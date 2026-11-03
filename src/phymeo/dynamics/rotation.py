"""Rotation algebra for validated body-to-world xyzw quaternions.

Kernels use FP64 intermediate arithmetic and preserve the input storage dtype.
"""

import numpy as np

# A participant with this contract drifts at its published final world angular velocity.
SEMI_IMPLICIT_WORLD_OMEGA = "semi_implicit_world_omega_v1"


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


def matrix_quaternion(rotation):
    """Return a unit XYZW quaternion for one validated body-to-world rotation."""
    diagonal = np.diag(rotation)
    trace = float(diagonal.sum())
    four_squares = np.r_[1.0 + 2.0 * diagonal - trace, 1.0 + trace]
    axis = int(np.argmax(four_squares))
    q = np.empty(4)
    q[axis] = 0.5 * np.sqrt(four_squares[axis])
    scale = 4.0 * q[axis]
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if axis == 3:
        q[:3] = skew / scale
    else:
        for other in range(3):
            if other != axis:
                q[other] = (rotation[axis, other] + rotation[other, axis]) / scale
        q[3] = skew[axis] / scale
    return q / np.hypot.reduce(q)
