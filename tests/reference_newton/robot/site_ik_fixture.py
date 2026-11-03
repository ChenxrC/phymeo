"""Two tool frames on separate nonparallel branches of one moving trunk, in SI units."""

import numpy as np


def mul(a, b):
    a, b = np.array(a), np.array(b)
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def qmatrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fixture(mirror=False):
    identity = np.array([0.0, 0, 0, 1])
    left = np.array([np.sqrt(0.1), 0, 0, np.sqrt(0.9)])  # Rx with cos=4/5, sin=3/5
    right = mul([0, np.sqrt(0.2), 0, np.sqrt(0.8)], [0, 0, 1, 0])  # Ry then local Rz(pi)
    offset = np.array(
        [[0.125, -0.25, 0], [-0.5, 0, 0.125], [0.25, 0, 0], [0.5, 0, -0.125], [0.375, 0, 0]]
    )
    frames = np.array([identity, left, identity, right, identity])
    axes = np.array([[0.0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    local = np.array([[0.125, 0.0625, 0.03125], [0.0625, -0.125, -0.03125]])
    mount = np.array([[0, np.sqrt(0.5), 0, np.sqrt(0.5)], [0.5, 0.5, 0.5, 0.5]])
    truth = np.array([0.25, np.arctan2(0.8, 0.6), 0.125, np.arctan2(-0.6, 0.8), -0.0625])
    goals = []
    rotations = []
    for k, (j, cosine, sine) in enumerate(((1, 0.6, 0.8), (3, 0.8, -0.6))):
        turn = np.array(
            [0.0, 0, np.copysign(np.sqrt((1 - cosine) / 2), sine), np.sqrt((1 + cosine) / 2)]
        )
        rot = mul(frames[j], turn)
        origin = offset[0] + [0, 0, truth[0]] + offset[j]
        goals.append(origin + qmatrix(rot) @ (offset[j + 1] + [truth[j + 1], 0, 0] + local[k]))
        rotations.append(mul(rot, mount[k]))
    goals = np.array(goals)
    rotations = np.array(rotations)
    if mirror:
        # Reflection sends positions and slide axes through S, axial hinge
        # vectors through -S, and each right-handed frame rotation to S R S.
        reflection = np.array([-1, 1, 1])
        offset *= reflection
        local *= reflection
        goals *= reflection
        axes *= reflection
        axes[[1, 3]] *= -1
        for values in (frames, mount, rotations):
            values[:, :3] *= -reflection
    return dict(
        parents=np.array([-1, 0, 1, 0, 3]),
        kinds=("prismatic", "revolute", "prismatic", "revolute", "prismatic"),
        anchors=offset,
        frames=frames,
        axes=axes,
        local=local,
        mount=mount,
        goals=goals,
        rotations=rotations,
        truth=truth,
        seed=np.array([-0.125, -0.65, -0.1, 0.8, 0.15]),
        lower=np.array([-0.25, -1.4, -0.15, -1.4, -0.15]),
        upper=np.array([0.45, 1.4, 0.25, 1.4, 0.25]),
    )
