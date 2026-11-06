"""Frozen independent inputs for capsule rotation against a closed finite mesh.

This module imports no Phymeo geometry implementation. Distances and rotations
used by the reference checks are independent of the new continuous producer.
"""

from dataclasses import dataclass
from math import acos, cos, pi, sin

import numpy as np

import phymeo as ph

BOX_FACES = (
    (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
    (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
    (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
)


def box(lower, upper):
    return tuple((x, y, z) for x in (lower[0], upper[0])
                 for y in (lower[1], upper[1]) for z in (lower[2], upper[2])), BOX_FACES


@dataclass(frozen=True)
class Case:
    name: str
    vertices: tuple
    triangles: tuple
    endpoints: tuple
    radius: float
    angle: float
    axis: tuple = (0.0, 0.0, 1.0)
    pivot_start: tuple = (0.0, 0.0, 0.0)
    pivot_end: tuple = (0.0, 0.0, 0.0)
    mesh_step: tuple = (0.0, 0.0, 0.0)


B01 = Case(
    "B01-orbit-shaft", *box((1.0, -1 / 8, -1 / 4), (2.0, 1 / 8, 1 / 4)),
    ((0.0, 3 / 2, -1.0), (0.0, 3 / 2, 1.0)), 1 / 8, -pi,
)


def public_case(case, *, world=0, reverse=False, reorder=False, capsule_world=None):
    """Construct only the existing public registrations and actual stored motions."""
    km = ph.CollisionKey(world, int(reverse), 0)
    kc = ph.CollisionKey(world if capsule_world is None else capsule_world, int(not reverse), 0)
    ids = tuple(range(len(case.vertices)))
    permutation = tuple(reversed(ids)) if reorder else ids
    mesh = ph.CollisionShape(km, "mesh", permutation, triangles=case.triangles)
    cap = ph.CollisionShape(kc, "capsule", (41, 93), case.radius)
    vertices = np.array([case.vertices[i] for i in permutation], dtype=float)
    end = None if not any(case.mesh_step) else vertices + case.mesh_step
    motion = ph.CollisionMotion(
        case.endpoints,
        turn=ph.RigidTurn(case.pivot_start, case.pivot_end, case.axis, case.angle),
    )
    return ph.CollisionModel((cap, mesh)), {
        km: ph.CollisionMotion(vertices, end), kc: motion
    }, mesh, cap


def stored_pose(path, time):
    """Independent ordinary Rodrigues evaluation of the *stored* public motion."""
    points = np.asarray(path.start)
    if path.turn is None:
        return points.copy() if path.end is None else points + time * (path.end - points)
    turn = path.turn
    axis = np.asarray(turn.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    pivot = np.asarray(turn.pivot_start)
    arms = points - pivot
    theta = time * turn.angle
    return (pivot + time * (np.asarray(turn.pivot_end) - pivot)
            + cos(theta) * arms + sin(theta) * np.cross(axis, arms)
            + (1 - cos(theta)) * np.outer(arms @ axis, axis))


def b01_first(stored_angle):
    # y_axis=3*cos(alpha)/2 first reaches y_face+r=1/4; x=sqrt(35)/4 is inside.
    return acos(1 / 6) / abs(stored_angle)


def b01_face_gap(stored_angle, time):
    # Used only where the independently checked x coordinate is inside [1,2].
    return 1.5 * cos(abs(stored_angle) * time) - 0.25
