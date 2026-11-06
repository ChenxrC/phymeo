"""Owned local inputs for one cloth/link force evaluation, with no cross-phase cache."""

import numpy as np

from ..constraints.collision_shapes import CollisionMotion
from ..geometry._static_capsule_face import static_capsule_face_sample


def _owned(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


class ClothLinkEvaluation:
    """Share only inputs that are identical within this invocation.

    The caller creates a fresh object for each entry, midpoint or endpoint force
    sample. No result, material history, operator or workspace is retained here.
    The exact static witness routine and the order of the velocity formula stay
    unchanged. CollisionMotion validation still occurs before a shape is queried.
    """

    def __init__(self, contacts, state, geo):
        cloth = state.domains[contacts.cloth_domain]
        self._triangles = _owned(cloth.position[contacts.faces])
        ordered_nodes = np.sort(contacts.faces, axis=1)
        self._ids = tuple(tuple(map(int, row)) for row in ordered_nodes)
        self._velocities = _owned(cloth.velocity[ordered_nodes])
        self._faces = tuple(contacts.triangles)
        self._capsules = tuple(contacts.capsules)
        ends, coms, linear, angular = [], [], [], []
        for link in contacts.links:
            _, _, com, jl, ja = geo[link.domain][:5]
            velocity = state.domains[link.domain].v
            index = link.link_index
            ends.append(_owned(contacts.endpoints(link, geo)))
            coms.append(_owned(com[index]))
            linear.append(_owned(jl[index] @ velocity))
            angular.append(_owned(ja[index] @ velocity))
        self._ends, self._coms = tuple(ends), tuple(coms)
        self._linear, self._angular = tuple(linear), tuple(angular)
        self._cap_motion = [None] * len(self._capsules)
        self._face_motion = [None] * len(self._faces)

    @property
    def triangles(self):
        return self._triangles

    def endpoints(self, link_index):
        return self._ends[link_index]

    def sample(self, link_index, face_index):
        cap_motion = self._cap_motion[link_index]
        if cap_motion is None:
            cap_motion = CollisionMotion(self._ends[link_index])
            self._cap_motion[link_index] = cap_motion
        face_motion = self._face_motion[face_index]
        if face_motion is None:
            face_motion = CollisionMotion(self._triangles[face_index])
            self._face_motion[face_index] = face_motion
        return static_capsule_face_sample(
            self._capsules[link_index], self._faces[face_index], cap_motion, face_motion,
        )[0]

    def relative_velocity(self, link_index, face_index, ids, weights, point):
        """Return the original COM arm and second-minus-first material velocity.

        A witness supplies all three sorted material node IDs, including zero
        weights. Reject a different address instead of interpreting row order as
        ownership. Keep weights @ node_velocity, then subtract the original
        linear + cross(angular, arm) expression; there is no regrouped reduction.
        """
        if tuple(ids) != self._ids[face_index]:
            raise ValueError("face material IDs do not match prepared geometry")
        weights = np.asarray(weights)
        if weights.shape != (3,) or not np.isfinite(weights).all():
            raise ValueError("invalid prepared material weights")
        arm = np.asarray(point) - self._coms[link_index]
        robot_v = self._linear[link_index] + np.cross(self._angular[link_index], arm)
        relative = weights @ self._velocities[face_index] - robot_v
        return arm, relative
