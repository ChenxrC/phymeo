"""RD01 registrations with stable material IDs, shared by the three backends."""

import numpy as np
from case_catalog import geometry_for

import phymeo as ph

NAMES = ("base", "clear", "permuted", "holdout", "holdout_permuted")
IDS = (7, 15, 31)


def scene(
    face_start, face_end, center_start, center_end, radius=0.25, *, ids=IDS, reverse=False, world=0
):
    face = ph.CollisionKey(world, 2 if reverse else 1, 0)
    ball = ph.CollisionKey(world, 1 if reverse else 2, 0)
    model = ph.CollisionModel(
        (ph.CollisionShape(face, "triangle", ids), ph.CollisionShape(ball, "sphere", (19,), radius))
    )
    paths = {
        face: ph.CollisionMotion(face_start, face_end),
        ball: ph.CollisionMotion([center_start], [center_end]),
    }
    return model, paths, face, ball


def authored(name, reverse=False, world=0):
    _, variant, geometry = geometry_for("RD01", name)
    a, b = geometry["A"], geometry["B"]
    order = variant.get("order_A", (0, 1, 2))
    return scene(
        a["vertices"],
        None,
        b["center_start"],
        b["center_end"],
        b["radius"],
        ids=tuple(IDS[i] for i in order),
        reverse=reverse,
        world=world,
    )


def deforming(world=0):
    # At t=1/2 the three vertices reach the reference triangle together. Before
    # that every vertex lies strictly below the bottom of the approaching ball.
    middle = np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 0]], float)
    velocity = np.array([[0.5, 0.25, 0.25], [-0.25, 0.25, -0.5], [0.25, -0.5, 0.75]])
    return scene(
        middle - velocity / 2, middle + velocity / 2, [0, 0, 1.25], [0, 0, -0.75], world=world
    )
