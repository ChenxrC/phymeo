"""Represent affine scalar-joint intervals with rigid motions of registered colliders."""

import math

from ..constraints.collision_shapes import CollisionMotion, RigidTurn
from ..core.robot import robot_array


def _registered_turn(asset, link, first, last, geometry, owner):
    ancestors = []
    index = link
    while index >= 0:
        ancestors.append(index)
        index = int(asset.model.parents[index])
    hinge = None
    slides = []
    for joint in reversed(ancestors):
        if first[joint] == last[joint]:
            continue
        name = asset.joint_names[joint]
        delta = float(last[joint]) - float(first[joint])
        if not math.isfinite(delta):
            raise ValueError(f"{owner}: joint {name!r} interval is not representable")
        if asset.model.joint_types[joint] == "revolute":
            if hinge is not None:
                raise ValueError(
                    f"{owner} moves revolute joints {asset.joint_names[hinge[0]]!r} and {name!r}; the supported interval allows one moving hinge"
                )
            hinge = (joint, delta)
        else:
            if hinge is not None:
                raise ValueError(
                    f"{owner} moves prismatic joint {name!r} downstream of moving revolute joint {asset.joint_names[hinge[0]]!r}"
                )
            slides.append((geometry.axis[joint], delta))
    try:
        # With no preceding moving hinge, each active slide has a fixed world axis.
        # A subsequent hinge and all its rigid descendants inherit this linear drift.
        shift = tuple(math.fsum(float(axis[k]) * delta for axis, delta in slides) for k in range(3))
        if not all(map(math.isfinite, shift)):
            raise OverflowError
        if hinge is None:
            if not any(shift):
                return None
            # One displacement for every vertex, including meshes with distant vertices.
            return RigidTurn((0.0, 0.0, 0.0), shift, (0.0, 0.0, 1.0), 0.0)
        joint, angle = hinge
        pivot = tuple(geometry.position[joint])
        end = tuple(float(p) + v for p, v in zip(pivot, shift, strict=True))
        if not all(map(math.isfinite, end)):
            raise OverflowError
        scale = max(map(abs, shift))
        if scale:
            # A moving-pivot descriptor stores two world positions. Bound the
            # displacement lost in that representation relative to the actual
            # slide motion, independently of the world's coordinate origin.
            error = math.hypot(
                *[((q - p) - v) / scale for p, q, v in zip(pivot, end, shift, strict=True)]
            )
            magnitude = math.hypot(*(v / scale for v in shift))
            if error > math.sqrt(math.ulp(1.0)) * magnitude:
                raise ValueError(
                    f"{owner} joint {asset.joint_names[joint]!r} moving pivot loses translation resolution"
                )
        return RigidTurn(pivot, end, tuple(geometry.axis[joint]), angle)
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{owner} has unrepresentable world pivot motion") from exc


def joint_interval_motion(asset, q_start, q_end, start_geometry, start_paths):
    """Continue the exact registered start poses over q(t) = q_start + t*(q_end-q_start).

    Any number of moving prismatic ancestors can precede one moving revolute
    ancestor. Other moving joints on a collider's own chain receive a named
    rejection. Unchanged joints still contribute their full initial FK transforms.
    The path retains the signed scalar angle, even when the end pose repeats.
    """
    first = robot_array(q_start, "q_start", ndim=1)
    last = robot_array(q_end, "q_end", ndim=1)
    if first.shape != (asset.model.dof_count,) or last.shape != first.shape:
        raise ValueError("joint interval endpoints must match the model DOF count")
    supported = {-1: None}
    result = {}
    for collider in asset.collisions:
        link = collider.link_index
        if link not in supported:
            owner = f"link {collider.link_name!r} collision {collider.name!r}"
            supported[link] = _registered_turn(asset, link, first, last, start_geometry, owner)
        path = start_paths[collider.shape.key]
        turn = supported[link]
        result[collider.shape.key] = (
            path
            if turn is None
            else CollisionMotion(
                path.start, turn=turn, body_orientation_xyzw=path.body_orientation_xyzw
            )
        )
    return result
