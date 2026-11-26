"""Fixed pickup cases on the two rails of the dual Panda grasp/insert cell.

Distances are in meters. The fork's local +Y follows the rails and +X crosses
the slot. Positive yaw is a right-handed rotation about world +Z. A reset
changes only the free part's initial pose. Each split uses separate seeds and
disjoint yaw strata. The records describe planned geometry before simulation.
"""

from dataclasses import replace
from functools import lru_cache
import math
from xml.etree import ElementTree as ET

import numpy as np

from _panda_assembly_scene import _free_transform, _quaternion


# Geometry from _panda_grasp_insert_scene. Verify these dimensions against the
# actual supplied XML before a planned case can become a simulator input.
_RAIL_X = .011
_RAIL_HALF = (.0045, .020, .010)
_RAIL_Z = .029
_SHOULDER_HALF = (.015, .015, .003)
_SHOULDER_Z = .016
_SHAFT_RADIUS = .004
_SHAFT_ENDS = (.019, .025)
_BASE_FACE_Z = .039

# seed, cross and along stratum centers (mm), world yaw stratum center (deg).
_STRATA = {
    "train": (
        (710401, 1.15, -13.5, -36.), (710402, -.85, 11., -22.),
        (710403, -1.30, -6., -7.), (710404, .80, 6., 7.),
        (710405, -1.15, -10., 22.), (710406, 1.25, 14., 36.),
    ),
    "holdout": (
        (860731, -1.05, 12.5, -29.), (860732, 1.10, -8., -15.),
        (860733, -.85, -12.5, 15.), (860734, .95, 8., 29.),
    ),
}
_JITTER = (.15, .8, 1.5)  # uniform half-widths in mm, mm, degrees


def geometry_margins(cross_m, along_m, yaw_deg):
    """Certify two finite contact patches and a supported projected COM.

    Each witness is a 3 × 6 mm rectangle on one rail, centered along the
    part's Y position. Map its four corners into the yawed square shoulder.
    Rectangles and shoulder are convex, so containing the corners contains
    the entire patch. The part's symmetric components put its projected COM
    at the shaft center. Its projection must lie inside the contact hull.
    """
    cross, along, yaw = map(float, (cross_m, along_m, yaw_deg))
    if not all(map(math.isfinite, (cross, along, yaw))):
        raise ValueError("pickup offsets and yaw must be finite")
    rail_inner = _RAIL_X - _RAIL_HALF[0]
    rail_outer = _RAIL_X + _RAIL_HALF[0]
    patch_inner, patch_outer, patch_half_y = .007, .010, .003
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    shoulder_clearances = []
    for side in (-1., 1.):
        # The fork's +Z points down, so world-positive yaw rotates the part by
        # -yaw in the fork frame. Its inverse maps the witness into the part.
        corners = [(side*x-cross, y) for x in (patch_inner, patch_outer)
                   for y in (-patch_half_y, patch_half_y)]
        shoulder_clearances.append(min(
            min(_SHOULDER_HALF[0] - abs(c*x-s*y), _SHOULDER_HALF[1] - abs(s*x+c*y))
            for x, y in corners))
    shaft_gaps = [rail_inner + cross - _SHAFT_RADIUS, rail_inner - cross - _SHAFT_RADIUS]
    shoulder_to_rail = (_RAIL_Z - _RAIL_HALF[2]) - (_SHOULDER_Z + _SHOULDER_HALF[2])
    tip_to_base = _BASE_FACE_Z - (_SHAFT_ENDS[1] + _SHAFT_RADIUS)
    patch_rail_x = min(patch_inner - rail_inner, rail_outer - patch_outer)
    patch_rail_y = _RAIL_HALF[1] - abs(along) - patch_half_y
    com_hull = min(patch_outer - abs(cross), patch_half_y)
    feasible = (min(*shaft_gaps, *shoulder_clearances, patch_rail_x, patch_rail_y,
                    com_hull, tip_to_base) > 0. and abs(shoulder_to_rail) <= 1e-12)
    return dict(feasible=feasible, shaft_rail_clearance_left_right_m=shaft_gaps,
                patch_shoulder_clearance_left_right_m=shoulder_clearances,
                patch_rail_side_margin_m=patch_rail_x, patch_rail_end_margin_m=patch_rail_y,
                guaranteed_patch_area_per_rail_m2=(patch_outer-patch_inner)*2.*patch_half_y
                if feasible else 0., com_support_hull_margin_m=com_hull,
                shoulder_to_rail_top_gap_m=shoulder_to_rail, tip_to_base_gap_m=tip_to_base)


def validate_reset(reset):
    """Recompute the geometric certificate from the three actual inputs."""
    geometry = geometry_margins(reset["cross_m"], reset["along_m"], reset["yaw_deg"])
    if not geometry["feasible"]:
        raise ValueError(f"part lacks certified pickup support or clearance: {geometry}")
    return geometry


def reset_bank(split):
    """Fresh JSON-compatible case records. No rollout history or global RNG."""
    if split not in _STRATA:
        raise ValueError("split must be 'train' or 'holdout'")
    records = []
    for i, (seed, cross, along, yaw) in enumerate(_STRATA[split], 1):
        rng = np.random.Generator(np.random.PCG64(seed))
        dx, dy, da = rng.uniform(-1., 1., 3) * _JITTER
        case = dict(id=f"{split}-{i:02d}", split=split, seed=seed,
                    cross_m=float((cross+dx)*.001), along_m=float((along+dy)*.001),
                    yaw_deg=float(yaw+da), plan="panda-fork-pickup-v1",
                    stratum_center_cross_mm_along_mm_yaw_deg=[cross, along, yaw],
                    uniform_halfwidth_cross_mm_along_mm_yaw_deg=list(_JITTER))
        case["geometry"] = validate_reset(case)
        records.append(case)
    return records


def _vec(element, name):
    return np.array([float(n) for n in element.attrib[name].split()])


@lru_cache(maxsize=2)
def _fork_pose(xml):
    """Bind the purely geometric checks to the actual fixed fixture."""
    world = ET.fromstring(xml).find("worldbody")
    cradle = world.find("body[@name='pickup_cradle']")
    part = world.find("body[@name='assembly_part']")
    if cradle is None or part is None or cradle.find("joint") is not None or cradle.find("freejoint") is not None:
        raise ValueError("expected a fixed pickup_cradle and an assembly_part")

    def check_geom(body, name, kind, **attributes):
        geom = body.find(f"geom[@name='{name}']")
        if geom is None or geom.get("type") != kind or any(
                key in geom.attrib for key in ("quat", "axisangle", "xyaxes", "zaxis", "euler")):
            raise ValueError(f"unexpected pickup geometry for {name}")
        for key, expected in attributes.items():
            actual = _vec(geom, key)
            if actual.shape != np.shape(expected) or not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(f"pickup geometry differs at {name}.{key}")
        return geom

    for side, name in ((-1., "pickup_rail_left"), (1., "pickup_rail_right")):
        check_geom(cradle, name, "box", pos=(side*_RAIL_X, 0., _RAIL_Z), size=_RAIL_HALF)
    check_geom(part, "part_handle", "box", pos=(0., 0., 0.), size=(.010, .012, .013))
    check_geom(part, "part_shoulder", "box", pos=(0., 0., _SHOULDER_Z), size=_SHOULDER_HALF)
    check_geom(part, "assembly_pin", "capsule", size=(_SHAFT_RADIUS,),
               fromto=(0., 0., _SHAFT_ENDS[0], 0., 0., _SHAFT_ENDS[1]))
    base = check_geom(cradle, "pickup_base", "box")
    top = _vec(base, "pos")[2] - _vec(base, "size")[2]
    if abs(top - _BASE_FACE_Z) > 1e-12:
        raise ValueError("pickup base top differs from the declared shaft clearance")
    x, y = _vec(cradle, "xyaxes").reshape(2, 3)
    rotation = np.column_stack((x, y, np.cross(x, y)))
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-10)
            or not np.allclose(rotation[:, 2], (0., 0., -1.), rtol=0., atol=1e-10)):
        raise ValueError("the fork must have horizontal rails and a downward +Z axis")
    return _vec(cradle, "pos"), rotation


def initial_state(scene, reset):
    """Return one native initial state for the existing compiled scene."""
    validate_reset(reset)
    origin, fork = _fork_pose(scene.xml)
    nominal = scene.initial_state()
    q = np.array(nominal.q, copy=True)
    if q.shape != (23,) or np.shape(nominal.v) != (22,):
        raise ValueError("expected two Panda arms and two finger slides before the free part")
    home = _free_transform(q[16:])
    if (not np.allclose(home[:3, 3], origin, rtol=0., atol=1e-10) or
            not np.allclose(home[:3, :3], fork, rtol=0., atol=1e-10)):
        raise ValueError("the nominal free part must coincide with the pickup fork frame")
    q[16:18] += fork[:2, :2] @ [reset["cross_m"], reset["along_m"]]
    yaw = math.radians(reset["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    world_turn = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    q[19:23] = _quaternion(world_turn @ home[:3, :3])
    return replace(nominal, q=q)
