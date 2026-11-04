"""Opt-in local URDF loading for acyclic mechanical mimic relationships."""

from ..dynamics.joint_mimic import MimicRelation, MimicRobotAsset
from ._robot_xml import _attributes, _child, _children, _name, _number
from ._urdf_fixed import folded_robot
from .urdf import _read_tree


def load_mimic_urdf(path, *, gravity=(0.0, 0.0, -9.81)):
    """Read a nonempty acyclic graph of affine, same-type, nonzero-ratio scalar linkages.

    Only the named independent joints accept actuator commands. The original URDF
    file is read once. Its other controls retain the ordinary importer's validation.
    """
    name, root, links, order, incoming = _read_tree(path)
    rows = []
    for child in order:
        _, joint_name, element = incoming[child]
        tag = _child(element, "mimic", f"mimic on {joint_name}", required=False)
        if tag is not None:
            where = f"mimic on {joint_name!r}"
            _attributes(tag, {"joint", "multiplier", "offset"}, where)
            _children(tag, set(), where)
            leader = _name(tag, "joint", where)
            a = _number(tag, "multiplier", where) if "multiplier" in tag.attrib else 1.0
            b = _number(tag, "offset", where) if "offset" in tag.attrib else 0.0
            rows.append((leader, joint_name, a, b))
            element.remove(tag)
    if not rows:
        raise ValueError("the mechanical importer requires at least one mimic relation")
    full = folded_robot(name, root, links, order, incoming, gravity, path)
    return MimicRobotAsset(
        full, *rows[0], additional_relations=tuple(MimicRelation(*row) for row in rows[1:])
    )
