"""Strict extraction of affine mimic tags from one already-read local URDF."""

from ._robot_xml import _attributes, _child, _children, _name, _number


def extract_mimic_relations(order, incoming):
    """Return raw relation tuples in tree order and remove only validated tags.

    Graph validation belongs to compile_mimic_coordinates. A malformed XML tag
    leaves all tags in this caller-owned tree intact; no file is rewritten.
    """
    rows, removals = [], []
    for child in order:
        _, joint_name, element = incoming[child]
        tag = _child(element, "mimic", f"mimic on {joint_name}", required=False)
        if tag is None:
            continue
        where = f"mimic on {joint_name!r}"
        _attributes(tag, {"joint", "multiplier", "offset"}, where)
        _children(tag, set(), where)
        driver = _name(tag, "joint", where)
        multiplier = _number(tag, "multiplier", where) if "multiplier" in tag.attrib else 1.0
        offset = _number(tag, "offset", where) if "offset" in tag.attrib else 0.0
        rows.append((driver, joint_name, multiplier, offset))
        removals.append((element, tag))
    if not rows:
        raise ValueError("the mechanical importer requires at least one mimic relation")
    for element, tag in removals:
        element.remove(tag)
    return tuple(rows)
