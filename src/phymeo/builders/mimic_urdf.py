"""Opt-in local URDF loading for acyclic mechanical mimic relationships."""

from ..dynamics.joint_mimic import MimicRelation, MimicRobotAsset
from ._mimic_xml import extract_mimic_relations
from ._urdf_fixed import folded_robot
from .urdf import _read_tree


def load_mimic_urdf(path, *, gravity=(0.0, 0.0, -9.81)):
    """Read a nonempty acyclic graph of affine, same-type, nonzero-ratio scalar linkages.

    Only the named independent joints accept actuator commands. The original URDF
    file is read once. Its other controls retain the ordinary importer's validation.
    """
    name, root, links, order, incoming = _read_tree(path)
    rows = extract_mimic_relations(order, incoming)
    full = folded_robot(name, root, links, order, incoming, gravity, path)
    return MimicRobotAsset(
        full, *rows[0], additional_relations=tuple(MimicRelation(*row) for row in rows[1:])
    )
