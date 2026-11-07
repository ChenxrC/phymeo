"""Opt-in local URDF loading for physical floating-root affine mechanisms."""

from ..dynamics.floating_mimic import FloatingMimicRobotAsset
from ._mimic_xml import extract_mimic_relations
from .floating_urdf import _floating_from_tree
from .urdf import _read_tree


def load_floating_mimic_urdf(path, *, gravity=(0.0, 0.0, -9.81), collapse_fixed_joints=False):
    """Read one original file, remove validated mimic tags, and assemble once.

    Fixed tooling remains explicit. Complete inertial/geometry assembly is the
    ordinary floating importer's shared path; no temporary URDF is generated.
    The returned wrapper has independent controls and a complete physical tree.
    """
    name, root, links, order, incoming = _read_tree(path)
    relations = extract_mimic_relations(order, incoming)
    tree = _floating_from_tree(
        name,
        root,
        links,
        order,
        incoming,
        gravity,
        path,
        collapse_fixed_joints=collapse_fixed_joints,
    )
    return FloatingMimicRobotAsset(tree, relations)
