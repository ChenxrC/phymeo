"""Affine graph identity around the existing full-tree material/bounds compiler."""

from dataclasses import dataclass, field

from ..dynamics.joint_mimic import MimicRobotAsset
from ._resident_affine_tree import AffineTreeWorldSpec, affine_model_token
from ._resident_tree_binding import ResidentTreeBinding, compile_tree_binding


@dataclass(frozen=True, eq=False)
class ResidentAffineTreeBinding(ResidentTreeBinding):
    """Current material in one real N-row physical group, never a K pseudo-body.

    Direct construction is an explicit installation with a full checkpoint
    topology. from_asset also proves the original graph, model and shape source.
    The inherited group-local point storage is immutable FP64 authored metadata.
    """

    _graph_token: tuple | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        if not isinstance(asset, MimicRobotAsset):
            raise TypeError("affine material source must be a MimicRobotAsset")
        # The ordinary compiler already handles folded sites, real physical
        # owners and original shape content. Retain its proof plus the graph.
        base = ResidentTreeBinding.from_asset(asset.tree, source_key, key=key, world=world)
        result = cls(base.key, base.world, base.group_index, base.local_points_m, topology=None)
        object.__setattr__(result, "_source_shape", base._source_shape)
        object.__setattr__(result, "_model_token", base._model_token)
        object.__setattr__(result, "_graph_token", affine_model_token(asset))
        return result


def compile_affine_tree_binding(shape, binding, spec):
    """Return the existing TreeBindingCompiled; no new geometry kernel or upload."""
    if not isinstance(binding, ResidentAffineTreeBinding) or not isinstance(spec, AffineTreeWorldSpec):
        raise TypeError("affine material compilation needs an affine binding and world")
    if binding._source_shape is not None and binding._graph_token != spec.graph_token:
        raise ValueError("affine source graph differs from its owned mechanical graph")
    return compile_tree_binding(shape, binding, spec.physical)
