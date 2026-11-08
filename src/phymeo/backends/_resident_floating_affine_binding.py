"""Original floating material ownership plus a constant mechanical graph identity."""

from dataclasses import dataclass, field

from ..dynamics.floating_mimic import FloatingMimicRobotAsset
from ._resident_floating_affine_tree import (
    FloatingAffineTreeWorldSpec,
    floating_affine_model_token,
)
from ._resident_floating_binding import (
    ResidentFloatingTreeBinding,
    compile_floating_tree_binding,
)
from ._resident_tree_binding import _token


@dataclass(frozen=True, eq=False)
class ResidentFloatingAffineTreeBinding(ResidentFloatingTreeBinding):
    """A real full body frame, with an explicit original affine graph token.

    Direct construction declares an installation against a complete checkpoint.
    from_asset additionally retains original collider/model ownership and folds
    a fixed authored material frame exactly once through the ordinary compiler.
    """

    graph_token: tuple = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.graph_token, tuple):
            raise ValueError("floating affine material requires its original graph token")
        object.__setattr__(self, "graph_token", _token(self.graph_token))

    @classmethod
    def from_asset(cls, asset, source_key, *, key, world):
        if not isinstance(asset, FloatingMimicRobotAsset):
            raise TypeError("floating affine material source must be FloatingMimicRobotAsset")
        base = ResidentFloatingTreeBinding.from_asset(asset.tree, source_key, key=key, world=world)
        result = cls(
            base.key,
            base.world,
            base.group_index,
            base.local_points_m,
            topology=None,
            graph_token=floating_affine_model_token(asset),
        )
        object.__setattr__(result, "_source_shape", base._source_shape)
        object.__setattr__(result, "_model_token", base._model_token)
        return result


def compile_floating_affine_tree_binding(shape, binding, spec):
    """Reuse the full-body FloatingBindingCompiled and the existing device bounds."""
    if not isinstance(binding, ResidentFloatingAffineTreeBinding) or not isinstance(
        spec, FloatingAffineTreeWorldSpec
    ):
        raise TypeError("floating affine material needs an affine floating binding and world")
    if binding.graph_token != spec.graph_token:
        raise ValueError("floating affine source graph differs from the owned graph")
    return compile_floating_tree_binding(shape, binding, spec.physical)
