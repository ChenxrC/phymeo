"""Phymeo rendering contracts and the public photographic renderer."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .photography import StudioRenderer as StudioRenderer
    from .preparation import PreparedScene as PreparedScene

__all__ = ['StudioRenderer', 'PreparedScene']


def __getattr__(name):
    if name == 'PreparedScene':
        from .preparation import PreparedScene
        globals()[name] = PreparedScene
        return PreparedScene
    if name == 'StudioRenderer':
        from .photography import StudioRenderer
        globals()[name] = StudioRenderer
        return StudioRenderer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
