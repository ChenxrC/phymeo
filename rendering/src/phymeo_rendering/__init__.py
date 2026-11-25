"""Phymeo rendering contracts and the public photographic renderer."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .photography import StudioRenderer as StudioRenderer

__all__ = ['StudioRenderer']


def __getattr__(name):
    if name == 'StudioRenderer':
        from .photography import StudioRenderer
        globals()[name] = StudioRenderer
        return StudioRenderer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
