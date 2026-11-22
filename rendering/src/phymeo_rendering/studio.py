"""Shared immutable neutral studio settings for explicit backend opt-in.

StudioLighting is an alias of the established SapienStudio value type, so old
imports and reference recipes remain compatible. It specifies emitter geometry
and radiance, not a promise that different renderers use the same estimator.
SAPIEN requires path tracing; native requires its resident studio extension.
Importing this module initializes neither renderer nor physics.
"""
from .backends._sapien_studio import SapienStudio as StudioLighting

__all__ = ['StudioLighting']
