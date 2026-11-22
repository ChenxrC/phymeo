"""One-way scene sources for authored fixtures, saved trajectories or a simulator."""

from collections.abc import Iterator
from typing import Protocol

from ..scene import SceneFrame, VisualScene


class SceneSource(Protocol):
    """All yielded frames reference the returned static scene."""

    def scene(self) -> VisualScene: ...

    def frames(self) -> Iterator[SceneFrame]: ...
