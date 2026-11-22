"""Finite authored scene values; no file loading, physics or rendering runtime."""

from collections.abc import Iterable, Iterator
from copy import deepcopy

from ..scene import FrameKey, SceneFrame, VisualScene, validate_scene_frame


def _check_repeated_frame(previous: SceneFrame, current: SceneFrame, context: str) -> None:
    if previous.time_s != current.time_s:
        raise ValueError(f"{context}: repeated FrameKey has conflicting time_s")
    old_nodes = {node.node_id: node.world_pose for node in previous.nodes}
    new_nodes = {node.node_id: node.world_pose for node in current.nodes}
    if old_nodes.keys() != new_nodes.keys():
        raise ValueError(f"{context}: repeated FrameKey has conflicting node_id membership")
    for node_id, old_pose in old_nodes.items():
        new_pose = new_nodes[node_id]
        if old_pose.position_m != new_pose.position_m:
            raise ValueError(
                f"{context}: repeated FrameKey has conflicting node {node_id!r}.position_m"
            )
        old_rotation = old_pose.orientation_xyzw
        new_rotation = new_pose.orientation_xyzw
        if new_rotation != old_rotation and new_rotation != tuple(-x for x in old_rotation):
            raise ValueError(
                f"{context}: repeated FrameKey has conflicting node "
                f"{node_id!r}.orientation_xyzw"
            )


def _check_progress(previous: SceneFrame, current: SceneFrame, context: str) -> None:
    if current.key.epoch < previous.key.epoch:
        raise ValueError(f"{context}: epoch regressed from {previous.key.epoch}")
    if current.key.epoch == previous.key.epoch:
        for name, old, new in (
            ("tick", previous.key.tick, current.key.tick),
            ("revision", previous.key.revision, current.key.revision),
            ("time_s", previous.time_s, current.time_s),
        ):
            if new < old:
                raise ValueError(f"{context}: {name} regressed from {old!r} to {new!r}")


class AuthoredSceneSource:
    """Own a finite snapshot of a static scene and authored frames.

    ``frames`` must be a finite iterable; it is fully consumed and validated
    before construction returns. Each ``frames()`` call starts an independent
    traversal. The source never infers time from tick, composes visual offsets,
    loads assets, or advances a physical source.

    Within each world's epoch, tick, revision and time are nondecreasing. A
    larger epoch may restart all three; an older epoch is rejected. Repeated
    full keys require equal time and equal node poses (node ordering and a
    quaternion's overall sign do not change a pose). Repeating an old key after
    that world has progressed is still a regression; use a new traversal to
    revisit earlier frames. No tolerance hides conflicting source values.
    """

    __slots__ = ("_scene", "_frames")

    def __init__(self, scene: VisualScene, frames: Iterable[SceneFrame]) -> None:
        if not isinstance(scene, VisualScene):
            raise TypeError("scene must be VisualScene")
        try:
            incoming = iter(frames)
        except TypeError as error:
            raise TypeError("frames must be a finite iterable of SceneFrame") from error

        # G1 values are frozen. Copy the entire value graph so neither the
        # caller's sequence nor its scene/frame objects remain source storage.
        owned_scene, owned_frames = deepcopy((scene, tuple(incoming)))
        seen: dict[FrameKey, SceneFrame] = {}
        latest: dict[str, SceneFrame] = {}
        for index, frame in enumerate(owned_frames):
            validate_scene_frame(owned_scene, frame)
            context = f"frame[{index}] world {frame.key.world_id!r}, key={frame.key!r}"
            if frame.key.phase != "authored":
                raise ValueError(f"{context}: AuthoredSceneSource requires phase='authored'")

            repeated = seen.get(frame.key)
            if repeated is not None:
                _check_repeated_frame(repeated, frame, context)
            previous = latest.get(frame.key.world_id)
            if previous is not None:
                _check_progress(previous, frame, context)
            seen.setdefault(frame.key, frame)
            latest[frame.key.world_id] = frame

        self._scene = owned_scene
        self._frames = owned_frames

    def scene(self) -> VisualScene:
        """Return the owned immutable static manifest."""
        return self._scene

    def frames(self) -> Iterator[SceneFrame]:
        """Return an independent iterator over the complete stored sequence."""
        return iter(self._frames)
