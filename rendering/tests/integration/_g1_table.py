"""Authored metre-scale samples used by the G1 scene and camera contracts."""

from math import sqrt
from pathlib import Path

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import (
    DirectionalLight,
    FrameKey,
    Instance,
    NodePose,
    Pose,
    SceneFrame,
    VisualScene,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "g1"
CAMERA_MOUNT = Pose((0.125, 0.0, 0.0))
ALUMINIUM_TOP = (0.0, 0.0, 0.0625)


def table_scene() -> VisualScene:
    return VisualScene(
        key="g1-table-v1",
        meshes=(
            Mesh("table", (FIXTURES / "table.gltf").as_uri()),
            Mesh("block", (FIXTURES / "block.gltf").as_uri()),
        ),
        materials=(
            Material("aluminium", (0.7, 0.7, 0.7), 1.0, 0.2),
            Material("plastic", (0.2, 0.3, 0.4), 0.0, 0.5),
            Material("rubber", (0.03, 0.03, 0.03), 0.0, 0.9),
        ),
        instances=(
            Instance(1, "table", "plastic", "base", Pose((0.0, 0.0, -0.0625))),
            Instance(11, "block", "aluminium", "base", Pose((-0.25, 0.0, 0.0))),
            Instance(12, "block", "plastic", "base"),
            Instance(13, "block", "rubber", "base", Pose((0.25, 0.0, 0.0))),
            Instance(21, "block", "aluminium", "base", Pose((0.3125, 0.03125, 0.125))),
        ),
        lights=(DirectionalLight((0.0, 0.0, -1.0), (1.0, 1.0, 1.0)),),
    )


def table_frames() -> tuple[SceneFrame, SceneFrame]:
    sin_45 = sqrt(0.5)
    wrist_poses = (
        Pose((-0.125, 0.0, 1.0625)),
        Pose((0.125, -0.125, 1.0625), (0.0, 0.0, sin_45, sin_45)),
    )
    first, second = (
        SceneFrame(
            FrameKey("g1-table-v1", "table", 0, tick, tick, "authored"),
            tick / 50.0,
            (NodePose("base", Pose()), NodePose("wrist", wrist)),
        )
        for tick, wrist in enumerate(wrist_poses)
    )
    return first, second
