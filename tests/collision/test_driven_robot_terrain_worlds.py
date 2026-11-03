"""One physical rail step supplies a tool path to two co-located collision worlds."""

import numpy as np
import pytest

import phymeo as ph


def tool_path(upper):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    limits = ph.JointLimits(np.array([-1.0]), np.array([upper]), 2048.0)
    drive = ph.DrivenRobotProgram(
        model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.zeros(1), limits=limits
    )
    entry = ph.RobotState(np.array([1 / 1024]), np.zeros(1))
    h = 1 / 32
    with drive.stepper(ph.StepSettings(substeps=1)) as stepper:
        result = stepper.step(drive.initial_state(entry), drive.inputs(np.zeros(1)), h)
    assert result.report.accepted, result.report.reason
    end = result.state.domains["robot"]
    force = -2.0 if upper == 0 else 0.0
    diagnostic = result.report.substeps[0].diagnostics
    np.testing.assert_allclose(diagnostic["stop_effort"], [force], rtol=0, atol=1e-12)
    np.testing.assert_allclose(diagnostic["stop_impulse"], [h * force], rtol=0, atol=1e-12)
    np.testing.assert_allclose(end.v, [h * force / 2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(end.q, entry.q + h * end.v, rtol=0, atol=1e-12)
    local_center = np.array([[3 / 16, 1 / 16, 255 / 2048]])
    centers = [
        ph.RobotPointPort(model, state, np.array([0]), local_center).positions()
        for state in (entry, end)
    ]
    return ph.CollisionMotion(*centers)


def check_worlds(backend, cache_root):
    grid = ph.Heightfield([[0, 0], [0, 0]], (1, 1))
    vertex_ids = (17, 29, 43, 61)
    shapes, paths = [], {}
    for world, upper in ((9, 0.0), (3, 1.0)):
        land = ph.CollisionKey(world, 0, 0)
        tool = ph.CollisionKey(world, 1, 0)
        shapes.extend(
            (
                ph.CollisionShape(tool, "sphere", (2,), 1 / 8),
                ph.CollisionShape(land, "heightfield", vertex_ids, heightfield=grid),
            )
        )
        paths[tool] = tool_path(upper)
        paths[land] = ph.CollisionMotion(grid.world_vertices())
    model = ph.CollisionModel(tuple(shapes))
    with ph.CollisionWorkspace(model, backend, cache_root=cache_root) as workspace:
        result = ph.detect(model, paths, ph.CollisionPolicy(), workspace)
        stats = workspace.stats()
    assert result.status == "ok" and result.required_candidates == 1
    cross = [p for p in result.pairs if p.first.world != p.second.world]
    assert len(cross) == 4 and all(p.reason == "different_world" and not p.candidate for p in cross)
    by_world = {p.first.world: p for p in result.pairs if p.first.world == p.second.world}
    assert by_world[3].status == "separated" and not by_world[3].candidate
    hit = by_world[9]
    assert hit.status == "hit" and hit.sweep.first_t == (0.5, 0.5)
    assert hit.sweep.start_gap_m == pytest.approx(1 / 2048, abs=1e-12)
    assert hit.sweep.end_gap_m == pytest.approx(-1 / 2048, abs=1e-12)
    entry = hit.heightfield.first_contact
    vertices = dict(zip(vertex_ids, grid.world_vertices(), strict=True))
    face = np.array([vertices[key] for key in entry.face.vertex_ids])
    contact = (3 / 16, 1 / 16, 0.0)
    for point in (
        entry.face.barycentric @ face,
        entry.witness.first_point,
        entry.witness.second_point,
    ):
        np.testing.assert_allclose(point, contact, rtol=0, atol=1e-12)
    np.testing.assert_allclose(entry.witness.normal, (0, 0, 1), rtol=0, atol=1e-12)
    assert hit.heightfield.last_contact.interval == (1.0, 1.0)
    if backend != "host":
        assert stats["heightfield_completed_launches"] > 0
    if backend == "cuda":
        assert stats["heightfield_downloads"] > 0 and stats["heightfield_synchronizations"] > 0


def test_driven_tool_terrain_world_isolation_on_host(tmp_path):
    check_worlds("host", tmp_path)


@pytest.mark.tyx
def test_driven_tool_terrain_world_isolation_on_selected_device(tyx_mode, tmp_path):
    check_worlds(tyx_mode, tmp_path)
