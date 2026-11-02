"""A folded connected sheet uses one particle state for both sides of a contact."""

from contextlib import ExitStack
from itertools import combinations

import numpy as np
import pytest
from adjacent_triangle_cases import hinge_path

import phymeo as ph


def model_for(points, velocities, faces, *, operator=None):
    dtype = points.dtype
    shapes = tuple(
        ph.CollisionShape(ph.CollisionKey(0, 3, i), "triangle", nodes)
        for i, nodes in enumerate(faces)
    )
    law = ph.NormalLayer(0.25, 16.0)
    system = ph.CollisionLoadSystem(
        ph.CollisionModel(shapes),
        {s.key: ph.ParticleAttachment(s.vertex_ids) for s in shapes},
        {(a.key, b.key): law for a, b in combinations(shapes, 2)},
        layer_operator=operator,
    )
    empty = np.empty((0, 3), dtype=dtype)
    state = ph.CoupledState(
        ph.ParticleState(points, velocities),
        ph.RigidBodyState(empty, np.empty((0, 4), dtype=dtype), empty, empty),
    )
    models = (
        ph.ParticleModel(np.ones(len(points), dtype=dtype), np.zeros(3, dtype=dtype)),
        ph.RigidBodyModel(np.empty(0, dtype=dtype), empty, np.zeros(3, dtype=dtype)),
    )
    return system, state, models


def folded_strip(mode="host", cache=None):
    dtype = np.float64 if mode == "host" else np.float32
    bottom = np.array([[-1, -1, 0], [1, -1, 0], [0, 2, 0]], dtype=dtype)
    points = np.concatenate((bottom, bottom + np.array([0, 0, 0.125], dtype=dtype)))
    faces = ((0, 1, 2), (3, 5, 4), (1, 0, 3), (1, 3, 4))
    edges = sorted({tuple(sorted((i, j))) for face in faces for i, j in combinations(face, 2)})
    edges = np.array(edges, dtype=np.int64)
    rest = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    material = ph.DistanceConstraints(edges, rest, np.ones(len(edges), dtype=dtype))
    with ExitStack() as stack:
        layer = (
            None
            if mode == "host"
            else stack.enter_context(ph.TyxNormalLayer(mode, cache_root=cache))
        )
        system, state, models = model_for(points, np.zeros_like(points), faces, operator=layer)
        solver = (
            ph.XPBDSolver(material, tolerance=5e-5)
            if mode == "host"
            else stack.enter_context(
                ph.TyxXPBDSolver(material, mode=mode, cache_root=cache, tolerance=5e-5)
            )
        )
        workspace = stack.enter_context(ph.CollisionWorkspace(system.model, mode, cache_root=cache))
        frame = system.step(
            state,
            ph.CoupledInputs.zeros(6, 0, dtype=dtype),
            models,
            (solver, None),
            1 / 64,
            workspace,
        )
        stats = workspace.stats()
        if layer is not None:
            stats["layer"] = layer.stats()
        return frame, stats


def verify_folded_strip(frame, atol):
    (contact,) = frame.loads.contacts
    assert (contact.first.shape, contact.second.shape) == (0, 1)
    assert contact.first.body == contact.second.body == 3 and len(contact.sites) == 3
    assert contact.force_n == pytest.approx(2.0, abs=atol)
    forces = frame.loads.inputs.particles.force
    np.testing.assert_allclose(forces[:, 2], [-2 / 3] * 3 + [2 / 3] * 3, atol=atol)
    np.testing.assert_allclose(forces.sum(0), 0, atol=atol)
    np.testing.assert_allclose(
        np.cross(frame.entry.particles.position, forces).sum(0), 0, atol=atol
    )
    v = frame.state.particles.velocity
    assert np.all(v[:3, 2] < 0) and np.all(v[3:, 2] > 0)
    np.testing.assert_allclose(v.sum(0), 0, atol=atol)
    assert [p.status for p in frame.path.pairs].count("adjacent") == 5


def test_connected_fold_transfers_one_pressure_to_its_two_remote_faces():
    frame, _ = folded_strip()
    verify_folded_strip(frame, 1e-10)


@pytest.mark.parametrize("remote", [False, True])
def test_a_shared_hinge_keeps_its_material_edge_and_rejects_an_otherwise_hidden_crossing(remote):
    original, paths = hinge_path(remote)
    faces = tuple(s.vertex_ids for s in original.shapes)
    start, end = np.zeros((4, 3)), np.zeros((4, 3))
    for shape in original.shapes:
        ids = list(shape.vertex_ids)
        start[ids] = paths[shape.key].start
        end[ids] = paths[shape.key].end
    h = 1 / 64
    system, state, models = model_for(start, (end - start) / h, faces)
    with ph.CollisionWorkspace(system.model) as work:
        external = ph.CoupledInputs.zeros(4, 0)
        assert system.sample(state, work).contacts == ()
        prior = work.last_result
        if remote:
            with pytest.raises(ph.SolverFailure, match="core path rejected"):
                system.step(state, external, models, (ph.XPBDSolver(), None), h, work)
            assert work.last_result is prior and system.last_frame is None
            np.testing.assert_array_equal(state.particles.position, start)
        else:
            out = system.step(state, external, models, (ph.XPBDSolver(), None), h, work)
            assert out.path.pairs[0].status == "adjacent" and out.loads.contacts == ()
            np.testing.assert_array_equal(out.state.particles.position, end)
