"""Public M1 model, bounded candidate search and typed contact results."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from m1_cases import mixed_scene, saved_inputs, verify_mixed

import phymeo as ph


def test_public_entry():
    for name in (
        "CollisionKey",
        "CollisionShape",
        "CollisionModel",
        "CollisionMotion",
        "RigidTurn",
        "CollisionPolicy",
        "CollisionWorkspace",
        "detect",
    ):
        assert hasattr(ph, name), f"M1 public entry is missing {name}"


def test_mixed_rr_rd_dd_are_one_canonical_query(tmp_path):
    model, paths, keys, expected = mixed_scene(reverse=True)
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        verify_mixed(out, keys, expected)
        assert work.last_result is out
        (tmp_path / "host-result.json").write_text(
            json.dumps(
                dict(result=asdict(out), inputs=saved_inputs(model, paths, ph.CollisionPolicy())),
                allow_nan=False,
            )
        )
    with pytest.raises(RuntimeError, match="closed"):
        ph.detect(model, paths, ph.CollisionPolicy(), work)


def test_capacity_retry_and_invalid_input_preserve_previous_publication():
    model, paths, keys, expected = mixed_scene()
    with ph.CollisionWorkspace(model) as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        small = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=1), work)
        assert small.status == "capacity_exceeded" and not small.pairs and not small.bounds
        assert (
            small.required_candidates == previous.required_candidates
            and work.last_result is previous
        )
        invalid = dict(paths)
        invalid[keys[0]] = ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]], [[0, -1, 0], [0, 1, 0]])
        with pytest.raises(ValueError, match="rigid"):
            ph.detect(model, invalid, ph.CollisionPolicy(), work)
        assert work.last_result is previous
        with pytest.raises(ValueError, match="motion"):
            ph.detect(
                model,
                {**paths, ph.CollisionKey(9, 9, 9): paths[keys[0]]},
                ph.CollisionPolicy(),
                work,
            )
        with pytest.raises(ValueError, match="model"):
            ph.detect(ph.CollisionModel(model.shapes), paths, ph.CollisionPolicy(), work)
        recovered = ph.detect(
            model, paths, ph.CollisionPolicy(candidate_capacity=small.required_candidates), work
        )
        assert recovered == previous
        verify_mixed(recovered, keys, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_static_sphere_face_witness_and_material_binding(reverse):
    sphere = ph.CollisionKey(0, 2 if reverse else 1, 0)
    face = ph.CollisionKey(0, 1 if reverse else 2, 0)
    model = ph.CollisionModel(
        (
            ph.CollisionShape(face, "triangle", (18, 4, 25)),
            ph.CollisionShape(sphere, "sphere", (3,), 0.25),
        )
    )
    tri = ph.CollisionMotion([[-2, -1, 0], [2, -1, 0], [0, 2, 0]])
    with ph.CollisionWorkspace(model) as work:
        for z, status in ((0.2, "overlap"), (0.25, "touching"), (0.3, "separated")):
            out = ph.detect(
                model,
                {face: tri, sphere: ph.CollisionMotion([[0, 0, z]])},
                ph.CollisionPolicy(margin=0.1),
                work,
            )
            (row,) = out.pairs
            assert row.candidate and row.family == "RD" and row.status == status
            g = row.witness
            assert g.source == "host-fp64" and g.gap_m == pytest.approx(z - 0.25)
            np.testing.assert_allclose(g.normal, [0, 0, 1 if reverse else -1])
            np.testing.assert_allclose(
                g.first_point if reverse else g.second_point, [0, 0, 0], atol=1e-12
            )
            weights = g.first_weights if reverse else g.second_weights
            assert dict(weights) == pytest.approx({4: 1 / 3, 18: 1 / 3, 25: 1 / 3})
        bad = ph.CollisionMotion([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        (row,) = ph.detect(
            model, {face: bad, sphere: ph.CollisionMotion([[0, 0, 0]])}, ph.CollisionPolicy(), work
        ).pairs
        assert row.candidate and row.status == "unresolved" and row.witness is None


def test_capsule_overlap_tangent_and_coincident_normal():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        (ph.CollisionShape(a, "capsule", (0, 1), 0.1), ph.CollisionShape(b, "capsule", (0, 1), 0.1))
    )
    first = ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]])
    with ph.CollisionWorkspace(model) as work:
        for z, status in ((0.2, "touching"), (0.15, "overlap"), (0.0, "overlap")):
            other = ph.CollisionMotion([[0, -1, z], [0, 1, z]])
            (row,) = ph.detect(model, {a: first, b: other}, ph.CollisionPolicy(), work).pairs
            assert row.status == status and row.witness.gap_m == pytest.approx(z - 0.2)
            assert dict(row.witness.first_weights) == pytest.approx({0: 0.5, 1: 0.5})
            if z == 0:
                assert row.witness.normal is None and row.witness.first_point is None
            else:
                np.testing.assert_allclose(row.witness.normal, [0, 0, 1])


def test_outward_fp32_bounds_preserve_large_translation_and_subnormal():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(tuple(ph.CollisionShape(k, "sphere", (1,), 0.125) for k in (a, b)))
    source = np.array([[2**24 + 1.0, 1e-45, 0.0]])
    first = ph.CollisionMotion(source)
    source[:] = 0
    with pytest.raises(ValueError):
        first.start.setflags(write=True)
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(
            model,
            {a: first, b: ph.CollisionMotion([[2**24 + 1.25, 1e-45, 0.0]])},
            ph.CollisionPolicy(),
            work,
        )
        (row,) = out.pairs
        assert row.candidate and row.status == "touching"
        box = out.bounds[0]
        assert box.lower[0] <= 2**24 + 1 - 0.125 and box.upper[0] >= 2**24 + 1 + 0.125
        tiny = ph.CollisionModel([ph.CollisionShape(a, "triangle", (0, 1, 2))])
        with ph.CollisionWorkspace(tiny) as other:
            data = ph.CollisionMotion([[1e-45, -1e-45, 0]] * 3)
            (box,) = ph.detect(tiny, {a: data}, ph.CollisionPolicy(), other).bounds
            assert 0 <= box.lower[0] <= 1e-45 <= box.upper[0]
            assert box.lower[1] <= -1e-45 <= box.upper[1] <= 0
        with pytest.raises(ValueError, match="FP32"):
            ph.detect(
                model,
                {a: ph.CollisionMotion([[1e100, 0, 0]]), b: first},
                ph.CollisionPolicy(),
                work,
            )
        assert work.last_result is out


def test_models_are_bounded_owned_and_versioned():
    k = ph.CollisionKey(0, 4, 0)
    with pytest.raises(ValueError):
        ph.CollisionKey(True, 0, 0)
    with pytest.raises(ValueError):
        ph.CollisionShape(k, "triangle", (7, 7, 9))
    with pytest.raises(ValueError):
        ph.RigidTurn([0, 0, 0], [0, 0, 0], [0, 0, 0], 1.0)
    with pytest.raises(ValueError):
        ph.CollisionMotion([[np.nan, 0, 0]])
    s = ph.CollisionShape(k, "tetrahedron", (0, 1, 2, 3), revision=7)
    with pytest.raises(ValueError):
        ph.CollisionModel([s, s])
    with pytest.raises(ValueError):
        ph.CollisionModel([s], [(k, ph.CollisionKey(0, 9, 9))])
    with pytest.raises(ValueError):
        ph.CollisionPolicy(candidate_capacity=-1)
    empty = ph.CollisionModel(())
    with ph.CollisionWorkspace(empty) as work:
        result = ph.detect(empty, {}, ph.CollisionPolicy(candidate_capacity=0), work)
        assert result.status == "ok" and not result.pairs and not result.bounds
    model = ph.CollisionModel([s])
    with ph.CollisionWorkspace(model) as work:
        (box,) = ph.detect(
            model, {k: ph.CollisionMotion(np.eye(4, 3))}, ph.CollisionPolicy(), work
        ).bounds
        assert box.key == k and box.revision == 7


def test_crossing_nearly_parallel_capsules_are_not_reported_separated():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel([ph.CollisionShape(k, "capsule", (0, 1), 1e-14) for k in (a, b)])
    # The axes intersect at (1/4, 0, 0), far inside both closed segments.
    paths = {
        a: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0]]),
        b: ph.CollisionMotion([[-0.25, -1e-13, 0], [0.75, 1e-13, 0]]),
    }
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.candidate and row.status == "overlap"
        assert row.witness.source == "host-rational-axis"
        assert row.witness.gap_m == -2e-14 and row.witness.normal is None
        assert dict(row.witness.first_weights) == {0: 0.375, 1: 0.625}
        assert dict(row.witness.second_weights) == {0: 0.5, 1: 0.5}


@pytest.mark.parametrize("sign", [-1, 1])
def test_full_turn_and_moving_pivot_keep_mid_arc_candidate(sign):
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel([ph.CollisionShape(k, "sphere", (0,), 0.0625) for k in (a, b)])
    paths = {
        a: ph.CollisionMotion(
            [[2, 0, 0]], turn=ph.RigidTurn([0, 0, 0], [1, 0, 0], [0, 0, 1], sign * 2 * np.pi)
        ),
        b: ph.CollisionMotion([[0.25, 2 * sign, 0]]),
    }
    # At t=1/4 the two centers coincide. Both start and end miss this target.
    with ph.CollisionWorkspace(model) as work:
        out = ph.detect(model, paths, ph.CollisionPolicy(), work)
        (row,) = out.pairs
        assert row.candidate and row.status == "unsupported" and row.witness is None
        assert out.bounds[0].lower[1] <= 2 * sign <= out.bounds[0].upper[1]


def test_thick_face_keeps_candidate_without_a_zero_thickness_witness():
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(a, "sphere", (0,), 0.125),
            ph.CollisionShape(b, "triangle", (0, 1, 2), 0.125),
        ]
    )
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(
            model,
            {
                a: ph.CollisionMotion([[0, 0, 0.2]]),
                b: ph.CollisionMotion([[-1, -1, 0], [1, -1, 0], [0, 1, 0]]),
            },
            ph.CollisionPolicy(),
            work,
        ).pairs
        assert row.candidate and row.status == "unsupported" and row.witness is None


@pytest.mark.parametrize("endpoint", ["start", "end"])
def test_shared_soft_vertex_requires_one_affine_path(endpoint):
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 1, 1)
    model = ph.CollisionModel(
        [ph.CollisionShape(a, "triangle", (3, 4, 5)), ph.CollisionShape(b, "triangle", (4, 3, 6))]
    )
    poses = {
        a: ph.CollisionMotion([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        b: ph.CollisionMotion([[1, 0, 0], [0, 0, 0], [0, -1, 0]]),
    }
    with ph.CollisionWorkspace(model) as work:
        previous = ph.detect(model, poses, ph.CollisionPolicy(), work)
        assert previous.pairs[0].candidate
        start = poses[b].start.copy()
        end = start.copy()
        moved = start if endpoint == "start" else end
        moved[1, 2] = 0.125
        with pytest.raises(ValueError, match="shared material vertex"):
            ph.detect(
                model, {**poses, b: ph.CollisionMotion(start, end)}, ph.CollisionPolicy(), work
            )
        assert work.last_result is previous
        assert ph.detect(model, poses, ph.CollisionPolicy(), work) == previous
