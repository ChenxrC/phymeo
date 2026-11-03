"""G3 exercises rigid, deformable and mixed narrow phases in one real world batch."""

import json

import numpy as np
import pytest

import phymeo as ph


def fixture_worlds():
    worlds = {}
    paths = {}
    for world in (90, 31, 7, 53):
        a, b = ph.CollisionKey(world, 0, 0), ph.CollisionKey(world, 1, 0)
        if world == 90:
            shapes, motion = (), {}
        elif world == 31:
            grid = ph.Heightfield([[0, 1, 0]] * 3, (1, 1), origin=(-1, -1, 0))
            shapes = (
                ph.CollisionShape(a, "heightfield", tuple(range(9)), heightfield=grid),
                ph.CollisionShape(b, "sphere", (0,), 5 / 16),
            )
            motion = {
                a: ph.CollisionMotion(grid.world_vertices()),
                b: ph.CollisionMotion([[-0.75, 0.125, 1.25]], [[0.75, 0.125, 1.25]]),
            }
        elif world == 7:
            shapes = (
                ph.CollisionShape(a, "sphere", (0,), 1 / 4),
                ph.CollisionShape(b, "triangle", (11, 7, 3)),
            )
            motion = {
                a: ph.CollisionMotion([[0, 0, 3 / 16]]),
                b: ph.CollisionMotion([[-1, -1, 0], [1, -1, 0], [0, 2, 0]]),
            }
        else:
            shapes = (
                ph.CollisionShape(a, "triangle", (11, 7, 3)),
                ph.CollisionShape(b, "triangle", (31, 41, 53)),
            )
            motion = {
                a: ph.CollisionMotion([[-1, 0, 0], [1, 0, 0], [0, 0, -0.25]]),
                b: ph.CollisionMotion(
                    [[0, -1, 0.5], [0, 1, 0.5], [0, 0, 1.5]],
                    [[0, -1, -1.5], [0, 1, -1.5], [0, 0, -0.5]],
                ),
            }
        worlds[world] = ph.CollisionModel(shapes[::-1])
        paths[world] = motion
    return worlds, paths


def measure(backend, tmp_path):
    worlds, paths = fixture_worlds()
    policies = {k: ph.CollisionPolicy(margin=0.5 if k == 31 else 0) for k in worlds}
    single = {}
    for k, model in worlds.items():
        with ph.CollisionWorkspace(model, backend, cache_root=tmp_path) as workspace:
            single[k] = ph.detect(model, paths[k], policies[k], workspace)

    with ph.WorldCollisionWorkspace(worlds.items(), backend, cache_root=tmp_path) as workspace:
        base = workspace.detect(paths, policies)
        assert list(base) == list(worlds) and base == single
        assert base[90].pairs == () and base[90].bounds == ()
        rr, rd, dd = (base[k].pairs[0] for k in (31, 7, 53))
        assert (rr.family, rd.family, dd.family) == ("RR", "RD", "DD")
        assert rr.sweep.first_t == (3 / 8, 3 / 8)
        assert rr.heightfield.last_contact.interval == (5 / 8, 5 / 8)
        assert rd.status == "overlap" and rd.witness.gap_m == -1 / 16
        assert rd.witness.first_point == (0.0, 0.0, -1 / 16)
        assert rd.witness.second_point == (0.0, 0.0, 0.0)
        assert dict(rd.witness.second_weights) == pytest.approx({11: 1 / 3, 7: 1 / 3, 3: 1 / 3})
        assert all(
            np.linalg.norm(p - np.array([0, 0, 3 / 16])) > 1 / 4
            for p in paths[7][rd.second].start
        )
        assert dd.sweep.first_t == (0.25, 0.25) and dd.sweep.end_gap_m == 0.25
        (point,) = dd.sweep.witness.intersection_points
        assert point.point == (0.0, 0.0, 0.0)
        assert dict(point.first_weights) == {11: 0.5, 7: 0.5, 3: 0.0}
        assert dict(point.second_weights) == {31: 0.5, 41: 0.5, 53: 0.0}

        raised = ph.CollisionMotion([[-0.75, 0.125, 1.5]], [[0.75, 0.125, 1.5]])
        near_paths = paths | {31: paths[31] | {rr.second: raised}}
        attempt = workspace.detect(near_paths, policies | {7: ph.CollisionPolicy(candidate_capacity=0)})
        assert attempt[31].pairs[0].sweep.first_t is None
        assert attempt[31].pairs[0].status == "separated"
        assert attempt[7].status == "capacity_exceeded" and attempt[7].required_candidates == 1
        assert attempt[7].pairs == () and workspace.last_result(7) is base[7]
        assert workspace.last_result(31) is attempt[31]
        assert attempt[53] == base[53] and attempt[90] == base[90]
        assert workspace.detect(paths, policies) == base
        stats = workspace.stats()
        assert stats["host_pair_table_slots"] == 3
        if backend == "cuda":
            assert stats["device_pair_table_slots"] == 3
            assert stats["heightfield_completed_launches"] == 3
            assert stats["triangle_completed_launches"] > 0 and stats["synchronizations"] > 0
        evidence = {
            "backend": backend,
            "registration": list(worlds),
            "families": [rr.family, rd.family, dd.family],
            "rigid_first": rr.sweep.first_t,
            "rigid_last": rr.heightfield.last_contact.interval,
            "rigid_deformable_gap_m": rd.witness.gap_m,
            "cloth_weights": rd.witness.second_weights,
            "cloth_cloth_first": dd.sweep.first_t,
            "cloth_cloth_point_m": point.point,
            "local_refusal": attempt[7].status,
            "pair_slots": stats["host_pair_table_slots"],
            "device_pair_slots": stats["device_pair_table_slots"],
            "completed_launches": stats["completed_launches"],
        }
        (tmp_path / "mixed-world-queries.json").write_text(json.dumps(evidence), encoding="utf-8")


def test_mixed_world_geometry_and_local_recovery_on_host(tmp_path):
    measure("host", tmp_path)


@pytest.mark.tyx
def test_mixed_world_geometry_and_local_recovery_on_device(tyx_mode, tmp_path):
    measure(tyx_mode, tmp_path)
