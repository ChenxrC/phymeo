"""Sparse pair-contact projection without publishing full negative diagnostic rows."""

from dataclasses import replace
from importlib import util

import numpy as np
import pytest
from g4_geometry_cases import copies, frozen
from test_mixed_world_queries import fixture_worlds

import phymeo as ph


def api():
    assert util.find_spec("phymeo.geometry.compact_contacts") is not None, (
        "missing compact world contact buffer"
    )
    from phymeo.geometry import compact_contacts

    return compact_contacts


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("name,n", [("sparse65", 0), ("dense34", 561)])
def test_omits_provably_negative_pair_objects_and_keeps_dense_contacts(
    name, n, backend, tmp_path, monkeypatch
):
    from phymeo.geometry import collision_pipeline as core

    m = api()
    source, frames = frozen(name)
    groups, paths = copies(source, frames, (37,))
    rules = {37: ph.CollisionPolicy()}
    with (
        ph.WorldCollisionWorkspace(groups, backend, cache_root=tmp_path) as full,
        m.CompactWorldContactWorkspace(groups, backend, cache_root=tmp_path) as compact,
    ):
        complete = full.detect(paths[0], rules)[37]
        assert len(complete.pairs) == (2080 if n == 0 else 561)
        original = core.CollisionPair.__init__
        calls = []

        def counted(self, *args, **kwargs):
            calls.append(1)
            original(self, *args, **kwargs)

        monkeypatch.setattr(core.CollisionPair, "__init__", counted)
        result = compact.detect(paths[0], rules)[37]
        assert result == m.compact_contacts(complete) and compact.last_result(37) is result
        assert len(calls) == n
        assert (
            result.status == "ok"
            and result.required_contacts == n
            and result.required_candidates == n
        )
        assert len(result.contacts) == n and result.diagnostics == ()
        assert tuple((p.first, p.second) for p in result.contacts) == tuple(
            sorted((p.first, p.second) for p in result.contacts)
        )
        for pair in result.contacts:
            assert pair.first.world == pair.second.world == 37 and pair.witness.normal == (
                0.0,
                1.0,
                0.0,
            )
            assert pair.witness.first_weights == pair.witness.second_weights == ((0, 1.0),)
        assert compact.stats()["compact_selected_slots"] == n


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_rr_rd_dd_and_the_empty_world_match_the_full_records(backend, tmp_path):
    m = api()
    worlds, paths = fixture_worlds()
    groups = list(worlds.items())
    rules = {k: ph.CollisionPolicy(margin=0.5 if k == 31 else 0) for k in worlds}
    with (
        ph.WorldCollisionWorkspace(groups, backend, cache_root=tmp_path) as full,
        m.CompactWorldContactWorkspace(groups, backend, cache_root=tmp_path) as compact,
    ):
        base = full.detect(paths, rules)
        actual = compact.detect(paths, rules)
        assert tuple(actual) == tuple(worlds)
        for world in worlds:
            assert actual[world] == m.compact_contacts(base[world])
        (surface,) = actual[7].contacts
        assert surface.family == "RD" and surface.witness.gap_m == pytest.approx(-1 / 16)
        assert len(surface.witness.second_weights) == 3
        np.testing.assert_allclose(
            sorted(w for _, w in surface.witness.second_weights), [1 / 3] * 3
        )
        (edge,) = actual[53].contacts
        assert edge.family == "DD" and edge.sweep.first_t[0] <= 1 / 4 <= edge.sweep.first_t[1]
        assert edge.sweep.witness.intersection_points
        assert actual[90].contacts == actual[90].diagnostics == ()
        assert actual[31].contacts[0].heightfield is not None


def problem_world(key, kind, filtering=None):
    from closed_mesh_cases import authored

    a, b = ph.CollisionKey(key, 2, 0), ph.CollisionKey(key, 7, 0)
    if kind == "folded":
        model, paths, _ = authored("base", world=key)
        sa, sb = model.shapes
        xy = paths[sa.key].start.copy()
        xy[10] = [-1.75, 1.5, 0.25]
        paths = {a: ph.CollisionMotion(xy), b: ph.CollisionMotion(paths[sb.key].start + [40, 0, 0])}
    elif kind == "flat":
        sa = ph.CollisionShape(a, "hull", (10, 20, 30, 40))
        sb = ph.CollisionShape(b, "triangle", (1, 2, 3))
        paths = {
            a: ph.CollisionMotion([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
            b: ph.CollisionMotion([[40, 0, 0], [41, 0, 0], [40, 1, 0]]),
        }
    elif kind == "unknown":
        sa = ph.CollisionShape(a, "cylinder", (47,), curved=ph.CurvedSolid("cylinder", (0.5, 0.25)))
        sb = ph.CollisionShape(b, "sphere", (91,), 0.0625)
        paths = {
            a: ph.CollisionMotion(
                [[-2, 0, 0]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), np.pi)
            ),
            b: ph.CollisionMotion([[0, 0, 0]]),
        }
    else:
        raise ValueError(kind)
    if filtering == "mask":
        sa = replace(sa, mask=0)
    return ph.CollisionModel((sa, sb), ((a, b),) if filtering == "exclude" else ()), paths


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_invalid_far_geometry_and_original_filter_priority(backend, tmp_path):
    import json
    from dataclasses import asdict

    m = api()
    specs = {
        21: ("folded", None),
        18: ("flat", "exclude"),
        26: ("unknown", None),
        5: ("flat", None),
        32: ("folded", "mask"),
        9: ("folded", "exclude"),
        4: ("flat", "mask"),
    }
    scenes = {k: problem_world(k, *s) for k, s in specs.items()}
    groups = [(k, model) for k, (model, _) in scenes.items()]
    paths = {k: p for k, (_, p) in scenes.items()}
    rules = dict.fromkeys(scenes, ph.CollisionPolicy())
    with (
        ph.WorldCollisionWorkspace(groups, backend, cache_root=tmp_path) as full,
        m.CompactWorldContactWorkspace(groups, backend, cache_root=tmp_path) as sparse,
    ):
        before = full.detect(paths, rules)
        current = sparse.detect(paths, rules)
        assert tuple(current) == tuple(specs)
        for k, (kind, filtering) in specs.items():
            assert current[k] == m.compact_contacts(before[k]) and current[k].contacts == ()
            if filtering:
                assert before[k].pairs[0].status == "filtered" and current[k].diagnostics == ()
            else:
                (d,) = current[k].diagnostics
                if kind == "unknown":
                    assert d.status == "unsupported" and d.candidate and d.sweep is None
                else:
                    assert (
                        d.status == "invalid_geometry"
                        and not d.candidate
                        and current[k].required_candidates == 0
                    )
        assert sparse.stats()["compact_validation_slots"] == 2
        (tmp_path / "compact-invalid.json").write_text(
            json.dumps({k: asdict(v) for k, v in current.items()}, indent=2), encoding="utf8"
        )


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_candidate_contact_and_issue_overflow_are_independent_and_transactional(backend, tmp_path):
    import json
    from dataclasses import asdict

    m = api()
    model, source = frozen("dense34")
    g, p = copies(model, source, (33,))
    other, failpath = problem_world(9, "flat")
    groups = [*g, (9, other)]
    paths = {**p[0], 9: failpath}
    policies = dict.fromkeys(paths, ph.CollisionPolicy())
    with (
        m.CompactWorldContactWorkspace(groups, backend, cache_root=tmp_path) as work,
        ph.WorldCollisionWorkspace(groups, backend, cache_root=tmp_path) as full,
    ):
        initial = work.detect(paths, policies)
        allrows = full.detect(paths, policies)
        limits = {33: m.ContactBufferLimits(contacts=560), 9: m.ContactBufferLimits(diagnostics=0)}
        bad = work.detect(paths, policies, limits_by_world=limits)
        assert bad[33] == m.compact_contacts(allrows[33], limits[33]) and bad[
            9
        ] == m.compact_contacts(allrows[9], limits[9])
        assert (
            bad[33].overflow == ("contacts",)
            and bad[33].required_contacts == 561
            and bad[33].contacts == ()
        )
        assert (
            bad[9].overflow == ("diagnostics",)
            and bad[9].required_diagnostics == 1
            and bad[9].diagnostics == ()
        )
        assert all(work.last_result(k) is v for k, v in initial.items())
        attempted = work.detect(paths, policies | {33: ph.CollisionPolicy(candidate_capacity=560)})
        assert (
            attempted[33].overflow == ("candidates",) and attempted[33].required_candidates == 561
        )
        assert attempted[33].required_contacts is None and work.last_result(33) is initial[33]
        assert work.last_result(9) is attempted[9]
        assert work.detect(paths, policies) == initial
        (tmp_path / "compact-overflow.json").write_text(
            json.dumps(
                dict(
                    contact=asdict(bad[33]),
                    diagnostic=asdict(bad[9]),
                    candidates=asdict(attempted[33]),
                ),
                indent=2,
            ),
            encoding="utf8",
        )


def test_future_status_and_incomplete_positive_contact_remain_visible():
    m = api()
    model, frames = frozen("dense34")
    g, p = copies(model, frames, (5,))
    with ph.WorldCollisionWorkspace(g) as work:
        full = work.detect(p[0], {5: ph.CollisionPolicy()})[5]
    one = full.pairs[0]
    unknown = replace(one, status="new_narrow_status", reason="future_contract")
    partial = replace(one, status="overlap", reason="support_depth_unresolved", witness=None)
    compact = m.compact_contacts(replace(full, pairs=(unknown, partial)))
    assert compact.contacts == (partial,) and compact.diagnostics == (unknown, partial)


def test_cuda_fault_does_not_commit_partial_compact_contacts(tmp_path, monkeypatch):
    m = api()
    worlds, paths = fixture_worlds()
    policies = dict.fromkeys(worlds, ph.CollisionPolicy())
    with m.CompactWorldContactWorkspace(worlds.items(), "cuda", cache_root=tmp_path) as work:
        old = work.detect(paths, policies)
        engine = work._work._engine

        def failed(*args):
            raise RuntimeError("injected compact-classification readback")

        monkeypatch.setattr(engine, "_read", failed)
        with pytest.raises(RuntimeError, match="compact-classification"):
            work.detect(paths, policies)
        assert all(work.last_result(k) is v for k, v in old.items())
        assert work.detect(paths, policies) == old and work._work._engine is not engine


def test_no_pair_registry_and_positive_gap_keep_their_distinct_meanings():
    m = api()
    with m.CompactWorldContactWorkspace([(19, ph.CollisionModel(()))]) as work:
        out = work.detect({19: {}}, {19: ph.CollisionPolicy()})[19]
        assert isinstance(out, m.CompactContactBuffer) and out.contacts == out.diagnostics == ()
        assert out.required_candidates == 0
    a, b = ph.CollisionKey(9, 0, 0), ph.CollisionKey(9, 1, 0)
    model = ph.CollisionModel(
        (ph.CollisionShape(a, "sphere", (0,), 1 / 8), ph.CollisionShape(b, "sphere", (0,), 1 / 8))
    )
    paths = {9: {a: ph.CollisionMotion([[0, 0, 0]]), b: ph.CollisionMotion([[5 / 16, 0, 0]])}}
    policies = {9: ph.CollisionPolicy(margin=1 / 8)}
    with (
        ph.WorldCollisionWorkspace([(9, model)]) as full,
        m.CompactWorldContactWorkspace([(9, model)]) as work,
    ):
        row = full.detect(paths, policies)[9]
        out = work.detect(paths, policies)[9]
        assert row.pairs[0].witness.gap_m == 1 / 16
        assert (
            out == m.compact_contacts(row) and out.required_candidates == 1 and out.contacts == ()
        )


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_noncontiguous_candidate_slots_across_mixed_worlds_keep_material_owners(backend, tmp_path):
    from compact_contact_cases import check_spheres, sphere_case

    m = api()
    groups = {48: sphere_case("sparse65-one", 48), 7: sphere_case("dense34-split", 7)}
    models = [(key, model) for key, (model, _) in groups.items()]
    fields = {key: paths for key, (_, paths) in groups.items()}
    rules = dict.fromkeys(groups, ph.CollisionPolicy())
    with (
        ph.WorldCollisionWorkspace(models, backend, cache_root=tmp_path) as full,
        m.CompactWorldContactWorkspace(models, backend, cache_root=tmp_path) as sparse,
    ):
        allrows = full.detect(fields, rules)
        results = sparse.detect(fields, rules)
        for key in groups:
            assert results[key] == m.compact_contacts(allrows[key])
            check_spheres(*groups[key], results[key])
        (hit,) = results[48].contacts
        assert (hit.first.body, hit.second.body) == (0, 64) and hit.witness.gap_m == -1 / 16
        assert results[7].required_contacts == 272
        assert sparse.stats()["compact_selected_slots"] == 273
        assert sparse.stats()["compact_skipped_slots"] == 2080 + 561 - 273


def test_tetrahedral_pair_occupancy_and_far_invalid_cells_survive_projection(tmp_path):
    from tetrahedral_solid_cases import authored

    m = api()
    valid, va = authored("base", world=12)
    invalid, vb = authored("collapsed", world=4)
    side = invalid.shapes[1].key
    other = vb[side]
    vb = dict(vb)
    vb[side] = ph.CollisionMotion(
        other.start + [40, 0, 0], None if other.end is None else other.end + [40, 0, 0]
    )
    worlds = ((12, valid), (4, invalid))
    paths = {12: va, 4: vb}
    rules = dict.fromkeys(paths, ph.CollisionPolicy())
    with (
        ph.WorldCollisionWorkspace(worlds, cache_root=tmp_path) as full,
        m.CompactWorldContactWorkspace(worlds, cache_root=tmp_path) as compact,
    ):
        oracle = full.detect(paths, rules)
        out = compact.detect(paths, rules)
        assert all(out[k] == m.compact_contacts(v) for k, v in oracle.items())
        (one,) = out[12].contacts
        assert one.solid is not None and "contains" in one.solid.start.relation
        assert len(oracle[12].solid_pairs) == 1
        (issue,) = out[4].diagnostics
        assert (
            not issue.candidate
            and issue.status == "invalid_geometry"
            and out[4].required_candidates == 0
        )


def test_singleton_bound_and_observed_nonunique_normal_have_different_buffers():
    m = api()
    a, b = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    s1, s2 = (
        ph.CollisionShape(a, "sphere", (0,), 1 / 8),
        ph.CollisionShape(b, "sphere", (0,), 1 / 8),
    )
    same = ph.CollisionMotion([[0, 0, 0]])
    with m.CompactWorldContactWorkspace([(0, ph.CollisionModel((s1,)))]) as work:
        empty = work.detect({0: {a: same}}, {0: ph.CollisionPolicy()})[0]
        assert (
            isinstance(empty, m.CompactContactBuffer) and empty.contacts == empty.diagnostics == ()
        )
        assert work.stats()["completed_launches"] == 0
    mod = ph.CollisionModel((s1, s2))
    with m.CompactWorldContactWorkspace([(0, mod)]) as work:
        ambiguous = work.detect({0: {a: same, b: same}}, {0: ph.CollisionPolicy()})[0]
        (hit,) = ambiguous.contacts
        assert (
            hit.status == "overlap"
            and hit.witness.normal is None
            and ambiguous.diagnostics == (hit,)
        )
