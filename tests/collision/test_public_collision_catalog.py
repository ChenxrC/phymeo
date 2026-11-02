"""Mixed-model acceptance of every registered RR, RD and DD case and variant."""

import json
from dataclasses import asdict, replace

import pytest
from adjacent_triangle_cases import authored as adjacent
from capsule_face_cases import authored as capsule_face
from case_catalog import catalog_entries
from closed_mesh_cases import authored as closed_mesh
from convex_hull_cases import authored as hulls
from hull_surface_cases import authored as hull_face
from rotating_round_cases import authored as arc
from round_cases import catalog_scene
from rounded_plate_cases import authored as plate
from sphere_face_cases import authored as sphere_face
from tetrahedral_solid_cases import authored as cells
from tetrahedral_surface_cases import scene as cell_face
from triangle_pair_cases import authored as triangles

import phymeo as ph


def case_scene(case, name):
    focus = ()
    if case in ("RR01", "RR02"):
        model, paths, *_ = catalog_scene(case, name)
    elif case in ("RD04", "DD04"):
        model, paths, key = cell_face(case, name)
        focus = (key,)
    elif case == "DD03":
        model, paths, focus = adjacent(name)
    elif case in ("DD01", "DD02"):
        model, paths = triangles(case, name)
    else:
        build = {
            "RR03": hulls,
            "RR04": arc,
            "RR05": closed_mesh,
            "RD01": sphere_face,
            "RD02": capsule_face,
            "RD03": plate,
            "RD05": hull_face,
            "DD05": cells,
        }[case]
        model, paths = build(name)[:2]
    return model, paths, focus


def run_catalog(backend, tmp_path):
    catalog = catalog_entries()
    expected = {f"{family}{n:02}" for family in ("RR", "RD", "DD") for n in range(1, 6)}
    assert set(catalog) == expected
    cases = []
    shapes = []
    motions = {}
    exclusions = []
    for case, family in catalog.items():
        for v in family["m0_variants"]:
            m, paths, focus = case_scene(case, v["name"])
            world = len(cases)
            keys = {s.key: replace(s.key, world=world) for s in m.shapes}
            shapes.extend(replace(s, key=keys[s.key]) for s in m.shapes)
            motions.update((keys[k], p) for k, p in paths.items())
            exclusions.extend((keys[a], keys[b]) for a, b in m.excluded)
            cases.append((case, v["name"], v["relation"], tuple(keys[k] for k in focus)))
    assert len(cases) == 74 and len({(c, n) for c, n, _, _ in cases}) == 74
    model = ph.CollisionModel(shapes, exclusions)
    with ph.CollisionWorkspace(model, backend, cache_root=tmp_path / "cache") as workspace:
        result = ph.detect(model, motions, ph.CollisionPolicy(margin=0.5), workspace)
        assert result.status == "ok" and workspace.last_result is result
        assert len(result.pairs) == len(shapes) * (len(shapes) - 1) // 2
        by_world = [[] for _ in cases]
        for p in result.pairs:
            if p.first.world != p.second.world:
                assert not p.candidate and p.status == "filtered" and p.reason == "different_world"
            else:
                by_world[p.first.world].append(p)
        records = []
        for (case, name, relation, focus), world in zip(cases, by_world, strict=True):
            target = [
                p
                for p in world
                if p.family == case[:2] and all(k in (p.first, p.second) for k in focus)
            ]
            assert target, (case, name)
            statuses = {p.status for p in target}
            if (case, name) == ("RD04", "internal_face_trap"):
                # This authored solid-overlap control has a separate exterior-surface contract.
                assert statuses == {"surface_clear"} and all(p.witness.gap_m > 0 for p in target)
                assert not any(set(f.face_ids) == {0, 1, 2} for p in target for f in p.facets)
            elif relation == "invalid":
                assert statuses == {"invalid_geometry"}, (case, name, statuses)
            elif relation == "filtered":
                assert statuses == {"filtered"} and all(p.reason == "excluded" for p in target)
            elif relation == "clear":
                assert statuses <= {"separated", "surface_clear", "adjacent"}, (
                    case,
                    name,
                    statuses,
                )
            else:
                assert relation == "hit" and statuses & {"hit", "overlap", "touching"}, (
                    case,
                    name,
                    statuses,
                )
                assert statuses.isdisjoint(
                    {"unsupported", "unresolved", "invalid_geometry", "filtered"}
                )
            records.append({"case": case, "variant": name, "target": [asdict(p) for p in target]})
        stats = workspace.stats()
        if backend != "host":
            assert stats["sweep_completed_launches"] and stats["hull_completed_launches"]
            assert stats["arc_chart_completed_launches"] and stats["triangle_completed_launches"]
            if backend == "cuda":
                assert (
                    stats["sweep_downloads"]
                    and stats["hull_downloads"]
                    and stats["synchronizations"]
                )
        (tmp_path / "catalog.json").write_text(
            json.dumps(
                dict(
                    records=records,
                    shapes=len(shapes),
                    pairs=len(result.pairs),
                    candidates=result.required_candidates,
                    stats=stats,
                ),
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_complete_public_catalog_host(tmp_path):
    run_catalog("host", tmp_path)


@pytest.mark.tyx
def test_complete_public_catalog_device(tyx_mode, tmp_path):
    run_catalog(tyx_mode, tmp_path)
