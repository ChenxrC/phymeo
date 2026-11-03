"""Contacts, material witnesses and refusal across a bounded paged population."""

import json

import pytest
from test_mixed_world_queries import fixture_worlds

import phymeo as ph
from phymeo.geometry.compact_contacts import CompactWorldContactWorkspace


def controls():
    worlds, paths = fixture_worlds()
    # A fifth world uses the very same local IDs. All its AABBs have strict clearance.
    world = 74
    keys = [ph.CollisionKey(world, k, 0) for k in range(3)]
    worlds[world] = ph.CollisionModel([ph.CollisionShape(k, "sphere", (0,), 0.125) for k in keys])
    paths[world] = {k: ph.CollisionMotion([[2 * i, 0, 0]]) for i, k in enumerate(keys)}
    rules = {k: ph.CollisionPolicy(margin=0.5 if k == 31 else 0) for k in worlds}
    return worlds, paths, rules


def compare_heterogeneous_pages(backend, tmp_path):
    worlds, paths, rules = controls()
    limits = ph.CollisionPageLimits(shapes=3, resident_pages=1)
    with ph.PagedWorldCollisionWorkspace(
        worlds.items(), backend, limits=limits, result_format="contacts", cache_root=tmp_path
    ) as work:
        result = work.detect(paths, rules)
        assert tuple(result) == tuple(worlds)
        assert all(
            isinstance(v, ph.CompactContactBuffer) and v.status == "ok" for v in result.values()
        )
        assert [result[k].required_contacts for k in worlds] == [0, 1, 1, 1, 0]
        assert result[31].contacts[0].sweep.first_t == (3 / 8, 3 / 8)
        assert result[7].contacts[0].witness.gap_m == -1 / 16
        assert result[53].contacts[0].sweep.first_t == (0.25, 0.25)
        (material,) = result[53].contacts[0].sweep.witness.intersection_points
        assert dict(material.first_weights) == {11: 0.5, 7: 0.5, 3: 0.0}
        assert dict(material.second_weights) == {31: 0.5, 41: 0.5, 53: 0.0}
        for key, m in worlds.items():
            with ph.CollisionWorkspace(m, backend, cache_root=tmp_path) as single:
                full = ph.detect(m, paths[key], rules[key], single)
            assert result[key] == ph.compact_contacts(full)
        stats = work.stats()
        assert stats["result_format"] == "contacts" and stats["peak_resident_pages"] == 1
        assert len(work.layout) == 4 and stats["evictions"] >= 3
        assert stats["backend_counts"]["compact_skipped_slots"] >= 3
        if backend != "host":
            assert stats["completed_launches"] > 0
        bad = work.detect(paths, rules, limits_by_world={53: ph.ContactBufferLimits(contacts=0)})
        assert bad[53].status == "capacity_exceeded" and bad[53].required_contacts == 1
        assert bad[53].contacts == () and work.last_result(53) is result[53]
        assert all(work.last_result(k) is bad[k] for k in (90, 31, 7, 74))
        assert work.detect(paths, rules) == result
        (tmp_path / "paged-contacts.json").write_text(
            json.dumps(
                dict(
                    backend=backend,
                    contacts={w: r.required_contacts for w, r in result.items()},
                    page_keys=[p.world_keys for p in work.layout],
                    evictions=stats["evictions"],
                    skipped=stats["backend_counts"]["compact_skipped_slots"],
                    device_launches=stats["completed_launches"],
                    refused_contacts=bad[53].required_contacts,
                )
            ),
            encoding="utf8",
        )


def test_contacts_cross_pages_and_keep_rr_rd_dd_material_owners(tmp_path):
    compare_heterogeneous_pages("host", tmp_path)


@pytest.mark.tyx
def test_contacts_cross_pages_on_device(tyx_mode, tmp_path):
    compare_heterogeneous_pages(tyx_mode, tmp_path)


def test_late_fault_and_unknown_capacity_key_leave_full_past_observation_intact(
    monkeypatch, tmp_path
):
    worlds, paths, rules = controls()
    limits = ph.CollisionPageLimits(shapes=3, resident_pages=1)
    with ph.PagedWorldCollisionWorkspace(
        worlds.items(), limits=limits, result_format="contacts"
    ) as work:
        old = work.detect(paths, rules)
        calls = work.stats()["page_dispatches"]
        with pytest.raises((ValueError, KeyError)):
            work.detect(paths, rules, limits_by_world={999: ph.ContactBufferLimits()})
        assert work.stats()["page_dispatches"] == calls
        real = CompactWorldContactWorkspace.detect

        def fail_late(leaf, local, policies, **kw):
            if 74 in local:
                raise RuntimeError("injected last page failure")
            return real(leaf, local, policies, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(CompactWorldContactWorkspace, "detect", fail_late)
            with pytest.raises(RuntimeError, match="injected last page"):
                work.detect(paths, rules)
        assert work.stats()["failed_queries"] == 1 and work.stats()["resident_pages"] == 0
        assert all(work.last_result(k) is old[k] for k in worlds)
        assert work.detect(paths, rules) == old
    with ph.PagedWorldCollisionWorkspace(worlds.items(), limits=limits) as full:
        with pytest.raises(ValueError, match="contacts"):
            full.detect(paths, rules, limits_by_world={53: ph.ContactBufferLimits(contacts=0)})
        assert full.stats()["page_builds"] == 0
    with pytest.raises(ValueError, match="result_format"):
        ph.PagedWorldCollisionWorkspace(worlds.items(), result_format="discard_unknown")
