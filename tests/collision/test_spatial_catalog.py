"""One spatial query must reproduce every public row of the 15-family catalog."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

import pytest
from test_public_collision_catalog import catalog_model

import phymeo as ph


def check_catalog(backend, tmp_path):
    model, paths, cases = catalog_model()
    policy = ph.CollisionPolicy(margin=0.5)
    with ph.CollisionWorkspace(model, backend, cache_root=tmp_path / "cache") as baseline:
        ref = ph.detect(model, paths, policy, baseline)
    assert len(cases) == 74 and len(ref.pairs) == 13366 and ref.required_candidates == 101
    modes = {}
    for mode in ("sap", "bvh"):
        with ph.CollisionWorkspace(
            model, backend, broad_phase=mode, mesh_midphase="bvh", cache_root=tmp_path / "cache"
        ) as accelerated:
            actual = ph.detect(model, paths, policy, accelerated)
            assert actual == ref
            modes[mode] = accelerated.stats()
            assert modes[mode]["mesh_facet_pairs_tested"] < modes[mode]["mesh_facet_pairs_total"]
    # Include all filtered rows, every bound and every volume-body aggregate.
    record = asdict(replace(ref, backend="canonical"))
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    (tmp_path / "catalog-parity.json").write_text(
        json.dumps(
            dict(
                count=len(cases),
                pairs=len(ref.pairs),
                candidates=ref.required_candidates,
                all_fields_sha256=sha256(encoded).hexdigest(),
                stats=modes,
            ),
            allow_nan=False,
        )
    )


def test_full_spatial_catalog_host(tmp_path):
    check_catalog("host", tmp_path)


@pytest.mark.tyx
def test_full_spatial_catalog_device(tyx_mode, tmp_path):
    check_catalog(tyx_mode, tmp_path)
