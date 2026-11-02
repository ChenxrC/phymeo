"""Load authored geometric cases and apply named variants for both HOST baselines."""

import json
from copy import deepcopy
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "docs/benchmarks/p65-general-collision-cases.json"


def catalog_entries():
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert document["schema"] == "phymeo-general-collision-cases/1"
    entries = {row["id"]: row for row in document["cases"]}
    assert len(entries) == len(document["cases"])
    expected = {f"{family}{i:02d}" for family in ("RR", "RD", "DD") for i in range(1, 6)}
    assert expected <= entries.keys()
    return entries


def geometry_for(case_id, variant_name):
    case = catalog_entries()[case_id]
    variant = next(v for v in case["m0_variants"] if v["name"] == variant_name)
    if "geometry" in variant:
        assert not variant["changes"]
    geometry = deepcopy(variant.get("geometry", case["geometry"]))
    for side, fields in variant["changes"].items():
        assert fields.keys() <= geometry[side].keys()
        geometry[side].update(fields)
    for side in ("A", "B"):
        if order := variant.get("order_" + side):
            assert sorted(order) == [0, 1, 2]
            for field in ("vertices", "vertices_start", "vertices_end"):
                if field in geometry[side]:
                    geometry[side][field] = [geometry[side][field][i] for i in order]
    if variant.get("swap"):
        geometry["A"], geometry["B"] = geometry["B"], geometry["A"]
    return case, variant, geometry
