"""B2 raw/effective geometry qualification; predeclared before product execution.

Original B1 gates: gap 1e-4 L, paired surfaces/moment 2e-4 L, normal 2e-3,
weights 1e-5. Both representations independently meet those same references.
No old K1 WIP imports or force-selected correspondences.
"""
from dataclasses import replace

import numpy as np
import pytest
import test_device_convex_patch_qualification as b1
import test_device_convex_query as k0

from phymeo.backends._resident_contact_bridge import ContactShapeFrame
from phymeo.backends._resident_device_patch_bridge import patches_from_device


def frames_for(result):
    source = result.source
    frames = {}
    for index, shape in enumerate(source["model"].shapes):
        world = shape.key.world
        origin = np.asarray(source["world_origins_m"][source["world_keys"].index(world)])
        start, end = source["vertex_offsets"][index:index + 2]
        declared = source["declared_local_points_m"][start:end]
        group = int(source["shape_groups"][index])
        if group < 0:
            points, position, rotation = declared, np.zeros(3), np.eye(3)
            reference = points.mean(axis=0)
        else:
            pose = source["pose"][index]
            position, rotation = origin + pose[:3], pose[3:].reshape(3, 3)
            points = declared @ rotation.T + position
            reference = position
        frames.setdefault(world, {})[shape.key] = ContactShapeFrame(
            shape, points, None if group < 0 else group, position, rotation,
            reference, np.zeros(3), np.zeros(3)
        )
    return frames


def assert_effective(converted, result, spec):
    row = result.rows[0]
    sample = converted.samples[row.pair]
    origin, scale = np.asarray(row.origin_m), spec["length_m"]
    actual = [np.r_[np.asarray(s.witness.first_point) - origin,
                    np.asarray(s.witness.second_point) - origin] for s in sample.sites]
    expected = [np.concatenate(s["surfaces"]) for s in spec["sites"]]
    match = b1.complete_match(actual, expected, 2e-4 * scale)
    assert sample.dimension == spec["dimension"]
    assert sample.source == "resident-device-patch-effective-v1"
    assert len(sample.sites) == len(row.patch.sites)
    moment = np.zeros(3)
    for i, site in enumerate(sample.sites):
        witness = site.witness
        assert abs(witness.gap_m - spec["gap_m"]) <= 1e-4 * scale
        np.testing.assert_allclose(witness.normal, spec["normal"], atol=2e-3, rtol=0)
        assert abs(site.weight - spec["sites"][match[i]]["weight"]) <= 1e-5
        assert abs(np.linalg.norm(witness.normal) - 1) <= 1e-12
        for side in ("first", "second"):
            raw = getattr(row.patch.sites[i], side)
            weights = getattr(witness, side + "_weights")
            assert dict(weights).keys() == dict(zip(raw.vertex_ids, raw.weights)).keys()
            assert len(weights) == len(raw.vertex_ids)
            np.testing.assert_allclose(
                [dict(weights)[node] for node in raw.vertex_ids],
                np.asarray(raw.weights) / sum(raw.weights), atol=1e-15, rtol=0
            )
        moment += site.weight * (np.asarray(site.load_point) - origin)
    wanted = sum((s["weight"] * (np.asarray(s["surfaces"][0]) + s["surfaces"][1]) / 2
                  for s in spec["sites"]), np.zeros(3))
    np.testing.assert_allclose(moment, wanted, atol=2e-4 * scale, rtol=0)
    assert converted.query is result
    assert len(converted.differences[row.pair]) == len(sample.sites)
    with pytest.raises(TypeError):
        converted.samples[row.pair] = sample
    assert not result.source["pose"].flags.writeable


@pytest.mark.tyx
@pytest.mark.parametrize("family", ("SS", "SC", "SB", "SH", "CC", "CB", "CH", "BB", "BH", "HH"))
@pytest.mark.parametrize("variant", ("identity", "cycle_reverse_ids"))
def test_all_families_raw_and_effective_share_independent_reference(
    family, variant, tyx_mode, tmp_path
):
    spec = getattr(b1.reference(), family.lower() + "_overlap")()
    case = k0.reference().family_case(family, spec["gap_m"])
    if variant != "identity":
        case = k0.reference().transformed(case, variant)
        spec = b1.reference().cycle_reverse_ids(spec)
    with b1.saved(tmp_path, "B2-" + family + "-" + variant, tyx_mode, case) as saved:
        with b1.prepared(case, tyx_mode, tmp_path) as (batch, broad, query, before):
            result = k0.query_attempt(batch, query, saved, "raw")
            b1.assert_patch(result.rows[0], spec, case)
            converted = patches_from_device(result, frames_for(result), candidates=(result.rows[0].pair,))
            assert_effective(converted, result, spec)
            saved["effective"] = converted
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("partial_bb", "partial_bh", "partial_hh", "trapezoid_refined"))
@pytest.mark.parametrize("variant", ("reverse", "tilt"))
def test_partial_faces_preserve_every_id_and_first_moment(name, variant, tyx_mode, tmp_path):
    case, spec = b1.reference().polygon_case(name, variant)
    with b1.saved(tmp_path, "B2-" + name + "-" + variant, tyx_mode, case) as saved:
        with b1.prepared(case, tyx_mode, tmp_path, requested_reverse=variant == "reverse") as (_, _, query, _):
            result = query.query()
            saved["raw"] = result
            b1.assert_patch(result.rows[0], spec, case)
            converted = patches_from_device(result, frames_for(result), candidates=(result.rows[0].pair,))
            assert_effective(converted, result, spec)
            saved["effective"] = converted


@pytest.mark.tyx
@pytest.mark.parametrize("fault", (
    "negative_bary", "missing_material", "nonunit_site_normal", "zero_weight",
    "missing_row", "duplicate_row", "declared_source", "uploaded_source", "frame_source",
    "resource", "unresolved", "surface", "quadrature"
))
def test_bad_representation_cannot_become_a_physical_site(fault, tyx_mode, tmp_path):
    spec = b1.reference().cc_overlap()
    case = k0.reference().family_case("CC", spec["gap_m"])
    with b1.prepared(case, tyx_mode, tmp_path) as (_, _, query, _):
        result = query.query()
        frames = frames_for(result)
        row, site = result.rows[0], result.rows[0].patch.sites[0]
        if fault == "negative_bary":
            site = replace(site, first=replace(site.first, weights=(-0.1, 1.1)))
        elif fault == "missing_material":
            site = replace(site, first=replace(site.first, vertex_ids=(2147483647,) + site.first.vertex_ids[1:]))
        elif fault == "nonunit_site_normal":
            site = replace(site, normal=(0, 0, 0))
        elif fault == "zero_weight":
            site = replace(site, weight=0)
        elif fault == "quadrature":
            site = replace(site, weight=site.weight + 0.1)
        elif fault == "surface":
            site = replace(site, first=replace(site.first, surface_relative_m=(100, 100, 100)))
        if fault in ("negative_bary", "missing_material", "nonunit_site_normal", "zero_weight", "surface", "quadrature"):
            # Mutate both representations consistently. These negatives must
            # reach semantic admission, not merely typed/raw binding checks.
            wire = dict(result.patch)
            for name in ("patch_weight", "patch_normal", "patch_ids", "patch_bary", "patch_surface"):
                wire[name] = np.array(wire[name], copy=True)
            wire["patch_weight"][0] = site.weight
            wire["patch_normal"][0] = site.normal
            count = len(site.first.vertex_ids)
            wire["patch_ids"][0, 0, :count] = site.first.vertex_ids
            wire["patch_bary"][0, 0, :count] = site.first.weights
            wire["patch_surface"][0, 0] = site.first.surface_relative_m
            site = replace(
                site, weight=float(wire["patch_weight"][0]),
                normal=tuple(map(float, wire["patch_normal"][0])),
                first=replace(site.first,
                    vertex_ids=tuple(map(int, wire["patch_ids"][0, 0, :count])),
                    weights=tuple(map(float, wire["patch_bary"][0, 0, :count])),
                    surface_relative_m=tuple(map(float, wire["patch_surface"][0, 0]))))
            row = replace(row, patch=replace(row.patch, sites=(site,) + row.patch.sites[1:]))
            result = replace(result, rows=(row,), patch=wire)
        elif fault in ("resource", "unresolved"):
            row = replace(row, patch=replace(row.patch, status="RESOURCE_LIMIT" if fault == "resource" else "UNRESOLVED"))
            result = replace(result, rows=(row,))
        elif fault == "missing_row":
            result = replace(result, rows=())
        elif fault == "duplicate_row":
            result = replace(result, rows=(row, row))
        elif fault in ("declared_source", "uploaded_source"):
            key = "declared_local_points_m" if fault == "declared_source" else "uploaded_local_points_m"
            source = dict(result.source)
            source[key] = np.array(source[key], copy=True)
            source[key][0, 0] += 0.1
            result = replace(result, source=source)
        elif fault == "frame_source":
            key = row.pair[0]
            original = frames[key.world][key]
            frames[key.world][key] = replace(original, world_points_m=original.world_points_m + [0.1, 0, 0])
        with pytest.raises((ValueError, RuntimeError)):
            patches_from_device(result, frames, candidates=(row.pair,))
