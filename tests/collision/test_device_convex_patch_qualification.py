"""B1 device patch qualification: AUTHORING ONLY until the parent's serial run.

K0 helper reuse is serialization, public installation and independent literals.
No old K0/K1 test function is collected by this new module.
"""
import hashlib
import importlib.util
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import test_device_convex_query as k0

import phymeo as ph


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).resolve().parents[1] / "reference_newton/rr/device_convex_patch_qualification_reference.py"
    spec = importlib.util.spec_from_file_location("b1_patch_independent_literals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def saved(tmp_path, identity, mode, inputs):
    with k0.saved(tmp_path, identity, mode, inputs) as record:
        record["qualification_sources"] = tuple(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (Path(__file__).resolve(), Path(reference().__file__).resolve()))
        yield record


@contextmanager
def prepared(case, mode, tmp_path, site_capacity=16, requested_reverse=False):
    world = case["keys"][0][0]
    quaternion = case.get("quaternion", k0.case_quaternion(case))
    tree = k0.ordinary_forest(anchor=case["positions"][0], orientation=quaternion)
    with ph.TyxWorldBatch([(world, tree)], mode, cache_root=tmp_path) as batch:
        before = batch.snapshot()
        model, bindings = k0.registrations(case, before[world].topology)
        keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
        request = tuple(reversed(keys)) if requested_reverse else keys
        with batch.prepare_collision(model, bindings) as broad:
            with closing(batch.prepare_convex_query(broad, pairs=(request,), site_capacity=site_capacity)) as plan:
                yield batch, broad, plan, before


def complete_match(actual, expected, tolerance):
    """Geometry-only bijection, preserving all points and multiplicities."""
    if len(actual) != len(expected):
        raise AssertionError("patch cardinality differs")
    candidates = [[j for j, wanted in enumerate(expected)
                   if np.linalg.norm(point - wanted) <= tolerance] for point in actual]
    assigned = {}

    def augment(i, seen):
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    assert all(augment(i, set()) for i in range(len(actual))), "patch lacks a complete geometric bijection"
    return {i: j for j, i in assigned.items()}


def assert_patch(row, spec, case):
    scale = spec["length_m"]
    patch = row.patch
    assert row.status == "RESOLVED" and patch.status == "RESOLVED", (row.status, patch.status, row.reason)
    assert patch.dimension == spec["dimension"]
    assert patch.device_resolved and patch.skip_host_patch_search and not patch.host_completed
    assert abs(row.gap_m - spec["gap_m"]) <= 1e-4 * scale
    np.testing.assert_allclose(row.normal, spec["normal"], atol=2e-3, rtol=0)
    actual = [np.concatenate((site.first.surface_relative_m, site.second.surface_relative_m)) for site in patch.sites]
    expected = [np.concatenate(site["surfaces"]) for site in spec["sites"]]
    matching = complete_match(actual, expected, 2e-4 * scale)
    weights = np.asarray([site.weight for site in patch.sites])
    assert np.all(weights > 0) and abs(float(weights.sum()) - 1) <= 64 * 2**-23
    origin = np.asarray(row.origin_m)
    points = {tuple(key): dict(zip(ids, world - origin, strict=True))
              for key, ids, world in zip(case["keys"], case["ids"], case["points"], strict=True)}
    radius = dict(zip(map(tuple, case["keys"]), case["radii"], strict=True))
    for i, site in enumerate(patch.sites):
        wanted = spec["sites"][matching[i]]
        assert abs(site.weight - wanted["weight"]) <= 1e-5
        assert abs(site.gap_m - spec["gap_m"]) <= 1e-4 * scale
        np.testing.assert_allclose(site.normal, spec["normal"], atol=2e-3, rtol=0)
        for sign, witness in zip((1, -1), (site.first, site.second), strict=True):
            key = (witness.key.world, witness.key.body, witness.key.shape)
            assert len(set(witness.vertex_ids)) == len(witness.vertex_ids)
            bary = np.asarray(witness.weights)
            assert np.all(bary >= 0) and abs(float(bary.sum()) - 1) <= 64 * 2**-23
            reconstructed = sum((w * points[key][int(node)] for node, w in
                                 zip(witness.vertex_ids, bary, strict=True)), np.zeros(3))
            expected_skin = sign * radius[key] * np.asarray(spec["normal"])
            np.testing.assert_allclose(witness.offset_m, expected_skin, atol=2e-4 * scale, rtol=0)
            np.testing.assert_allclose(reconstructed + witness.offset_m, witness.surface_relative_m,
                                       atol=2e-4 * scale, rtol=0)
        np.testing.assert_allclose(site.common_m,
                                   .5 * (np.asarray(site.first.surface_relative_m) + site.second.surface_relative_m),
                                   atol=2e-4 * scale, rtol=0)
    actual_moment = sum((site.weight * np.asarray(site.common_m) for site in patch.sites), np.zeros(3))
    expected_moment = sum((site["weight"] * .5 * (np.asarray(site["surfaces"][0]) + site["surfaces"][1])
                           for site in spec["sites"]), np.zeros(3))
    np.testing.assert_allclose(actual_moment, expected_moment, atol=2e-4 * scale, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("family", ("BB", "BH", "HH"))
@pytest.mark.parametrize("variant", ("reverse", "cycle", "tilt", "permute", "cycle_reverse_ids"))
def test_remaining_face_covariance(family, variant, tyx_mode, tmp_path):
    spec = getattr(reference(), family.lower() + "_overlap")()
    case = k0.reference().family_case(family, spec["gap_m"])
    if variant != "reverse":
        case = k0.reference().transformed(case, variant)
        spec = getattr(reference(), {"cycle": "cycled", "tilt": "tilted", "permute": "permuted",
                                     "cycle_reverse_ids": "cycle_reverse_ids"}[variant])(spec)
    with saved(tmp_path, f"B1-face-{family}-{variant}", tyx_mode, case) as record:
        with prepared(case, tyx_mode, tmp_path, requested_reverse=variant in ("reverse", "cycle_reverse_ids")) as (batch, broad, plan, before):
            result = k0.query_attempt(batch, plan, record, "face")
            assert_patch(result.rows[0], spec, case)
            assert result.request_reversed == (variant == "reverse",)
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)
            assert plan.require_current(result) is result


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("partial_bb", "partial_bh", "partial_hh"))
@pytest.mark.parametrize("variant", ("identity", "reverse", "cycle", "tilt", "permute"))
def test_partial_finite_support_polygons(name, variant, tyx_mode, tmp_path):
    case, spec = reference().polygon_case(name, variant)
    with saved(tmp_path, f"B1-{name}-{variant}", tyx_mode, dict(case=case, expected=spec)) as record:
        with prepared(case, tyx_mode, tmp_path, requested_reverse=variant == "reverse") as (batch, broad, plan, before):
            result = k0.query_attempt(batch, plan, record, "partial")
            assert_patch(result.rows[0], spec, case)
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)
            assert result.request_reversed == (variant == "reverse",)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("identity", "cycle", "tilt"))
def test_trapezoid_area_moment_survives_edge_refinement(variant, tyx_mode, tmp_path):
    outputs = []
    for name in ("trapezoid", "trapezoid_refined"):
        case, spec = reference().polygon_case(name, variant)
        assert spec["area_m2"] == pytest.approx(1.5, abs=1e-13)
        with saved(tmp_path, f"B1-{name}-{variant}", tyx_mode, dict(case=case, expected=spec)) as record:
            with prepared(case, tyx_mode, tmp_path) as (batch, broad, plan, before):
                result = k0.query_attempt(batch, plan, record, "q04")
                row = result.rows[0]
                assert_patch(row, spec, case)
                # Frozen K1 algebra check, not a physical delivery or contact step.
                force = sum((-100 * site.gap_m * site.weight * np.asarray(site.normal)
                             for site in row.patch.sites), np.zeros(3))
                torque = sum((np.cross(site.common_m, -100 * site.gap_m * site.weight * np.asarray(site.normal))
                              for site in row.patch.sites), np.zeros(3))
                np.testing.assert_allclose(force, spec["normal"], atol=1e-2, rtol=0)
                expected_torque = np.cross(spec["centroid"], spec["normal"])
                np.testing.assert_allclose(torque, expected_torque, atol=2e-3 * spec["length_m"], rtol=0)
                outputs.append((force, torque))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-3 * spec["length_m"], rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("partial_bb", "partial_hh", "trapezoid_refined"))
def test_site_capacity_is_exact_and_never_truncates(name, tyx_mode, tmp_path):
    case, spec = reference().polygon_case(name)
    needed = len(spec["sites"])
    with saved(tmp_path, f"B1-capacity-{name}", tyx_mode, dict(case=case, needed=needed)) as record:
        for capacity in (needed, needed - 1):
            with prepared(case, tyx_mode, tmp_path, site_capacity=capacity) as (batch, broad, plan, before):
                result = k0.query_attempt(batch, plan, record, f"capacity-{capacity}")
                row = result.rows[0]
                if capacity == needed:
                    assert_patch(row, spec, case)
                else:
                    assert row.status == "RESOLVED"
                    assert row.patch.status == "RESOURCE_LIMIT"
                    assert row.patch.sites == () and not row.patch.device_resolved
                    assert not row.patch.skip_host_patch_search
                assert k0.canonical(batch.snapshot()) == k0.canonical(before)


@pytest.mark.tyx
@pytest.mark.parametrize("field", ("weight", "id", "bary", "gap", "normal", "padding", "late_read"))
def test_late_patch_failure_keeps_both_publications(field, tyx_mode, tmp_path, monkeypatch):
    case, spec = reference().polygon_case("partial_hh")
    with saved(tmp_path, f"B1-atomic-{field}", tyx_mode, case) as record:
        with prepared(case, tyx_mode, tmp_path) as (batch, broad, plan, before):
            accepted = k0.query_attempt(batch, plan, record, "accepted")
            assert_patch(accepted.rows[0], spec, case)
            broad_accepted = broad.last_result
            original = plan._read_patch

            def corrupt():
                payload = original()
                if field == "late_read":
                    raise RuntimeError("injected late complete patch read")
                if field == "weight":
                    payload["patch_weight"][0] = -1
                elif field == "id":
                    payload["patch_ids"][0, 0, 0] = 2147483647
                elif field == "bary":
                    payload["patch_bary"][0, 0, 0] = -1
                elif field == "gap":
                    payload["patch_gap"][0] += 1
                elif field == "normal":
                    payload["patch_normal"][0] *= -1
                elif field == "padding":
                    payload["patch_ids"][len(spec["sites"]), 0, 0] = 1
                record["corrupt_payload"] = payload
                return payload

            monkeypatch.setattr(plan, "_read_patch", corrupt)
            with pytest.raises(RuntimeError):
                k0.query_attempt(batch, plan, record, "rejected")
            assert plan.last_result is accepted and broad.last_result is broad_accepted
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)
            assert plan.require_current(accepted) is accepted
            monkeypatch.setattr(plan, "_read_patch", original)
            retry = k0.query_attempt(batch, plan, record, "retry")
            assert_patch(retry.rows[0], spec, case)
            assert retry is not accepted and plan.last_result is retry


def test_geometry_bijection_rejects_missing_replaced_and_duplicate_points():
    expected = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    assert len(complete_match(expected[::-1], expected, 1e-4)) == 3
    for wrong in (expected[:2], np.asarray([[0., 0.], [1., 0.], [1., 0.]]),
                  np.asarray([[0., 0.], [1., 0.], [0., 2.]])):
        with pytest.raises(AssertionError):
            complete_match(wrong, expected, 1e-4)
