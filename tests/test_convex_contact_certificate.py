"""Independent original geometry certificates, including adversarial records."""

import copy
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from phymeo.validation.convex_contact import certify_contact, match_contact_pairs

CASES = json.loads((Path(__file__).parent / "fixtures/m16_geometry_contract.json").read_text())


def arguments(case):
    result = {
        k: np.asarray(case[k])
        for k in (
            "positions",
            "rotations",
            "material_ids",
            "weights",
            "surfaces",
            "normal",
            "receipt",
        )
    }
    result.update(
        vertices=np.concatenate(case["points"]),
        offsets=np.array([0, len(case["points"][0]), sum(map(len, case["points"]))]),
        pair=np.array([0, 1]),
        gap=case["gap"],
    )
    return result


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("swap", [False, True])
def test_saved_unique_intersections_and_side_reversal(index, swap):
    args = arguments(CASES[index])
    expected = certify_contact(**args)
    if swap:
        for k in ("pair", "material_ids", "weights", "surfaces"):
            args[k] = args[k][::-1].copy()
        args["receipt"][1:] = args["receipt"][1:][::-1]
        args["normal"] *= -1
    got = certify_contact(**args)
    assert got.unique
    assert got.point_error < 2e-9
    np.testing.assert_allclose(got.point, expected.point, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_id",
        "duplicate_id",
        "negative_weight",
        "sum",
        "unused_weight",
        "surface",
        "normal",
        "gap",
        "nan",
        "unique_lie",
        "bad_rotation",
        "float_id",
    ],
)
def test_corrupt_certificate_rejected(mutation):
    a = arguments(CASES[1])
    if mutation == "foreign_id":
        a["material_ids"][0, 0] = a["offsets"][1]
    elif mutation == "duplicate_id":
        a["material_ids"][0, 1] = a["material_ids"][0, 0]
    elif mutation == "negative_weight":
        a["weights"][0, 0] = -1e-12
    elif mutation == "sum":
        a["weights"][0, 0] += 0.1
    elif mutation == "unused_weight":
        a["weights"][0, 3] = 0.1
    elif mutation == "surface":
        a["surfaces"][0, 0] += 1e-5
    elif mutation == "normal":
        a["normal"] *= -1
    elif mutation == "gap":
        a["gap"] += 1e-5
    elif mutation == "nan":
        a["weights"][0, 0] = np.nan
    elif mutation == "unique_lie":
        a["receipt"][1:] = 4
    elif mutation == "bad_rotation":
        a["rotations"][0, 0, 0] += 0.01
    elif mutation == "float_id":
        a["material_ids"] = a["material_ids"].astype(float)
    with pytest.raises(ValueError):
        certify_contact(**a)


def nonunique():
    cube = np.array(list(product((-1.0, 1.0), repeat=3)))
    # The overlapping parallel faces admit many different valid midpoints.
    return dict(
        vertices=np.concatenate([cube, cube]),
        offsets=np.array([0, 8, 16]),
        positions=np.array([[0.0, 0, 0], [1.9, 0, 0]]),
        rotations=np.array([np.eye(3)] * 2),
        pair=np.array([0, 1]),
        material_ids=np.array([[4, 5, 6, 7], [8, 9, 10, 11]]),
        weights=np.ones((2, 4)) / 4,
        surfaces=np.array([[1.0, 0, 0], [0.9, 0, 0]]),
        normal=np.array([1.0, 0, 0]),
        gap=-0.1,
        receipt=np.array([1, 4, 4]),
    )


def test_nonunique_face_choice_is_valid_but_never_unique():
    a = nonunique()
    cert = certify_contact(**a)
    assert not cert.unique and cert.kind == "nonunique_choice"
    a["receipt"][0] = 2
    with pytest.raises(ValueError, match="topology"):
        certify_contact(**a)


def test_nonunique_bad_closure_cannot_pass_same_support_planes():
    a = nonunique()
    a["weights"][1] = [0.5, 0.5, 0, 0]
    a["surfaces"][1, 1] = -1
    with pytest.raises(ValueError, match="closure"):
        certify_contact(**a)


def test_nonunique_choice_changes_torque_without_changing_net_force():
    a = nonunique()
    c = certify_contact(**a)
    b = copy.deepcopy(a)
    b["weights"][:] = [0.5, 0.5, 0, 0]
    b["surfaces"][:, 1] = -1
    d = certify_contact(**b)
    force = np.array([1.0, 0, 0])
    assert not c.unique and not d.unique
    assert np.linalg.norm(np.cross(c.point, force) - np.cross(d.point, force)) == 1


@pytest.mark.parametrize("other", [[[0, 1]], [[0, 1], [0, 1]], [[0, 1], [2, 4]]])
def test_contact_missing_extra_duplicate_rejected(other):
    with pytest.raises(ValueError):
        match_contact_pairs(np.array([[0, 1], [2, 3]]), np.array(other))


def test_pair_mapping_is_bijective_and_not_order_dependent():
    assert match_contact_pairs(np.array([[2, 3], [0, 1]]), np.array([[0, 1], [2, 3]])).tolist() == [
        1,
        0,
    ]


def test_near_parallel_unique_edges_refused():
    a = arguments(CASES[1])
    a["vertices"][a["material_ids"][1, 1]] = a["vertices"][a["material_ids"][1, 0]]
    # Reconstruction also checks the changed geometry; neither failure may
    # produce a certificate or silently reinterpret it as nonunique.
    with pytest.raises(ValueError):
        certify_contact(**a)
