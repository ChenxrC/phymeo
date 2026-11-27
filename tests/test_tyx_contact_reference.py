"""Capture boundary rejects and independent native mechanics reconstruction."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from phymeo.validation.convex_contact import ContactCertificate
from phymeo.validation.tyx_contact_reference import (
    certify_world,
    evaluate_candidate,
    install_certified_geometry,
)

CASE = json.loads((Path(__file__).parent / "fixtures/m16_geometry_contract.json").read_text())[1]


def capture(world=0):
    points = np.concatenate(CASE["points"])
    cat = SimpleNamespace(points=points, offsets=np.array([0, len(CASE["points"][0]), len(points)]))
    ids = np.array(CASE["material_ids"])
    ids[ids >= 0] += world * len(points)
    normal = np.array(CASE["normal"])
    normal /= np.linalg.norm(normal)
    pair = np.array([[0, 1]])
    point = np.array(CASE["surfaces"]).mean(0)
    q = dict(
        world=np.array([world]),
        geom_ids=pair,
        meta=np.ones((1, 16), int),
        gap=np.array([[CASE["gap"]]]),
        material_ids=ids[None],
        skin_offset=np.zeros((1, 2, 3)),
        barycentric=np.array(CASE["weights"])[None],
        surface=np.array(CASE["surfaces"])[None],
        surface_low=np.zeros((1, 2, 3)),
        normal=normal[None],
        witness_receipt=np.array(CASE["receipt"])[None],
    )
    c = dict(
        counts=np.ones(world + 1, int),
        pairs=np.tile(pair, (world + 1, 1)),
        status=np.zeros(world + 1, int),
        geometry=np.tile(np.r_[CASE["gap"], point, normal], (world + 1, 1)),
        point_low=np.zeros((world + 1, 3)),
    )
    data = SimpleNamespace(
        geom_xpos=np.array(CASE["positions"]),
        geom_xmat=np.array(CASE["rotations"]).reshape(2, 9),
        contact=[
            SimpleNamespace(
                geom=pair[0], pos=point.copy(), dist=CASE["gap"], frame=np.r_[normal, np.zeros(6)]
            )
        ],
    )
    return SimpleNamespace(geom_type=np.array([7, 7])), data, cat, q, c, world


@pytest.mark.parametrize("world", [0, 1, 7])
def test_global_material_ids_and_contact_order_admitted(world):
    args = capture(world)
    certs, report = certify_world(*args)
    assert len(certs) == len(report) == 1 and certs[0][1].unique


@pytest.mark.parametrize(
    "kind",
    ["foreign_world", "skin", "unresolved", "assembled_point", "missing", "box", "native_gap"],
)
def test_capture_corruption_cannot_reach_reference(kind):
    m, d, cat, q, c, w = capture(1)
    if kind == "foreign_world":
        q["material_ids"][0, 0, 0] -= len(cat.points)
    elif kind == "skin":
        q["skin_offset"][0, 0, 0] = 0.001
    elif kind == "unresolved":
        q["meta"][0, 0] = 0
    elif kind == "assembled_point":
        c["geometry"][1, 1] += 0.001
    elif kind == "missing":
        d.contact = []
    elif kind == "box":
        m.geom_type[:] = 6
    elif kind == "native_gap":
        d.contact[0].dist += 0.001
    with pytest.raises(ValueError):
        certify_world(m, d, cat, q, c, w)


def native_scene():
    m = mj.MjModel.from_xml_string("""<mujoco><option cone="elliptic" gravity="0 0 0"/>
    <worldbody><body><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body pos=".195 .01 .01"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body pos="1 0 0"><joint type="slide"/><geom type="sphere" size=".05"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    d.qvel[:] = np.linspace(-0.1, 0.1, m.nv)
    mj.mj_forward(m, d)
    # Each midpoint moved toward the center of the valid face overlap. These
    # are explicitly nonunique test points, not claimed unique mesh receipts.
    certs = []
    for i, c in enumerate(d.contact):
        p = c.pos.copy()
        p[1:] = 0.5 * p[1:] + 0.0025
        certs.append(
            (
                i,
                ContactCertificate(
                    tuple(c.geom),
                    tuple(p),
                    tuple(c.frame[:3]),
                    c.dist,
                    False,
                    "nonunique_test_face",
                    0,
                    0,
                    0,
                    0,
                ),
            )
        )
    return m, d, certs


def test_reference_rebuilds_islands_and_preserves_material_mass_and_force_state():
    m, d, certs = native_scene()
    mass = d.qM.copy()
    material = np.array([np.r_[c.friction, c.solref, c.solimp] for c in d.contact])
    install_certified_geometry(mj, m, d, certs)
    np.testing.assert_array_equal(mass, d.qM)
    np.testing.assert_array_equal(
        material, np.array([np.r_[c.friction, c.solref, c.solimp] for c in d.contact])
    )
    qacc = d.qacc.copy()
    force = d.efc_force.copy()
    qforce = d.qfrc_constraint.copy()
    good, _ = evaluate_candidate(mj, m, d, qacc)
    bad, _ = evaluate_candidate(mj, m, d, qacc + 100)
    assert good["balance_passed"] and not bad["balance_passed"]
    np.testing.assert_array_equal(qacc, d.qacc)
    np.testing.assert_array_equal(force, d.efc_force)
    np.testing.assert_array_equal(qforce, d.qfrc_constraint)


def test_partial_geometry_installation_rejected():
    m, d, certs = native_scene()
    with pytest.raises(ValueError, match="complete"):
        install_certified_geometry(mj, m, d, certs[:-1])


def test_stale_island_jacobians_fail_reference_self_balance(monkeypatch):
    m, d, certs = native_scene()
    monkeypatch.setattr(mj, "mj_island", lambda *args: None)
    with pytest.raises(ValueError, match="own force balance"):
        install_certified_geometry(mj, m, d, certs)
