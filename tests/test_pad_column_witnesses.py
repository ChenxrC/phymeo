"""Compare inclined, reversing and parallel material columns against the public exact face query."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.experiments.compliant_cloth import trace_start
from phymeo.experiments.compliant_handoff import build_handoff
from phymeo.geometry.pad_columns import rounded_pad_columns


@pytest.mark.parametrize(
    "mid,arm",
    [
        ((0.125, 0.18, 0.25), (0.0625, -0.012, 0.035)),
        ((0.55, 0.125, -0.03125), (0.0, 0.037, 0.09)),
        ((-0.6, -0.6, -0.125), (0.09, -0.03, 0.0)),
        ((-0.8, 0.25, 0.25), (0.25, 0.01, -0.125)),
        ((0.0, 0.51, 0.0625), (0.7, 0.0, 0.0)),
        ((0.4, 0.1, 0.15), (0.0, 0.0, 0.04)),
    ],
)
def test_column_distance_matches_complete_capsule_triangle_fan(mid, arm):
    mid, arm = np.array(mid), np.array(arm)
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    normal = np.array([0.0, 0.0, 1.0])
    radius = 0.0625
    sites = rounded_pad_columns(vertices, normal, radius, mid[None, :], arm[None, :])
    capsule = ph.CollisionShape(ph.CollisionKey(0, 0, 0), "capsule", (10, 11), radius=radius)
    boundary = (
        ph.CollisionShape(ph.CollisionKey(0, 1, 0), "triangle", (0, 1, 2)),
        ph.CollisionShape(ph.CollisionKey(0, 1, 1), "triangle", (0, 2, 3)),
    )
    model = ph.CollisionModel((capsule, *boundary))
    paths = {
        capsule.key: ph.CollisionMotion([mid - arm, mid + arm]),
        **{
            s.key: ph.CollisionMotion(vertices[list(s.vertex_ids)] - radius * normal)
            for s in boundary
        },
    }
    with ph.CollisionWorkspace(model) as ws:
        rows = ph.detect(model, paths, ph.CollisionPolicy(margin=2.0), ws).pairs
    exact = min((p.witness.gap_m for p in rows if p.first == capsule.key))
    assert sites.gap_m[0] == pytest.approx(exact, abs=5e-13)
    np.testing.assert_allclose(mid + sites.fraction[0] * arm, sites.points_m[0], atol=2e-15)


def test_opt_in_pad_configuration_keeps_original_front_and_records_local_normals():
    case = build_handoff(robot="articulated", fabric="soft", finger_pad_radius_m=0.002)
    assert len(case.program.finite_pads) == 4
    record = trace_start(case, "finite-pad")
    for patch in record["patches"]:
        if patch["name"] == "table":
            assert "radius_m" not in patch
        else:
            assert patch["radius_m"] == 0.002
            contact = record["frames"][0]["contacts"][patch["name"]]
            assert np.array(contact["site_normals"]).shape == np.array(contact["force_n"]).shape
