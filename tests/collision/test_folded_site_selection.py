"""Finite fold topology and physical normal-layer activation without pair or frame scheduling."""

import numpy as np
import pytest

import phymeo as ph


def setup_mesh(*, second=True, two_probes=False, reverse=False, unsupported=False):
    faces = []
    attach = {}
    xyz = []
    masses = []
    for index, cx in enumerate((0.0, 4.0) if second else (0.0,)):
        offset = len(xyz)
        body = 10 + index
        xyz.extend([[cx, -1, 0], [cx, 1, 0], [cx - 1, 0, -1 / 16], [cx + 1, 0, -1 / 16]])
        masses.extend([96.0, 96.0, 64.0, 64.0])
        for row, ids, indices in (
            (9, (7, 17, 31), (offset + 2, offset, offset + 1)),
            (3, (17, 49, 31), (offset, offset + 3, offset + 1)),
        ):
            # The catalogue, rather than numeric face ID, fixes the material edge orientation.
            shape = ph.CollisionShape(ph.CollisionKey(3, body, row), "triangle", ids)
            faces.append(shape)
            attach[shape.key] = ph.ParticleAttachment(indices)
    for i in range(2 if two_probes else 1):
        j = len(xyz)
        xyz.append([-0.22 + 0.42 * i, 0, 0.09])
        masses.append(1.0)
        shape = ph.CollisionShape(ph.CollisionKey(3, 70 + i, 4), "particle", (101,))
        faces.append(shape)
        attach[shape.key] = ph.ParticleAttachment((j,))
    if unsupported:
        o = len(xyz)
        xyz.extend([[-2, -1, 0.02], [2, -1, 0.02], [0, 1, 0.02]])
        masses.extend([2.0, 2.0, 2.0])
        shape = ph.CollisionShape(ph.CollisionKey(3, 99, 1), "triangle", (1, 2, 3))
        faces.append(shape)
        attach[shape.key] = ph.ParticleAttachment((o, o + 1, o + 2))
    return (
        ph.CollisionModel(faces[::-1] if reverse else faces),
        attach,
        np.asarray(xyz, dtype=float),
        np.asarray(masses),
    )


def test_selector_infers_hinges_and_exact_activation_interval_from_the_actual_sweep():
    from phymeo.geometry.folded_contact_sites import FoldedSiteSelector

    model, bindings, x, _ = setup_mesh()
    select = FoldedSiteSelector(model, bindings, width_m=1 / 16)
    top = 8
    tangent = np.array([1, 0, 1 / 16])
    tangent /= np.linalg.norm(tangent)
    n = np.cross(tangent, [0, 1, 0])
    surface = np.array([-0.22, 0, -0.22 / 16])
    x[top] = surface + 3 / 32 * n
    end = x.copy()
    end[top] = surface + 1 / 32 * n
    into = select.select(x, end)
    assert into.eligible == 2 and into.broad_rejected == 1
    (site,) = into.sites
    assert site.key.point == (3, 70, 101) and site.key.body == (3, 10)
    assert set(site.key.edge) == {17, 31}
    (band,) = site.intervals
    assert band.t0 == pytest.approx(0.5, abs=2e-12) and band.t1 == 1
    assert band.entry_bracket[0] <= 0.5 <= band.entry_bracket[1] and band.exit_bracket is None
    assert band.start.normal_gap_m == pytest.approx(
        1 / 16
    ) and band.end.normal_gap_m == pytest.approx(1 / 32)
    assert band.end.spatial[0].distance_m > 0
    (departing,) = select.select(end, x).sites
    assert departing.key == site.key
    assert departing.intervals[0].exit_bracket[0] <= 0.5 <= departing.intervals[0].exit_bracket[1]
    np.testing.assert_array_equal(select.select(x, x).sites, ())
    # Same topology, untouched identities. A geometrically translated point now chooses the other patch.
    later = end.copy()
    later[top, 0] += 4
    selected = select.select(later)
    assert len(selected.sites) == 1 and selected.sites[0].key.body == (3, 11)


def test_unregistered_faces_and_capacity_fail_visibly_without_stale_coverage():
    from phymeo.geometry.folded_contact_sites import FoldedSiteSelector

    model, bindings, x, _ = setup_mesh(second=False, two_probes=True)
    x[-2:, 2] = 0.025
    selector = FoldedSiteSelector(model, bindings, width_m=1 / 16)
    with pytest.raises(ph.SolverFailure, match="capacity"):
        selector.select(x, capacity=1)
    assert len(selector.select(x, capacity=2).sites) == 2
    model, bindings, x, _ = setup_mesh(second=False, unsupported=True)
    x[4, 2] = 0.03
    bad = FoldedSiteSelector(model, bindings, width_m=1 / 16)
    with pytest.raises(ph.SolverFailure, match="unsupported.*face|unpaired"):
        bad.select(x)
