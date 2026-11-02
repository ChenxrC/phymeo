"""Regression controls for changing the cloth while keeping the experiment comparable."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.experiments.cloth_drape import build_drape
from phymeo.experiments.compliant_handoff import build_handoff
from phymeo.experiments.sheet_fabric import REFERENCE, SOFT, material_for_sheet
from phymeo.solvers.compliant_cloth_contact import _elastic


def test_bending_changes_without_softening_the_in_plane_edges():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    faces = [(0, 1, 2), (1, 0, 3)]
    stiff, soft = (material_for_sheet(rest, faces, spec) for spec in (REFERENCE, SOFT))
    v = np.zeros_like(rest)
    flat = 1.01 * rest
    a, b = (_elastic(m, flat, v, 0.0) for m in (stiff, soft))
    assert np.linalg.norm(a) > 1.0
    np.testing.assert_array_equal(a, b)
    folded = rest.copy()
    folded[3] = [0.0, -np.cos(0.2), np.sin(0.2)]
    a, b = (_elastic(m, folded, v, 0.0) for m in (stiff, soft))
    ratio = SOFT.bend_compliance / REFERENCE.bend_compliance
    assert ratio >= 20
    assert np.linalg.norm(b) > 0
    np.testing.assert_allclose(a, ratio * b, atol=2e-10)


def test_fabric_and_grid_are_separately_configurable():
    original = build_handoff()
    assert original.program.cloth.particle_count == 6
    np.testing.assert_array_equal(original.receiver_nodes, [3, 5])
    variant = replace(SOFT, bend_compliance=SOFT.bend_compliance * 2)
    case = build_handoff(fabric=variant, grid=(7, 3), initial_tilt_deg=0.0)
    x = case.initial.domains["cloth"].position
    assert x.shape == (21, 3) and case.program.faces.shape == (24, 3)
    assert sum(1 / case.program.cloth.inverse_mass) == pytest.approx(0.048)
    np.testing.assert_allclose(x[case.receiver_nodes, 0], 0.06)
    np.testing.assert_allclose(x[case.receiver_nodes, 2], [0.0265, 0.0515])
    assert np.all(case.program.material.bending.compliance == variant.bend_compliance)
    assert np.all(case.program.material.distance.compliance == REFERENCE.stretch_compliance)
    with pytest.raises(ValueError):
        build_handoff(fabric="typo")
    with pytest.raises(ValueError):
        build_handoff(grid=(1, 3))
    with pytest.raises(ValueError):
        replace(SOFT, stretch_compliance=0.0)


def test_gravity_drape_comparison_changes_only_bending():
    hard, soft = build_drape("reference"), build_drape("soft")
    np.testing.assert_array_equal(hard.program.faces, soft.program.faces)
    np.testing.assert_array_equal(
        hard.initial.domains["cloth"].position, soft.initial.domains["cloth"].position
    )
    np.testing.assert_array_equal(hard.program.cloth.inverse_mass, soft.program.cloth.inverse_mass)
    assert np.all(hard.program.cloth.inverse_mass > 0)
    assert hard.dt == soft.dt and hard.substeps == soft.substeps
    for a, b in zip(hard.program.patches, soft.program.patches, strict=True):
        np.testing.assert_array_equal(a.geometry.local_vertices, b.geometry.local_vertices)
        assert a.geometry.key == b.geometry.key and a.friction == b.friction
