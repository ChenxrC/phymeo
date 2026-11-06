"""One-evaluation reuse must preserve actual geometry, material addresses and forces."""

from collections.abc import Mapping

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph
from phymeo.solvers.cloth_link_contact import ClothLinkContacts


def assert_same(actual, expected):
    """Compare every published field without rounding or dropping material metadata."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert tuple(actual) == tuple(expected)
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_same(a, b)
    else:
        assert actual == expected


def contacts(program, *, local_reuse):
    return ClothLinkContacts(
        program.cloth_domain, program.surfaces, program.cloth_links.areas,
        program.robot_contact, program.cloth_links.law, program.cloth_links.operator,
        local_reuse=local_reuse,
    )


def test_local_reuse_parameter_preserves_real_two_newton_split_face_load():
    # Existing public geometry and force sampling must work before the new API RED.
    program, state, _ = make_scene(split=True)
    geometry = program._link_geometry(state)
    expected = program.cloth_links.evaluate(state, geometry)
    np.testing.assert_allclose(expected[0].sum(0), [0, 0, -2], rtol=0, atol=1e-13)
    np.testing.assert_allclose(expected[1]["cloth_link_efforts"]["arm"], [2], rtol=0, atol=1e-13)
    assert len(expected[1]["cloth_links"]) == 2
    assert all(row["normal_force_n"] > 0 for row in expected[1]["cloth_links"])
    # The frozen first RED is unexpected keyword local_reuse on this existing class.
    reused = contacts(program, local_reuse=True)
    assert_same(reused.evaluate(state, geometry), expected)
