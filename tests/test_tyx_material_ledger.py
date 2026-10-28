"""The audit never substitutes for actual interpret/C++/CUDA candidates."""

import pytest
from test_material_ledger import collect, require

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize(
    "kind", ["shared", "fixed", "fixed_compliant", "coincident", "none", "empty"]
)
def test_actual_material_equations_and_owned_report(tyx_mode, kind):
    c = collect(tyx_mode, kind)
    r = require(c)
    assert r["max_residual"] <= c["settings"].tolerance
    assert c["diag"]["execution"]["completed_launches"] > 0


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"mode": "interpret", "compute_dtype": "float32", "completed_launches": True},
        {"mode": "wrong", "compute_dtype": "float32", "completed_launches": 8},
    ],
)
def test_actual_execution_metadata_must_be_typed_and_bound(tyx_mode, bad):
    from test_material_ledger import ledger, mutable

    import phymeo as ph

    c = collect(tyx_mode)
    book = ledger(c)
    d = mutable(c["diag"])
    if isinstance(bad, dict) and bad["mode"] == "interpret":
        bad = dict(bad, mode=tyx_mode)
    d["execution"] = bad
    with pytest.raises(ph.SolverFailure):
        require(c, book=book, diag=d)


def test_actual_nonrepresentable_fixed_coordinate_restoration_is_explicit(tyx_mode):
    import numpy as np
    from test_material_path import material_case

    model, initial, constraints = material_case("fixed")
    initial.velocity[0] = 0
    initial.position += np.array([0.1, 0.2, 0.3])
    c = collect(tyx_mode, setup=(model, initial, constraints))
    r = require(c)
    assert np.any(r["conversion_displacement"][0] != 0)
    np.testing.assert_array_equal(
        r["conversion_displacement"][0], -r["fixed_restoration_displacement"][0]
    )
    np.testing.assert_array_equal(c["final"].position[0], initial.position[0])


@pytest.mark.parametrize("failure", ["path", "close"])
def test_actual_audited_late_failure_and_retry(tyx_mode, failure):
    from test_material_ledger import check_audited_transaction

    check_audited_transaction(tyx_mode, failure)


def test_actual_bounded_material_ledger(tyx_mode):
    import numpy as np

    import phymeo as ph

    setup = (
        ph.ParticleModel(np.ones(2), np.zeros(3)),
        ph.ParticleState(np.array([[0.0, 0, 0], [2.0, 0, 0]]), np.zeros((2, 3))),
        ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1)),
    )
    c = collect(tyx_mode, setup=setup)
    r = require(c)
    np.testing.assert_allclose(r["material_impulses"], [[5, 0, 0], [-5, 0, 0]], atol=1e-6)


def test_actual_triangle_robot_prescribed_pair_ledger(tyx_mode):
    from test_material_ledger import check_triangle_robot_requested_pair

    check_triangle_robot_requested_pair(tyx_mode)
