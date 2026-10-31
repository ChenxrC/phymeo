"""Fixed whole/step/rebuild grid; analytic geometry paths, not dynamics."""

import dataclasses
from fractions import Fraction as F

import numpy as np
import pytest
from test_shell_boundary import VARIANTS, control, payload, save, vec

from phymeo.constraints.conforming_shell import ShellUnresolved
from phymeo.constraints.shell_boundary import BoundaryProofLimits, SupportHint
from phymeo.geometry.shell_boundary import certify_shell_boundary
from phymeo.geometry.shell_boundary_sweep import certify_shell_boundary_sweep
from phymeo.validation.shell_boundary import verify_shell_boundary, verify_shell_boundary_sweep


def state(name, ref, x, rotation, time):
    result = x.copy()
    if name == "C1":
        result += np.array([1 / 4, 1 / 8, -1 / 16]) @ rotation.T * float(time)
    else:
        delta = abs(time - F(1, 8))
        result[ref.node_ids >= 4] += np.array([float(delta), 0, 0]) @ rotation.T
    return result


@pytest.mark.parametrize("name", ("C1", "C2"))
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("h", (F(1, 64), F(1, 128), F(1, 256)))
def test_full_analytic_grid_and_two_predeclared_rebuilds(name, variant, h, tmp_path):
    # Pair by authored name/variant/h/rebuild/ordinal; retain every payload field.
    snapshots = []
    for rebuild in range(2):
        ref, x, hint, rotation, _ = control(name, variant)
        end = F(1, 8) if name == "C1" else F(1, 4)
        joins = (F(0), end) if name == "C1" else (F(0), F(1, 8), end)
        segments = tuple(
            np.stack((state(name, ref, x, rotation, a), state(name, ref, x, rotation, b)))
            for a, b in zip(joins, joins[1:])
        )
        durations = tuple(float(b - a) for a, b in zip(joins, joins[1:]))
        calls = [("whole", segments, durations)]
        for step in range(int(end / h)):
            a, b = step * h, (step + 1) * h
            calls.append(
                (
                    f"step{step:03d}",
                    (
                        np.stack(
                            (state(name, ref, x, rotation, a), state(name, ref, x, rotation, b))
                        ),
                    ),
                    (float(h),),
                )
            )
        rebuilt = []
        for label, path, duration in calls:
            receipt = certify_shell_boundary_sweep(ref, path, durations=duration, witness_hint=hint)
            checked = verify_shell_boundary_sweep(
                receipt,
                ref,
                path,
                durations=duration,
                witness_hint=hint,
                limits=BoundaryProofLimits(),
            )
            assert checked
            assert checked.verification_kind == "sweep"
            assert checked.created_cells == sum(len(s.cells) for s in receipt.segments)
            assert 0 < checked.operations <= 2_000_000
            assert receipt.status == "weak_material_path_certified"
            assert len(receipt.segments) == len(path)
            assert receipt.operations <= 2_000_000
            assert receipt.path_kind == "constant-derived-director-v1"
            save(
                tmp_path / f"{name}-{variant}-h{h.denominator}-r{rebuild}-{label}.json",
                (ref, path, duration, hint, receipt, checked),
            )
            rebuilt.append(payload((receipt, checked)))
        snapshots.append(rebuilt)
    assert snapshots[0] == snapshots[1]


def _exact_middle_at(start, end, time):
    return tuple(
        tuple(
            F.from_float(float(p)) + time * (F.from_float(float(q)) - F.from_float(float(p)))
            for p, q in zip(a, b, strict=True)
        )
        for a, b in zip(start, end, strict=True)
    )


def _box_xy(points, rows):
    selected = tuple(points[i] for i in rows)
    assert all(p[2] == 0 for p in selected)
    return (
        min(p[0] for p in selected),
        max(p[0] for p in selected),
        min(p[1] for p in selected),
        max(p[1] for p in selected),
    )


def _certified_endpoint(ref, positions, hint, path):
    certificate = certify_shell_boundary(ref, positions, witness_hint=hint)
    checked = verify_shell_boundary(
        certificate, ref, positions, witness_hint=hint, limits=BoundaryProofLimits()
    )
    assert checked and checked.verification_kind == "static"
    assert certificate.status == checked.status == "weak_material_geometry_certified"
    save(path, (ref, positions, hint, certificate, checked))


def test_legal_endpoints_cannot_hide_constant_director_component_crossing(tmp_path):
    ref, x, hint, _, _ = control("C2")
    x[ref.node_ids >= 4, 1] -= 2
    x[ref.node_ids >= 4, 0] += 2
    y = x.copy()
    y[ref.node_ids < 4, 0] += 4
    y[ref.node_ids >= 4, 0] -= 4
    start = _exact_middle_at(x, y, F(0))
    finish = _exact_middle_at(x, y, F(1))
    assert _box_xy(start, range(4)) == (F(-2), F(-1), F(-1), F(1))
    assert _box_xy(start, range(4, 8)) == (F(1), F(2), F(-1), F(1))
    assert _box_xy(finish, range(4)) == (F(2), F(3), F(-1), F(1))
    assert _box_xy(finish, range(4, 8)) == (F(-3), F(-2), F(-1), F(1))
    assert F(1) - F(-1) == 2 > 0 and F(2) - F(-2) == 4 > 0
    # Each endpoint has a valid complete separator for its actual ordering.
    # All cluster/fan/material hints retain their original complete coverage.
    start_pair = dataclasses.replace(
        hint.pairs[0], supports=(SupportHint(vec((1, 0, 0), 0), (1, 1)),), matching=()
    )
    first_hint = dataclasses.replace(hint, pairs=(start_pair,))
    end_pair = dataclasses.replace(
        hint.pairs[0], supports=(SupportHint(vec((-1, 0, 0), 0), (0, 1)),), matching=()
    )
    last_hint = dataclasses.replace(hint, pairs=(end_pair,))
    _certified_endpoint(ref, x, first_hint, tmp_path / "crossing-start.json")
    _certified_endpoint(ref, y, last_hint, tmp_path / "crossing-end.json")
    # Fixed exact interior time: both open boxes coincide. This analytic
    # counterexample is not a sampled approximation to continuous safety.
    interior = _exact_middle_at(x, y, F(3, 8))
    expected = (F(-1, 2), F(1, 2), F(-1), F(1))
    assert _box_xy(interior, range(4)) == _box_xy(interior, range(4, 8)) == expected
    rho = F(1, 32)
    assert all(F.from_float(float(v)) == rho for v in ref.half_thickness_m)
    assert (expected[1] - expected[0]) * (expected[3] - expected[2]) * (2 * rho) == 4 * rho > 0
    with pytest.raises(ShellUnresolved):
        certify_shell_boundary_sweep(ref, (np.stack((x, y)),), durations=(1.0,), witness_hint=hint)


@pytest.mark.parametrize(
    "kind", ("changing_director", "zero_star_inside", "bad_join", "zero_duration")
)
def test_actual_path_domain_and_full_closed_join_failures(kind, tmp_path):
    ref, x, hint, _, _ = control("C2")
    y = x.copy()
    if kind == "changing_director":
        y[0, 2] += 1 / 8
        paths = (np.stack((x, y)),)
        times = (1 / 8,)
    elif kind == "zero_star_inside":
        center = x[:4].mean(axis=0)
        y[:4] = 2 * center - x[:4]
        paths = (np.stack((x, y)),)
        times = (1 / 8,)
        # The endpoint motion is a proper pi rotation in the middle plane.
        # It retains +ez directors and the same geometric box, but changes the
        # contacting MATERIAL corner from A node2 to A node0.
        end_pair = dataclasses.replace(
            hint.pairs[0],
            supports=(
                SupportHint(vec((1, 0, 0), 0), (0, 1)),
                SupportHint(vec((0, 1, 0), 0), (0, 1)),
            ),
            matching=(((0, -1), (4, -1)), ((0, 1), (4, 1))),
        )
        exterior = tuple(
            (
                simplex,
                tuple(
                    tuple(-q for q in component) if axis < 2 else component
                    for axis, component in enumerate(direction)
                ),
            )
            if all(v[0] < 4 for v in simplex)
            else (simplex, direction)
            for simplex, direction in hint.exterior
        )
        end_hint = dataclasses.replace(hint, pairs=(end_pair,), exterior=exterior)
        first = _exact_middle_at(x, y, F(0))
        last = _exact_middle_at(x, y, F(1))
        assert _box_xy(first, range(4)) == _box_xy(last, range(4)) == (F(-2), F(-1), F(-1), F(1))
        assert (
            _box_xy(first, range(4, 8)) == _box_xy(last, range(4, 8)) == (F(-1), F(0), F(1), F(3))
        )
        assert first[2] == last[0] == (F(-1), F(1), F(0))
        _certified_endpoint(ref, x, hint, tmp_path / "zero-star-start.json")
        _certified_endpoint(ref, y, end_hint, tmp_path / "zero-star-end.json")
        # Every A middle edge is (1-2t) times its original edge, hence every
        # actual face/star is (1-2t)^2 times its nonzero original value.
        assert (F(1) - 2 * F(0)) ** 2 == (F(1) - 2 * F(1)) ** 2 == 1
        assert (F(1) - 2 * F(1, 2)) ** 2 == 0
        middle = _exact_middle_at(x, y, F(1, 2))
        assert middle[:4] == ((F(-3, 2), F(0), F(0)),) * 4
        assert middle[4:] == first[4:]
    elif kind == "bad_join":
        paths = (np.stack((x, x)), np.stack((x + 1 / 16, x + 1 / 16)))
        times = (1 / 8, 1 / 8)
    else:
        paths = (np.stack((x, x)),)
        times = (0.0,)
    with pytest.raises((ShellUnresolved, ValueError, TypeError)):
        certify_shell_boundary_sweep(ref, paths, durations=times, witness_hint=hint)


def test_missing_time_node_or_endpoint_is_not_a_certificate():
    ref, x, hint, _, _ = control("C2")
    paths = (np.stack((x, x)),)
    certificate = certify_shell_boundary_sweep(ref, paths, durations=(1 / 8,), witness_hint=hint)
    forged = dataclasses.replace(certificate, segments=())
    assert not verify_shell_boundary_sweep(
        forged, ref, paths, durations=(1 / 8,), witness_hint=hint, limits=BoundaryProofLimits()
    )


def test_independent_nonmidpoint_partition_covers_the_same_constant_path():
    from phymeo.constraints.shell_boundary import BoundaryTimeCell

    ref, x, hint, _, _ = control("C2")
    paths = (np.stack((x, x)),)
    receipt = certify_shell_boundary_sweep(ref, paths, durations=(1 / 8,), witness_hint=hint)
    segment = receipt.segments[0]
    root = segment.cells[0]
    assert root.children == ()
    # All actual points/stars/supports are constant. Reuse their exact constant
    # polynomial facts as an externally authored proof, not a producer rerun.
    assert all(len(row[1]) == 1 for row in root.polynomial_facts)
    first = BoundaryTimeCell((F(0), F(1, 3)), 1, (), root.polynomial_facts, root.sqrt_bits)
    second = BoundaryTimeCell((F(1, 3), F(1)), 1, (), root.polynomial_facts, root.sqrt_bits)
    nodes = (dataclasses.replace(root, children=(1, 2)), first, second)
    alternate = dataclasses.replace(
        receipt, segments=(dataclasses.replace(segment, cells=nodes),), created_cells=6
    )
    assert verify_shell_boundary_sweep(
        alternate, ref, paths, durations=(1 / 8,), witness_hint=hint, limits=BoundaryProofLimits()
    )
    missing = dataclasses.replace(
        alternate, segments=(dataclasses.replace(segment, cells=nodes[:-1]),)
    )
    assert not verify_shell_boundary_sweep(
        missing, ref, paths, durations=(1 / 8,), witness_hint=hint, limits=BoundaryProofLimits()
    )


def test_changing_director_f1_is_explicitly_outside_first_path_domain():
    from phymeo.geometry._shell_contact_field import FieldArithmetic, Meter
    from phymeo.geometry.shell_boundary_sweep import _actual_polynomials, _director_identity

    # Original F1: q(t)=1/8+t/8. Independently c_+=(0,-2q,2),
    # c_-=(0,2q,2), S_0=S_1=(0,0,4), S_2=c_+, S_3=c_-.
    # dS_2 cross S_2(0)=(-1/2,0,0), dS_3 cross S_3(0)=(1/2,0,0).
    original = {"nodes": (0, 1, 2, 3), "faces": ((10, (0, 1, 2)), (20, (1, 0, 3)))}
    first = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 1 / 8], [0, -1, 1 / 8]], dtype=np.float64)
    second = first.copy()
    second[2:, 2] = 1 / 4
    limits = BoundaryProofLimits()
    meter = Meter(limits.strict_limits)
    a = FieldArithmetic((), meter, limits)
    _, _, _, stars, _ = _actual_polynomials(
        original, np.stack((first, second)), meter, (0, 1, 2, 3)
    )
    assert stars[2] == ((F(0),), (F(-1, 4), F(-1, 4)), (F(2),))
    assert stars[3] == ((F(0),), (F(1, 4), F(1, 4)), (F(2),))
    expected = {
        0: (F(0), F(0), F(4)),
        1: (F(0), F(0), F(4)),
        2: (F(0), F(-1, 4), F(2)),
        3: (F(0), F(1, 4), F(2)),
    }
    squared = {0: F(16), 1: F(16), 2: F(65, 16), 3: F(65, 16)}
    with pytest.raises(ShellUnresolved) as raised:
        _director_identity(original, stars, (None, None, expected, squared), a)
    assert raised.value.category == "unsupported_constant_director"
    # This unit control exercises the actual input-derived domain predicate;
    # it does not use an unrelated malformed cluster hint as a rejection oracle.


@pytest.mark.parametrize(
    "mutation", ("transport", "zero_identity", "endpoint", "whole_map", "same_count_cells")
)
def test_whole_path_facts_cannot_be_replaced_by_endpoint_or_hash_claims(mutation):
    ref, x, hint, _, _ = control("C2")
    paths = (np.stack((x, x)),)
    certificate = certify_shell_boundary_sweep(ref, paths, durations=(1 / 8,), witness_hint=hint)
    segment = certificate.segments[0]
    if mutation == "transport":
        changed = dataclasses.replace(segment, component_translations=())
    elif mutation == "zero_identity":
        changed = dataclasses.replace(segment, cross_identities=segment.cross_identities[:-1])
    elif mutation == "endpoint":
        changed = dataclasses.replace(segment, endpoint_points=segment.endpoint_points[:1])
    elif mutation == "whole_map":
        changed = dataclasses.replace(segment, map_functions=segment.map_functions[:-1])
    else:
        root = segment.cells[0]
        changed = dataclasses.replace(
            segment, cells=(dataclasses.replace(root, span=(F(0), F(1, 2))),)
        )
    forged = dataclasses.replace(certificate, segments=(changed,))
    assert (
        verify_shell_boundary_sweep(
            forged, ref, paths, durations=(1 / 8,), witness_hint=hint, limits=BoundaryProofLimits()
        )
        is False
    )


def test_inner_recheck_is_in_the_original_aggregate_created_cell_budget():
    from phymeo.constraints.conforming_shell import ShellProofLimits

    ref, x, hint, _, _ = control("C2")
    limits = BoundaryProofLimits(strict_limits=ShellProofLimits(max_time_cells=1))
    # A constant segment needs one producer root AND one independent checker root.
    # A reset between those passes would incorrectly return a successful receipt.
    with pytest.raises(ShellUnresolved) as raised:
        certify_shell_boundary_sweep(
            ref, (np.stack((x, x)),), durations=(1 / 8,), witness_hint=hint, limits=limits
        )
    assert raised.value.category == "time_cells_exhausted"
