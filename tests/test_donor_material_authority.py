"""Symmetric physical controls, oriented finite jaw observations and held-out admission failures."""

import numpy as np
import pytest

from phymeo.experiments.donor_material_authority import (
    DonorPulse,
    pad_coordinates,
    qualify_response,
)


def test_common_bounded_pulse_changes_sign_and_rotates_with_the_observed_direction():
    pulse = DonorPulse.toward(
        (2.0, 1.0), (5.0, 5.0), duration=0.5, peak_speed=0.3, peak_acceleration=2.0
    )
    # A triangular speed profile hits .3 m/s, covering .075 m in .5 s.
    assert pulse.distance == pytest.approx(0.075)
    p, v = pulse.sample(0.25, 1)
    np.testing.assert_allclose(p, [0.0225, 0.030])
    np.testing.assert_allclose(v, [0.18, 0.24])
    p_end, v_end = pulse.sample(2.0, -1)
    np.testing.assert_allclose(p_end, [-0.045, -0.060])
    np.testing.assert_array_equal(v_end, [0, 0])
    reflected = DonorPulse.toward(
        (-2.0, 1.0), (-5.0, 5.0), duration=0.5, peak_speed=0.3, peak_acceleration=2.0
    )
    for t in (0.0, 0.1, 0.3, 0.49, 0.5):
        for a, b in zip(pulse.sample(t, 1), reflected.sample(t, 1), strict=True):
            np.testing.assert_allclose(a * [-1, 1], b, atol=1e-14)
        for a, b in zip(pulse.sample(t, 1), pulse.sample(t, -1), strict=True):
            np.testing.assert_array_equal(a, -b)
    with pytest.raises(ValueError):
        DonorPulse.toward((0, 0), (0, 0), duration=0.5, peak_speed=0.3, peak_acceleration=2.0)


def jaws(s):
    a = np.array([[-1.0, -0.2, -0.5], [-1.0, -0.2, 0.5], [1.0, -0.2, 0.5], [1.0, -0.2, -0.5]])
    b = a[::-1].copy()
    b[:, 1] = 0.2
    if s < 0:
        a = (a * [-1, 1, 1])[::-1]
        b = (b * [-1, 1, 1])[::-1]
    return a, b


@pytest.mark.parametrize("s", [1.0, -1.0])
def test_measured_material_site_moves_across_a_finite_pad_edge_and_axial_margin(s):
    first, second = jaws(s)
    old = pad_coordinates(first, second, [1.25 * s, 0.0, 0.125], 0.05)
    new = pad_coordinates(first, second, [0.75 * s, 0.125, 0.125], 0.05)
    assert old["side_margin_m"] == pytest.approx(-0.30) and new["side_margin_m"] == pytest.approx(
        0.20
    )
    assert old["axial_margin_m"] == pytest.approx(0.15) and new["axial_margin_m"] == pytest.approx(
        0.025
    )


def test_mirrored_two_mass_material_response_requires_real_transmitted_spring_force():
    # Two accepted symplectic steps. The first changes the 2 kg donor alone;
    # the next transmits the spring force to the 1 kg material.
    for sign in (-1, 1):
        for k in (0.0, 2.0):
            x = np.zeros(2)
            v = np.zeros(2)
            h = 0.25
            for index in (0, 1):
                traction = k * (x[0] - x[1])
                v += h * np.array([(2 * sign - traction) / 2, traction])
                x += h * v
                assert 2 * v[0] + v[1] == pytest.approx((index + 1) * h * 2 * sign)
                if index == 0:
                    assert x[1] == 0
            assert x[1] == pytest.approx(sign / 128 if k else 0.0)
            a, b = jaws(1)
            obs = pad_coordinates(a, b, [1.0 - x[1], 0, 0], 0)
            assert obs["side_margin_m"] == pytest.approx(x[1])


def measurement(t, respond, *, axis=(1.0, 0.0, 0.0), support=0.4, gap=0.001):
    return dict(
        end_s=t,
        material_m=(respond * np.asarray(axis)).tolist(),
        jaw_center_m=[0, 0, 0],
        donor_m=[respond, 0],
        min_A_weaker_pad_n=support,
        mean_A_up_n=0.5,
        min_structure_m=gap,
        first_structure_force_s=None,
        shared=False,
    )


def test_gate_demands_two_adjacent_opposed_responses_and_actual_structural_relief():
    normal = [1, 0, 0]
    approach = [0, 0, 1]
    base = [measurement(i * 0.01, 0.0, gap=-0.0008) for i in (1, 2, 3)]
    plus = [measurement(i * 0.01, d, gap=0.0002) for i, d in enumerate((0.0, 0.001, 0.0012), 1)]
    minus = [measurement(i * 0.01, d, gap=-0.001) for i, d in enumerate((0.0, -0.001, -0.0009), 1)]

    def assess():
        return qualify_response(
            base,
            plus,
            minus,
            normal,
            approach,
            width=0.0015,
            weight=1.0,
            dt=0.01,
            baseline_contact_index=2,
        )

    assert assess()["go"] and assess()["qualifying_signs"] == [1]
    plus[-1]["min_structure_m"] = -0.0005
    assert not assess()["go"] and assess()["authority_signs"] == [1]
    plus[-1]["min_A_weaker_pad_n"] = 0.1
    assert not assess()["authority_signs"]
    plus[-1]["min_A_weaker_pad_n"] = 0.4
    minus[1]["material_m"] = [0.0005, 0, 0]
    assert not assess()["authority_signs"]
