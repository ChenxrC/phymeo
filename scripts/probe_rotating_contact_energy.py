"""Exact interval evidence for a rotating finite-contact candidate, not a solver.

No phymeo/native execution, tolerance modification, or performance claim.
The exact ideal angle atan(3/4) is enclosed by an alternating rational series.
"""

import argparse
import json
from fractions import Fraction as F
from pathlib import Path


def atan_interval(x, terms=80):
    assert 0 < x < 1 and terms % 2 == 0
    lower = sum(((-1) ** k * x ** (2 * k + 1) / (2 * k + 1) for k in range(terms)), F(0))
    return lower, lower + x ** (2 * terms + 1) / (2 * terms + 1)


def record(bounds):
    lower, upper = bounds
    assert lower <= upper
    return dict(
        exact_lower=str(lower), exact_upper=str(upper), midpoint_display=float((lower + upper) / 2)
    )


def evidence():
    lo, hi = atan_interval(F(3, 4))
    assert hi - lo < F(1, 10**20)
    # m=I=h=1; x0=(1,0); x1=(2,3/2); pre v=(1,3/2); pre omega=theta.
    # Initial vy=3/2-p, omega0=theta+p. J0 lever=1; J1 lever=5/2.
    p = ((F(3, 2) - hi) / 2, (F(3, 2) - lo) / 2)
    closing = (F(5, 2) * lo - F(3, 5), F(5, 2) * hi - F(3, 5))
    response = F(29, 4)
    eta = tuple(v / response for v in closing)
    loss = (closing[0] ** 2 / (2 * response), closing[1] ** 2 / (2 * response))
    gain = (p[0] ** 2, p[1] ** 2)
    total = (gain[0] - loss[1], gain[1] - loss[0])
    assert total[0] > F(1, 10)  # A strict analytic energy violation, not a fitted tolerance.
    # Same final geometry but independent initial velocities for prescribed p=1.
    good_held = (F(1, 2) - hi, F(1, 2) - lo)
    good_total = (good_held[0] - loss[1], good_held[1] - loss[0])
    assert good_total[1] < -F(1, 5)
    # Symbolic coefficient checks in a free variable theta: L=theta+3/2 for both states.
    assert F(2) * F(3, 2) - F(3, 2) * F(1) == F(3, 2)
    assert F(2) * F(4, 5) - F(3, 2) * (-F(3, 5)) == F(5, 2)
    assert F(1) + F(5, 2) ** 2 == response
    return dict(
        status="candidate_research_not_automatic_contact_acceptance",
        native_calls=0,
        theta=record((lo, hi)),
        model=dict(mass_kg=1, inertia_kg_m2=1, duration_s=1, friction=0, radius_m=0),
        geometry=dict(
            local_triangle=[[0, 0, -1], [3, 0, 0], [0, 0, 1]],
            node_start=[1, 0, 0],
            node_end=[2, 1.5, 0],
            final_normal=[-0.6, 0.8, 0],
            final_lever_m=2.5,
        ),
        path_proof="tan(theta*t)<=t*tan(theta)=3*t/4<=3*t/(2*(1+t)); hence signed gap>=0 for every t in [0,1]. Projection stays in [0,5/2] inside the finite triangle. This is an analytic one-sided proof, not unsigned positive-clearance CCD or sampled safety.",
        bad=dict(
            held_p_Ns=record(p),
            held_delta_energy_J=record(gain),
            terminal_eta_Ns=record(eta),
            terminal_energy_loss_J=record(loss),
            total_delta_energy_J=record(total),
        ),
        good=dict(
            held_p_Ns=1,
            held_delta_energy_J=record(good_held),
            total_delta_energy_J=record(good_total),
        ),
        angular_momentum="Initial and held Lz=theta+3/2. Terminal node moment=+(5/2)*eta, rotor moment=-(5/2)*eta, so total Lz unchanged.",
        limitation="Ideal exact-angle prescribed vectors only. No production solve, native backend acceptance, general tree energy proof, event partition, feature lifecycle, or full H1 claim. The good control is the same scene with changed initial velocities, not a second independent scene.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = evidence()
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "native_calls": 0,
                "bad_delta_energy_J": data["bad"]["total_delta_energy_J"]["midpoint_display"],
                "good_delta_energy_J": data["good"]["total_delta_energy_J"]["midpoint_display"],
            }
        )
    )


if __name__ == "__main__":
    main()
