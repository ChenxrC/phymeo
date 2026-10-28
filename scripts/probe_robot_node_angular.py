"""Manufactured counterexample for a candidate robot-node time discretization.

No production solver is invoked: this rejects a proposed extension, not the
accepted fixed-environment robot-plane implementation.
"""

import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for h in (0.1, 0.05, 0.025):
        point = np.array([np.cos(h), np.sin(h), 0.0])
        impulse = np.array([-np.sin(h), np.cos(h), 0.0])
        particle_v1 = np.array([1.0, 0, 0])
        particle_v0 = particle_v1 - impulse
        particle_x0 = point - h * particle_v1
        omega0, omega1 = 2.0, 1.0
        angular0 = omega0 + np.cross(particle_x0, particle_v0)[2]
        angular1 = omega1 + np.cross(point, particle_v1)[2]
        angular_change = angular1 - angular0
        kinetic_change = 0.5 * (
            omega1**2 + particle_v1 @ particle_v1 - omega0**2 - particle_v0 @ particle_v0
        )
        work = impulse @ (particle_v1 - impulse * omega1)
        np.testing.assert_allclose(angular_change, -h * np.cos(h), atol=1e-14)
        np.testing.assert_allclose(kinetic_change, -2 - np.sin(h), atol=1e-14)
        assert work < 0 and kinetic_change < 0 and abs(angular_change) > h * 0.9
        rows.append(
            dict(
                h=h,
                initial_gap=float(np.sin(h)),
                final_gap=0.0,
                angular_change=float(angular_change),
                analytic_angular_change=float(-h * np.cos(h)),
                contact_work=float(work),
                kinetic_change=float(kinetic_change),
            )
        )
    path = Path("docs/benchmarks/p65-robot-node-angular-candidate.json")
    path.write_text(
        json.dumps(
            dict(
                status="candidate_scheme_refuted_not_product_failure",
                assumption="constant scalar I=m=1, q1=q0+h*w1, x1=x0+h*v1, equal_opposite impulse at final robot J1",
                rows=rows,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
