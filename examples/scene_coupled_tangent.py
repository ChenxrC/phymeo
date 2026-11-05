"""Public finite clamp with actual normal loads and persistent tangent state.

Run with a source checkout or an installed Phymeo wheel. HOST FP64 only.
Explicit stations are the existing degree-two triangle-fan quadrature of each
finite rectangular patch; no state is pinned or overwritten during stepping.
"""

import json

import numpy as np

from phymeo import (
    CollisionKey,
    ContactSceneBuilder,
    MaterialStationRegistration,
    MaterialStationRegistry,
    NormalLayer,
    StationKey,
)


def run(*, mass=1.0, force_n=0.5, steps=48):
    b = ContactSceneBuilder(gravity=(0.0, 0.0, 0.0))
    b.add_box(10, (0.3, 0.3, 0.1), mass=None, position=(0, 0, -0.2))
    b.add_box(20, (0.1, 0.1, 0.05), mass=mass)
    b.add_box(30, (0.3, 0.3, 0.1), mass=None, position=(0, 0, 0.2))
    law = NormalLayer(0.06, 200.0, 0.2, 0.5)
    b.pair(10, 20, law)
    b.pair(20, 30, law)
    b.pair(10, 30, None)
    corners = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    anchors = [
        wl * corners[e] + wr * corners[(e + 1) % 4]
        for e in range(4)
        for wl, wr in ((1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3))
    ]
    stations = []
    for side, (a, other, z) in enumerate(((10, 20, -0.05), (20, 30, 0.05))):
        for index, (x, y) in enumerate(anchors):
            stations.append(
                MaterialStationRegistration(
                    StationKey(0, 12 * side + index),
                    CollisionKey(0, a, 0),
                    CollisionKey(0, other, 0),
                    "second" if side == 0 else "first",
                    (x, y, z),
                    (1, 0, 0),
                    2 * side,
                    2 * side + 1,
                    first_feature=(1, 3, 5, 7),
                    second_feature=(0, 2, 4, 6),
                )
            )
    with b.build() as scene:
        scene.enable_coupled_tangent(MaterialStationRegistry(stations), stiffness_n_per_m=200.0)
        row = scene.rigid_rows[0, 20]
        external = scene.inputs()
        external.rigid.force[row, 0] = force_n
        maximum_energy = maximum_motion = momentum_error = 0.0
        minimum_normal = float("inf")
        slides = 0
        for _ in range(steps):
            before = scene.state
            frame = scene.step(1 / 1024, external)
            receipt = scene.tangent_receipt
            assert receipt.accepted
            assert {r.identity.generation for r in scene.tangent_history.active} == {1}
            minimum_normal = min(minimum_normal, sum(c.force_n for c in frame.loads.contacts))
            maximum_energy = max(maximum_energy, abs(receipt.energy_residual_j))
            maximum_motion = max(maximum_motion, receipt.material_motion_error_m)
            slides += sum(r.update.regime == "slide" for r in scene.tangent_history.active)
            momentum_error = max(
                momentum_error,
                float(
                    np.max(
                        np.abs(
                            mass * (frame.state.rigid.velocity[row] - before.rigid.velocity[row])
                            - (external.rigid.force[row] + frame.loads.inputs.rigid.force[row])
                            / 1024
                        )
                    )
                ),
            )
        assert minimum_normal > 3.999 and maximum_energy < 1e-8 and momentum_error < 1e-12
        return dict(
            mass_kg=mass,
            external_force_n=force_n,
            steps=steps,
            station_count=len(scene.tangent_history.active),
            sliding_station_steps=slides,
            displacement_x_m=float(frame.state.rigid.position[row, 0]),
            final_tangent_force_x_n=float(frame.loads.inputs.rigid.force[row, 0]),
            minimum_normal_n=minimum_normal,
            maximum_motion_error_m=maximum_motion,
            maximum_energy_residual_j=maximum_energy,
            maximum_momentum_error=momentum_error,
        )


if __name__ == "__main__":
    print(json.dumps([run(mass=m, force_n=f) for m, f in ((1.0, 0.5), (2.0, 3.0))], indent=2))
