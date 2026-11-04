"""A frictionless sphere's normal traction passes through its geometric centre."""

import numpy as np
import phymeo as ph


def test_tilted_registered_surface_has_no_spurious_centred_sphere_moment():
    q = np.array([9.010823123389855e-5, -9.010821668198332e-5, 7.278733865634379e-11, 1.0])
    b = ph.ContactSceneBuilder(gravity=(0, 0, 0))
    b.add_box(2, (0.5, 0.375, 0.125), mass=2, orientation_xyzw=q)
    b.add_sphere(1, 0.125, mass=1, position=(0.03125, 0.015625, 0.28125))
    b.pair(1, 2, ph.NormalLayer(0.0625, 128))
    with b.build() as scene:
        sample = scene.sample()
        i = scene.rigid_rows[0, 1]
        assert sample.loads.contacts[0].force_n > 0
        np.testing.assert_array_equal(sample.loads.inputs.rigid.torque[i], 0)
