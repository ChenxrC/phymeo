"""Compare the accepted nodal angular impulse over the same externally driven folded interval."""

import json

import numpy as np
import pytest
from test_folded_material_seam import H, fixture

import phymeo as ph


@pytest.mark.parametrize("count", [2, 4])
def test_folded_discrete_angular_impulse_at_smaller_substeps(count, tmp_path):
    op, start, rows, masses, rotation = fixture()
    load = ph.StepInputs.zeros(5)
    load.force[rows[0]] = [0, 0, -2]
    h = H / count
    state = start
    external_entry = np.zeros(3)
    external_path = np.zeros(3)
    internal_entry = np.zeros(3)
    internal_path = np.zeros(3)
    summed_node_angular = 0.0
    events = []
    samples = []
    for i in range(count):
        result = op.step(state, load, h)
        assert result.report.accepted
        x = state.particles.position[rows]
        xf = result.state.particles.position[rows]
        external_entry += h * np.cross(x, load.force[rows]).sum(axis=0)
        external_path += h * np.cross((x + xf) / 2, load.force[rows]).sum(axis=0)
        internal_entry += result.discrete_angular_impulse_nms
        summed_node_angular += float(
            np.linalg.norm(np.cross(x, result.nodal_contact_impulse_ns[rows]), axis=1).sum()
        )
        for leg in result.legs:
            internal_path += leg.integrated_angular_impulse_nms
        if len(result.legs) == 2:
            events.append((i + result.legs[0].t1) / count)
        samples.append(
            dict(
                tick=i + 1,
                pressure_n=result.normal_force_n,
                start_angle=result.legs[0].start.fold_angle_rad,
                end_angle=result.legs[-1].end.fold_angle_rad,
                entry_internal_nms=result.discrete_angular_impulse_nms,
                endpoint_position=xf.tolist(),
                endpoint_velocity=result.state.particles.velocity[rows].tolist(),
            )
        )
        state = result.state

    def L(p):
        q = p.particles
        return np.cross(q.position[rows], masses[:, None] * q.velocity[rows]).sum(axis=0)

    change = L(state) - L(start)
    np.testing.assert_allclose(change - external_entry, internal_entry, atol=3e-10, rtol=0)
    np.testing.assert_allclose(internal_path, 0, atol=3e-10, rtol=0)
    assert len(events) == 1 and state.history.active_steps == count
    (tmp_path / ("angular-fold-" + str(count) + ".json")).write_text(
        json.dumps(
            dict(
                substeps=count,
                dt_s=h,
                window_s=H,
                world_force=load.force.tolist(),
                events=events,
                initial_position=start.particles.position.tolist(),
                initial_velocity=start.particles.velocity.tolist(),
                actual_delta_L_nms=change.tolist(),
                entry_external_nms=external_entry.tolist(),
                drift_integrated_external_nms=external_path.tolist(),
                entry_internal_nms=internal_entry.tolist(),
                drift_integrated_internal_nms=internal_path.tolist(),
                actual_less_drift_external_nms=(change - external_path).tolist(),
                sum_entry_node_angular_norms_nms=summed_node_angular,
                samples=samples,
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )
