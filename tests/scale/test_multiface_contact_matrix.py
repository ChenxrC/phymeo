"""World/face-resolved force readings from the same accepted ragged material batch."""

from contextlib import ExitStack

import numpy as np
import pytest
from multiface_load_cases import H, make_world

import phymeo as ph


@pytest.mark.tyx
def test_true_multiface_batch_reads_one_sensor_table_per_physical_phase(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path))
        worlds = {5: make_world(5, 2, layer), 2: make_world(2, 8, layer)}
        batch = stack.enter_context(
            ph.TyxMultifaceContactBatch(
                [(w, ph.ClothLinkWorld(spec.program, spec.initial)) for w, spec in worlds.items()],
                tyx_mode,
                settings=ph.StepSettings(
                    substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True
                ),
                cache_root=tmp_path,
            )
        )
        programs = {w: spec.program for w, spec in worlds.items()}
        frames = batch.step(H, {w: spec.controls[0] for w, spec in worlds.items()})
        r = [ph.ContactSelector(w, ("link", 0), "shaft") for w in worlds]
        cols = [
            ph.ContactSelector(w, "cloth", key) for w, spec in worlds.items() for key in spec.keys
        ]
        table = ph.measure_cloth_link_matrix(frames, programs, r, cols)
        assert table.phase == "substep-midpoint" and table.force_n.shape == (2, 10, 3)
        for i, (w, spec) in enumerate(worlds.items()):
            want = ph.measure_cloth_link_wrench(frames[w], spec.program, r[i], world=w)
            np.testing.assert_allclose(table.total_force_n[i], want.force_n, atol=1e-12)
            np.testing.assert_allclose(table.force_n[i].sum(0), want.force_n, atol=1e-12)
            np.testing.assert_allclose(table.total_friction_n[i], want.friction_force_n, atol=1e-12)
            for j, col in enumerate(cols):
                if col.world != w:
                    assert table.weight_n[i, j] == 0 and not np.any(table.position_m[i, j])
                    continue
                one = ph.measure_cloth_link_wrench(
                    frames[w], spec.program, r[i], world=w, counterparts=col
                )
                np.testing.assert_allclose(table.force_n[i, j], one.force_n, atol=1e-12)
                report = frames[w].report.substeps[0].diagnostics["robot_contact"]["cloth_links"]
                face = next(v for v in report if v["face"] == col.shape)
                points = np.array([s["point_m"] for s in face["sites"]])
                forces = np.array([s["force_on_cloth_n"] for s in face["sites"]])
                # The capsule and finite-thickness cloth surface are separated by gap along n.
                middle = points - (face["half_thickness_m"] + face["gap_m"] / 2) * face["normal"]
                np.testing.assert_allclose(
                    table.position_m[i, j],
                    np.average(middle, axis=0, weights=np.linalg.norm(forces, axis=1)),
                    atol=2e-12,
                    rtol=0,
                )
        with pytest.raises(ValueError, match="same nonempty"):
            ph.measure_cloth_link_matrix(frames, {5: programs[5]}, r, cols)
