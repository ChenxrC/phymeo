from contextlib import ExitStack
from dataclasses import replace

import pytest
from multiface_load_cases import STEPS, certify, make_world, physical_hash, whole_window

import phymeo as ph
from phymeo.backends.tyx_world_contact import ClothLinkWorld, TyxWorldContactBatch


@pytest.mark.parametrize("faces", [2, 8])
def test_connected_faces_all_bear_measured_load_and_change_shape(faces):
    worlds = [make_world(i, faces, None) for i in (0, 1)]
    assert all(len(w.initial.domains["cloth"].position) == faces + 2 for w in worlds)
    for w in worlds:
        assert sum(w.areas) == pytest.approx(0.25)
        with pytest.raises(ValueError, match="one three-node face"):
            TyxWorldContactBatch([(7, ClothLinkWorld(w.program, w.initial))], "cuda")
    with ExitStack() as stack:
        steppers = [
            stack.enter_context(
                w.program.stepper(
                    ph.StepSettings(
                        substeps=1,
                        position_iterations=1,
                        velocity_iterations=1,
                        record_substeps=True,
                    )
                )
            )
            for w in worlds
        ]
        ordinary = whole_window(worlds, steppers, isolated=True)
        interleaved = whole_window(worlds, steppers)
    for world, reference, observed in zip(worlds, ordinary, interleaved, strict=True):
        proof = certify(world, observed)
        assert len(proof["midpoints"]) == STEPS and len(proof["observed_changes"]) == faces
        assert [physical_hash(out) for out in reference] == [physical_hash(out) for out in observed]
    # Only values in the timing dictionary may change unnoticed.
    frame = ordinary[0][0]
    sub = frame.report.substeps[0]

    def with_sub(s):
        return replace(frame, report=replace(frame.report, substeps=(s,)))

    new_clocks = {key: value + 1.0 for key, value in sub.timings.items()}
    assert physical_hash(with_sub(replace(sub, timings=new_clocks))) == physical_hash(frame)
    assert physical_hash(
        with_sub(replace(sub, timings={**new_clocks, "extra_clock": 1.0}))
    ) != physical_hash(frame)
    assert physical_hash(
        with_sub(replace(sub, diagnostics={**sub.diagnostics, "extra_force": 1.0}))
    ) != physical_hash(frame)
