from contextlib import ExitStack
from dataclasses import replace


import pytest
from multiface_load_cases import H, STEPS, certify, make_world, physical_hash, whole_window

import phymeo as ph
from phymeo.core.scene import fingerprint


SETTINGS = ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1, record_substeps=True)


def specs(worlds):
    return [(k, ph.ClothLinkWorld(w.program, w.initial)) for k, w in worlds.items()]


def inputs(worlds, tick):
    return {k: w.controls[tick] for k, w in worlds.items()}


def moved(world, height):
    state = world.initial
    points = state.domains["cloth"]
    initial = ph.SceneState(
        {**state.domains, "cloth": ph.ParticleState(points.position + [0, 0, height], points.velocity)}
    )
    return replace(world, initial=initial)


def hashes(states):
    return {k: fingerprint(v) for k, v in states.items()}


def test_four_loaded_multiface_worlds_pack_one_device_page_per_phase(tyx_mode, tmp_path):
    with ExitStack() as stack:
        normal = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path))
        worlds = {i: make_world(i, 8, normal) for i in (3, 0, 2, 1)}
        ordinary = whole_window(
            list(worlds.values()),
            [stack.enter_context(w.program.stepper(SETTINGS)) for w in worlds.values()],
        )
        reference = normal.stats()
        assert reference["completed_launches"] == 48 and reference["sites"] == 416
        batch = stack.enter_context(
            ph.TyxMultifaceContactBatch(specs(worlds), tyx_mode, settings=SETTINGS, cache_root=tmp_path)
        )
        frames = {k: [] for k in worlds}
        for t in range(STEPS):
            output = batch.step(H, inputs(worlds, t))
            assert tuple(output) == tuple(worlds)
            for k in worlds:
                frames[k].append(output[k])
            for phase, expected in zip(batch.last_phases, ("entry", "midpoint", "endpoint"), strict=True):
                assert phase.phase == expected and phase.substep == 0
                offset = 0
                for (key, start, stop), (world_id, w) in zip(phase.world_offsets, worlds.items(), strict=True):
                    assert key == world_id and start == offset and stop > start
                    faces = [g for g in phase.faces if g.world == key]
                    assert tuple(g.face for g in faces) == w.keys
                    for face, ids in zip(faces, w.faces, strict=True):
                        assert face.nodes == ids and face.link == "shaft" and face.robot == "link"
                        assert face.start == offset < face.stop <= face.start + 2
                        offset = face.stop
                    assert offset == stop
                assert phase.sites == offset
        stats = batch.stats()
        assert stats["completed_launches"] == 12 and stats["sites"] == reference["sites"]
        for field in ("uploads", "downloads", "uploaded_bytes", "downloaded_bytes"):
            assert reference[field] == 4 * stats[field]
        if tyx_mode == "cuda":
            assert stats["uploads"] == stats["downloads"] == 12
            assert stats["uploaded_bytes"] == 79872 and stats["downloaded_bytes"] == 30720
        for (key, w), expected in zip(worlds.items(), ordinary, strict=True):
            assert list(map(physical_hash, frames[key])) == list(map(physical_hash, expected))
            assert len(certify(w, frames[key])["observed_changes"]) == 8
        assert normal.stats() == reference


def test_heterogeneous_keys_empty_rows_shared_ids_reset_and_retry(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path))
        worlds = {91: make_world(0, 2, layer), 7: make_world(0, 8, layer), 36: moved(make_world(0, 2, layer), -0.5)}
        native = {k: stack.enter_context(w.program.stepper(SETTINGS)) for k, w in worlds.items()}
        batch = stack.enter_context(
            ph.TyxMultifaceContactBatch(specs(worlds), tyx_mode, settings=SETTINGS, cache_root=tmp_path)
        )
        initial = hashes(batch.snapshot())
        assert initial == hashes({k: w.initial for k, w in worlds.items()})
        with pytest.raises(ValueError, match="site capacity"):
            batch.step(H, inputs(worlds, 0), site_capacity=1)
        assert hashes(batch.snapshot()) == initial and not batch.last_phases
        assert batch.stats()["completed_launches"] == 0
        states = {k: w.initial for k, w in worlds.items()}
        for tick in (0, 1):
            ordinary = {k: native[k].step(s, worlds[k].controls[tick], H) for k, s in states.items()}
            out = batch.step(H, inputs(worlds, tick), site_capacity=32)
            assert {k: physical_hash(v) for k, v in out.items()} == {k: physical_hash(v) for k, v in ordinary.items()}
            for phase in batch.last_phases:
                assert phase.world_offsets[-1] == (36, phase.sites, phase.sites)
                assert {g.world for g in phase.faces} == {91, 7}
            diag = out[36].report.substeps[0].diagnostics["robot_contact"]
            assert not diag["cloth_links"] and initial[36] != fingerprint(out[36].state)
            states = {k: v.state for k, v in ordinary.items()}
            # Reset only one actual configuration; other accepted states must survive.
            if tick == 0:
                before = hashes(batch.snapshot())
                batch.reset([7])
                states[7] = worlds[7].initial
                assert hashes(batch.snapshot()) == {**before, 7: initial[7]}
                # The returned result is caller-owned, even for a shared cloth node.
                out[91].state.domains["cloth"].position[:] = 99
                assert hashes(batch.snapshot())[91] == before[91]


def test_separated_cloths_advance_without_launch_and_late_path_verdict_is_preserved(tyx_mode, tmp_path):
    with ExitStack() as resources:
        layer = resources.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path))
        clear = moved(make_world(0, 2, layer), -0.5)
        empty = resources.enter_context(
            ph.TyxMultifaceContactBatch(specs({4: clear}), tyx_mode, settings=SETTINGS, cache_root=tmp_path)
        )
        frame = empty.step(H, {4: clear.controls[0]})[4]
        assert frame.report.accepted and fingerprint(frame.state) != fingerprint(clear.initial)
        assert len(empty.last_phases) == 3 and all(s.sites == 0 for s in empty.last_phases)
        assert empty.stats()["completed_launches"] == 0

        # Both initially have a finite nondegenerate shaft-to-face separation.
        # One actual endpoint trajectory exceeds its frozen hard penetration budget.
        invalid = moved(make_world(2, 2, layer), 0.11)
        good = make_world(1, 8, layer)
        worlds = {8: good, 1: invalid}
        serial = {k: resources.enter_context(w.program.stepper(SETTINGS)).step(w.initial, w.controls[0], H) for k, w in worlds.items()}
        assert serial[8].report.accepted and not serial[1].report.accepted
        assert "penetration" in serial[1].report.reason
        group = resources.enter_context(
            ph.TyxMultifaceContactBatch(specs(worlds), tyx_mode, settings=SETTINGS, cache_root=tmp_path)
        )
        out = group.step(H, inputs(worlds, 0))
        assert {k: physical_hash(v) for k, v in out.items()} == {k: physical_hash(v) for k, v in serial.items()}
        assert len(group.last_phases) == 3
        assert hashes(group.snapshot()) == hashes({8: out[8].state, 1: invalid.initial})
        # A late configuration error in another registered world cannot publish its neighbour.
        old_states, old_trace = hashes(group.snapshot()), group.last_phases
        with pytest.raises(ValueError, match="domains"):
            group.step(H, {8: good.controls[1], 1: ph.SceneInputs({})})
        assert hashes(group.snapshot()) == old_states and group.last_phases is old_trace


def test_fixed_page_boundary_packs_across_worlds_and_substeps(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path))
        w = make_world(0, 8, layer)
        settings = replace(SETTINGS, substeps=2)
        model = ph.ClothLinkWorld(w.program, w.initial)
        batch = stack.enter_context(ph.TyxMultifaceContactBatch(
            [(i, model) for i in range(9)], tyx_mode, settings=settings, cache_root=tmp_path
        ))
        expected = stack.enter_context(w.program.stepper(settings)).step(w.initial, w.controls[0], H)
        out = batch.step(H, {k: w.controls[0] for k in range(9)})
        assert all(physical_hash(frame) == physical_hash(expected) for frame in out.values())
        assert [p.substep for p in batch.last_phases] == [0, 0, 0, 1, 1, 1]
        assert batch.last_phases[0].sites == 144
        launches = sum((p.sites + 127) // 128 for p in batch.last_phases)
        assert batch.stats()["completed_launches"] == launches and launches > 6
