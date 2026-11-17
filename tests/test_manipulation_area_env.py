"""I3j CUDA task lifecycle with fresh full-volume pressure geometry.

This new protocol cube uses the registered I3i robot, not its old sphere or
supplied surfaces. The support pair is frictionless area contact; fingers and
the separate native baseline retain native contact. No grasp skill is claimed.
"""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from test_manipulation_area_observation import H
from test_manipulation_area_observation import _scene as protocol_scene
from test_manipulation_env import (
    evidence,
    metadata,
    owned_host,
    physical,
    record,
    semantic_transition,
)
from test_manipulation_env import (
    step as env_step,
)
from test_mujoco_tyx_area_pressure_lifecycle import _same

from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import AreaPressureIdentity, quadrature_area_pressure
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh
from phymeo.training.manipulation_checkpoint import ManipulationEnvCheckpoint
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_observation import (
    AreaPressureRoleBinding,
    ManipulationSuccessReader,
)
from phymeo.training.manipulation_tasks import Primitive, evaluate_goal
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxEnvError

pytestmark = pytest.mark.cuda
KEYS = (41, 7, 305)
MODEL_KEYS = ("area-soft", "native", "area-stiff")
PAIR = ("object", "support-body")
AREA_FIELDS = (
    "area_meta",
    "area_ids",
    "area_data",
    "area_measure_words",
    "area_declared_mask",
)
# Outward faces of the cube. Connecting each to node 8 partitions the full
# volume into twelve tetrahedra; every exterior node has zero pressure.
FACES = (
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)
CELL_IDS = ("bottom-0", "bottom-1") + tuple(f"side-or-top-{i}" for i in range(10))


def _cube_scene():
    original = protocol_scene()
    replacements = (
        ('<mujoco model="area-success-protocol">', '<mujoco model="area-env-protocol">'),
        (
            '<body name="support-body"><geom name="support" type="plane" size="1 1 .05"/></body>',
            '<body name="support-body" pos="0 0 -.005"><geom name="support" type="box" size=".01 .01 .005"/></body>',
        ),
        (
            '<geom name="object-0" type="sphere" size=".05" mass=".08"/>',
            '<geom name="object-0" type="box" size=".05 .05 .05" mass=".08"/>',
        ),
    )
    xml = original.mjcf
    for before, after in replacements:
        assert xml.count(before) == 1
        xml = xml.replace(before, after)
    return replace(
        original,
        key="area-env-protocol",
        mjcf=xml,
        object_shapes=(Primitive("box", (0.05, 0.05, 0.05)),),
        object_label_zh="协议测试立方体",
        object_label_en="protocol cube",
    )


def _asset(model, center_pressure):
    vertices = np.array(
        (
            (-0.05, -0.05, -0.05),
            (0.05, -0.05, -0.05),
            (0.05, 0.05, -0.05),
            (-0.05, 0.05, -0.05),
            (-0.05, -0.05, 0.05),
            (0.05, -0.05, 0.05),
            (0.05, 0.05, 0.05),
            (-0.05, 0.05, 0.05),
            (0, 0, 0),
        )
    )
    pressure = TetraPressureMesh(
        tuple(f"corner-{i}" for i in range(8)) + ("center",),
        vertices,
        tuple((8, *face) for face in FACES),
        (0,) * 8 + (center_pressure,),
        CELL_IDS,
        pressure_domain="full",
    )
    rigid = RigidPressureMesh(
        ("plate-0", "plate-1", "plate-2", "plate-3"),
        ((-0.01, -0.01, 0.005), (0.01, -0.01, 0.005), (0.01, 0.01, 0.005), (-0.01, 0.01, 0.005)),
        ((0, 1, 2), (0, 2, 3)),
        ("plate-a", "plate-b"),
    )
    return BodyAreaPressureAsset(
        AreaPressureIdentity(*PAIR, model + "-full-pressure", "finite-rigid-support"),
        pressure,
        rigid,
    )


def _declarations():
    scene = _cube_scene()
    compilation = compile_mujoco_xml(scene.mjcf, require_asset=False)
    specs = tuple(
        MuJoCoTyxModelSpec(
            name,
            compilation,
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=() if name == "native" else (PAIR,),
        )
        for name in MODEL_KEYS
    )
    assets = {
        "area-soft": (_asset("area-soft", 100000),),
        "area-stiff": (_asset("area-stiff", 125000),),
    }
    return dict(
        scenes=dict.fromkeys(KEYS, scene),
        specs=specs,
        worlds=tuple(MuJoCoTyxWorldSpec(k, m) for k, m in zip(KEYS, MODEL_KEYS, strict=True)),
        assets=assets,
        roles={
            name: (AreaPressureRoleBinding(rows[0].identity, "A", "support"),)
            for name, rows in assets.items()
        },
        action_specs={
            key: MuJoCoTyxActionSpec(
                (0, 1, 2), scale=np.full(3, 0.1, np.float32), bias=np.full(3, 0.01, np.float32)
            )
            for key in KEYS
        },
    )


def _env_arguments(declarations, *, auto_reset=True, horizons=None):
    scenes = declarations["scenes"]
    return dict(
        goals={key: scene.goals[scene.default_goal] for key, scene in scenes.items()},
        action_specs=declarations["action_specs"],
        horizons=horizons or dict.fromkeys(KEYS, 8),
        auto_reset=auto_reset,
    )


def _make(stack, declarations, *, as_env=True, assets=None, auto_reset=True, horizons=None):
    batch = stack.enter_context(
        MuJoCoTyxHeterogeneousBatch(
            declarations["specs"],
            declarations["worlds"],
            mode="cuda",
        )
    )
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(
        batch,
        declarations["assets"] if assets is None else assets,
    )
    env = None
    if as_env:
        env = stack.enter_context(
            HeterogeneousManipulationEnv(
                batch,
                declarations["scenes"],
                area_scene=scene,
                area_roles=declarations["roles"],
                **_env_arguments(declarations, auto_reset=auto_reset, horizons=horizons),
            )
        )
    return env, batch, scene


def _native_controls(batch):
    return {
        key: MuJoCoControl(
            np.full(row.nu, 0.01, np.float32),
            qfrc_applied=np.zeros(row.nv, np.float32),
            xfrc_applied=np.zeros((row.nbody, 6), np.float32),
        )
        for key, row in batch.layout.items()
    }


def _check_error(error, phase, *, state):
    assert error.phase == phase and error.action_accepted is True
    assert error.state_accepted is state and error.reset_accepted is False
    assert error.needs_recovery is True and error.transition is None


def _check_no_success(transition):
    assert not transition.terminated.any().item() and not transition.reward.any().item()
    assert all(not result.success for result in transition.results.values())


def test_area_env_matches_independent_auto_scene_reader_and_native_baseline(tmp_path):
    declarations = _declarations()
    with ExitStack() as stack, evidence(tmp_path / "area-env-independent.json.gz") as records:
        env, batch, scene = _make(stack, declarations)
        _, reference_batch, reference_scene = _make(stack, declarations, as_env=False)
        ws = stack.enter_context(
            reference_batch.prepare_device_workspace(
                include_area_pressure=True, include_bodies=True
            )
        )
        reader = ManipulationSuccessReader(
            ws, declarations["scenes"], area_roles=declarations["roles"]
        )
        assert not reader.read(
            ws.refresh(expected_revision=reference_batch.checkpoint_revision),
            episodes=dict.fromkeys(KEYS, 0),
        )
        assert len(scene.asset_sha256) == 64 and scene.asset_sha256 == reference_scene.asset_sha256
        with pytest.raises(AttributeError):
            scene.asset_sha256 = "0" * 64
        histories = dict.fromkeys(KEYS, None)
        records.append(dict(registered=declarations["scenes"][41].mjcf, assets=scene.assets))
        for tick in range(2):
            transition = env_step(env)
            reference_batch.set_controls(_native_controls(reference_batch))
            assert reference_scene.step(
                H, expected_revision=reference_batch.checkpoint_revision
            ) == dict.fromkeys(KEYS, True)
            frame = ws.refresh(expected_revision=reference_batch.checkpoint_revision)
            values = ws.readback(frame)
            observed = reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
            expected = {
                key: evaluate_goal(
                    declarations["scenes"][key], env.goals[key], observed[key], histories[key]
                )
                for key in KEYS
            }
            histories = {key: value.history for key, value in expected.items()}
            record(
                records,
                env,
                batch,
                transition,
                independent=reference_batch.integration_snapshot(),
                geometry=scene.last_geometry,
            )
            _same(physical(batch), physical(reference_batch))
            _same(scene.last_geometry, reference_scene.last_geometry)
            _same(transition.success_frames, observed)
            _same(transition.results, expected)
            _same(env.history, histories)
            for name, value in values.items():
                _same(owned_host(transition.observation[name]), value)
                _same(owned_host(transition.final_observation[name]), value)
            assert transition.input_frame.ticks == (tick,) * 3
            assert transition.output_frame.ticks == (tick + 1,) * 3
            assert transition.evaluation_frame is transition.output_frame
            assert set(scene.last_geometry) == {41, 305}
            assert owned_host(transition.observation["area_declared_mask"])[:, 0].tolist() == [
                1,
                0,
                1,
            ]
            for name in AREA_FIELDS:
                assert transition.observation[name].is_cuda
            _check_no_success(transition)
            for key in (41, 305):
                result = scene.last_geometry[key][0]
                quad = quadrature_area_pressure(result.surface)
                assert quad.total_area_m2 == pytest.approx(0.0004, rel=1e-9, abs=1e-12)
                assert {cell for cell, _ in quad.source_ids} <= {"bottom-0", "bottom-1"}
                np.testing.assert_allclose(
                    quad.normals, np.tile((0, 0, 1), (len(quad.dA), 1)), rtol=0, atol=1e-12
                )
                assert transition.success_frames[key].support_up_force_n > 0
                assert result.surface.stamp.step_index == tick
            # The native world is a separately declared real native baseline;
            # its device area payload is empty, not a synthetic success label.
            assert frame.area_catalogs[7] == ()
            assert not values["area_meta"][1].any()
            assert transition.success_frames[7].support_up_force_n > 0
        assert all(value.tick == 2 for value in batch.integration_snapshot().values())


def test_area_env_horizons_auto_and_explicit_selected_reset(tmp_path):
    declarations = _declarations()
    horizons = {41: 1, 7: 3, 305: 2}
    with ExitStack() as stack, evidence(tmp_path / "area-env-reset.json.gz") as records:
        automatic, auto_batch, auto_scene = _make(stack, declarations, horizons=horizons)
        manual, manual_batch, manual_scene = _make(
            stack, declarations, horizons=horizons, auto_reset=False
        )
        for sequence, done in ((1, (41,)), (2, (41, 305))):
            left, right = env_step(automatic), env_step(manual)
            record(records, automatic, auto_batch, left, explicit=right)
            _check_no_success(left)
            _check_no_success(right)
            _same(owned_host(left.final_observation), owned_host(right.final_observation))
            _same(left.success_frames, right.success_frames)
            _same(left.results, right.results)
            assert owned_host(left.final_mask).tolist() == [key in done for key in KEYS]
            assert owned_host(left.truncated).tolist() == [key in done for key in KEYS]
            assert dict(manual.pending_reset) == {key: key in done for key in KEYS}
            assert not any(automatic.pending_reset.values())
            before = physical(manual_batch)
            with pytest.raises(MuJoCoTyxEnvError) as blocked:
                env_step(manual)
            assert blocked.value.phase == "action-preflight" and not blocked.value.needs_recovery
            _same(physical(manual_batch), before)
            manual.reset(done, expected_frame=manual.current_frame)
            _same(physical(auto_batch), physical(manual_batch))
            _same(owned_host(automatic.observation), owned_host(manual.observation))
            _same(metadata(automatic), metadata(manual))
            assert not auto_scene.last_geometry and not manual_scene.last_geometry
            assert automatic.current_frame.ticks == ((0, 1, 1) if sequence == 1 else (0, 2, 0))
            assert automatic.episode_ids[41] == sequence
            assert automatic.episode_ids[305] == (1 if sequence == 2 else 0)
        assert automatic.episode_lengths[7] == 2 and automatic.episode_ids[7] == 0


def test_area_env_fresh_paired_restore_and_asset_role_preflight(tmp_path, monkeypatch):
    declarations = _declarations()
    with ExitStack() as stack, evidence(tmp_path / "area-env-restore.json.gz") as records:
        source, source_batch, source_scene = _make(stack, declarations)
        env_step(source)
        capsule = ManipulationEnvCheckpoint.from_bytes(source.checkpoint().to_bytes())
        saved_observation = owned_host(source.observation)
        expected = env_step(source)
        expected_state, expected_geometry = physical(source_batch), source_scene.last_geometry
        fresh, fresh_batch, fresh_scene = _make(stack, declarations)
        fresh.restore(capsule, expected_revision=fresh_batch.checkpoint_revision)
        assert not fresh.needs_recovery and fresh.last_transition is None
        assert not fresh_scene.last_geometry
        _same(owned_host(fresh.observation), saved_observation)
        actual = env_step(fresh)
        record(
            records,
            fresh,
            fresh_batch,
            actual,
            expected=expected,
            geometry=fresh_scene.last_geometry,
        )
        _same(semantic_transition(actual), semantic_transition(expected))
        _same(physical(fresh_batch), expected_state)
        _same(fresh_scene.last_geometry, expected_geometry)
        # These two owners have the same valid configuration and asset digest;
        # an owner guard, not a different-model checkpoint error, must stop CAS.
        revisions = (fresh_batch.checkpoint_revision, source_batch.checkpoint_revision)
        saved_frame = fresh.current_frame
        try:
            fresh_scene.batch = source_batch
            with pytest.raises(ValueError):
                fresh.restore(capsule, expected_revision=fresh_batch.checkpoint_revision)
            assert (fresh_batch.checkpoint_revision, source_batch.checkpoint_revision) == revisions
            _same(physical(fresh_batch), expected_state)
            _same(physical(source_batch), expected_state)
            assert fresh.current_frame is saved_frame and not fresh.needs_recovery
        finally:
            fresh_scene.batch = fresh_batch
        changed_assets = dict(declarations["assets"])
        original = changed_assets["area-soft"][0]
        changed_assets["area-soft"] = (
            BodyAreaPressureAsset(
                original.identity,
                replace(original.pressure_local, nodal_pressure=(0,) * 8 + (100001,)),
                original.rigid_local,
            ),
        )
        changed, changed_batch, changed_scene = _make(stack, declarations, assets=changed_assets)
        assert changed_scene.asset_sha256 != source_scene.asset_sha256
        before = physical(changed_batch)

        def forbidden_restore(*args, **kwargs):
            raise AssertionError("asset mismatch reached physical restore")

        with monkeypatch.context() as patch:
            patch.setattr(changed_batch, "restore", forbidden_restore)
            with pytest.raises(ValueError, match="binding"):
                changed.restore(capsule, expected_revision=changed_batch.checkpoint_revision)
        _same(physical(changed_batch), before)
        assert not changed.needs_recovery
        # A dataclass can be deliberately bypassed with object.__setattr__.
        # The pressure remains a valid full field, but its identity no longer
        # matches the captured asset. Reject before accepting even controls.
        owned_pressure = changed_scene.assets["area-soft"][0].pressure_local
        original_pressure = owned_pressure.nodal_pressure
        mutated_pressure = original_pressure.copy()
        mutated_pressure[-1] += 1
        revision, old_frame = changed_batch.checkpoint_revision, changed.current_frame
        old_metadata = metadata(changed)
        old_cursor = changed.checkpoint().read(changed.binding)[1]["cursor_records"]
        try:
            object.__setattr__(owned_pressure, "nodal_pressure", mutated_pressure)
            with pytest.raises(ValueError):
                env_step(changed)
            assert changed_batch.checkpoint_revision == revision
            assert changed.current_frame is old_frame
            assert not changed.needs_recovery and changed.last_transition is None
            _same(physical(changed_batch), before)
            _same(metadata(changed), old_metadata)
        finally:
            object.__setattr__(owned_pressure, "nodal_pressure", original_pressure)
        assert changed_scene.asset_sha256 != source_scene.asset_sha256
        assert changed.checkpoint().read(changed.binding)[1]["cursor_records"] == old_cursor
        assert set(changed.observation) >= set(AREA_FIELDS)
        _, candidate_batch, candidate_scene = _make(stack, declarations, as_env=False)
        arguments = _env_arguments(declarations)
        for bad_roles in (
            None,
            {
                **declarations["roles"],
                "area-soft": (replace(declarations["roles"]["area-soft"][0], role="other"),),
            },
            {
                **declarations["roles"],
                "area-soft": (
                    replace(
                        declarations["roles"]["area-soft"][0],
                        identity=replace(original.identity, compliant_material="wrong-material"),
                    ),
                ),
            },
        ):
            with pytest.raises(ValueError):
                HeterogeneousManipulationEnv(
                    candidate_batch,
                    declarations["scenes"],
                    area_scene=candidate_scene,
                    area_roles=bad_roles,
                    **arguments,
                )
        with pytest.raises(ValueError):
            HeterogeneousManipulationEnv(
                candidate_batch,
                declarations["scenes"],
                area_scene=None,
                area_roles=declarations["roles"],
                **arguments,
            )
        with pytest.raises(ValueError):
            HeterogeneousManipulationEnv(
                candidate_batch,
                declarations["scenes"],
                area_scene=source_scene,
                area_roles=declarations["roles"],
                **arguments,
            )
        assert all(state.tick == 0 for state in candidate_batch.integration_snapshot().values())


def test_area_env_late_geometry_and_post_physics_export_require_restore(tmp_path, monkeypatch):
    import phymeo.backends.mujoco_area_pressure_scene as geometry

    declarations = _declarations()
    with ExitStack() as stack, evidence(tmp_path / "area-env-failures.json.gz") as records:
        env, batch, scene = _make(stack, declarations)
        previous = env_step(env)
        checkpoint = env.checkpoint()
        before, semantic_before = physical(batch), metadata(env)
        poses = geometry._current_body_poses
        visited = []
        object_body = tuple(env.workspace.model_metadata["area-stiff"]["body_names"]).index(
            "object"
        )

        def bad_last_pose(owner):
            result = poses(owner)
            visited.extend(result)
            if 305 in result:
                result = {key: (p.copy(), r.copy()) for key, (p, r) in result.items()}
                result[305][0][object_body] = 1e20
            return result

        with monkeypatch.context() as patch:
            patch.setattr(geometry, "_current_body_poses", bad_last_pose)
            with pytest.raises(MuJoCoTyxEnvError) as failed:
                env_step(env)
        _check_error(failed.value, "physics", state=False)
        assert 305 in visited
        assert env.last_transition is previous
        assert not scene.last_geometry  # Accepted action revision invalidated old geometry.
        _same(physical(batch), before)
        _same(metadata(env), semantic_before)
        record(
            records,
            env,
            batch,
            failure=repr(failed.value),
            action_accepted=failed.value.action_accepted,
            state_accepted=failed.value.state_accepted,
        )
        for operation in (
            lambda: env_step(env),
            env.checkpoint,
            lambda: env.reset(expected_frame=env.current_frame),
        ):
            with pytest.raises(MuJoCoTyxEnvError, match="restore"):
                operation()
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        assert not env.needs_recovery

        def fail_export(frame):
            raise RuntimeError("injected post-physical area export failure")

        with monkeypatch.context() as patch:
            patch.setattr(env.workspace, "to_torch", fail_export)
            with pytest.raises(MuJoCoTyxEnvError) as exported:
                env_step(env)
        _check_error(exported.value, "task-export", state=True)
        assert env.last_transition is None
        assert all(state.tick == 2 for state in batch.integration_snapshot().values())
        _same(metadata(env), semantic_before)
        assert all(rows[0].surface.stamp.step_index == 1 for rows in scene.last_geometry.values())
        record(
            records,
            env,
            batch,
            failure=repr(exported.value),
            action_accepted=exported.value.action_accepted,
            state_accepted=exported.value.state_accepted,
            geometry=scene.last_geometry,
        )
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        replay = env_step(env)
        record(records, env, batch, replay)
        assert replay.sequence == 2 and not env.needs_recovery
        assert all(frame.tick == 1 for frame in replay.success_frames.values())
        _check_no_success(replay)
