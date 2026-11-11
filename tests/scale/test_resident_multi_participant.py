"""ROB1b R01: independent free-body spring law and complete public transactions.

Frozen before product implementation: m=(2,1), I=1/8, r=1/8, x=+-7/64,
incoming v=+-1/64, k=64, d=mu=g=0, h=1/1024, 40 uninterrupted steps.
No product geometry/dynamics routine supplies the radial spring oracle.
"""
import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLDS = (503, 91, 1207, 6)
PARTICIPANTS = (7, 29)
MASSES = np.array([2.0, 1.0])
INERTIA = 1 / 8
RADIUS = 1 / 8
STEPS = 40


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape,
                    bytes_hex=value.tobytes(order="C").hex(), values=value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if type(value) is object:
        return dict(opaque_process_identity=True)
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[(encode(k), encode(v)) for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unhandled evidence type: " + type(value).__name__)


def save(directory, name, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(encode(payload), stream, allow_nan=False, separators=(",", ":"))


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def inputs(count=1, *, reverse=False, transformed=False, external=False):
    rotation = np.eye(3)
    translation = np.zeros(3)
    orientation = np.array([0.0, 0, 0, 1])
    if transformed:
        rotation = np.array([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]])
        translation = np.array([3.0, -2, 0.5])
        orientation = np.array([0.0, 0, 2**-0.5, 2**-0.5])
    positions = np.array([[-7 / 64, 0, 0], [7 / 64, 0, 0]]) @ rotation.T + translation
    velocities = np.array([[1 / 64, 0, 0], [-1 / 64, 0, 0]]) @ rotation.T
    owners, shapes, bindings, controls, laws, definitions = [], [], [], {}, {}, {}
    for world in WORLDS[:count]:
        keys = []
        for side, participant in enumerate(PARTICIPANTS):
            key = ph.ResidentParticipantKey(world, participant)
            model = ph.FloatingRobotModel(
                ph.RigidBodyModel(np.array([1 / MASSES[side]]),
                                  np.full((1, 3), INERTIA), np.zeros(3)),
                np.zeros(3),
            )
            initial = ph.FloatingRobotState(
                np.concatenate((positions[side], orientation)),
                np.concatenate((velocities[side], np.zeros(3))),
            )
            tree = ph.FloatingRobotTree(model, initial, kp=np.zeros(0),
                                       kd=np.zeros(0), effort_limit=np.zeros(0))
            owner = ph.ResidentParticipant(key, tree, (101 + 2 * side,))
            shape_key = ph.CollisionKey(world, owner.body_ids[0], 11)
            shape = ph.CollisionShape(shape_key, "sphere", (41,), RADIUS, revision=3)
            binding = ph.ResidentFloatingTreeBinding(
                shape_key, world, 0, np.zeros((1, 3)), topology=owner.compiled.topology
            )
            force = np.zeros((1, 3))
            if external:
                force[0] = (1 / 32) * rotation[:, 1]
            control = tree.inputs(np.zeros((0, 3)),
                                  ph.FloatingRobotInputs(np.zeros(0), force, np.zeros((1, 3))))
            owners.append(owner)
            shapes.append(shape)
            bindings.append(ph.ResidentParticipantBinding(key, binding))
            controls[key] = control
            definitions[key] = dict(mass=MASSES[side], inertia=INERTIA, model=model,
                                    initial=initial, body_id=owner.body_ids[0], shape=shape,
                                    binding=binding, control=control)
            keys.append(shape_key)
        laws[tuple(sorted(keys))] = ph.RobotContactMaterial(64, 0, 0)
    if reverse:
        owners.reverse()
        shapes.reverse()
        bindings.reverse()
    return dict(participants=tuple(owners), model=ph.CollisionModel(tuple(shapes)),
                bindings=tuple(bindings), controls=controls, laws=laws,
                definitions=definitions, position=positions, velocity=velocities,
                orientation=orientation, rotation=rotation, translation=translation,
                worlds=tuple(sorted(WORLDS[:count])), transformed=transformed,
                registration_reversed=reverse, h=H, steps=STEPS)


def radial_step(position, velocity):
    delta = position[1] - position[0]
    distance = np.linalg.norm(delta)
    assert distance > 0
    normal = delta / distance
    force = 64 * max(0.0, 2 * RADIUS - distance)
    wrench = np.zeros((2, 6))
    wrench[0, :3] = -force * normal
    wrench[1, :3] = force * normal
    next_velocity = velocity + H * wrench[:, :3] / MASSES[:, None]
    return position + H * next_velocity, next_velocity, wrench


def physical_state(snapshot):
    return {world: dict(point.participants) for world, point in snapshot.items()}


def record(scene, plan, before):
    return dict(entry=before, accepted=scene.snapshot(), receipts=plan.last_accepted,
                status=plan.last_status, plan_stats=plan.stats(), scene_stats=scene.stats())


def audit_world(data, world, tick, before, after, receipt, expected_position,
                expected_velocity, expected_wrench):
    keys = tuple(ph.ResidentParticipantKey(world, p) for p in PARTICIPANTS)
    assert tuple(before.participants) == tuple(after.participants) == keys
    assert (before.tick, after.tick, receipt.entry_tick, receipt.exit_tick) == (tick, tick + 1, tick, tick + 1)
    assert set(receipt.participants) == set(keys)
    assert len(receipt.sites) == 1 and not receipt.static_reactions
    site = receipt.sites[0]
    assert site.pair == tuple(sorted(data["laws"])[data["worlds"].index(world)])
    close(site.first_force_n, expected_wrench[0, :3])
    close(site.second_force_n, expected_wrench[1, :3])
    close(site.first_force_n + site.second_force_n, np.zeros(3))
    for side, key in enumerate(keys):
        first, last, item = before.participants[key], after.participants[key], receipt.participants[key]
        assert first.world == last.world == world
        assert item.participant == key
        assert item.entry_state == first and item.accepted_state == last
        assert last.tick == tick + 1
        assert last.control == first.control and last.history == first.history
        close(last.position[:3], expected_position[side], atol=WINDOW_ATOL)
        close(last.velocity[:3], expected_velocity[side], atol=WINDOW_ATOL)
        close(last.velocity[3:], np.zeros(3))
        close(last.position[3:], data["orientation"])
        close(item.law_wrench_fp64, expected_wrench[side:side + 1])
        close(item.uploaded_wrench_fp32, item.law_wrench_fp64)
        close(item.contact_generalized, expected_wrench[side])
        literal_mass = np.diag([MASSES[side]] * 3 + [INERTIA] * 3)
        close(item.entry_mass, literal_mass, atol=2e-5)
        close(item.entry_bias, np.zeros(6))
        close(item.entry_linear, np.eye(6)[:3][None, :, :])
        close(item.entry_angular, np.eye(6)[3:][None, :, :])
        close(item.input_external, np.zeros(6))
        diagnostic = item.diagnostic
        close(diagnostic.mass, literal_mass, atol=2e-5)
        close(diagnostic.acceleration, expected_wrench[side] /
              np.array([MASSES[side]] * 3 + [INERTIA] * 3))
        close(diagnostic.mass @ diagnostic.acceleration + diagnostic.bias,
              diagnostic.external)
        close(np.asarray(last.velocity) - np.asarray(first.velocity),
              H * np.asarray(diagnostic.acceleration))
        close(item.entry_geometry.position, first.position)
        close(item.accepted_geometry.position, last.position)
    actual_positions = np.array([after.participants[k].position[:3] for k in keys])
    actual_velocities = np.array([after.participants[k].velocity[:3] for k in keys])
    close(np.sum(MASSES[:, None] * actual_velocities, axis=0),
          np.sum(MASSES[:, None] * data["velocity"], axis=0), atol=WINDOW_ATOL)
    close(np.sum(np.cross(actual_positions, MASSES[:, None] * actual_velocities), axis=0),
          np.sum(np.cross(data["position"], MASSES[:, None] * data["velocity"]), axis=0),
          atol=WINDOW_ATOL)


@pytest.mark.tyx
@pytest.mark.parametrize("count,reverse,transformed",
                         [(1, False, False), (1, True, True),
                          (2, False, True), (4, True, False)])
def test_r01_two_free_bodies_complete_40step(count, reverse, transformed, tyx_mode, tmp_path):
    data = inputs(count, reverse=reverse, transformed=transformed)
    records, failure = [], None
    with ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(data["controls"])
        with scene.prepare_contacts(data["model"], data["bindings"], data["laws"],
                                    candidate_capacity=count, site_capacity=128,
                                    narrow_phase="host_patch") as plan:
            initial_stats = dict(scene.stats())
            oracle = {w: (data["position"].copy(), data["velocity"].copy()) for w in data["worlds"]}
            try:
                assert scene.world_keys == data["worlds"]
                assert scene.participant_keys == tuple(sorted(data["controls"]))
                for tick in range(STEPS):
                    before = scene.snapshot()
                    expected = {w: radial_step(*oracle[w]) for w in data["worlds"]}
                    assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
                    after = scene.snapshot()
                    row = record(scene, plan, before)
                    row["independent"] = expected
                    records.append(row)
                    for world in data["worlds"]:
                        q, v, wrench = expected[world]
                        if tick == 0:
                            close(wrench[:, :3], np.array([[-2.0, 0, 0], [2.0, 0, 0]]) @ data["rotation"].T,
                                  atol=1e-12, rtol=0)
                        audit_world(data, world, tick, before[world], after[world],
                                    plan.last_accepted[world], q, v, wrench)
                        oracle[world] = q, v
                stats = scene.stats()
                for name in ("entry_mechanics_stages", "candidate_solve_stages",
                             "candidate_finalize_stages"):
                    assert stats[name] - initial_stats[name] == STEPS
                assert stats["state_epoch"] - initial_stats["state_epoch"] == STEPS
            except BaseException as exc:
                failure = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                save(tmp_path, "R01-complete-window",
                     dict(inputs=data, mode=tyx_mode, records=records, failure=failure,
                          retained=scene.snapshot(), stats=scene.stats()))
    assert scene.stats()["device_owned_bytes"] == 0
