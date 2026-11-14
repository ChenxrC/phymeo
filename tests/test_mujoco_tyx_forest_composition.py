"""Forest cross-feature contracts; authoring only, execution belongs to parent.

Four parameterized nodes reuse the accepted native oracle/receipt helpers.
The original forest module and product sources are intentionally untouched.
"""

import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import test_mujoco_tyx_forest as forest
from test_mujoco_tyx_ball_constraints import case as scalar

from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoInitialState


def free_pair_source():
    root = ET.fromstring(forest.source("two-free", object_first=False))
    root.find("default/geom").set("condim", "3")
    for index, position in ((0, "0 0 .8"), (1, ".138 0 .8")):
        body = root.find(f".//body[@name='object{index}']")
        body.set("pos", position)
        geom = body.find(f"geom[@name='payload{index}']")
        geom.set("contype", str(index + 1))
        geom.set("conaffinity", str(2 - index))
        if index == 0:
            geom.set("type", "sphere")
            geom.set("size", ".1")
            # Sphere center y=.025, box center y=0: the x-normal at y=.025
            # misses BOTH FREE body origins, not merely their offset COMs.
            geom.set("pos", "0 .025 0")
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("mode", forest.MODES)
def test_independent_free_pair_contact_has_both_angular_reactions(mode, tmp_path):
    mj = forest.normal._deps()
    xml = free_pair_source()
    model = mj.MjModel.from_xml_string(xml)
    free_ids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    assert len(free_ids) == 2 and (model.nq, model.nv) == (17, 15)
    free_columns = [
        np.arange(int(model.jnt_dofadr[joint]), int(model.jnt_dofadr[joint]) + 6)
        for joint in free_ids
    ]
    robot_columns = np.setdiff1d(np.arange(model.nv), np.concatenate(free_columns))
    with forest.runtime(xml, mode, {19: forest.initial(mj, model, contact=True)}) as owner:
        start = owner.integration_snapshot()[19]
        continuous = forest.native_data(mj, model, start)
        for columns in free_columns:
            assert np.all(start.v[columns] != 0)
        for tick in range(8):
            before = owner.integration_snapshot()[19]
            command = forest.packet(model, tick)
            expected = forest.mixed.native(mj, model, before.q, before.v, command, before.time)
            local = forest.native_data(mj, model, before)
            for data in (local, continuous):
                forest.normal._set(data, command)
                mj.mj_step(model, data)
            owner.set_controls({19: command})
            assert owner.step(forest.H) == {19: True}
            after = owner.integration_snapshot()[19]
            row = after.last_step
            forest._save(
                tmp_path / "free-pair.jsonl",
                dict(
                    source=xml if tick == 0 else None,
                    entry=before,
                    actual=after,
                    oracle=expected,
                    continuous_q=continuous.qpos.copy(),
                    continuous_v=continuous.qvel.copy(),
                ),
            )
            forest.mixed.assert_receipt(row, expected, model)
            forest.assert_successor(mj, model, before, after, command, local, continuous)
            np.testing.assert_allclose(row.qfrc_constraint[robot_columns], 0, rtol=0, atol=1e-7)
            if tick == 0:
                assert len(row.contacts) == 1
                contact = row.contacts[0]
                assert contact.dimension == 3 and contact.normal_force > 0.01
                assert set(contact.body_ids) == set(map(int, model.jnt_bodyid[free_ids]))
                for columns in free_columns:
                    assert np.linalg.norm(row.qfrc_constraint[columns[:3]]) > 0.001
                    assert np.linalg.norm(row.qfrc_constraint[columns[3:]]) > 0.0001
                    response = row.qacc[columns[3:]] - row.dynamics.smooth_acceleration[columns[3:]]
                    assert np.linalg.norm(response) > 0.001
                    assert np.linalg.norm(contact.jacobian[:, columns]) > 0.01
                for body in contact.body_ids:
                    lever = contact.position - row.dynamics.body_position[body]
                    assert np.linalg.norm(np.cross(lever, contact.frame[0])) > 0.02
                np.testing.assert_allclose(
                    contact.force_on_geometry1_world + contact.force_on_geometry2_world,
                    np.zeros(3),
                    rtol=0,
                    atol=1e-6,
                )


def constrained_source():
    root = ET.fromstring(forest.source("rp", object_first=False))
    joint = root.find(".//joint[@name='j0']")
    joint.set("limited", "true")
    joint.set("range", "-.02 .02")
    joint.set("margin", ".002")
    joint.set("frictionloss", ".03")
    drive = root.find("actuator/general[@name='drive0']")
    drive.set("dyntype", "filterexact")
    drive.set("dynprm", ".004")
    drive.set("actearly", "false")
    drive.set("actlimited", "true")
    drive.set("actrange", "-.3 .3")
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("mode", forest.MODES)
def test_first_fixed_joint_constraints_activation_and_nonleading_free_restore(mode, tmp_path):
    mj = forest.normal._deps()
    xml = constrained_source()
    model = mj.MjModel.from_xml_string(xml)
    first = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "j0")
    free = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "free0")
    assert first == 0 and free == 2 and model.na == 1
    assert model.actuator_trnid[0, 0] == first and model.actuator_actadr[0] == 0
    assert model.jnt_limited[first] and model.dof_frictionloss[0] > 0
    qa = int(model.jnt_qposadr[free]) + 3
    assert qa > 3  # The FREE quaternion is not the legacy q[3:7] slice.
    keys = (3, 29)
    starts = {}
    for slot, key in enumerate(keys):
        base = forest.initial(mj, model, slot=slot)
        q = base.q.copy()
        q[int(model.jnt_qposadr[first])] = 0.0195
        starts[key] = MuJoCoInitialState(
            q, base.v, act=np.array([0.07 + 0.01 * slot]), time=base.time
        )
    journal = scalar.Journal(tmp_path / "forest-composition.jsonl.gz")
    try:
        journal.emit("source", xml=xml, initial=starts)
        with forest.runtime(xml, mode, starts) as owner:
            stored_initial = owner.integration_snapshot()
            references = scalar.references(mj, model, stored_initial)
            for tick in range(3):
                commands = {
                    key: forest.packet(model, 2 * tick, slot) for slot, key in enumerate(keys)
                }
                output = scalar.advance(
                    owner,
                    mj,
                    model,
                    commands,
                    references,
                    journal,
                    f"step-{tick + 1}",
                    tiny=True,
                )
                for key in keys:
                    rows = output[key].last_step.constraints
                    assert any(row.constraint_type == 1 and row.source_id == 0 for row in rows)
                    assert any(row.constraint_type == 3 and row.source_id == first for row in rows)
                    assert abs(output[key].act[0] - stored_initial[key].act[0]) > 1e-5
            commands = {key: forest.packet(model, 6, slot) for slot, key in enumerate(keys)}
            owner.set_controls(commands)
            saved_state = owner.integration_snapshot()
            saved = MuJoCoTyxCheckpoint.from_bytes(owner.checkpoint().to_bytes())
            saved_references = {key: copy.copy(data) for key, data in references.items()}
            fourth = scalar.advance(
                owner,
                mj,
                model,
                commands,
                references,
                journal,
                "step-4",
                tiny=True,
            )
            owner.restore(saved, expected_revision=owner.checkpoint_revision)
            scalar.exact(owner.integration_snapshot(), saved_state, "complete forest restore")
            references = {key: copy.copy(data) for key, data in saved_references.items()}
            repeated = scalar.advance(
                owner,
                mj,
                model,
                commands,
                references,
                journal,
                "repeated-step-4",
                tiny=True,
            )
            scalar.exact(fourth, repeated, "complete forest state and receipt successor")
            checkpoint_before = owner.checkpoint()
            owner.reset((29,))
            reset = owner.integration_snapshot()
            scalar.exact(
                reset[3], repeated[3], "unselected world retains activation and constraints"
            )
            scalar.checkpoint_world_equal(checkpoint_before, owner.checkpoint(), 0)
            assert reset[29].tick == 0 and reset[29].last_step is None
            for field in ("q", "v", "act", "time"):
                scalar.exact(getattr(reset[29], field), getattr(stored_initial[29], field), field)
            np.testing.assert_array_equal(
                reset[29].q[qa : qa + 4], stored_initial[29].q[qa : qa + 4]
            )
            assert abs(reset[29].q[qa : qa + 4] @ reset[29].q[qa : qa + 4] - 1) < 2e-6
            references[29] = scalar.references(mj, model, {29: reset[29]})[29]
            scalar.advance(
                owner,
                mj,
                model,
                commands,
                references,
                journal,
                "partial-reset-successor",
                tiny=True,
            )
    finally:
        journal.close()
