"""Recompute the issued pickup commands from frozen weights and S4000 input states."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_pickup_student import PickupStudent
from panda_pickup_from_reach import (
    OBSERVATION_CONTRACT, PickupFeatures, apply_action, read_handoffs, sha, write_json,
)


def verify(evidence, student_path, handoffs, scene):
    run = json.loads((evidence / "summary.json").read_text(encoding="utf8"))
    for path, field in ((scene, "model_sha256"), (student_path, "student_sha256"), (handoffs, "handoffs_sha256")):
        if sha(path) != run[field]:
            raise ValueError(f"{field} differs from the device run")
    model = mj.MjModel.from_xml_path(str(scene))
    policy = PickupStudent.load(student_path)
    if policy.observation_contract != OBSERVATION_CONTRACT:
        raise ValueError("unexpected pickup observation layout")
    starts = read_handoffs(handoffs, model)
    result = {}
    first_actions = []
    for key, world in run["worlds"].items():
        start = starts[int(key)]
        features = PickupFeatures(model, start.qpos)
        d = mj.MjData(model)
        for branch in world["branches"].values():
            if sha(evidence / branch["trace"]) != branch["sha256"]:
                raise ValueError(f"device trace checksum mismatch in world {key}")
        path = evidence / world["branches"]["student"]["trace"]
        target = start.target
        repeat, steps = run["action_repeat"], run["decisions"]
        with np.load(path, allow_pickle=False) as saved:
            q, actions, targets, obs = (saved[k] for k in ("qpos", "action", "motor_target", "decision_observation"))
            if (q.shape != (steps * repeat, 23) or actions.shape != (len(q), 16) or targets.shape != actions.shape
                    or obs.shape != (steps, len(policy.mean))):
                raise ValueError("incomplete policy replay")
            state_err = act_err = 0.
            for i in range(steps):
                j = i * repeat
                d.qpos[:] = q[j]
                mj.mj_kinematics(model, d)
                x = features.observe(d, target)
                a = policy.act(x)
                state_err = max(state_err, float(np.max(np.abs(x - obs[i]))))
                act_err = max(act_err, float(np.max(np.abs(a - actions[j]))))
                np.testing.assert_allclose(x, obs[i], rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(a, actions[j], rtol=2e-5, atol=2e-5)
                block = slice(j, j + repeat)
                np.testing.assert_array_equal(actions[block], np.broadcast_to(actions[j], (repeat, 16)))
                target = apply_action(model, target, actions[j])
                np.testing.assert_array_equal(targets[block], np.broadcast_to(target, (repeat, 16)))
            first_actions.append(actions[0])
            result[key] = dict(decisions=steps, scalar_motor_actions=steps * 16,
                               max_abs_observation_delta=state_err, max_abs_action_delta=act_err,
                               held_substep_commands_match=True, motor_targets_reconstructed=True)
    return dict(schema="pickup-weight-replay-v1", student_sha256=sha(student_path),
                device_summary_sha256=sha(evidence / "summary.json"), worlds=result,
                first_action_range_across_worlds=float(np.ptp(first_actions, axis=0).max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "student", "handoffs", "scene", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, verify(args.evidence, args.student, args.handoffs, args.scene))
