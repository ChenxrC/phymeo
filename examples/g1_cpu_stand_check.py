"""Compare a two-second native G1 stand against the original physical acceptance gates."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import g1_mujoco_tyx_policy as g1
import mujoco as mj
import numpy as np
import prepare_g1_stand_assets as assets_spec
from _g1_stand_balance import HIP_KD_NM_S_PER_RAD, HIP_KP_NM_PER_RAD, HIP_MOTOR_FRACTION, HipAnkleBalance


def run(assets, initial_journal):
    h = .002
    if mj.__version__ != "3.3.7":
        raise ValueError("the archived independent reference used MuJoCo 3.3.7")
    opener = gzip.open if initial_journal.suffix == ".gz" else open
    with opener(initial_journal, "rt", encoding="utf-8") as source:
        records = (json.loads(line) for line in source)
        metadata = next(records)
        if metadata["event"] != "sources" or metadata["step_s"] != h:
            raise ValueError("incompatible seed trajectory")
        seed = next(r for r in records if r["event"] == "initial")
    xml_sha = g1.binding(assets / g1.ASSET_NAME)["sha256"]
    yaml_sha = hashlib.sha256((assets / "deploy.yaml").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if xml_sha != metadata["inputs"][g1.ASSET_NAME]["sha256"] or yaml_sha != assets_spec.DEPLOY_LF_SHA:
        raise ValueError("controller settings or model differ from the pinned source")
    _, model, _ = g1.compile_source(assets, h)
    protocol = g1.Protocol(model, assets, stand_only=True)
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = seed["q"], seed["v"]
    pd = g1.Controller(protocol, data.qpos, h, stand=True)
    controller = HipAnkleBalance(protocol, data.qpos)
    stats = g1.Stats(protocol, h, stand=True)
    stats.foot_origin = g1.endpoint_foot_centers(protocol, data.qpos)[:, :2].copy()
    loads = None
    snapshots = {}
    first_step_failure = None
    first_gate_failure = None
    peak = np.zeros((2, 2))
    for tick in range(1000):
        q, v, t = data.qpos.copy(), data.qvel.copy(), data.time
        proposal, feedback = controller.correct(pd.propose(q, v, t, tick), q, v, loads)
        peak = np.maximum(peak, np.abs(feedback["hip_applied_joint_delta_nm"]))
        packet = g1.packet_for(protocol, proposal, tick * h, 0)
        g1.set_native(data, packet)
        mj.mj_step(model, data)
        pd.commit(proposal)
        native = g1.native_fields(model, data, q, v, packet, full=False)
        forces = g1.support(protocol, native["contacts"], v, native["body_com"])
        loads = forces["foot_force_world"][:, 2].copy()
        _, _, failures, _ = stats.update(data.qpos, data.qvel, data.time, t, forces, native["energy"],
                                          g1.endpoint_foot_centers(protocol, data.qpos))
        if abs(data.time - (tick + 1) * h) > 2e-12 or any(w.number for w in data.warning):
            raise ValueError(f"MuJoCo warning or unexpected clock at tick {tick + 1}")
        if failures and first_step_failure is None:
            first_step_failure = {"tick": tick + 1, "reasons": failures}
        if tick + 1 >= 250:
            summary = stats.summary(assessed=True)
            if summary["checkpoint_failures"] and first_gate_failure is None:
                first_gate_failure = {"tick": tick + 1, "reasons": summary["checkpoint_failures"]}
            if (tick + 1) % 250 == 0:
                snapshots[str(tick + 1)] = summary
    return {
        "status": "PASS" if first_step_failure is None and first_gate_failure is None else "FAIL",
        "backend": "native CPU MuJoCo", "version": mj.__version__, "controller": "hip-ankle",
        "h_s": h, "steps": 1000, "model_sha256": xml_sha, "deploy_normalized_sha256": yaml_sha,
        "initial_journal_sha256": g1.binding(initial_journal)["sha256"],
        "initial": {"q": seed["q"], "v": seed["v"]}, "first_step_failure": first_step_failure,
        "first_checkpoint_failure": first_gate_failure, "cumulative_checkpoints": snapshots,
        "hip_kp_nm_per_rad": HIP_KP_NM_PER_RAD, "hip_kd_nm_s_per_rad": HIP_KD_NM_S_PER_RAD,
        "hip_control_range_fraction": HIP_MOTOR_FRACTION, "max_abs_applied_hip_delta_nm": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--initial-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.assets, args.initial_journal)
    with args.output.open("x", encoding="utf8") as out:
        json.dump(g1.json_value(result), out, indent=2, allow_nan=False)
    for tick, row in result["cumulative_checkpoints"].items():
        print(tick, row["maximum_reference_xy_error"], row["maximum_tilt"], row["checkpoint_failures"])
    print(result["status"], args.output)
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
