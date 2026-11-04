"""Read both raw engines; distinguish momentum-derived cut loads from contact force rows."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    report = json.loads(path.read_text(encoding="utf8"))
    assert json.loads(path.with_name("final.json").read_text(encoding="utf-8-sig"))["raw_exit"] == 0
    state = np.asarray(report["states_xyz_v_m_s"], np.float32)
    assert state.shape == (17, 7, 6) and np.isfinite(state).all() and report["steps"] == 16
    keys = [tuple(k) for k in report["keys"]]
    loads = []
    impact = []
    for tick, native in enumerate(report["native_contacts"]):
        forces = {(None, 8): 0.0, (5, 8): 0.0, (2, 5): 0.0}
        for row in native["23"]["rows"]:
            a, b = row["first"], row["second"]
            key = (None, b if a is None else a) if a is None or b is None else tuple(sorted((a, b)))
            if key not in forces:
                continue
            high = {(None, 8): 8, (5, 8): 5, (2, 5): 2}[key]
            if report["engine"] == "phymeo":
                signed = np.asarray(row["normal"]) * row["normal_impulse_n_s"] / report["dt_s"]
                up_force = signed if high == b else -signed
            else:
                signed = np.asarray(row["native_force_on_first_n"])
                up_force = signed if high == a else -signed
            forces[key] += float(up_force[2])
        loads.append([forces[(2, 5)], forces[(5, 8)], forces[(None, 8)]])
        upper = 0.0
        for row in native["17"]["rows"]:
            if report["engine"] == "phymeo":
                j = np.asarray(row["normal"]) * row["normal_impulse_n_s"]
                large = j if row["second"] == 81 else -j
            else:
                j = np.asarray(row["native_force_on_first_n"]) * report["dt_s"]
                large = j if row["first"] == 81 else -j
            upper += float(large @ np.asarray([0.6, 0.8, 0]))
        impact.append(upper)
    return (
        report,
        state,
        keys,
        dict(
            native_loads_top_middle_ground_last8_n=np.mean(loads[-8:], axis=0).tolist(),
            native_initial_impulse_on_heavy_n_s=impact[0],
            all_native_cut_loads_n=loads,
        ),
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--reports", nargs=4, type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if a.out.exists():
        cli.error("fresh file required")
    group = {}
    for path in a.reports:
        row = read(path)
        group[row[0]["engine"], row[0]["device"]] = row
    assert set(group) == {(e, d) for e in ("phymeo", "newton") for d in ("cpu", "cuda:0")}
    result = {}
    for device in ("cpu", "cuda:0"):
        p, x, keys, pn = group["phymeo", device]
        n, y, other, nn = group["newton", device]
        assert keys == other
        diffs = {}
        for world in (17, 23, 4):
            indices = [i for i, k in enumerate(keys) if k[0] == world]
            delta = abs(x[:, indices].astype(float) - y[:, indices])
            diffs[world] = dict(
                max_position_m=float(np.max(delta[:, :, :3])),
                max_velocity_m_s=float(np.max(delta[:, :, 3:])),
            )
        result[device] = dict(
            phymeo=p["observed"],
            newton=n["observed"],
            phymeo_native=pn,
            newton_native=nn,
            state_differences=diffs,
        )
    with a.out.open("x", encoding="utf8") as f:
        json.dump(result, f, allow_nan=False, indent=2)
    for k, v in result.items():
        print(k, v["phymeo"]["gates"], v["newton"]["gates"], v["state_differences"])


if __name__ == "__main__":
    main()
