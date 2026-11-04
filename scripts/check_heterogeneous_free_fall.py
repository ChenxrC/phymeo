"""Compare raw W128 cross-engine trajectories and discrete mechanics before timing ratios."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path, engine):
    report = json.loads(path.read_text(encoding="utf8"))
    guard = json.loads(path.with_name("final.json").read_text(encoding="utf-8-sig"))
    assert guard["raw_exit"] == 0 and report["engine"] == engine and report["device"] == "cuda:0"
    assert report["phase"] == "measure" and (
        report["steps_per_window"],
        report["samples"],
        report["warmup_windows"],
    ) == (32, 20, 5)
    (run,) = report["runs"]
    assert run["asymmetric"] and run["worlds"] == 128 and run["final_tick"] == 801
    assert run["dt_s"] == 1 / 128 and (
        run["max_position_error_m"],
        run["max_velocity_error_m_s"],
    ) == (0, 0)
    assert report["observation"] == ("kinematics" if engine == "phymeo" else "native_rigid_state")
    data = np.load(path.with_name(run["trajectory_file"]), allow_pickle=False)
    assert data.shape == (802, 128, 6) and data.dtype == np.float32
    assert np.unique(data[0, :, 2]).size == np.unique(data[0, :, 5]).size == 128
    assert run["verified_world_steps"] == 128 * 20 * 32
    return (
        report,
        run,
        data,
        datetime.fromisoformat(guard["start"]),
        datetime.fromisoformat(guard["end"]),
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--phymeo", nargs="+", type=Path, required=True)
    cli.add_argument("--newton", nargs="+", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    if len(a.phymeo) != len(a.newton) or len(a.phymeo) < 2 or a.out.exists():
        cli.error("paired independent sessions and a fresh output are required")
    rows, intervals = [], []
    reference = None
    for pp, npth in zip(a.phymeo, a.newton, strict=True):
        ph, p, pstate, ps, pe = read(pp, "phymeo")
        nt, n, nstate, ns, ne = read(npth, "newton")
        assert nt["runtime"]["newton"] == "1.7.0" and nt["runtime"]["warp"] == "1.17.0"
        assert p["application_order"] == n["application_order"] == sorted(p["registration_order"])
        assert (
            p["registration_order"] == n["registration_order"]
            and len(set(p["application_order"])) == 128
        )
        assert p["cold"]["output_sha256"] == n["cold"]["output_sha256"]
        assert [v["output_sha256"] for v in p["measurements"]] == [
            v["output_sha256"] for v in n["measurements"]
        ]
        np.testing.assert_array_equal(pstate.view(np.uint32), nstate.view(np.uint32))
        if reference is not None:
            np.testing.assert_array_equal(pstate.view(np.uint32), reference.view(np.uint32))
        reference = pstate
        assert pe <= ns or ne <= ps
        intervals.extend(((ps, pe), (ns, ne)))
        # Independent per-transition impulse, semi-implicit drift, and the energy loss per step relative
        # to continuous gravitational free fall.
        s = pstate.astype(np.float64)
        x, v = s[:, :, :3], s[:, :, 3:]
        g = np.array([0, 0, -8.0])
        mass, dt = 2.0, 1 / 128
        impulse = np.max(np.abs(mass * np.diff(v, axis=0) - mass * dt * g))
        drift = np.max(np.abs(np.diff(x, axis=0) - dt * v[1:]))
        energy = 0.5 * mass * np.sum(v * v, axis=2) - mass * (x @ g)
        energy_error = np.max(np.abs(np.diff(energy, axis=0) + 0.5 * mass * (g @ g) * dt**2))
        assert impulse == drift == energy_error == 0
        metrics = {}
        for label, record in (("phymeo", p), ("newton", n)):
            metrics[label] = dict(
                cold_step_ms=record["cold"]["wall_ns"] / 1e6,
                setup_ms=record["setup_ns"] / 1e6,
                p50_ms=record["warm_p50_window_ns"] / 1e6,
                p95_ms=record["warm_p95_window_ns"] / 1e6,
                world_steps_s=record["verified_world_steps_per_s"],
                counts=record["warm_totals"],
            )
        rows.append(
            dict(
                first="phymeo" if ps < ns else "newton",
                step_impulse_residual_n_s=float(impulse),
                semiimplicit_drift_residual_m=float(drift),
                energy_change_residual_j=float(energy_error),
                newton_over_phymeo_throughput=metrics["newton"]["world_steps_s"]
                / metrics["phymeo"]["world_steps_s"],
                samples=[str(pp), str(npth)],
                **metrics,
            )
        )
    assert {r["first"] for r in rows} == {"phymeo", "newton"}
    intervals.sort()
    assert all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    report = dict(positions_velocities_shape=list(reference.shape), rounds=rows)
    with a.out.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(
        [
            (
                r["first"],
                r["phymeo"]["p50_ms"],
                r["newton"]["p50_ms"],
                r["newton_over_phymeo_throughput"],
            )
            for r in rows
        ]
    )


if __name__ == "__main__":
    main()
