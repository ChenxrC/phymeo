"""Prescribed rotational terminal-boundary probes, not an automatic contact solver.

Uses public native participant operations; the old all-prismatic public
TerminalImpulseLedger contract is not changed or claimed to accept these probes.
"""

import argparse
import json
from math import cos, sin
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance


def arr(v):
    return np.asarray(v, dtype=np.float64)


def mass(q):
    if len(q) == 1:
        return arr([[1]])
    c = cos(float(q[1]))
    return arr([[3.5 + c, 1.25 + 0.5 * c], [1.25 + 0.5 * c, 1.25]])


def solve(m, rhs):
    if len(rhs) == 1:
        return rhs / m[0, 0]
    a, b, d = m[0, 0], m[0, 1], m[1, 1]
    det = a * d - b * b
    return arr([(d * rhs[0] - b * rhs[1]) / det, (a * rhs[1] - b * rhs[0]) / det])


def point_and_j(q):
    a = float(q[0])
    if len(q) == 1:
        p = arr([cos(a), sin(a), 0])
        return p, arr([[-sin(a)], [cos(a)], [0]])
    b = a + float(q[1])
    p = arr([cos(a) + 0.75 * cos(b), sin(a) + 0.75 * sin(b), 0])
    return p, arr([[-p[1], -0.75 * sin(b)], [p[0], 0.75 * cos(b)], [0, 0]])


def require_close(actual, expected, tol):
    error = float(np.max(np.abs(actual - expected)))
    assert error <= tol * max(1.0, float(np.max(np.abs(expected)))), (error, tol)
    return error


def run_case(count, mode):
    h = 0.25 if count == 1 else 0.125
    q0 = arr([0.25] if count == 1 else [0.25, 0.5])
    v0 = arr([0.5] if count == 1 else [0.5, -1])
    local = arr([[1 if count == 1 else 0.75, 0, 0]])
    robot = ph.RobotModel(
        np.arange(count) - 1,
        ("revolute",) * count,
        arr([[0, 0, 0]] if count == 1 else [[0, 0, 0], [1, 0, 0]]),
        np.tile(arr([0, 0, 0, 1]), (count, 1)),
        np.tile(arr([0, 0, 1]), (count, 1)),
        np.ones(count),
        np.zeros((count, 3)) if count == 1 else np.tile(arr([0.5, 0, 0]), (count, 1)),
        np.ones((count, 3)),
        np.zeros(3),
    )
    bias = np.zeros(count)
    if count == 2:
        b = 0.5 * sin(float(q0[1]))
        bias = arr([-b * (2 * v0[0] * v0[1] + v0[1] ** 2), b * v0[0] ** 2])
    expected_v = v0 + h * solve(mass(q0), -bias)
    expected_q = q0 + h * expected_v
    endpoint, _ = point_and_j(expected_q)
    node_v = arr([[0.125, -0.25, 0]])
    nodes = ph.ParticleModel(arr([1]), np.zeros(3))
    node_x = endpoint[None, :] - h * node_v
    rstate, nstate = ph.RobotState(q0, v0), ph.ParticleState(node_x, node_v)
    original = [v.tobytes() for v in (q0, v0, node_x, node_v)]
    if mode == "numpy":
        rf, nf = ph.RobotParticipant(robot), ph.ClothParticipant(nodes, None)
    else:
        rf, nf = ph.TyxRobotParticipant(robot, mode), ph.TyxClothParticipant(nodes, None, mode)
    context = ph.BackendContext(rf.capabilities.backend)
    tol = 1e-8 if mode == "numpy" else 2e-5
    settings = ph.StepSettings(
        substeps=1, position_iterations=1, velocity_iterations=1, tolerance=tol
    )
    workspaces, closed = [], 0
    try:
        rw = rf.create_workspace("robot", rstate, ph.RobotInputs.zeros(count), h, settings, context)
        workspaces.append(rw)
        nw = nf.create_workspace(
            "nodes", nstate, ph.StepInputs(np.zeros((1, 3))), h, settings, context
        )
        workspaces.append(nw)
        for w in workspaces:
            w.predict()
            w.position_sweep()
            assert w.position_metrics().max_residual <= tol
            w.prepare_velocity()
            w.velocity_sweep()
            assert w.velocity_metrics().max_residual <= tol
        before_r, before_n = rw.result(), nw.result()
        pre_q, pre_v = before_r.q.copy(), before_r.v.copy()
        pre_x, pre_nv = before_n.position.copy(), before_n.velocity.copy()
        errors = dict(
            pre_q=require_close(pre_q, expected_q, tol), pre_v=require_close(pre_v, expected_v, tol)
        )
        current_point, jac = point_and_j(pre_q)
        native_point = rw.point_port(np.array([count - 1]), local).positions()[0]
        errors["native_point"] = require_close(native_point, current_point, tol)
        errors["paired_point"] = require_close(pre_x[0], current_point, tol)
        p = arr([0.25, -0.5, 0])
        m1 = mass(pre_q)
        delta = solve(m1, -(jac.T @ p))
        # Duplicate rows aggregate before the live native solve, not independent solves.
        rw.apply_velocity_impulses(
            np.array([count - 1, count - 1]), np.repeat(local, 2, axis=0), np.tile(-p / 2, (2, 1))
        )
        nw.apply_velocity_impulses(np.array([0, 0]), None, np.tile(p / 2, (2, 1)))
        after_r, after_n = rw.result(), nw.result()
        assert after_r.q.tobytes() == pre_q.tobytes()
        assert after_n.position.tobytes() == pre_x.tobytes()
        errors["robot_velocity"] = require_close(after_r.v, pre_v + delta, tol)
        errors["node_velocity"] = require_close(after_n.velocity, pre_nv + p, tol)
        u = pre_nv[0] - jac @ pre_v
        response = p - jac @ delta
        predicted_dk = float(p @ u + 0.5 * (p @ response))
        observed_dk = float(
            (
                after_r.v @ m1 @ after_r.v
                - pre_v @ m1 @ pre_v
                + np.sum(after_n.velocity**2)
                - np.sum(pre_nv**2)
            )
            / 2
        )
        errors["terminal_energy"] = require_close(arr([observed_dk]), arr([predicted_dk]), tol)
        dl = float(
            m1[0] @ (after_r.v - pre_v) + np.cross(pre_x[0], after_n.velocity[0] - pre_nv[0])[2]
        )
        errors["paired_axis_momentum"] = require_close(arr([dl]), arr([0]), tol)
        mass_change = float(np.max(np.abs(m1 - mass(q0))))
        if count == 2:
            assert mass_change > 0.01
        if mode != "numpy":
            for value in (
                pre_q,
                pre_v,
                pre_x,
                pre_nv,
                after_r.q,
                after_r.v,
                after_n.position,
                after_n.velocity,
            ):
                assert np.array_equal(value, value.astype(np.float32).astype(np.float64))
        assert original == [v.tobytes() for v in (q0, v0, node_x, node_v)]
        execution = [w.engine.stats() for w in workspaces] if mode != "numpy" else []
        result = dict(
            case="single_rotor" if count == 1 else "two_revolute_mass_change",
            backend=context.backend,
            h=h,
            tolerance=tol,
            max_errors=errors,
            endpoint_mass_change=mass_change,
            predicted_terminal_energy=predicted_dk,
            observed_terminal_energy=observed_dk,
            execution=execution,
            applications_per_domain=1,
            duplicate_rows_per_domain=2,
        )
    finally:
        for w in reversed(workspaces):
            w.close()
            closed += 1
    result["workspaces_closed"] = closed
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("numpy", "interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = provenance()
    with np.errstate(all="raise"):
        cases = [run_case(n, args.mode) for n in (1, 2)]
    after = provenance()
    assert before["source_sha256"] == after["source_sha256"]
    args.output.write_text(
        json.dumps(
            dict(
                status="prescribed_native_boundary_complete_not_automatic_contact",
                source=before,
                source_after=after,
                cases=cases,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "cases": len(cases),
                "workspaces_closed": sum(r["workspaces_closed"] for r in cases),
                "source_sha256": before["source_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
