"""Fixed saved-node derivative controls for the SC2 common-path peak residual."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from check_p65_common_motion import integral, times
from check_p65_sc2_common_entry import evaluate
from p65_common_motion import need, tables, unpack
from probe_p65_common_motion import saved
from probe_p65_joint_readback import save
from probe_p65_sc2_common_entry import original
from run_p65_sc2_step_tail import filehash
from verify_p65_contact_layer_screen import sha

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"
PLAN = BASE / "p65-sc2-peak-current-tangent-protocol.json"


def baseline():
    plan = json.loads(PLAN.read_bytes())
    path = BASE / plan["parent"]
    need(sha(path, normalize=True) == plan["parent_sha256_lf"], "wrong baseline")
    result = json.loads(path.read_bytes())
    return BASE.parents[1] / result["evidence_root"], result


def pack(material, pressures, rows):
    return np.concatenate((material, pressures, rows.ravel()))


def analytic_reference(model, q, v):
    # Holomorphic extension of the declared smooth branch. Products are bilinear.
    x = q[:12].reshape(4, 3)
    idx = model.material.bending.indices[0]
    a, b, c, d = x[idx]
    e, u, w = b - a, c - a, d - a
    eu, ew, uw, ee = np.dot(e, u), np.dot(e, w), np.dot(u, w), np.dot(e, e)
    le = np.sqrt(ee)
    triple = np.dot(e, np.cross(u, w))
    X = eu * ew - ee * uw
    Y = -le * triple
    need(X.real > 0, "oriented atan branch")
    gx = np.array([ew * u + eu * w - 2 * uw * e, ew * e - ee * w, eu * e - ee * u])
    gy = np.array(
        [-triple * e / le - le * np.cross(u, w), -le * np.cross(w, e), -le * np.cross(e, u)]
    )
    three = (X * gy - Y * gx) / (X * X + Y * Y)
    G = np.vstack((-three.sum(axis=0), three))
    theta = np.arctan(Y / X)
    mat = np.zeros(13, dtype=q.dtype)
    mf = mat[:12].reshape(4, 3)
    np.add.at(mf, idx, -(theta - float(model.rest)) * G / model.material.bending.compliance[0])
    for (i, j), rest, alpha in zip(
        model.material.distance.indices,
        model.material.distance.rest_length,
        model.material.distance.compliance,
        strict=True,
    ):
        edge = x[j] - x[i]
        length = np.sqrt(np.dot(edge, edge))
        force = ((length - rest) / (alpha * length)) * edge
        mf[i] += force
        mf[j] -= force
    loads = []
    ps = []
    ez = np.array([0.0, 0.0, 1.0])
    for ids in model.faces:
        i, j, k = ids
        left, right = x[j] - x[i], x[k] - x[i]
        raw = np.cross(left, right)
        length = np.sqrt(np.dot(raw, raw))
        n = raw / length
        adj = (ez - n[2] * n) / length
        dj = -model.rho * np.cross(right, adj)
        dk = -model.rho * np.cross(adj, left)
        for node in ids:
            row = np.zeros(13, dtype=q.dtype)
            row[3 * i : 3 * i + 3] = -dj - dk
            row[3 * j : 3 * j + 3] = dj
            row[3 * k : 3 * k + 3] = dk
            row[3 * node + 2] += 1
            row[-1] = -1
            gap = x[node, 2] - q[-1] + model.rho * (1 - n[2])
            rate = np.dot(row, v)
            need(0 < gap.real < 0.002 and rate.real < 0.001, "complex oracle outside active chart")
            p = (500 / 6) * (0.002 - gap) * (1 - 1000 * rate)
            ps.append(p)
            loads.append(p * row)
    return pack(mat, np.asarray(ps), np.asarray(loads))


def reference_columns(model, q, v):
    length = len(q)
    eps = 2.0**-72
    value = analytic_reference(model, q, v)
    cols = []
    for j in range(2 * length):
        x = q.astype(complex)
        u = v.astype(complex)
        if j < length:
            x[j] += 1j * eps
        else:
            u[j - length] += 1j * eps
        cols.append(analytic_reference(model, x, u).imag / eps)
    return value, np.stack(cols, axis=1)


def original_weights(nodes):
    exact = list(map(lambda z: Q(float(z)), nodes))
    n = len(exact)
    vv = np.zeros((n, n))
    pp = np.zeros((n, n))
    for column, s in enumerate(exact):
        cardinal = (Q(1),)
        for j, t in enumerate(exact):
            if column != j:
                cardinal = times(cardinal, (-t / (s - t), 1 / (s - t)))
        v = integral(cardinal)
        p = integral(v)
        vv[:, column] = [float(evaluate(v, t)) for t in exact]
        pp[:, column] = [float(evaluate(p, t)) for t in exact]
    return vv, pp


def err(a, b):
    return float(np.max(abs(a - b) / (1 + abs(b))))


def force_partials(jet):
    zeros = np.zeros_like(jet.mq)
    left = np.concatenate((jet.mq, jet.pq, jet.rq.reshape(-1, len(zeros))), axis=0)
    right = np.concatenate((zeros, jet.pv, jet.rv.reshape(-1, len(zeros))), axis=0)
    return np.hstack((left, right))


def sum_jac(cols, n):
    return cols[:n] + cols[n + 6 :].reshape(6, n, 2 * n).sum(axis=0)


def check(root, red):
    prev, meta = baseline()
    source = prev / "peak-proposal.json.gz"
    digest = next(x["sha256"] for x in meta["raw"] if x["name"] == source.name)
    need(filehash(source) == digest, "unbound old trial")
    model, _, _, _ = original()
    try:
        data = saved(source)
        chart = data["chart"]
        need(
            data["input_sha"] == meta["d"]["endpoint"] == meta["peak_rejection"]["inlet"]
            and chart["owners"] == list(model.owners),
            "wrong saved inlet",
        )
        state = chart["iterations"][-1]
        nodes = chart["nodes"]
        n = len(model.mass)
        h = float(Q(data["h"]))
        V, P = original_weights(nodes)
        _, _, vs, ps = tables(len(nodes) - 1)
        mass = model.mass
        true_matrix = np.eye(len(nodes) * n)
        valerr = frameerr = field_error = frozen_error = map_drop = rate_drop = 0.0
        symmetry = translation = 0.0
        qs = []
        us = []
        q_in, v_in, _ = unpack(
            np.frombuffer((prev / "d.bin").read_bytes(), dtype="<i4"), model.signature
        )
        fixed = model.linearization(np.asarray(q_in, float), np.asarray(v_in, float), None)
        for k, (q, v) in enumerate(zip(state["q"], state["v"], strict=True)):
            val, cols = reference_columns(model, q, v)
            actual = model.forces(q, v)
            frameerr = max(frameerr, err(pack(actual.material, actual.pressures, actual.rows), val))
            fk = sum_jac(cols, n)
            frozen_error = max(frozen_error, err(np.hstack(fixed), fk))
            for column in range(len(nodes)):
                true_matrix[k * n : (k + 1) * n, column * n : (column + 1) * n] -= (
                    h * h * P[k, column] * fk[:, :n] + h * V[k, column] * fk[:, n:]
                ) / mass[:, None]
            if red:
                continue
            from p65_peak_tangent import force_jet

            jet = force_jet(model, q, v)
            valerr = max(
                valerr, err(pack(jet.value.material, jet.value.pressures, jet.value.rows), val)
            )
            jc = force_partials(jet)
            field_error = max(field_error, err(jc, cols))
            qs.append(jet.mq + jet.rq.sum(axis=0))
            us.append(jet.rv.sum(axis=0))
            symmetry = max(
                symmetry,
                float(np.max(abs(jet.mq - jet.mq.T))),
                float(np.max(abs(jet.gap_hessian - jet.gap_hessian.transpose(0, 2, 1)))),
            )
            for axis in range(3):
                direction = np.zeros(n)
                direction[axis:12:3] = 1
                direction[-1] = int(axis == 2)
                translation = max(
                    translation,
                    float(np.max(abs(qs[-1] @ direction))),
                    float(np.max(abs(us[-1] @ direction))),
                )
            badq = jet.rq - jet.value.pressures[:, None, None] * jet.gap_hessian
            map_drop = max(map_drop, err(badq.reshape(-1, n), cols[n + 6 :, :n]))
            omit = jet.pq + ((500 / 6) * 1000 * (0.002 - jet.gap))[:, None] * np.einsum(
                "aij,j->ai", jet.gap_hessian, v
            )
            rate_drop = max(rate_drop, err(omit, cols[n : n + 6, :n]))
        if red:
            save(
                root / "old-tangent-red.json",
                dict(old_scaled_error=frozen_error, expected_tolerance=2e-10),
            )
            need(
                frozen_error <= 2e-10,
                f"frozen inlet derivative fails the saved-state oracle: {frozen_error:.8g}",
            )
            return
        from p65_peak_tangent import residual_matrix

        derivative = residual_matrix(h, vs, ps, mass, np.stack(qs), np.stack(us))
        materr = err(derivative, true_matrix)
        report = dict(
            source_sha=digest,
            fixed_states=len(nodes),
            real_reference_evaluations=len(nodes),
            complex_reference_evaluations=26 * len(nodes),
            coordinate_columns=26,
            material_owner_field_rows=13 + 6 + 78,
            residual_matrix=list(derivative.shape),
            value_error=valerr,
            original_full_port_error=frameerr,
            component_derivative_error=field_error,
            residual_derivative_error=materr,
            exact_cardinal_error=max(err(vs, V), err(ps, P)),
            symmetry=symmetry,
            translation=translation,
            old_tangent_error=frozen_error,
            omitted_map_term_error=map_drop,
            omitted_rate_geometry_error=rate_drop,
        )
        report["accepted"] = (
            max(valerr, frameerr) <= 2e-11
            and max(field_error, materr) <= 2e-10
            and symmetry < 1e-10
            and translation < 1e-9
            and min(frozen_error, map_drop, rate_drop) > 2e-10
        )
        save(root / "derivative.json", report)
        need(report["accepted"], str(report))
        print(json.dumps(report, indent=2))
    finally:
        model.parts[3].close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--red", action="store_true")
    args = cli.parse_args()
    check(args.root, args.red)
