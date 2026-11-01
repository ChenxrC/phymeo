"""Fresh disk readback of common motion frames, exact endpoint integrals and transaction controls."""

import argparse
import gzip
import json
from fractions import Fraction as Q
from hashlib import sha256
from math import comb
from pathlib import Path

import numpy as np
from p65_common_motion import DEGREE, GRID, unpack
from probe_p65_common_motion import ClothPlatform, saved, verified_config
from probe_p65_joint_readback import fixture, save


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def bernstein_moments(n):
    # Direct expansion of T_n(2s-1) in degree-n Bernstein coefficients.
    weights = [Q((-1) ** (n - k) * comb(2 * n, 2 * k), comb(n, k)) for k in range(n + 1)]
    return sum(weights) / Q(n + 1), sum((n + 1 - k) * w for k, w in enumerate(weights)) / Q(
        (n + 1) * (n + 2)
    )


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def public_numbers(state):
    c, r = state["domains"]["cloth"], state["domains"]["robot"]
    return np.r_[c["position"].ravel(), r["q"]], np.r_[c["velocity"].ravel(), r["v"]]


def readback(root, tag, model, *, predecessor_checkpoint=None):
    ap = root / f"{tag}-attempt.json.gz"
    rp = root / f"{tag}-result.json.gz"
    op = root / f"{tag}-output.bin"
    attempt = saved(ap)
    result = saved(rp)
    record = result["report"]["substeps"][0]
    need(
        result["report"]["accepted"] and record["accepted"] and len(result["substep_states"]) == 1,
        "actual public frame not accepted",
    )

    # Compare the serialized immutable public diagnostics and out-of-band diagnostic archive.
    def raw(path):
        return json.loads(gzip.decompress(path.read_bytes()))

    demo = raw(rp)
    need(
        demo["report"]["substeps"][0]["diagnostics"]["common_motion"] == raw(ap)
        and demo["state"] == demo["substep_states"][0],
        "returned diagnostics/state binding",
    )
    curve = attempt["curve"]
    cert = attempt["certificate"]
    need(
        cert["accepted"]
        and all(cert["checks"].values())
        and curve["owner_charges"] == 486
        and cert["owner_proof_enclosures"] == 384,
        "gates or resources",
    )
    incoming = attempt["inlet"]["history"]["motion"]
    out = result["state"]["history"]["motion"]
    need(out.tobytes() == op.read_bytes(), "checkpoint differs from returned public state")
    q0, v0, t0 = unpack(incoming, model.signature)
    q1, v1, t1 = unpack(out, model.signature)
    h = t1 - t0
    public_q, public_v = public_numbers(result["state"])
    need(
        np.array_equal(public_q, np.asarray(q1, float))
        and np.array_equal(public_v, np.asarray(v1, float))
        and float(h) == curve["h"],
        "checkpoint projection or duration",
    )
    need(
        np.array_equal(curve["incoming_velocity"], np.asarray(v0, float))
        and np.array_equal(curve["incoming_position"], np.asarray(q0, float)),
        "actual inlet state",
    )
    if tag == "first":
        need(t0 == 0 and all(v == 0 for v in v0), "original rest start")
    else:
        predecessor = (
            root / "first-output.bin" if predecessor_checkpoint is None else predecessor_checkpoint
        )
        need(
            incoming.tobytes() == predecessor.read_bytes() and any(v != 0 for v in v0),
            "lost restart",
        )
    js = []
    bs = []
    mom = [bernstein_moments(k) for k in range(DEGREE + 1)]
    for o in range(len(model.owners)):
        j = sum(h * a * int(curve["pressure"][k, o]) / GRID for k, (a, b) in enumerate(mom))
        b = sum(h * h * b * int(curve["pressure"][k, o]) / GRID for k, (a, b) in enumerate(mom))
        need(float(j) == cert["j_ns"][o] and float(b) == cert["b_ns2"][o], "owner readback moment")
        js.append(float(j))
        bs.append(float(b))
    for i, mass in enumerate(model.mass):
        coeff = [
            Q(int(curve["material"][k, i]) + sum(map(int, curve["rows"][k, :, i])), GRID)
            for k in range(DEGREE + 1)
        ]
        coeff[0] += Q(float(curve["external"][i]))
        j = sum(h * a * c for (a, _), c in zip(mom, coeff, strict=True))
        b = sum(h * h * w * c for (_, w), c in zip(mom, coeff, strict=True))
        need(
            Q(float(mass)) * (v1[i] - v0[i]) == j
            and Q(float(mass)) * (q1[i] - q0[i] - h * v0[i]) == b,
            "node force and motion disagree",
        )
    rows = curve["iterations"]
    need(
        len(rows) == 4
        and curve["force_calls"] == 1 + sum(len(x["q"]) - 1 for x in rows)
        and all(
            np.array_equal(x["q"][0], np.asarray(q0, float))
            and np.array_equal(x["v"][0], np.asarray(v0, float))
            for x in rows
        ),
        "evaluation or inlet accounting",
    )
    return dict(
        accepted=True,
        tag=tag,
        start_s=float(t0),
        end_s=float(t1),
        inlet_sha256=sha256(incoming.tobytes()).hexdigest(),
        output_sha256=file_sha(op),
        attempt_sha256=file_sha(ap),
        result_sha256=file_sha(rp),
        public_position=public_q.tolist(),
        public_velocity=public_v.tolist(),
        true_inlet_velocity=list(map(float, v0)),
        checkpoint_wire_bytes=len(op.read_bytes()),
        checkpoint_content_bytes=34 + int.from_bytes(op.read_bytes()[:2], "big"),
        solver_owner_evaluations=curve["owner_charges"],
        proof_owner_enclosures=cert["owner_proof_enclosures"],
        four_collocation_residuals_m_s2=[r["residual"] for r in rows],
        public_frame_residual=max(
            max(record["position"]["residuals"].values()),
            max(record["velocity"]["residuals"].values()),
        ),
        j_ns=js,
        b_ns2=bs,
        certificate=cert,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("first-audit", "next-audit", "fault-audit"), required=True
    )
    args = parser.parse_args()
    root = args.root
    config, _ = verified_config()
    parts = fixture(config)
    model = ClothPlatform(parts)
    try:
        if args.phase == "fault-audit":
            answers = {}
            for name in ("inlet", "owner", "bubble"):
                r = saved(root / f"{name}-result.json.gz")
                need(not r["report"]["accepted"] and not r["substep_states"], "forgery committed")
                orig = saved(root / f"{name}-attempt.json.gz")["inlet"]
                for a, b in zip(public_numbers(orig), public_numbers(r["state"]), strict=True):
                    need(np.array_equal(a, b), "rollback did not preserve real state")
                need(
                    np.array_equal(orig["history"]["motion"], r["state"]["history"]["motion"]),
                    "rollback history",
                )
                answers[name] = r["report"]["reason"]
            save(root / "fault-audit.json", answers)
            print(answers)
        else:
            tag = args.phase.split("-")[0]
            report = readback(root, tag, model)
            save(root / f"{tag}-audit.json", report)
            print(
                tag,
                "exact all-coordinate moments, public return and checkpoint certified",
                report["output_sha256"],
            )
    finally:
        parts[3].close()


if __name__ == "__main__":
    main()
