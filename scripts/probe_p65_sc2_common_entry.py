"""Original SC2 inlet, first positive owner and a persisted exact bridge checkpoint."""

import argparse
import json
from copy import deepcopy
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

import numpy as np
from check_p65_sc2_common_entry import evaluate, isolate, physical_bridge, state_polynomials
from p65_common_motion import Forces, checkpoint, fit, need, unpack
from probe_p65_common_motion import ClothPlatform, coordinates, joint_state, saved
from probe_p65_joint_readback import PLAN as BASE
from probe_p65_joint_readback import fixture, save
from verify_p65_contact_layer_screen import sha

import phymeo as ph
from phymeo.core.scene import fingerprint

SPEC = BASE.with_name("p65-sc2-common-motion-entry-protocol.json")


def original():
    spec = json.loads(SPEC.read_bytes())
    need(
        sha(BASE, normalize=True) == spec["fixture_sha256_lf"]
        and Path(ph.__file__).resolve().parent == (BASE.parents[2] / "src/phymeo").resolve(),
        "wrong original fixture or source",
    )
    f = json.loads(BASE.read_bytes())
    for node in f["cloth"]["nodes"]:
        node[2] += float(Q(spec["z0"]))
    f["cloth"].update(gravity_z="0", rest_angle_rad="0")
    parts = fixture(f)
    model = ClothPlatform(parts)
    q0, v0 = coordinates(parts[1])
    v0[2:12:3] = list(map(lambda t: float(Q(t)), spec["vz"]))
    q, v = list(map(Q, q0)), list(map(Q, v0))
    init = joint_state(q, v, checkpoint(q, v, Q(0), model.signature))
    binding = fingerprint((sha(SPEC, normalize=True), model.signature, init, parts[2], Q_TO_WIRE))
    return model, q, v, binding


Q_TO_WIRE = "SC2 offset motion, exact event and two moments v1"


class FreeForces(ClothPlatform):
    def forces(self, q, v):
        actual = super().forces(q, v)
        # Virtual extension for root isolation. Only its prefix before the first root is delivered.
        return Forces(actual.material, np.zeros_like(actual.rows), np.zeros_like(actual.pressures))


def propose(root):
    model, q, v, bind = original()
    h = Q(1, 64)
    try:
        free = fit(FreeForces(model.parts), q, v, h, bind, degree=4, sweeps=3)
        _, pos, vel = state_polynomials(free, q, v, model, h)
        br, first, end, _ = isolate(pos, model, 32)
        mid = sum(br[first]) / 2
        pp = np.array([float(evaluate(p, mid)) for p in pos])
        vv = np.array([float(evaluate(p, mid)) for p in vel])
        D, _, rate = model.maps(pp, vv)
        r = rate[first]
        slope = Q(round((500 / 6) * (-r) * (1 - 1000 * r) * (1 << 32)), 1 << 32)
        dt = h * (end - mid)
        ramp = dict(
            slope=str(slope),
            map=[str(Q(round(d * (1 << 48)), 1 << 48)) for d in D[first]],
            J=str(slope * dt**2 / 2),
            B=str(slope * dt**3 / 6),
        )
        data = dict(
            binding=bind,
            free=free,
            roots=[[str(left), str(right)] for left, right in br],
            first=first,
            end=str(end),
            ramp=ramp,
        )
        save(root / "proposal.json.gz", data)
    finally:
        model.parts[3].close()


def check(root):
    model, q, v, bind = original()
    try:
        data = saved(root / "proposal.json.gz")
        need(data["binding"] == data["free"]["binding"] == bind, "input binding")
        br = [tuple(map(Q, pair)) for pair in data["roots"]]
        h = Q(1, 64)

        def accept(obj, first, roots, ramp, vin):
            need(
                vin == v
                and len(obj["iterations"]) == 3
                and obj["force_calls"] == 13
                and obj["owner_charges"] == 78,
                "initial velocity or evaluation ledger",
            )
            return physical_bridge(obj, q, vin, model, h, roots, first, Q(data["end"]), ramp)

        qp, vp, cert = accept(data["free"], data["first"], br, data["ramp"], v)
        packet = checkpoint(qp, vp, Q(cert["bridge_end_s"]), model.signature)
        need(
            unpack(packet, model.signature) == (qp, vp, Q(cert["bridge_end_s"])),
            "checkpoint mismatch",
        )
        # Refuse earlier at the relevant boundary; do not repeat the whole successful force audit.
        bad = deepcopy(data["free"])
        bad["owners"] = bad["owners"][:-1]
        fake = deepcopy(data["free"])
        fake["pressure"][0, 0] = 1
        wrong = list(br)
        scalar = (Q(3, 1024) - Q(1, 500)) * 8 / h
        wrong[data["first"]] = (scalar, scalar)
        broken = dict(data["ramp"], J=str(Q(data["ramp"]["J"]) * 2))
        trials = [
            (
                "wrong-first",
                data["free"],
                (data["first"] + 1) % 6,
                br,
                data["ramp"],
                v,
                "wrong earliest",
            ),
            ("ballistic", data["free"], data["first"], wrong, data["ramp"], v, "wrong earliest"),
            ("owner", bad, data["first"], br, data["ramp"], v, "material address"),
            ("early-load", fake, data["first"], br, data["ramp"], v, "prebirth pressure"),
            (
                "lost-velocity",
                data["free"],
                data["first"],
                br,
                data["ramp"],
                [Q(0)] * 13,
                "initial velocity",
            ),
            ("forged-impulse", data["free"], data["first"], br, broken, v, "force moments"),
        ]
        negative = {}
        for name, obj, first, roots, ramp, vin, reason in trials:
            try:
                accept(obj, first, roots, ramp, vin)
            except ValueError as exc:
                need(reason in str(exc), "unexpected rejection: " + str(exc))
                negative[name] = str(exc)
            else:
                raise ValueError("accepted forgery: " + name)
        with (root / "bridge.bin").open("xb") as out:
            out.write(packet.tobytes())
        cert.update(
            negative=negative,
            checkpoint_sha256=sha256(packet.tobytes()).hexdigest(),
            proposal_sha256=sha256((root / "proposal.json.gz").read_bytes()).hexdigest(),
            input_binding=bind,
            protocol_sha256_lf=sha(SPEC, normalize=True),
        )
        save(root / "entry.json", cert)
        print(json.dumps(cert, indent=2))
    finally:
        model.parts[3].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("proposal", "check"))
    args = parser.parse_args()
    (propose if args.phase == "proposal" else check)(args.root)


if __name__ == "__main__":
    main()
