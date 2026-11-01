"""Run the two registered SC-L1 entry steps without access to reference trajectories."""

import argparse
import json
import math
from fractions import Fraction as Q
from math import comb
from pathlib import Path

from probe_p65_layer_joint_publication import SIZE, advance_piece, decode, packet, ratio
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, fit, sha

PLAN = INPUT.with_name("p65-contact-layer-entry-protocol.json")
PLAN_SHA = "9fc803c5468982c580dac55bb567dd755bf16132242208bb967b5d2221052067"


def fields(method):
    return dict(
        force_charges=0,
        convolution_terms=0,
        conversion_terms=0,
        half_subdivision_means=0,
        method=method,
    )


def traction_series(g, y, dt, M, layer, work):
    # Powers are in normalized time, so each derivative recurrence includes dt.
    x = [layer.delta - g]
    factors = [y]
    p = []
    for k in range(8):
        p.append(layer.k * sum(x[j] * factors[k - j] for j in range(k + 1)))
        work["convolution_terms"] += k + 1
        work["force_charges"] += 1
        if k < 7:
            x.append(dt * (factors[k] - int(k == 0)) / (layer.d * (k + 1)))
            factors.append(-dt * layer.d * p[k] / (M * (k + 1)))
    controls = []
    for i in range(8):
        value = sum(p[j] * Q(comb(i, j), comb(7, j)) for j in range(i + 1))
        work["conversion_terms"] += i + 1
        if value < 0:
            raise ArithmeticError("negative predicted traction ordinate")
        controls.append(Q(float(value)))
    return controls


def half_controls(p, side, work):
    first, last = [p[0]], [p[-1]]
    while len(p) > 1:
        p = [(x + y) / 2 for x, y in zip(p, p[1:])]
        work["half_subdivision_means"] += len(p)
        first.append(p[0])
        last.append(p[-1])
    return first if side == 0 else list(reversed(last))


def append(file, data):
    offset = file.tell()
    file.write(data)
    file.flush()
    return offset


def read(path, offset):
    with path.open("rb") as source:
        source.seek(offset)
        return source.readline()


def line(obj):
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("ascii")


def run(directory, initial, begin, h, method, M, layer, rate_max, cap):
    directory.mkdir(parents=True, exist_ok=False)
    states = directory / "states.jsonl"
    macros = directory / "forces.jsonl"
    ledger = directory / "pieces.jsonl"
    work = fields(method)
    initial_time, g0, y0 = decode(initial)
    if initial_time != begin:
        raise ValueError("seed time mismatch")
    event = None
    if g0 > layer.delta:
        closing = (y0 - 1) / layer.d
        if closing <= 0:
            raise ValueError("registered rising approach required")
        event = begin + (g0 - layer.delta) / closing
        if event > begin + h:
            raise ValueError("registered step has no layer entry")
    active_start = begin if event is None else event
    span = begin + h - active_start
    rate = max(
        float(layer.k * layer.d * layer.delta / M),
        math.sqrt(float(layer.k * (1 + layer.d * rate_max) / M)),
    )
    count = max(1, math.ceil(8 * float(span) * rate))
    active_dt = span / count
    groups = []
    if event is not None:
        groups.append(("free", event - begin, 1))
    group_size = 2 if method == "series7" else 1
    for i in range(0, count, group_size):
        size = min(group_size, count - i)
        groups.append(("active", size * active_dt, size))

    largest = len(initial)
    with states.open("xb") as motion, macros.open("xb") as forces, ledger.open("xb") as pieces:
        state_offset = append(motion, initial)
        for index, (phase, duration, parts) in enumerate(groups):
            # Both halves start from their actual persisted state. Only their force record is shared.
            t0, g, y = decode(read(states, state_offset))
            if phase == "free":
                controls = [Q(0)]
            elif method == "affine":
                if work["force_charges"] + 3 > cap:
                    raise ArithmeticError("force budget")
                _, _, left, right, _, _, calls = advance_piece(g, y, duration, M, layer)
                work["force_charges"] += calls
                controls = [left, right]
            else:
                if work["force_charges"] + 8 > cap:
                    raise ArithmeticError("coefficient budget")
                controls = traction_series(g, y, duration, M, layer, work)
            words = [float(z).hex() for z in controls]
            macro_offset = append(
                forces,
                line(dict(phase=phase, t=str(t0), dt=str(duration), parts=parts, words=words)),
            )
            dt = duration / parts
            for side in range(parts):
                t, g, y = decode(read(states, state_offset))
                if t != t0 + side * dt:
                    raise ValueError("broken readback continuity")
                saved = json.loads(read(macros, macro_offset))
                coeff = [Q(float.fromhex(word)) for word in saved["words"]]
                if parts == 2:
                    coeff = half_controls(coeff, side, work)
                n = len(coeff)
                j = dt * sum(coeff) / n
                b = dt**2 * sum((n - i) * f for i, f in enumerate(coeff)) / (n * (n + 1))
                v = (y - 1) / layer.d
                data = packet(t + dt, g - dt * v + b / M, y - layer.d * j / M)
                largest = max(largest, len(data))
                state_offset = append(motion, data)
                append(pieces, line(dict(macro=index, side=side, j=ratio(j), b=ratio(b))))
        final = read(states, state_offset)
    if decode(final)[0] != begin + h or largest > SIZE:
        raise ValueError("end packet mismatch")
    (directory / "end.state.json").write_bytes(final)
    hashes = {
        p: sha(directory / p)
        for p in ("end.state.json", "states.jsonl", "forces.jsonl", "pieces.jsonl")
    }
    return dict(
        parts=count + int(event is not None),
        active_parts=count,
        macros=len(groups),
        event=str(event) if event is not None else None,
        start=str(begin),
        end=str(begin + h),
        work=work,
        max_motion_bytes=largest,
        hashes=hashes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=("affine", "series7"), required=True)
    parser.add_argument("--phase", choices=("entry", "successor"), required=True)
    args = parser.parse_args()
    if sha(PLAN, normalize=True) != PLAN_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered inputs changed")
    if args.method == "series7":
        first = json.loads((args.root / "affine" / "audit.json").read_bytes())
        if first["protocol_sha256_lf"] != PLAN_SHA or not first["comparison_eligible"]:
            raise ValueError("conditional comparison has not been opened")
    fixture = json.loads(INPUT.read_bytes())
    controls = fixture["controls"]
    rows = [*fixture["fit"]["static"], fixture["fit"]["compression"]]
    layer = fit(rows, [Q(row["pressure_pa"]) for row in rows])
    plan = json.loads(PLAN.read_bytes())
    limits = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    h = Q("1/64")
    if (
        controls["area_m2"] != "1"
        or str(h) not in controls["report_h_s"]
        or Q(controls["sc2"]["compressive_load_n"]) != 0
    ):
        raise ValueError("unexpected original control")
    if args.phase == "entry":
        begin = Q(0)
        initial = packet(
            begin,
            Q(controls["sc2"]["gap_initial_m"]),
            1 + layer.d * Q(controls["sc2"]["closing_speed_m_s"]),
        )
        origin = "original scalar impact inlet"
    else:
        begin = h
        origin = args.method + "/entry/end.state.json"
        initial = (args.root / origin).read_bytes()
    directory = args.root / args.method / args.phase
    result = run(
        directory,
        initial,
        begin,
        h,
        args.method,
        2 * Q("1/2"),
        layer,
        Q(fixture["rate_range_m_s"][1]),
        limits["force_evaluations_per_outer_step"],
    )
    result.update(
        origin=origin,
        method=args.method,
        phase=args.phase,
        protocol_sha256_lf=PLAN_SHA,
        source_sha256_lf=sha(Path(__file__), normalize=True),
    )
    (directory / "index.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(args.method, args.phase, result["parts"], result["work"])


if __name__ == "__main__":
    main()
