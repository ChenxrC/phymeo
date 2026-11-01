"""Advance the registered scalar tail witnesses using one serializable motion/traction packet."""

import argparse
import json
import math
from fractions import Fraction as Q
from pathlib import Path

from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, fit, sha

PLAN = INPUT.with_name("p65-contact-layer-publication-protocol.json")
PLAN_SHA = "58f93ed30dc2733cfb5814428442718fb0ade03c15e26d4d43cdf6d6dcc617af"
SCHEMA = "sc-l1-motion-v1"
BITS = 512
SIZE = 1024


def ratio(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > BITS:
        raise ValueError("motion or moment integer exceeds packet limit")
    return str(value)


def project(value):
    word = float(value)
    return [word.hex(), ratio(value - Q(word))]


def packet(t, gap, y):
    state = dict(schema=SCHEMA, t=ratio(t), gap=project(gap), y=project(y))
    data = (json.dumps(state, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
    if len(data) > SIZE:
        raise ValueError("motion packet exceeds byte limit")
    return data


def decode(data):
    if len(data) > SIZE:
        raise ValueError("motion packet exceeds byte limit")
    row = json.loads(data)
    t = Q(row["t"])
    gap, y = (Q(float.fromhex(word)) + Q(remainder) for word, remainder in (row["gap"], row["y"]))
    if packet(t, gap, y) != data:
        raise ValueError("noncanonical or incomplete motion packet")
    return t, gap, y


def advance_piece(gap, y, dt, mass, layer):
    """Three rounded law evaluations. Only the two final forces act on the motion."""
    v = (y - 1) / layer.d
    evaluations = 0

    def law(g, factor):
        nonlocal evaluations
        evaluations += 1
        if g < 0:
            raise ValueError("hard core violation in force evaluation")
        return Q(float(layer.static_pressure(g) * max(factor, 0)))

    def end(right):
        j = dt * (left + right) / 2
        b = dt**2 * (2 * left + right) / 6
        return gap - dt * v + b / mass, y - layer.d * j / mass, j, b

    left = law(gap, y)
    pg, py, _, _ = end(left)
    predicted_right = law(pg, py)
    cg, cy, _, _ = end(predicted_right)
    right = law(cg, cy)
    g1, y1, j, b = end(right)
    return g1, y1, left, right, j, b, evaluations


def name(h, step):
    return f"h{Q(h).denominator}-step{step}"


def produce(directory, seed, step_start, h, layer, fixture, limits):
    directory.mkdir(exist_ok=False, parents=True)
    states, loads = directory / "states.jsonl", directory / "loads.jsonl"
    mass = 2 * Q("1/2")  # original impact witness has relative mass 2m, F=0
    rate_max = Q(fixture["rate_range_m_s"][1])
    rate = max(
        float(layer.k * layer.d * layer.delta / mass),
        math.sqrt(float(layer.k * (1 + layer.d * rate_max) / mass)),
    )
    count = max(1, math.ceil(8 * float(h) * rate))
    dt = h / count
    if decode(seed)[0] != step_start:
        raise ValueError("seed time differs from the requested step")
    evaluations = 0
    largest = 0
    with states.open("xb") as sink, loads.open("x", encoding="ascii", newline="\n") as ledger:
        offset = 0
        sink.write(seed)
        with states.open("rb") as published:
            for index in range(count):
                sink.flush()
                published.seek(offset)
                data = published.readline(SIZE + 1)
                t, gap, y = decode(data)
                if t != step_start + index * dt:
                    raise ValueError("discontinuous published time")
                if evaluations + 3 > limits["force_evaluations_per_outer_step"]:
                    raise ValueError("force evaluation limit")
                g1, y1, left, right, j, b, calls = advance_piece(gap, y, dt, mass, layer)
                evaluations += calls
                outgoing = packet(t + dt, g1, y1)
                largest = max(largest, len(data), len(outgoing))
                offset = sink.tell()
                sink.write(outgoing)
                row = dict(
                    p0=float(left).hex(), p1=float(right).hex(), j=ratio(j), b=ratio(b), calls=calls
                )
                ledger.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            published.seek(offset)
            final = published.readline(SIZE + 1)
            if decode(final)[0] != step_start + h:
                raise ValueError("incomplete original step")
    (directory / "end.state.json").write_bytes(final)
    return dict(
        parts=count,
        calls=evaluations,
        max_packet_bytes=largest,
        states_sha256=sha(states),
        loads_sha256=sha(loads),
        end_sha256=sha(directory / "end.state.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", choices=("primary", "remaining"), required=True)
    parser.add_argument("--mode", choices=("witness", "successor"), required=True)
    args = parser.parse_args()
    if sha(PLAN, normalize=True) != PLAN_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered protocol or material changed")
    plan = json.loads(PLAN.read_bytes())
    limits = plan["wire_limits"]
    if limits["bits_per_exported_integer"] != BITS or limits["bytes_per_motion_packet"] != SIZE:
        raise ValueError("wire limits changed")
    if args.group == "remaining":
        primary = json.loads((args.root / "primary-audit.json").read_bytes())
        if primary["protocol_sha256_lf"] != PLAN_SHA or not primary["passed"]:
            raise ValueError("primary has no accepted audit")
    fixtures = json.loads(INPUT.read_bytes())
    rows = [*fixtures["fit"]["static"], fixtures["fit"]["compression"]]
    layer = fit(rows, [Q(row["pressure_pa"]) for row in rows])
    ids = [plan["primary"]] if args.group == "primary" else plan["others"]
    parent = args.root / args.group / args.mode
    parent.mkdir(exist_ok=False, parents=True)
    if args.mode == "witness":
        witness_file = INPUT.with_name(plan["old_results"])
        inlets = {
            (r["h"], r["step"]): (Q(r["g0"]), Q(r["y0"]))
            for r in json.loads(witness_file.read_bytes())["outer_failures"]
            if r["case"] == "impact" and r["m"] == "1/2"
        }
    reports = []
    for grid, step in ids:
        h = Q(grid)
        case = name(h, step)
        if args.mode == "witness":
            t = step * h
            seed = packet(t, *inlets[grid, step])
            origin = "registered witness inlet"
        else:
            t = (step + 1) * h
            origin = args.root / args.group / "witness" / case / "end.state.json"
            seed = origin.read_bytes()
            origin = origin.relative_to(args.root).as_posix()
        result = produce(parent / case, seed, t, h, layer, fixtures, limits)
        reports.append(dict(id=case, origin=origin, start=str(t), end=str(t + h), **result))
    summary = dict(
        group=args.group,
        mode=args.mode,
        source_sha256_lf=sha(Path(__file__), normalize=True),
        protocol_sha256_lf=PLAN_SHA,
        runs=reports,
    )
    (parent / "index.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(
        f"{args.group} {args.mode}: {len(reports)} intervals, {sum(r['parts'] for r in reports)} internal pieces"
    )


if __name__ == "__main__":
    main()
