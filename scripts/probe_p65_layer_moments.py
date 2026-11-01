"""Produce SC-L1 scalar piece paths without reading any reference trajectories."""

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, fit, sha

PROTOCOL = INPUT.with_name("p65-contact-layer-intrastep-protocol.json")
PROTOCOL_SHA = "d65d0d6138a2f25fa10f64a6dd3dc76f57a8cbeb1364fcc03f70e2f430048e87"


@dataclass(frozen=True)
class State:
    gap: float
    y: float  # 1+d*closing_speed


# Positive closed quadrature exact for polynomials through degree seven. The
# law on an active trial is degree five; the time-weighted law is degree six.
def quadrature():
    weights = (41, 216, 27, 272, 27, 216, 41)
    for index, weight in enumerate(weights):
        s = index / 6
        a = 1 - s
        yield (
            weight / 840,
            4 - 6 * s,
            6 * s - 2,
            (a * a, 2 * s * a, s * s),
            (a**3, 3 * a * a * s, 3 * a * s * s, s**3),
        )


QUADRATURE = tuple(quadrature())


class PieceSolver:
    def __init__(self, initial, duration, mass, external, layer, budget):
        self.initial = initial
        self.duration = float(duration)
        self.mass = mass
        self.external = external
        self.width = float(layer.delta)
        self.d = float(layer.d)
        self.k = float(layer.k)  # the registered coupon has unit area
        self.budget = budget
        self.calls = 0

    def compression_controls(self, middle, end):
        start = self.width - self.initial.gap
        factor = self.duration / (3 * self.d)
        second = start + factor * (self.initial.y - 1)
        third = second + factor * (middle - 1)
        return start, second, third, third + factor * (end - 1)

    def system(self, middle, end):
        if self.calls >= self.budget:
            raise ArithmeticError("outer step residual evaluation budget exhausted")
        self.calls += 1
        start = self.initial.y
        z = self.compression_controls(middle, end)
        scale = 2 * self.mass / (self.d * self.duration)
        left = scale * (middle - start) - self.external
        right = scale * (end - middle) - self.external
        aa, ab, ba, bb = scale, 0.0, -scale, scale
        factor = self.duration / (3 * self.d)
        for weight, wl, wr, basis_y, basis_z in QUADRATURE:
            y = basis_y[0] * start + basis_y[1] * middle + basis_y[2] * end
            x = sum(c * b for c, b in zip(z, basis_z, strict=True))
            # This polynomial extension is only used to propose a root. The
            # independent observer tests the signed geometry and clipped law.
            pressure = self.k * x * y
            dm = self.k * (factor * (basis_z[2] + basis_z[3]) * y + x * basis_y[1])
            de = self.k * (factor * basis_z[3] * y + x * basis_y[2])
            left += weight * wl * pressure
            right += weight * wr * pressure
            aa += weight * wl * dm
            ab += weight * wl * de
            ba += weight * wr * dm
            bb += weight * wr * de
        return (left, right), (aa, ab, ba, bb)

    def solve(self, limits):
        middle = end = self.initial.y
        tolerance = 1e-12 * (
            1 + abs(self.external) + abs(self.k * self.compression_controls(middle, end)[0] * end)
        )
        residual, jacobian = self.system(middle, end)
        for iteration in range(limits["newton_updates_per_piece"] + 1):
            norm = max(map(abs, residual))
            if norm <= tolerance:
                speed_sum = math.fsum((self.initial.y - 1, middle - 1, end - 1))
                gap = self.initial.gap - self.duration * speed_sum / (3 * self.d)
                return middle, State(gap, end)
            if iteration == limits["newton_updates_per_piece"]:
                break
            a, b, c, d = jacobian
            det = a * d - b * c
            if not math.isfinite(det) or det == 0:
                raise ArithmeticError("singular moment Jacobian")
            r, s = residual
            dm, de = (b * s - d * r) / det, (c * r - a * s) / det
            for backtrack in range(limits["backtracks_per_update"] + 1):
                factor = 0.5**backtrack
                m, e = middle + factor * dm, end + factor * de
                trial, derivative = self.system(m, e)
                if max(map(abs, trial)) < norm:
                    middle, end = m, e
                    residual, jacobian = trial, derivative
                    break
            else:
                raise ArithmeticError("moment update did not reduce the residual")
        raise ArithmeticError("moment iteration limit")


def produce(case, mass, grid, layer, fixture, limits, stream):
    controls = fixture["controls"]
    if Fraction(controls["area_m2"]) != 1:
        raise ValueError("this reduced control requires unit reference area")
    M = 2 * float(mass)
    width, k, d = map(float, (layer.delta, layer.k, layer.d))
    if case == "support":
        state = State(0.0, 1.0)
        external = float(mass) / 4
    else:
        closing = float(Fraction(controls["sc2"]["closing_speed_m_s"]))
        state = State(float(Fraction(controls["sc2"]["gap_initial_m"])), 1 + d * closing)
        external = 0.0
    rate_max = float(Fraction(fixture["rate_range_m_s"][1]))
    rate = max(k * d * width / M, math.sqrt(k * (1 + d * rate_max) / M))
    window = Fraction(controls["window_s"])
    count = window / grid
    if count.denominator != 1:
        raise ValueError("incomplete original grid")

    def emit(step, t0, t1, old, new, middle, calls, phase):
        row = dict(
            case=case,
            m=str(mass),
            h=str(grid),
            step=step,
            t0=str(t0),
            t1=str(t1),
            phase=phase,
            g0=old.gap,
            g1=new.gap,
            y0=old.y,
            yb=middle,
            ye=new.y,
            evaluations=calls,
        )
        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    costs = []
    for step in range(int(count)):
        left, right = grid * step, grid * (step + 1)
        used = 0
        if state.gap > width:
            if external or state.y <= 1:
                raise ArithmeticError("unsupported free branch in registered controls")
            v = (Fraction(state.y) - 1) / Fraction(d)
            until_entry = (Fraction(state.gap) - Fraction(width)) / v
            dt = min(right - left, until_entry)
            new = State(float(Fraction(state.gap) - v * dt), state.y)
            emit(step, left, left + dt, state, new, state.y, 0, "free")
            left, state = left + dt, new
        if left < right:
            parts = max(1, math.ceil(8 * float(right - left) * rate))
            dt = (right - left) / parts
            for part in range(parts):
                solver = PieceSolver(
                    state,
                    dt,
                    M,
                    external,
                    layer,
                    limits["residual_evaluations_per_outer_step"] - used,
                )
                try:
                    middle, new = solver.solve(limits)
                except ArithmeticError as error:
                    return dict(
                        case=case,
                        m=str(mass),
                        h=str(grid),
                        passed=False,
                        failed_step=step,
                        failed_piece=part,
                        evaluations=used + solver.calls,
                        reason=str(error),
                    )
                used += solver.calls
                emit(
                    step,
                    left + part * dt,
                    left + (part + 1) * dt,
                    state,
                    new,
                    middle,
                    solver.calls,
                    "active",
                )
                state = new
        costs.append(used)
    return dict(
        case=case,
        m=str(mass),
        h=str(grid),
        passed=True,
        steps=len(costs),
        max_step_evaluations=max(costs),
        total_evaluations=sum(costs),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(PROTOCOL, normalize=True) != PROTOCOL_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered protocol or material changed")
    fixture = json.loads(INPUT.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    fit_rows = [*fixture["fit"]["static"], fixture["fit"]["compression"]]
    layer = fit(fit_rows, [Fraction(row["pressure_pa"]) for row in fit_rows])
    args.output_dir.mkdir(exist_ok=False)
    path = args.output_dir / "pieces.jsonl"
    with path.open("x", encoding="utf8", newline="\n") as stream:
        runs = [
            produce(case, Fraction(mass), Fraction(h), layer, fixture, protocol["limits"], stream)
            for mass in fixture["controls"]["trace_masses_kg"]
            for case in ("support", "impact")
            for h in fixture["controls"]["report_h_s"]
        ]
    report = dict(
        schema="p65-scalar-moment-producer-v1",
        runs=runs,
        pieces_sha256=sha(path),
        source_sha256_lf=sha(Path(__file__), normalize=True),
        protocol_sha256_lf=PROTOCOL_SHA,
    )
    (args.output_dir / "producer.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    for row in runs:
        print(row["case"], row["m"], row["h"], row.get("reason", "completed"))
    return 0 if all(row["passed"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
