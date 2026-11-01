"""A separate two-body spring and finite contact layer use the unchanged common-path kernel."""

import argparse
import json
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

import numpy as np
from check_p65_common_motion import (
    add,
    bounds,
    integral,
    one_integral,
    outward,
    power,
    scale,
    sub,
    times,
)
from p65_common_motion import Forces, checkpoint, endpoint, fit, need, unpack
from probe_p65_joint_readback import save
from verify_p65_contact_layer_screen import sha

from phymeo.validation.bending_bounds import Interval

PLAN = (
    Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-common-motion-transfer-protocol.json"
)
PLAN_SHA = "eabe09344e3d8112b0f050c5aacd2706a3fcdd14bd917d3a43b33e433dcdc2fc"


class SpringPair:
    def __init__(self, c):
        self.mass = np.array([float(Q(x)) for x in c["masses_kg"]])
        self.external = self.mass * float(Q(c["gravity_m_s2"]))
        self.owners = ("surface-A/surface-B",)
        self.k = float(Q(c["material_k_n_m"]))
        self.rest = float(Q(c["material_rest_m"]))
        self.kc = float(Q(c["layer_k_n_m"]))
        self.width = float(Q(c["layer_width_m"]))

    def forces(self, q, v):
        gap = q[0] - q[1]
        need(0 <= gap < self.width, "outside active hard layer")
        material = -self.k * (gap - self.rest)
        p = self.kc * (self.width - gap)
        return Forces(np.array([material, -material]), np.array([[p, -p]]), np.array([p]))

    def linearization(self, q, v, first):
        return -(self.k + self.kc) * np.array([[1.0, -1.0], [-1.0, 1.0]]), np.zeros((2, 2))


def trig(z, order):
    # cos(sqrt(z)), sin(sqrt(z))/sqrt(z); alternating terms decrease for this case.
    need(0 < z < 1, "series not bounded by first unused term")
    n = 18
    a = sum((-z) ** i / factorial(2 * i + order) for i in range(n))
    b = a + (-z) ** n / factorial(2 * n + order)
    return Interval(min(a, b), max(a, b))


def error(value, box):
    return max(abs(value - box.low), abs(value - box.high))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    need(sha(PLAN, normalize=True) == PLAN_SHA, "transfer specification changed")
    spec = json.loads(PLAN.read_bytes())
    model = SpringPair(spec)
    q0 = list(map(Q, spec["initial_positions_m"]))
    v0 = list(map(Q, spec["initial_velocities_m_s"]))
    h = Q(spec["h_s"])
    sig = PLAN_SHA
    initial = checkpoint(q0, v0, Q(0), sig)
    qin, vin, _ = unpack(initial, sig)
    curve = fit(model, qin, vin, h, sig)
    q1, v1 = endpoint(curve, qin, vin)
    packet = checkpoint(q1, v1, h, sig)
    need(unpack(packet, sig) == (q1, v1, h), "nonzero inlet or exact endpoint identity")
    m1, m2 = map(Q, spec["masses_kg"])
    km = Q(spec["material_k_n_m"])
    kc = Q(spec["layer_k_n_m"])
    rest = Q(spec["material_rest_m"])
    width = Q(spec["layer_width_m"])
    g0 = q0[0] - q0[1]
    w0 = v0[0] - v0[1]
    mu = m1 + m2
    equilibrium = (km * rest + kc * width) / (km + kc)
    a = g0 - equilibrium
    w2 = (km + kc) * (1 / m1 + 1 / m2)
    need(w2 == 63, "original frequency")
    C = trig(w2 * h * h, 0)
    S = trig(w2 * h * h, 1)
    gap = equilibrium + a * C + w0 * h * S
    speed = -a * w2 * h * S + w0 * C
    gravity = Q(spec["gravity_m_s2"])
    xc = (m1 * q0[0] + m2 * q0[1]) / mu
    vc = (m1 * v0[0] + m2 * v0[1]) / mu
    center = xc + h * vc + gravity * h * h / 2
    center_v = vc + gravity * h
    qref = [center + m2 / mu * gap, center - m1 / mu * gap]
    vref = [center_v + m2 / mu * speed, center_v - m1 / mu * speed]
    J = kc * ((width - equilibrium) * h - a * h * S - w0 * (1 - C) / w2)
    B = kc * ((width - equilibrium) * h * h / 2 - a * (1 - C) / w2 - w0 * h * (1 - S) / w2)
    qpoly = []
    work = 0
    P = power(curve["pressure"][:, 0])
    mat = power(curve["material"][:, 0])
    for i, m in enumerate((m1, m2)):
        f = add((m * gravity,), add(power(curve["material"][:, i]), power(curve["rows"][:, 0, i])))
        v = integral(scale(f, h / m), v0[i])
        q = integral(scale(v, h), q0[i])
        need(sum(q) == q1[i] and sum(v) == v1[i], "all-coordinate readback")
        qpoly.append(q)
        work += m * gravity * (q1[i] - q0[i])
    path_gap = sub(qpoly[0], qpoly[1])
    gaps = bounds(path_gap)
    min_gap = min(iv.low for iv in gaps)
    need(
        min_gap > 0 and max(iv.high for iv in gaps) < width and min(iv.low for iv in bounds(P)) > 0,
        "whole-path active layer",
    )
    dP = add(P, scale(sub(path_gap, (width,)), kc))
    dmat = add(mat, scale(sub(path_gap, (rest,)), km))
    actual_j = h * one_integral(P)
    actual_b = h * h * one_integral(times((1, -1), P))
    dg = max(error(x, y) for x, y in zip(q1, qref, strict=True))
    dv = max(error(x, y) for x, y in zip(v1, vref, strict=True))
    jerr = error(actual_j, J)
    berr = error(actual_b, B) / min(m1, m2)
    law = h * max(iv.magnitude for iv in bounds(dP))
    merr = h * h / 2 * max(iv.magnitude for iv in bounds(dmat)) / min(m1, m2)

    def U(g):
        return km * (g - rest) ** 2 / 2 + kc * (width - g) ** 2 / 2

    energy = (
        sum(m * (b * b - a * a) / 2 for m, b, a in zip((m1, m2), v1, v0, strict=True))
        + U(q1[0] - q1[1])
        - U(g0)
        - work
    )
    need(
        dg < Q("1e-8")
        and dv < Q("2e-6")
        and max(jerr, law) < Q("1e-10")
        and berr < Q("1e-10")
        and merr < Q("1e-8")
        and energy < Q("1e-8"),
        "closed form discriminators",
    )
    save(
        root / "transfer-result.json",
        dict(
            schema="p65-common-motion-transfer-results-v1",
            accepted=True,
            model="two translational masses, one owner, independent undamped spring and active finite layer",
            protocol_sha256_lf=PLAN_SHA,
            initial_state=[q0, v0],
            final_state=[q1, v1],
            analytic_positions_m=[outward(a) for a in qref],
            analytic_velocities_m_s=[outward(a) for a in vref],
            position_error_upper_m=float(dg),
            speed_error_upper_m_s=float(dv),
            impulse_error_upper_ns=float(jerr),
            time_moment_error_upper_per_mass_m=float(berr),
            full_path_law_impulse_error_upper_ns=float(law),
            full_path_material_position_error_upper_m=float(merr),
            full_path_minimum_gap_lower_m=float(min_gap),
            signed_original_energy_defect_j=float(energy),
            owner_evaluations=curve["owner_charges"],
            verification="analytical oscillator, rational alternating trig enclosures, whole-path polynomial laws",
        ),
    )
    print("independent two-body transfer accepted", float(dg), float(jerr), curve["owner_charges"])


if __name__ == "__main__":
    main()
