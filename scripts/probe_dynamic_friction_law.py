"""Exact algebra for proposed split friction laws; not a native/automatic solver.

This probe imports no phymeo APIs and advances no production simulation. All
impulses are prescribed rational candidates, not outputs of a contact algorithm.
"""

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def add(*vectors):
    return tuple(sum(values, F(0)) for values in zip(*vectors, strict=True))


def scale(a, vector):
    return tuple(a * v for v in vector)


def kinetic(node, robots):
    return (dot(node, node) + dot(robots, robots)) / 2


def support():
    h, p = F(1, 4), F(1, 4)
    free_v = (h, -h)
    high_t, low_t = -h, -p / 4
    high = add(free_v, (high_t, p))
    low = add(free_v, (low_t, p))
    robot = h / 2 - p / 2
    terminal_only_position = scale(h, add(free_v, (F(0), p)))
    assert high == (0, 0) and robot == 0
    assert abs(high_t) <= 2 * p
    assert low == (F(3, 16), 0)
    assert scale(h, low) == (F(3, 64), 0)
    assert terminal_only_position == (F(1, 16), 0)
    return dict(
        h=h,
        normal_held=p,
        terminal_normal=F(0),
        high_mu=F(2),
        high_tangent=high_t,
        high_velocity=high,
        low_mu=F(1, 4),
        low_tangent=low_t,
        low_velocity=low,
        low_position=scale(h, low),
        terminal_only_position_drift=terminal_only_position,
    )


def material_oblique_impact():
    h, p, eta, compliance = F(1, 4), F(3, 7), F(1, 3), F(1, 16)
    y0 = (F(1, 8), F(9, 8))
    prediction = (y0[0] + h * (-1 + p), y0[1] - h)
    alpha = compliance / (h * h)
    multiplier = -(prediction[1] - prediction[0] - 1) / (2 + alpha)
    y = (prediction[0] - multiplier, prediction[1] + multiplier)
    vy = tuple((b - a) / h for a, b in zip(y0, y, strict=True))
    robot_y, robot_vy = -h * p / 2, -(p + eta) / 2
    tangent = -(p + eta) / 2
    a = (1 + tangent, vy[0] + eta)
    b = (F(1), vy[1])
    potential = (y[1] - y[0] - 1) ** 2 / (2 * compliance)
    energy = (dot(a, a) + dot(b, b)) / 2 + robot_vy**2 + potential
    assert y == (F(-3, 56), F(51, 56)) and robot_y == y[0]
    assert a == (F(13, 21), F(-8, 21)) and b == (F(1), F(-6, 7))
    assert tangent == -F(8, 21) and robot_vy == -F(8, 21)
    assert a[1] + b[1] + 2 * robot_vy == -2
    assert energy == F(1135, 882)
    return dict(
        h=h,
        normal_held=p,
        normal_terminal=eta,
        tangent_held=F(0),
        tangent_terminal=tangent,
        x_position=(h, h),
        y_position=y,
        robot_y=robot_y,
        node_velocity=(a, b),
        robot_vy=robot_vy,
        material_multiplier=multiplier,
        total_energy=energy,
        y_momentum=F(-2),
    )


def cross_contact(tangent):
    n1, n2 = (F(0), F(1)), (F(-3, 5), F(4, 5))
    v0, robot0 = (F(21, 10), F(-9, 5)), (F(1), F(6))
    gap0 = F(28, 5)
    p2 = (33 + 10 * tangent) / 28
    p1 = (13 - 2 * tangent) / 14
    before = add(v0, scale(p1, n1), scale(p2, n2), (tangent, F(0)))
    robot_pre = (robot0[0] - p1, robot0[1] - p2)
    position = before  # h=1 and x0=0, with no material or external force
    gaps = (dot(position, n1) - robot_pre[0], gap0 + dot(position, n2) - robot_pre[1])
    assert gaps == (0, 0)
    eta2 = F(14, 5)
    after = add(before, scale(eta2, n2))
    robot_post = (robot_pre[0], robot_pre[1] - eta2)
    assert dot(after, n2) - robot_post[1] == 0
    assert dot(after, n1) - robot_post[0] > 0
    terminal_energy = kinetic(after, robot_post) - kinetic(before, robot_pre)
    assert terminal_energy == -F(196, 25)
    assert kinetic(after, robot_post) < kinetic(v0, robot0)
    return dict(
        normal_held=(p1, p2),
        tangent_held=tangent,
        normal_terminal=(F(0), eta2),
        node_pre=before,
        node_post=after,
        robot_pre=robot_pre,
        robot_post=robot_post,
        final_gaps=gaps,
        held_tangent_work=tangent * before[0],
        total_tangent_dot_final_slip=tangent * after[0],
        terminal_energy_change=terminal_energy,
        total_energy_after=kinetic(after, robot_post),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output path; evidence is not overwritten")
    phasewise = cross_contact(-F(1, 2))
    endpoint = cross_contact(F(201, 550))
    assert phasewise["node_pre"] == (1, 0)
    assert phasewise["node_post"] == (-F(17, 25), F(56, 25))
    assert phasewise["held_tangent_work"] < 0
    assert phasewise["total_tangent_dot_final_slip"] == F(17, 50) > 0
    assert endpoint["node_pre"][0] == F(42, 25)
    assert endpoint["node_post"][0] == 0
    assert endpoint["held_tangent_work"] == F(4221, 6875) > 0
    assert abs(endpoint["tangent_held"]) < endpoint["normal_held"][0] / 2
    result = dict(
        status="exact_algebra_verified",
        scope="prescribed rational candidates only; no production integration or backend execution",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        static_support_and_terminal_only_counterexample=support(),
        material_oblique_impact=material_oblique_impact(),
        phasewise_not_endpoint_coulomb=phasewise,
        endpoint_not_phasewise_passive=endpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, default=str, indent=2) + "\n", encoding="utf-8")
    print(
        "Exact support/chain and two cross-contact candidate checks passed; no native simulation."
    )


if __name__ == "__main__":
    main()
