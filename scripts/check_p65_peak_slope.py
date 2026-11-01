"""Independent polynomial reproduction and prior between-node failure gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as ch
from p65_common_motion import need, tables
from probe_p65_joint_readback import save
from verify_p65_contact_layer_screen import sha

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"
PLAN = BASE / "p65-sc2-peak-slope-protocol.json"


def run(root, red):
    spec = json.loads(PLAN.read_bytes())
    inp = BASE / spec["previous"]
    need(sha(inp, normalize=True) == spec["previous_sha256_lf"], "changed representation falsifier")
    previous = json.loads(inp.read_bytes())
    lower = [
        z["exact_nodal_interpolation_J_error_lower_ns"]
        for z in previous["representation"]["owners"]
    ]
    if red:
        need(max(lower) <= 1e-10, f"even exact old nodal pressures fail: {max(lower):.8g} Ns")
        return
    from p65_peak_slope import additional_modes, primitives

    nodes, card, _, _ = tables(13)
    exact = np.eye(21)
    values = ch.chebvander(2 * nodes - 1, 20) @ exact
    slopes = ch.chebvander(2 * nodes - 1, 19) @ (2 * ch.chebder(exact, axis=0))
    low = card @ values
    high = additional_modes(nodes, card, values, slopes)
    full = high.copy()
    full[: len(low)] += low
    err = float(np.max(abs(full - exact)))
    zero = float(np.max(abs(ch.chebvander(2 * nodes - 1, 20) @ high)))
    v, x = primitives(high, nodes, 1.0)
    # Compare time integrals against independently constructed exact monomial antiderivatives.
    from fractions import Fraction as Q

    from check_p65_common_motion import basis, integral
    from check_p65_sc2_common_entry import evaluate

    basis_terms = basis()
    vref = []
    xref = []
    for s in map(lambda a: Q(float(a)), nodes):
        vref.append([float(evaluate(integral(p), s)) for p in basis_terms])
        xref.append([float(evaluate(integral(integral(p)), s)) for p in basis_terms])
    _, _, V, P = tables(13)
    ev = float(np.max(abs(v + V @ values - np.array(vref))))
    ex = float(np.max(abs(x + P @ values - np.array(xref))))
    need(max(err, zero, ev, ex) < 5e-10, "degree-20 reconstruction or force moments")
    report = dict(
        accepted=True,
        polynomial_controls=21,
        max_coefficient_error=err,
        value_at_node_error=zero,
        impulse_error=ev,
        backward_error=ex,
    )
    save(root / "basis-check.json", report)
    print(report)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--red", action="store_true")
    argv = cli.parse_args()
    run(argv.root, argv.red)
