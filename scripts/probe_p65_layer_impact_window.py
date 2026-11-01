"""Continue the accepted SC-L1 impact kernel with the actual on-disk predecessor."""

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import probe_p65_layer_entry as kernel
from verify_p65_contact_layer_screen import INPUT, INPUT_SHA, fit, sha

PLAN = INPUT.with_name("p65-contact-layer-impact-window-protocol.json")
PLAN_SHA = "efb8347aae20d4bccc8207251794f8f3ee50ae88ccc168e3b2bbed809e1c176d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=range(2, 8), required=True)
    args = parser.parse_args()
    if sha(PLAN, normalize=True) != PLAN_SHA or sha(INPUT, normalize=True) != INPUT_SHA:
        raise ValueError("registered source or fixture changed")
    plan = json.loads(PLAN.read_bytes())
    prior_file = INPUT.with_name(plan["prefix_result"])
    if (
        sha(prior_file, normalize=True) != plan["prefix_result_sha256_lf"]
        or sha(Path(kernel.__file__), normalize=True) != plan["kernel_sha256_lf"]
    ):
        raise ValueError("prefix or integration kernel changed")
    previous = json.loads(prior_file.read_bytes())
    if not previous["series7"]["passed"]:
        raise ValueError("prefix has not been accepted")
    fixture = json.loads(INPUT.read_bytes())
    c = fixture["controls"]
    case = plan["case"]
    h, m, T = (Q(case[key]) for key in ("h_s", "m_kg", "T_s"))
    if (
        T / h != 8
        or T != Q(c["window_s"])
        or str(m) not in c["trace_masses_kg"]
        or str(h) not in c["report_h_s"]
        or Q(c["area_m2"]) != 1
        or Q(c["sc2"]["compressive_load_n"]) != 0
        or Q(c["sc2"]["gap_initial_m"]) != Q(case["g0_m"])
        or Q(c["sc2"]["closing_speed_m_s"]) != Q(case["closing_speed_m_s"])
    ):
        raise ValueError("unexpected impact control")
    if args.step == 2:
        origin = previous["raw_directory"] + "/series7/successor/end.state.json"
        source = INPUT.parents[2] / origin
        digest = plan["resume_sha256"]
    else:
        origin = f"step-{args.step - 1}/end.state.json"
        source = args.root / origin
        pred = json.loads(source.with_name("index.json").read_bytes())
        if pred["step"] != args.step - 1 or pred["protocol_sha256_lf"] != PLAN_SHA:
            raise ValueError("wrong predecessor")
        digest = pred["hashes"]["end.state.json"]
    if sha(source) != digest:
        raise ValueError("unexpected predecessor bytes")
    seed = source.read_bytes()
    rows = [*fixture["fit"]["static"], fixture["fit"]["compression"]]
    layer = fit(rows, [Q(r["pressure_pa"]) for r in rows])
    wire = json.loads(INPUT.with_name(plan["wire_from"]).read_bytes())["wire_limits"]
    folder = args.root / f"step-{args.step}"
    result = kernel.run(
        folder,
        seed,
        args.step * h,
        h,
        "series7",
        2 * m,
        layer,
        Q(fixture["rate_range_m_s"][1]),
        wire["force_evaluations_per_outer_step"],
    )
    result.update(
        step=args.step,
        origin=origin,
        seed_sha256=digest,
        protocol_sha256_lf=PLAN_SHA,
        driver_sha256_lf=sha(Path(__file__), normalize=True),
        kernel_sha256_lf=plan["kernel_sha256_lf"],
    )
    (folder / "index.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(
        f"impact step {args.step}, {result['parts']} pieces, {result['work']['force_charges']} coefficients"
    )


if __name__ == "__main__":
    main()
