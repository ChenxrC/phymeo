"""One registered regression on the saved SC2 failure and its reconstructed replacement."""

import argparse
import json
from pathlib import Path

from verify_p65_contact_layer_screen import sha

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"
PLAN = BASE / "p65-sc2-event-cell-protocol.json"


def check(root, candidate):
    plan = json.loads(PLAN.read_bytes())
    parent = BASE / plan["source"]
    if sha(parent, normalize=True) != plan["source_sha256_lf"]:
        raise ValueError("wrong reference")
    old = json.loads(parent.read_bytes())
    if not candidate:
        witness = old["window"]["first_spurious_or_negative"][1]
        force = witness["p_n"]
        gap = witness["gap_m"][0]
    else:
        cell = json.loads((root / "accepted.json").read_bytes())
        assert cell["accepted"] and cell["same_inlet"] and cell["all_original_owners"]
        assert cell["new_endpoint"] != cell["old_endpoint"] and all(cell["gates"].values())
        assert cell["early_witness"]["owner"] == old["counterexample"]["owner"]
        force = cell["early_witness"]["force_n"]
        gap = cell["early_witness"]["gap_m"][0]
    assert gap > 0.002 and force == 0, f"outside layer: {gap} m, published {force} N"


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--candidate", action="store_true")
    args = cli.parse_args()
    check(args.root, args.candidate)
