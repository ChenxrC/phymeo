"""Use the same zero-load regression on old and reconstructed SC2 neighbor events."""

import argparse
import json
from pathlib import Path

from verify_p65_contact_layer_screen import sha

BASE = Path(__file__).resolve().parents[1] / "docs/benchmarks"
PLAN = BASE / "p65-sc2-causal-entry-protocol.json"


def regression(root, candidate):
    conf = json.loads(PLAN.read_bytes())
    path = BASE / conf["failure_control"]
    assert sha(path, normalize=True) == conf["failure_sha256_lf"]
    old = json.loads(path.read_bytes())["first_unverified_successor"]
    wanted = {r["owner"] for r in old["owners"] if r.get("witness")}
    assert len(wanted) == 4
    if not candidate:
        premature = {
            r["owner"]: r["witness"]["pressure_n"] for r in old["owners"] if r.get("witness")
        }
    else:
        result = json.loads((root / "cluster-accepted.json").read_bytes())
        assert result["accepted"] and result["inlet"] == old["source_cp_sha256"]
        assert {r["owner"] for r in result["events"]} == wanted and len(result["events"]) == 4
        assert all(result["gates"].values()) and result["distinct_and_ordered"]
        premature = {r["owner"]: r["prebirth_force_n"] for r in result["events"]}
    assert all(p == 0 for p in premature.values()), (
        f"represented pressure before its owner event: {premature}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    regression(args.root, args.candidate)
