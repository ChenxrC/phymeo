"""Inspect retained K0 final-run bytes without importing a solver or rerunning queries."""

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts/p65-k0-final-20261109-01"
OLD = ROOT / "artifacts/p65-generic-collision-20261107-01"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_arrays(value):
    count = 0
    if isinstance(value, dict):
        if {"dtype", "shape", "bytes_hex", "values"} <= value.keys():
            raw = bytes.fromhex(value["bytes_hex"])
            itemsize = int(value["dtype"][2:])
            assert len(raw) == math.prod(value["shape"]) * itemsize
            count += 1
        for child in value.values():
            count += check_arrays(child)
    elif isinstance(value, list):
        for child in value:
            count += check_arrays(child)
    return count


def main():
    membership = read(OLD / "k0-predeclared-execution-membership-v1.json")
    identities = [
        identity
        for group in membership["groups"]
        for selection in group["selections"]
        for identity in selection["ids"]
    ]
    assert len(identities) == len(set(identities)) == 142
    report = {"schema": "k0-final-retained-audit-v1", "backends": {}, "product_members": []}
    for mode, task in (("interpret", "08-interpret"), ("cuda", "09-cuda")):
        run = EVIDENCE / task
        receipt = read(run / "final.json")
        assert receipt["raw_exit"] == 0
        assert "142 passed" in (run / "stdout.log").read_text(encoding="utf-8")
        for before, after in zip(receipt["sources"], receipt["sources_after"], strict=True):
            assert before["path"] == after["path"]
            assert before["sha256"] == after["sha256"]
            assert digest(Path(before["path"]).read_bytes()).upper() == before["sha256"]
        paths = list((run / "payloads").rglob("*.json.gz"))
        expected = {f"{identity}-{mode}.json.gz" for identity in identities}
        assert len(paths) == len(expected) and {path.name for path in paths} == expected
        statuses, touch_statuses, inventory = Counter(), Counter(), []
        arrays = attempts = 0
        for path in sorted(paths):
            payload = read(path)
            identity = payload["identity"]
            assert identity in identities and payload["mode"] == mode
            assert payload["completed"] is True and payload["failure"] is None
            assert payload["schema"] == "K0-query-evidence-v1" and "inputs" in payload
            for source in payload["source"]:
                original = bytes.fromhex(source["bytes_hex"])
                assert digest(original) == source["sha256"]
                assert original == Path(source["path"]).read_bytes()
            arrays += check_arrays(payload)
            attempts += len(payload["attempts"])
            for attempt in payload["attempts"]:
                result = attempt.get("result")
                if result is None:
                    continue
                assert {"raw", "source", "stats", "rows"} <= result.keys()
                for row in result["rows"]:
                    key = (row["status"], row["relation"], row["reason"])
                    statuses[str(key)] += 1
                    if identity.endswith("-touch"):
                        touch_statuses[str(key)] += 1
            inventory.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(path.read_bytes()),
                    "bytes": path.stat().st_size,
                }
            )
        report["backends"][mode] = {
            "payloads": len(paths),
            "attempts": attempts,
            "encoded_arrays": arrays,
            "row_statuses": dict(statuses),
            "touch_statuses": dict(touch_statuses),
            "inventory": inventory,
        }
    install = read(EVIDENCE / "wheel-install.json")
    wheel = Path(install["wheel"])
    report["wheel_sha256"] = digest(wheel.read_bytes())
    with ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("phymeo/") or member.endswith("/"):
                continue
            data = archive.read(member)
            assert data == (ROOT / "src" / member).read_bytes()
            assert data == (Path(install["site"]) / member).read_bytes()
            report["product_members"].append({"member": member, "sha256": digest(data)})
    for mode in ("interpret", "cuda"):
        consumer = read(EVIDENCE / f"wheel-{mode}.json.gz")
        assert consumer["accepted_queries"] == 4 and consumer["rejected_queries"] == 0
        assert consumer["wheel_sha256"] == report["wheel_sha256"]
        assert Path(install["site"]) in Path(consumer["phymeo_file"]).parents
    report["status"] = "passed"
    report["scope"] = "K0 default distance/witness query; optional K1 patch authoring is unaccepted"
    target = EVIDENCE / "retained-audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"PASS: 284 complete payloads; {len(report['product_members'])} source/wheel/installed members; 8 installed queries"
    )


if __name__ == "__main__":
    main()
