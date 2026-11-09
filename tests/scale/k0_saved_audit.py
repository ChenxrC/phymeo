"""Read-only check of the K0 Newton short reference and installed-wheel calls.

Stdlib file, gzip, zip and hash inspection only. It does not import Phymeo,
Newton, or TYX, and it does not repeat a query.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT = ROOT / "artifacts" / "p65-generic-collision-20261107-01" / "k0-closeout"
OUTSIDE = ROOT.parent / "k0-closeout-outside"
WHEEL = OUTSIDE / "wheel" / "phymeo-0.1.0a0-py3-none-any.whl"
INSTALLED = OUTSIDE / "site" / "Lib" / "site-packages"
MEMBERS = (
    "phymeo/__init__.py",
    "phymeo/backends/tyx_world_batch.py",
    "phymeo/backends/_resident_convex_source.py",
    "phymeo/backends/_device_convex_query.py",
    "phymeo/backends/_tyx_convex_query_kernels.py",
    "phymeo/backends/_tyx_convex_support.py",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    errors = []
    newton_path = CLOSEOUT / "newton-short.json"
    interpret_path = CLOSEOUT / "wheel-interpret.json.gz"
    cuda_path = CLOSEOUT / "wheel-cuda.json.gz"
    newton = load_json(newton_path)
    interpret = load_json(interpret_path)
    cuda = load_json(cuda_path)
    if newton.get("schema") != "k0-newton-short-v1" or newton.get("newton_version") != "1.7.0":
        errors.append("newton record is not the pinned 1.7.0 short reference")
    devices = newton.get("devices", [])
    if [item.get("device") for item in devices] != ["cpu", "cuda"]:
        errors.append("newton record does not contain a cpu call and a cuda call")
    for item in devices:
        if item.get("error") or item.get("collide_calls") != 4 or item.get("within_1e_4_L") != 4:
            errors.append(f"newton {item.get('device')} did not call all four controls inside 1e-4 m")
    wheel_bytes = WHEEL.read_bytes()
    wheel_hash = sha256(wheel_bytes)
    for name, record in (("interpret", interpret), ("cuda", cuda)):
        if record.get("schema") != "k0-wheel-consumer-v1":
            errors.append(f"{name} wheel record has the wrong schema")
        if record.get("accepted_queries") != 4 or record.get("rejected_queries") != 0:
            errors.append(f"{name} wheel call is not 4 accepted and 0 rejected")
        if record.get("wheel_sha256") != wheel_hash:
            errors.append(f"{name} wheel record hash does not match the built wheel")
        loaded = Path(record.get("phymeo_file", ""))
        if INSTALLED not in loaded.parents or ROOT in loaded.parents:
            errors.append(f"{name} loaded Phymeo from outside the installed site")
    compared = []
    with ZipFile(WHEEL) as archive:
        for member in MEMBERS:
            packed = archive.read(member)
            disk = (ROOT / "src" / member).read_bytes()
            installed = (INSTALLED / member).read_bytes()
            same = sha256(packed) == sha256(disk) == sha256(installed)
            compared.append({"member": member, "sha256": sha256(disk), "bytes": len(disk), "same_bytes": same})
            if not same:
                errors.append(f"{member} differs between source, wheel and installed site")
    report = {
        "schema": "k0-saved-audit-v1",
        "status": "complete-saved-evidence" if not errors else "failed",
        "errors": errors,
        "newton_calls": sum(item.get("collide_calls", 0) for item in devices),
        "wheel_accepted_queries": interpret.get("accepted_queries", 0) + cuda.get("accepted_queries", 0),
        "wheel_rejected_queries": interpret.get("rejected_queries", 0) + cuda.get("rejected_queries", 0),
        "wheel_sha256": wheel_hash,
        "wheel_bytes": len(wheel_bytes),
        "install": "zip extract of the built wheel; the execution venv has no pip module",
        "members": compared,
        "omitted": "The 142 pytest gzip files from the identity queue were not retained, so this audit does not count them.",
        "inputs": [str(newton_path), str(interpret_path), str(cuda_path), str(WHEEL)],
    }
    target = CLOSEOUT / "audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(target)
    print(report["status"], "errors", len(errors), "newton_calls", report["newton_calls"], "wheel_accepted", report["wheel_accepted_queries"])
    if errors:
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
