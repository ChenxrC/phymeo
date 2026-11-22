"""Build a fixed, relocatable example ZIP without importing an engine or SDK."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import zipfile


def _entry():
    path = Path(__file__).resolve().parents[1] / "examples/portable_panda_replay.py"
    spec = importlib.util.spec_from_file_location("portable_panda_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_bundle(rendering_root, output):
    entry = _entry()
    root, output = Path(rendering_root).resolve(), Path(output).resolve()
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError(output if output.exists() else checksum)
    payloads, records = {}, {}
    total = 0
    for relative, spec in sorted(entry.SOURCES.items()):
        data = entry.source_payload(root, spec)
        total += len(data)
        if total > entry.MAX_TOTAL_BYTES:
            raise ValueError("source bundle exceeds total byte limit")
        payloads[relative] = data
        records[relative] = {"source_relative": spec["source"],
                             "sha256": entry.digest(data), "bytes": len(data)}
    manifest = {"schema": entry.SCHEMA,
                "engine": {"distribution": "phymeo-rendering", "source": "installed-wheel-only",
                           "panda_manifest_sha256": entry.PANDA_MANIFEST_SHA256},
                "files": records,
                "original_film_commit": "ab5ebaa3e86006c851bcd557d1b85ccf6c83d9de",
                "source_clock": {"recorded_duration_s": 10, "source_exposures": 300, "playback_fps": 30},
                "text_representation": "LF; binary images, trace, manifest and URDF unchanged",
                "integrity_scope": "fixed source pins and file SHA256; not an authenticated signature"}
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data, compresslevel=6)
    data = archive.getvalue()
    sha = entry.digest(data)
    # Validation finishes before either publication. Never replace an old ZIP.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    with checksum.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{sha}  {output.name}\n")
    return {"schema": "phymeo-portable-panda-build-v1", "path": str(output),
            "sha256": sha, "bytes": len(data), "files": len(payloads),
            "uncompressed_bytes": sum(map(len, payloads.values()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendering-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_bundle(args.rendering_root, args.output), indent=2))


if __name__ == "__main__":
    main()
