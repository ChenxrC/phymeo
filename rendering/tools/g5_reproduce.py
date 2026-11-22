"""Offline G5 wheel inspection and source-external reproduction (stdlib coordinator).

No downloads or SDK discovery fallback. Device paths are opt-in through main().
The installed probe is copied out before execution and invoked with Python -I.
"""
from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

MANIFEST_SHA256 = "d6c5f952ec5e293403d37962a5a462a70fb1359307479c87e744dfb80b4846e4"
ASSET_PREFIX = "phymeo_rendering/_panda_assets/"
SOURCE_EXTENSIONS = {".cmake", ".cpp", ".h", ".hpp", ".inc", ".vert", ".frag", ".glsl", ".comp", ".geom", ".tesc", ".tese", ".rgen", ".rchit", ".rmiss", ".rahit", ".rint", ".rcall"}
BINARY_EXTENSIONS = {".dll", ".pyd", ".so", ".dylib", ".spv", ".exe", ".lib", ".a", ".o", ".obj"}
F0_FILES = ("integration/__init__.py", "integration/_g1_table.py", "integration/_r0_table.py",
            "fixtures/g1/table.gltf", "fixtures/g1/block.gltf")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"wheel/resource path is not contained: {name!r}")
    return name


def inspect_wheel(path, rendering_root=None):
    """Verify RECORD, optional dependencies, pinned assets and complete source bytes.

    rendering_root is the rendering/ directory of the candidate checkout. When
    supplied, every Python/native/shader source and all Panda files must match.
    This is package verification, not a claim that the source builds or renders.
    """
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel member")
        for info in archive.infolist():
            _safe_name(info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("wheel contains a symbolic link")
        files = {name: archive.read(name) for name in names if not name.endswith("/")}
    binaries = [name for name in files if PurePosixPath(name).suffix.lower() in BINARY_EXTENSIONS]
    if binaries:
        raise ValueError(f"platform/compiled binary in source-only wheel: {binaries}")
    records = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("wheel must have one RECORD")
    record = records[0]
    seen = set()
    for row in csv.reader(io.StringIO(files[record].decode("utf-8"))):
        if len(row) != 3 or row[0] in seen or row[0] not in files:
            raise ValueError("RECORD contains missing, duplicate or malformed rows")
        name, digest, size = row
        seen.add(name)
        if name == record:
            if digest or size:
                raise ValueError("RECORD self entry must have empty digest and size")
            continue
        actual = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b"=").decode()
        if digest != actual or size != str(len(files[name])):
            raise ValueError(f"RECORD digest/size mismatch: {name}")
    if seen != set(files):
        raise ValueError("RECORD does not cover all wheel files")
    dist = record.rsplit("/", 1)[0]
    headers = BytesParser().parsebytes(files[dist + "/WHEEL"])
    if headers.get("Root-Is-Purelib") != "true" or headers.get_all("Tag") != ["py3-none-any"]:
        raise ValueError("source-only wheel requires purelib py3-none-any metadata")
    metadata = BytesParser().parsebytes(files[dist + "/METADATA"])
    if metadata.get("Name") != "phymeo-rendering":
        raise ValueError("wrong wheel distribution")
    dependencies = metadata.get_all("Requires-Dist", [])
    optional = re.compile(r"(?:sapien==3\.0\.3|numpy==1\.26\.4)\s*;\s*extra\s*==\s*(['\"])sapien\1")
    if any(optional.fullmatch(row) is None for row in dependencies):
        raise ValueError("base dependencies or undeclared optional dependency in rendering wheel")
    if len(dependencies) != 2 or set(metadata.get_all("Provides-Extra", [])) != {"sapien"}:
        raise ValueError("wheel must preserve the two pinned optional reference dependencies")

    def require(name, digest=None):
        _safe_name(name)
        if name not in files:
            raise ValueError(f"missing wheel resource: {name}")
        if digest is not None and sha256(files[name]) != digest:
            raise ValueError(f"SHA256 digest mismatch: {name}")
        return files[name]

    manifest = json.loads(require(ASSET_PREFIX + "manifest.json", MANIFEST_SHA256))
    resources = [(r["file"], r["sha256"]) for r in manifest["licenses"] + manifest["visual_placement_sources"] + manifest["meshes"]]
    resources += [(r["archive_file"], r["archive_sha256"]) for r in manifest["sources"]]
    for name, digest in resources:
        require(ASSET_PREFIX + _safe_name(name), digest)
    native_prefix = "phymeo_rendering/backends/"
    for name in ("_native/CMakeLists.txt", "_native/renderer.cpp", "_native/renderer.h", "_native/resident.inc",
                 "_shaders/native.vert", "_shaders/native.frag", "_shaders/resident.vert", "_shaders/resident.frag"):
        require(native_prefix + name)
    checked_sources = []
    if rendering_root is not None:
        root = Path(rendering_root).resolve()
        package = root / "src/phymeo_rendering"
        for source in sorted(package.rglob("*")):
            if source.is_file() and (source.suffix == ".py" or source.suffix in SOURCE_EXTENSIONS or source.name == "CMakeLists.txt"):
                name = source.relative_to(root / "src").as_posix()
                require(name, sha256(source.read_bytes()))
                checked_sources.append(name)
        for source in sorted((root / "assets/panda").rglob("*")):
            if source.is_file():
                require(ASSET_PREFIX + source.relative_to(root / "assets/panda").as_posix(), sha256(source.read_bytes()))
    return {
        "schema": "phymeo-rendering-wheel-audit-v1", "file": str(path), "sha256": sha256(path.read_bytes()),
        "version": metadata.get("Version"), "tag": "py3-none-any", "platform_binaries": [],
        "base_dependencies": [], "optional_dependencies": dependencies,
        "native_sources": sorted(n for n in files if n.startswith(native_prefix + "_native/") or n.startswith(native_prefix + "_shaders/")),
        "source_files_compared": checked_sources,
        "panda": {"manifest_sha256": MANIFEST_SHA256, "files_verified": len(resources) + 1,
                  "instances": len(manifest["instances"]), "meshes": len(manifest["meshes"])},
    }


def validate_external_workdir(work_dir, source_root):
    work, source = Path(work_dir).resolve(), Path(source_root).resolve()
    if work.is_relative_to(source) or source.is_relative_to(work):
        raise ValueError("G5 work directory must be outside the source checkout and its ancestors")
    return work


def _save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reproduce(args):
    source = Path(args.source_root).resolve()
    rendering = source / "rendering"
    work = validate_external_workdir(args.work_dir, source)
    audit = inspect_wheel(args.wheel, rendering)
    # Never reuse a previous installation or silently merge its evidence.
    work.mkdir(parents=True, exist_ok=False)
    report = {"schema": "phymeo-rendering-g5-reproduction-v1", "status": "running", "wheel": audit,
              "source_root": str(source), "work_dir": str(work), "commands": [], "paths": {},
              "scope": "offline installed-wheel F0; no physics, native PT or unseen device claims"}
    report_path = work / "report.json"
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.validation_layer_path:
        env["VK_LAYER_PATH"] = str(Path(args.validation_layer_path).resolve())
    sequence = 0

    def run(label, command):
        nonlocal sequence
        sequence += 1
        log = work / f"{sequence:02d}-{label}.log"
        item = {"label": label, "argv": [str(x) for x in command], "cwd": str(work), "log": log.name}
        report["commands"].append(item)
        _save(report_path, report)
        with log.open("wb") as stream:
            try:
                completed = subprocess.run(item["argv"], cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
                item["returncode"] = completed.returncode
            except OSError as exc:
                item["error"] = str(exc)
                _save(report_path, report)
                raise
        _save(report_path, report)
        if completed.returncode:
            raise RuntimeError(f"{label} failed ({completed.returncode}); see {log}")

    try:
        bundle = work / "fixture"
        bundle.mkdir()
        provenance = {}
        for relative in F0_FILES:
            original = rendering / "tests" / relative
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            provenance[relative] = sha256(target.read_bytes())
        probe = work / "installed_probe.py"
        shutil.copyfile(Path(__file__).with_name("_g5_installed.py"), probe)
        provenance[probe.name] = sha256(probe.read_bytes())
        shutil.copyfile(Path(__file__), work / "coordinator.py")
        provenance["coordinator.py"] = sha256((work / "coordinator.py").read_bytes())
        report["copied_fixture_sha256"] = provenance
        pinned_wheel = work / Path(args.wheel).name
        shutil.copyfile(args.wheel, pinned_wheel)
        if sha256(pinned_wheel.read_bytes()) != audit["sha256"]:
            raise ValueError("wheel changed during copy")
        run("create-venv", [sys.executable, "-I", "-m", "venv", str(work / "env")])
        python = work / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run("install-base", [python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", pinned_wheel])
        base = [python, "-I", probe, "--bundle", bundle, "--output", work / "captures", "--forbid-source", source]
        run("base-probe", [*base, "--mode", "base"])
        report["base"] = json.loads((work / "captures/base.json").read_text())
        package = Path(report["base"]["package_directory"])
        requested = [] if args.base_only else (args.run_path or ["native-raster", "sapien-raster", "sapien-pt"])
        report["requested_paths"] = requested
        if "native-raster" in requested:
            try:
                build = work / "native-build"
                run("native-configure", [args.cmake, "-S", package / "backends/_native", "-B", build, *args.cmake_arg])
                run("native-build", [args.cmake, "--build", build, "--config", "Release"])
                filenames = ("phymeo_native.dll", "libphymeo_native.so", "libphymeo_native.dylib")
                libraries = [build / name for name in filenames if (build / name).is_file()]
                if len(libraries) != 1:
                    raise ValueError("installed-source build must produce exactly one native library")
                run("native-f0", [*base, "--mode", "native-raster", "--library", libraries[0],
                                  "--shader-directory", build, "--native-device-index", str(args.native_device_index)])
                report["paths"]["native-raster"] = json.loads((work / "captures/native-raster.json").read_text())
                report["native_build_artifacts"] = {p.name: sha256(p.read_bytes()) for p in sorted(build.iterdir())
                                                     if p.is_file() and (p.suffix in {".dll", ".so", ".dylib", ".spv"} or p.name == "CMakeCache.txt")}
            except Exception as exc:
                report["paths"]["native-raster"] = {"status": "failed", "error": str(exc)}
        reference = [name for name in requested if name.startswith("sapien-")]
        if reference:
            try:
                if not args.wheelhouse:
                    raise ValueError("reference reproduction requires an explicit offline --wheelhouse")
                wheelhouse = Path(args.wheelhouse).resolve()
                report["offline_wheelhouse"] = {"directory": str(wheelhouse), "wheels": {
                    p.name: sha256(p.read_bytes()) for p in sorted(wheelhouse.glob("*.whl"))}}
                run("install-reference", [python, "-I", "-m", "pip", "install", "--no-index", "--only-binary=:all:",
                                           "--find-links", wheelhouse, "sapien==3.0.3", "numpy==1.26.4"])
                for name in reference:
                    try:
                        run(name + "-f0", [*base, "--mode", name, "--sapien-device", args.sapien_device])
                        report["paths"][name] = json.loads((work / "captures" / (name + ".json")).read_text())
                    except Exception as exc:
                        report["paths"][name] = {"status": "failed", "error": str(exc)}
            except Exception as exc:
                for name in reference:
                    report["paths"][name] = {"status": "failed", "error": str(exc)}
        run("installed-freeze", [python, "-I", "-m", "pip", "freeze", "--all"])
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["paths"].values()) else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _save(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--source-root", required=True, help="candidate checkout, used only to bind/copy sources and fixtures")
    parser.add_argument("--work-dir", required=True, help="NEW directory outside every candidate checkout")
    parser.add_argument("--wheelhouse", help="local wheels including all SAPIEN transitives; no index fallback")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--run-path", action="append", choices=("native-raster", "sapien-raster", "sapien-pt"))
    parser.add_argument("--native-device-index", type=int, default=0)
    parser.add_argument("--sapien-device", default="cuda:0")
    parser.add_argument("--validation-layer-path")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cmake-arg", action="append", default=[], help="repeat as --cmake-arg=-DNAME=value")
    args = parser.parse_args()
    if args.base_only and args.run_path:
        parser.error("--base-only and --run-path are mutually exclusive")
    report = reproduce(args)
    print(json.dumps({"status": report["status"], "report": str(Path(args.work_dir).resolve() / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
