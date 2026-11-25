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
import importlib.util
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
STUDIO_PREFIX = "phymeo_rendering/_studio_assets/"
STUDIO_ASSETS = {
    "university_workshop_1k.hdr": "a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548",
    "NOTICE.md": "5f5e06c50f199c223dd1588946f66899ec30a4d07909407d6ab0108d0e4912a0",
}
PREPARATION_SOURCES = tuple("phymeo_rendering/" + name for name in (
    "preparation.py", "_prepare_geometry.py", "_prepare_textures.py", "_prepare_cache.py"))
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
    optional = re.compile(r"([A-Za-z0-9_.-]+)\s*([^;]+?)\s*;\s*extra\s*==\s*(['\"])([^'\"]+)\3")
    allowed = {("sapien==3.0.3", "sapien"), ("numpy==1.26.4", "sapien"),
               ("pillow==12.2.0", "textures"), ("numpy==1.26.4", "ibl"),
               ("pillow==12.2.0", "prepare"), ("vtk<10,>=9.3", "prepare"),
               ("numpy<3,>=1.26", "prepare")}
    observed = set()
    for row in dependencies:
        matched = optional.fullmatch(row)
        # Packaging backends can reorder comma-separated version bounds.
        pair = None
        if matched:
            bounds = ",".join(sorted(part.strip() for part in matched[2].split(",")))
            pair = (matched[1].lower() + bounds, matched[4])
        if pair not in allowed or pair in observed:
            raise ValueError("base dependencies or undeclared optional dependency in rendering wheel")
        observed.add(pair)
    if observed != allowed or set(metadata.get_all("Provides-Extra", [])) != {extra for _, extra in allowed}:
        raise ValueError("wheel must preserve the reference, texture, IBL and preparation dependencies")

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
    for name, digest in STUDIO_ASSETS.items():
        require(STUDIO_PREFIX + name, digest)
    for name in PREPARATION_SOURCES:
        require(name)
    for asset in ('index.html', 'viewer.css', 'viewer.js'):
        relative = 'phymeo_rendering/viewer_assets/' + asset
        digest = sha256((Path(rendering_root) / 'src' / relative).read_bytes()) if rendering_root else None
        require(relative, digest)
    native_prefix = "phymeo_rendering/backends/"
    for name in ("_native/CMakeLists.txt", "_native/renderer.cpp", "_native/renderer.h", "_native/resident.inc",
                 "_shaders/native.vert", "_shaders/native.frag", "_shaders/resident.vert", "_shaders/resident.frag"):
        require(native_prefix + name)
    checked_sources = []
    omitted_source_controls = {}
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
                relative = source.relative_to(root / "assets/panda").as_posix()
                digest = sha256(source.read_bytes())
                # Exact root Git controls affect checkout behavior, not the
                # installed asset. Manifest resources were already required
                # above. All other dotfiles and nested names remain mandatory;
                # if a Git control is packaged, its bytes are checked too.
                if relative in {".gitattributes", ".gitignore"} and ASSET_PREFIX + relative not in files:
                    omitted_source_controls[relative] = digest
                else:
                    require(ASSET_PREFIX + relative, digest)
    return {
        "schema": "phymeo-rendering-wheel-audit-v1", "file": str(path), "sha256": sha256(path.read_bytes()),
        "version": metadata.get("Version"), "tag": "py3-none-any", "platform_binaries": [],
        "base_dependencies": [], "optional_dependencies": dependencies,
        "native_sources": sorted(n for n in files if n.startswith(native_prefix + "_native/") or n.startswith(native_prefix + "_shaders/")),
        "source_files_compared": checked_sources,
        "preparation_sources": list(PREPARATION_SOURCES),
        "studio_asset_sha256": dict(STUDIO_ASSETS),
        "panda": {"manifest_sha256": MANIFEST_SHA256, "files_verified": len(resources) + 1,
                  "instances": len(manifest["instances"]), "meshes": len(manifest["meshes"]),
                  "source_control_files_omitted": omitted_source_controls},
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
    features = None
    if getattr(args, "feature_smoke", False):
        spec = importlib.util.spec_from_file_location("g5_feature_smoke", Path(__file__).with_name("g5_feature_smoke.py"))
        features = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features)
    requested = [] if args.base_only else (args.run_path or ["native-raster", "sapien-raster", "sapien-pt"])
    feature_plan = (features.plan_features(requested, skip_features=getattr(args, "skip_feature", ()) or ())
                    if features else None)
    env = features.feature_environment(work, os.environ) if features else dict(os.environ)
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
        if features:
            feature_bundle = work/"feature-fixture"
            report["feature_fixture"] = features.copy_feature_bundle(rendering, feature_bundle,
                enabled=[kind for kind, row in feature_plan.items() if row["status"] == "required"])
            feature_launcher = work/"feature_launcher.py"
            shutil.copyfile(Path(__file__).with_name("g5_feature_smoke.py"), feature_launcher)
            provenance["feature_launcher.py"] = sha256(feature_launcher.read_bytes())
            provenance["feature-fixture/manifest.json"] = sha256((feature_bundle/"manifest.json").read_bytes())
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

        report["requested_paths"] = requested
        if features:
            report["feature_plan"] = feature_plan
            report["features"] = {kind: ({"status": "not_run", "reason": "prerequisite not completed"}
                if row["status"] == "required" else dict(row)) for kind, row in feature_plan.items()}
            if not args.wheelhouse:
                raise ValueError("feature acceptance requires an explicit offline --wheelhouse")
            wheelhouse = Path(args.wheelhouse).resolve()
            report["offline_wheelhouse"] = {"directory": str(wheelhouse), "wheels": {
                p.name: sha256(p.read_bytes()) for p in sorted(wheelhouse.glob("*.whl"))}}
            run("install-feature-extras", [python, "-I", "-m", "pip", "install", "--no-index",
                "--only-binary=:all:", "--find-links", wheelhouse, str(pinned_wheel)+"[ibl,textures]"])

            def run_feature(kind, library=None):
                if feature_plan[kind]["status"] != "required":
                    return
                target = work/"captures"/"features"/kind
                try:
                    command = features.feature_command(
                        python, feature_launcher, kind, bundle=feature_bundle, output=target,
                        source_root=source, library=library, device_index=args.native_device_index,
                        sapien_device=args.sapien_device)
                    run(kind+"-installed", command)
                    report["features"][kind] = json.loads(target.with_name(target.name+".installed.json").read_text())
                except Exception as error:
                    report["features"][kind] = {"status": "failed", "error": str(error)}
                _save(report_path, report)

            run_feature("host")
            if report["features"]["host"]["status"] != "passed":
                raise RuntimeError("source-external feature HOST preparation failed; no device feature was run")
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
                if features:
                    run_feature("uv-native", libraries[0])
                    run_feature("ibl-native", libraries[0])
                    run_feature("roughness-native", libraries[0])
                    run_feature("ray-query-native", libraries[0])
                    run_feature("neutral-native", libraries[0])
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
                reference_requirements = [str(pinned_wheel)+"[sapien]"] if features else ["sapien==3.0.3", "numpy==1.26.4"]
                run("install-reference", [python, "-I", "-m", "pip", "install", "--no-index", "--only-binary=:all:",
                                           "--find-links", wheelhouse, *reference_requirements])
                for name in reference:
                    try:
                        run(name + "-f0", [*base, "--mode", name, "--sapien-device", args.sapien_device])
                        report["paths"][name] = json.loads((work / "captures" / (name + ".json")).read_text())
                        if features:
                            run_feature("uv-"+name)
                            run_feature("roughness-"+name)
                            if name == "sapien-pt":
                                run_feature("ibl-sapien")
                    except Exception as exc:
                        report["paths"][name] = {"status": "failed", "error": str(exc)}
                if features:
                    run_feature("neutral-sapien")
            except Exception as exc:
                for name in reference:
                    report["paths"][name] = {"status": "failed", "error": str(exc)}
        run("installed-freeze", [python, "-I", "-m", "pip", "freeze", "--all"])
        checked = list(report["paths"].values()) + list(report.get("features", {}).values())
        report["status"] = ("failed" if any(row["status"] not in ("passed", "not_verified") for row in checked)
                            else "partial" if any(row["status"] == "not_verified" for row in checked) else "passed")
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
    parser.add_argument("--feature-smoke", action="store_true", help="run copied UV/HDRI/roughness/Neutral/hardware-shadow protocols in this same installed environment")
    parser.add_argument("--skip-feature", action="append", default=[], metavar="KIND:REASON",
                        help="explicitly leave a feature unverified; requires --feature-smoke; overall result is partial")
    parser.add_argument("--run-path", action="append", choices=("native-raster", "sapien-raster", "sapien-pt"))
    parser.add_argument("--native-device-index", type=int, default=0)
    parser.add_argument("--sapien-device", default="cuda:0")
    parser.add_argument("--validation-layer-path")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cmake-arg", action="append", default=[], help="repeat as --cmake-arg=-DNAME=value")
    args = parser.parse_args()
    if args.base_only and (args.run_path or args.feature_smoke):
        parser.error("--base-only cannot request device paths or feature smoke")
    if args.skip_feature and not args.feature_smoke:
        parser.error("--skip-feature requires --feature-smoke")
    report = reproduce(args)
    print(json.dumps({"status": report["status"], "report": str(Path(args.work_dir).resolve() / "report.json")}))
    return 0 if report["status"] in ("passed", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
