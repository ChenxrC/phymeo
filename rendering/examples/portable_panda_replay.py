"""Offline, movable Panda replay entry; the installed wheel supplies the engine.

Only the asset-directory argument of the unchanged film helper is relocated.
All actual camera generation and capture remain in panda_realtime_film.capture.
SHA256 is an integrity check, not an authenticity signature.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

SCHEMA = "phymeo-portable-panda-example-v1"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
PANDA_MANIFEST_SHA256 = "d6c5f952ec5e293403d37962a5a462a70fb1359307479c87e744dfb80b4846e4"

# Keys are ZIP destinations; source paths are relative to rendering/. Fixed
# hashes bind the existing evidence and helpers, independently of manifest.json.
SOURCES = {
    "replay.py": {"source": "examples/portable_panda_replay.py", "sha256": None, "text": True},
    "README.md": {"source": "examples/portable_panda/README.md", "sha256": None, "text": True},
    "licenses/CC0-1.0.txt": {"source": "examples/portable_panda/CC0-1.0.txt", "sha256": "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499", "text": True},
    "examples/panda_realtime_film.py": {"source": "examples/panda_realtime_film.py", "sha256": "78b0ff2cdeab5c24e3715ff7ed880cb7bcb720d8376a939c2d399c3569b3e61a", "text": True},
    "examples/_panda_scene.py": {"source": "examples/_panda_scene.py", "sha256": "e781b754a72a47fd8cc1f3efeef7d03fd7d30e4634236d34a752c051a94196a6", "text": True},
    "examples/_panda_textures.py": {"source": "examples/_panda_textures.py", "sha256": "95539eadbeb10b6c9b921d117113eb5c3cb8a28e7343b94977c352ee2f783e5f", "text": True},
    "examples/panda_movies.py": {"source": "examples/panda_movies.py", "sha256": "4d297bc4b6dbfc41bb27c7651f6b2de7929a0b62aa1e97211c050b96bf2107aa", "text": True},
    "examples/_panda_motion.py": {"source": "examples/_panda_motion.py", "sha256": "f430b5e5d35ffe4f82c4d1df9e129afe489031602b66190444abc7fed456da67", "text": True},
    "tests/fixtures/panda_replay/panda.urdf": {"source": "tests/fixtures/panda_replay/panda.urdf", "sha256": "2b5bd00abb4ba68e91d532d2d778da64d1c942a9b22aa7acd58dd12a713971fd", "text": False},
    "tests/fixtures/panda_replay/LICENSE": {"source": "tests/fixtures/panda_replay/LICENSE", "sha256": "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9", "text": True},
    "tests/fixtures/panda_replay/NOTICE": {"source": "tests/fixtures/panda_replay/NOTICE", "sha256": "d8ded0c268405e7847c281faa15042d381fbf7d308ec91d9941fbd943a6e1053", "text": True},
    "tests/fixtures/panda_replay/SOURCES.md": {"source": "tests/fixtures/panda_replay/SOURCES.md", "sha256": "f456f3f9addeb33ca33cf0bc11acd5c1a9506a89a640cc24bbee8c77e874d394", "text": True},
    "tests/fixtures/panda_film/trace.json": {"source": "tests/fixtures/panda_film/trace.json", "sha256": "58073ab38acab8f5057061fa487117aa0284f81b097a4f5bb0473f890515f118", "text": False},
    "tests/fixtures/panda_film/manifest.json": {"source": "tests/fixtures/panda_film/manifest.json", "sha256": "2c98f9020f6becf9942e1a2df8dc8eb40ceb51d71d3916c8665ef117b8dfb10e", "text": False},
    "tests/fixtures/panda_film/recording.json": {"source": "tests/fixtures/panda_film/recording.json", "sha256": "33cec03417a0f6d3b622cb9b01c76a16df3257de6bc3c07d9820e18d91827cfe", "text": False},
    "tests/fixtures/panda_film/SOURCES.md": {"source": "tests/fixtures/panda_film/SOURCES.md", "sha256": "dddba29bdd3f95b5c27d130d1606d9a2e29d7a9b87f1f9eb1cf3bc494759f25d", "text": True},
    "examples/materials/wood_table_001_diff_1k.jpg": {"source": "examples/materials/wood_table_001_diff_1k.jpg", "sha256": "460dd08d240f4a1f02982415048c3c5c200f385db212da18dc7e2df68bf4d0be", "text": False},
    "examples/materials/wood_table_001_rough_1k.jpg": {"source": "examples/materials/wood_table_001_rough_1k.jpg", "sha256": "a2ca9ae89dfe7b8e2e5e92f0aa43467845417de8a50205b5852ef0d57c7624f9", "text": False},
    "examples/materials/university_workshop_1k.hdr": {"source": "examples/materials/university_workshop_1k.hdr", "sha256": "a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548", "text": False},
    "examples/materials/NOTICE.md": {"source": "examples/materials/NOTICE.md", "sha256": "5f5e06c50f199c223dd1588946f66899ec30a4d07909407d6ab0108d0e4912a0", "text": True},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("expected a contained relative file path")
    rel = Path(relative)
    if rel.is_absolute() or any(part in (".", "..") for part in rel.parts):
        raise ValueError("relative path escapes the bundle")
    path = root / rel
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise ValueError("symlinked bundle resources are not supported")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("resource escapes the bundle")
    return path


def read_bounded(path, limit=MAX_FILE_BYTES):
    if not path.is_file():
        raise ValueError(f"missing regular resource: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"resource exceeds byte limit: {path.name}")
    return data


def source_payload(root, spec):
    data = read_bounded(contained(root, spec["source"]))
    # Git's declared LF text representation, never image/trace/URDF conversion.
    if spec["text"]:
        data = data.replace(b"\r\n", b"\n")
    if spec["sha256"] is not None and digest(data) != spec["sha256"]:
        raise ValueError(f"pinned source SHA256 mismatch: {spec['source']}")
    return data


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads(read_bounded(root / "manifest.json", 128 * 1024))
    if manifest.get("schema") != SCHEMA or set(manifest.get("files", {})) != set(SOURCES):
        raise ValueError("bundle schema or fixed file set differs")
    expected_engine = {"distribution": "phymeo-rendering", "source": "installed-wheel-only",
                       "panda_manifest_sha256": PANDA_MANIFEST_SHA256}
    if manifest.get("engine") != expected_engine:
        raise ValueError("bundle engine provenance differs")
    actual = set()
    # Stop enumeration at the fixed bound, including directories, before reading.
    for index, path in enumerate(root.rglob("*")):
        if index > 128 or path.is_symlink():
            raise ValueError("extra or symlinked bundle resources")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(SOURCES) | {"manifest.json"}:
        raise ValueError("missing or extra bundle files")
    total = 0
    for relative, spec in SOURCES.items():
        record = manifest["files"][relative]
        if not isinstance(record, dict) or set(record) != {"source_relative", "sha256", "bytes"}:
            raise ValueError("invalid file record")
        if record["source_relative"] != spec["source"]:
            raise ValueError("source identity differs")
        data = read_bounded(contained(root, relative))
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("bundle exceeds total byte limit")
        actual_digest = digest(data)
        if type(record["bytes"]) is not int or record["bytes"] != len(data) or record["sha256"] != actual_digest:
            raise ValueError(f"bundle SHA256/length mismatch: {relative}")
        if spec["sha256"] is not None and actual_digest != spec["sha256"]:
            raise ValueError(f"pinned bundle SHA256 mismatch: {relative}")
    return manifest


def relocate_panda(film):
    """Keep original scene construction, choosing the installed asset default."""
    original = film.build_scene
    expected = (film.ROOT / "assets/panda").resolve()

    def installed_assets(prepared, *args, **kwargs):
        if Path(prepared).resolve() != expected:
            raise ValueError("unexpected Panda asset relocation source")
        return original(None, *args, **kwargs)

    film.build_scene = installed_assets


def installed_engine():
    """Reject source/editable fallbacks; public loading verifies every robot part."""
    import phymeo_rendering
    from phymeo_rendering.panda_visuals import panda_visual_assets

    package = Path(phymeo_rendering.__file__).resolve().parent
    distribution = importlib.metadata.distribution("phymeo-rendering")
    installed_init = Path(distribution.locate_file("phymeo_rendering/__init__.py")).resolve()
    if installed_init != package / "__init__.py":
        raise ValueError("the imported engine does not match the installed distribution")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise ValueError("use the released wheel, not an editable source installation")
    files = {str(path).replace("\\", "/") for path in distribution.files or ()}
    if "phymeo_rendering/_panda_assets/manifest.json" not in files:
        raise ValueError("installed wheel does not include the declared Panda assets")
    asset_dir = package / "_panda_assets"
    manifest = read_bounded(asset_dir / "manifest.json", 512 * 1024)
    if digest(manifest) != PANDA_MANIFEST_SHA256:
        raise ValueError("installed Panda manifest SHA256 differs")
    meshes, materials, instances = panda_visual_assets()
    record = distribution.read_text("RECORD")
    return {"distribution": "phymeo-rendering", "version": distribution.version,
            "module_path": str(package), "init_sha256": digest(read_bounded(installed_init)),
            "installed_record_sha256": digest(record.encode("utf-8")) if record else None,
            "panda_manifest_sha256": digest(manifest),
            "meshes": len(meshes), "materials": len(materials), "instances": len(instances),
            "asset_directory": str(asset_dir),
            "verification_scope": "public Panda loader verifies all robot resources; RECORD digest identifies installed metadata, not a wheel signature"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "studio"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", choices=("540p", "720p", "1080p"), default="720p")
    parser.add_argument("--camera", choices=("fixed", "wide", "detail", "sequence"), default="sequence")
    parser.add_argument("--tiles", type=int, choices=(1, 2, 4))
    parser.add_argument("--shadow-mode", choices=("software", "ray_query"), default="software")
    parser.add_argument("--area-light-grid", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--cube-size", type=int, choices=(16, 32, 64, 128, 256), default=128)
    parser.add_argument("--validation", action="store_true")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--frame", type=int, choices=range(300), metavar="0..299")
    select.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "native" and args.library is None:
        parser.error("native mode requires --library with its matching SPIR-V directory")
    if not 1 <= args.samples <= 4096:
        parser.error("--samples must be in 1..4096")
    if args.mode != "native" and (args.shadow_mode != "software" or args.area_light_grid != 2):
        parser.error("native shadow/grid options cannot select reference behavior")
    if args.mode == "native" and args.tiles not in (None, 1):
        parser.error("tiling is reserved for the offline reference entry")
    return args


def film_arguments(args, root):
    return argparse.Namespace(
        mode=args.mode, output=args.output.resolve(),
        excerpt=Path(root) / "tests/fixtures/panda_film/trace.json",
        library=args.library.resolve() if args.library else None,
        device_index=args.device_index, device=args.device, validation=args.validation,
        shadow_mode=args.shadow_mode, area_light_grid=args.area_light_grid,
        samples=args.samples, reference_seed=None, surface_roughness=True,
        display_transform="pbr-neutral", native_studio=False, workshop=False,
        photographic=True, camera=args.camera, resolution=args.resolution,
        tiles=args.tiles if args.tiles is not None else (1 if args.mode == "native" else 2),
        daylight=True, light_strength=.5, light_yaw=25., cube_size=args.cube_size, stride=1,
        frame=None if args.all else (60 if args.frame is None else args.frame))


def main(argv=None):
    sys.dont_write_bytecode = True
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    manifest = verify_bundle(root)
    output = args.output.resolve()
    if output.is_relative_to(root) or output == root:
        raise ValueError("choose a new output outside the immutable example bundle")
    if output.exists():
        raise FileExistsError(output)
    engine = installed_engine()
    sys.path.insert(0, str(root / "examples"))
    import panda_realtime_film as film

    if Path(film.__file__).resolve() != root / "examples/panda_realtime_film.py":
        raise ValueError("film helper import did not bind the verified bundle")
    relocate_panda(film)
    receipt = {"schema": "phymeo-portable-panda-capture-v1", "status": "failed",
               "bundle_manifest_sha256": digest(read_bounded(root / "manifest.json")),
               "files": manifest["files"], "installed_engine": engine,
               "relocation": "film.build_scene(asset_dir=None); original camera and capture functions",
               "physics_steps_requested": 0, "mode": args.mode,
               "requested_source_frame": None if args.all else (60 if args.frame is None else args.frame)}
    try:
        film.capture(film_arguments(args, root))
        if any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules):
            raise RuntimeError("a physics package was unexpectedly imported")
        receipt["capture_sha256"] = digest(read_bounded(output / "capture.json"))
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if output.is_dir():
            with (output / "portable-source.json").open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    main()
