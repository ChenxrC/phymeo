"""Build inputs and binary identity for primary panel sampling and contour AA."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from preview_lowres_filter._helpers import digest, markers, read, save, verify
from preview_lowres_filter.runtime import check_seal as check_parent


def prepare(parent, output, samples, contours):
    parent, base = check_parent(parent)
    output = Path(output).resolve()
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("a separate output directory is required")
    output.mkdir(parents=True, exist_ok=False)
    for name in base["generated_sha256"]:
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, dest)
    from preview_panel_shadow.build_source import apply as primary_panels
    contract = {"panels": primary_panels(output / "source", samples=samples)}
    if contours:
        from preview_contour_aa.build_source import apply as contour_resolve
        contract["contours"] = contour_resolve(output / "source")
    manifest = deepcopy(base)
    manifest["contract"]["surface_quality"] = contract
    manifest["surface_parent"] = dict(path=str(parent), seal_sha256=digest(parent / "lowres-seal.json"))
    manifest["generated_sha256"] = {
        p.relative_to(output).as_posix(): digest(p)
        for name in ("source", "runtime") for p in sorted((output / name).rglob("*")) if p.is_file()
    }
    save(output / "source-manifest.json", manifest)


def binary_hashes(root):
    return {p.name: digest(p) for p in sorted((root / "build").glob("*.spv"))}


def seal(output):
    root, manifest = verify(output)
    parent, _ = check_parent(manifest["surface_parent"]["path"])
    if digest(parent / "lowres-seal.json") != manifest["surface_parent"]["seal_sha256"]:
        raise ValueError("parent seal changed")
    policy = manifest["contract"]["surface_quality"]
    before, after = binary_hashes(parent), binary_hashes(root)
    panels = {name for name in before if name.endswith("_indirect_progressive.frag.spv")}
    expected = panels | ({"preview_history.comp.spv"} if "contours" in policy else set())
    if len(panels) != 4 or before.keys() != after.keys() or {n for n in before if before[n] != after[n]} != expected:
        raise ValueError("unexpected set of changed GPU shaders")
    for name in expected:
        old = markers((parent / "build" / name).read_bytes())
        new = markers((root / "build" / name).read_bytes())
        key, label, value = (("107", "PR_PRIMARY_PANEL_SEQUENCE_SAMPLES", policy["panels"]["primary_samples_per_panel"])
                             if name in panels else ("104", "PR_PREVIEW_CONTOUR_AA_ABI", 1))
        if any(new.get(k) != v for k, v in old.items()) or new.get(key) != dict(name=label, value=[value]):
            raise ValueError("feature marker or parent shader contract changed: " + name)
    save(root / "surface-seal.json", dict(manifest_sha256=digest(root / "source-manifest.json"),
         dll_sha256=digest(root / "build/phymeo_native.dll"), shaders=after))


def check_package(directory):
    root, manifest = verify(directory)
    record = read(root / "surface-seal.json")
    if (record["manifest_sha256"] != digest(root / "source-manifest.json")
            or record["dll_sha256"] != digest(root / "build/phymeo_native.dll")
            or record["shaders"] != binary_hashes(root)):
        raise ValueError("compiled surface-quality package differs from its seal")
    return root, manifest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    new = sub.add_parser("prepare")
    new.add_argument("--parent", type=Path, required=True)
    new.add_argument("--output", type=Path, required=True)
    new.add_argument("--samples", type=int, choices=(8, 16), default=16)
    new.add_argument("--contours", action="store_true")
    sub.add_parser("seal").add_argument("--package", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.parent, args.output, args.samples, args.contours)
    else:
        seal(args.package)


if __name__ == "__main__":
    main()
