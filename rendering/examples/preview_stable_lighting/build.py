"""Build the stable lighting shaders against the sealed surface-quality transport."""
import argparse
from pathlib import Path
import shutil
import subprocess

from preview_lowres_filter._helpers import digest, markers, read, save, verify
from preview_surface_quality.build import binary_hashes, check_package as check_surface

from .shaders import configure_shaders


DEFAULT_PACKAGE = Path(__file__).resolve().parents[3] / "artifacts/rendering/panda-stable-surface/default"


def check_package(directory):
    root, manifest = verify(directory)
    record = read(root / "stable-seal.json")
    if (record["manifest_sha256"] != digest(root / "source-manifest.json")
            or record["shaders"] != binary_hashes(root)):
        raise ValueError("stable lighting shader package differs from its seal")
    parent, _ = check_surface(manifest["parent"])
    if digest(parent / "surface-seal.json") != manifest["parent_seal_sha256"]:
        raise ValueError("surface transport changed since the stable lighting build")
    return root, manifest


def build(parent, output, radius):
    parent, base = check_surface(parent)
    quality = base["contract"]["surface_quality"]
    if "contours" not in quality or quality["panels"]["primary_samples_per_panel"] != 16:
        raise ValueError("stable motion preview requires the 16-point soft-shadow and four-sample contour package")
    output = Path(output).resolve()
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("a separate package directory is required")
    output.mkdir(parents=True, exist_ok=False)
    for name in base["generated_sha256"]:
        if name.startswith("source/"):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent / name, target)
    policy = configure_shaders(output / "source/_shaders", radius)
    generated = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted((output / "source").rglob("*")) if p.is_file()}
    manifest = dict(parent=str(parent), parent_seal_sha256=digest(parent / "surface-seal.json"),
                    stable_lighting=policy, generated_sha256=generated)
    save(output / "source-manifest.json", manifest)

    # Use the same compiler, target and macro matrix as the parent. Only shader
    # source changed. The verified parent still owns the DLL and Python runtime.
    cache = {}
    for line in (parent / "build/CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            declaration, value = line.split("=", 1)
            cache[declaration.split(":", 1)[0]] = value
    tool = cache["CMAKE_COMMAND"]
    cmd = [tool, "-S", str(output / "source/_native"), "-B", str(output / "build"), "-G", cache["CMAKE_GENERATOR"]]
    if cache.get("CMAKE_GENERATOR_PLATFORM"):
        cmd.extend(["-A", cache["CMAKE_GENERATOR_PLATFORM"]])
    cmd += [f"-D{key}={cache[key]}" for key in ("VULKAN_INCLUDE_DIR", "VULKAN_LIBRARY", "GLSLANG_VALIDATOR")]
    subprocess.run(cmd, check=True)
    subprocess.run([tool, "--build", str(output / "build"), "--config", "Release",
                    "--target", "native_shaders", "--parallel", "1"], check=True)
    before, after = binary_hashes(parent), binary_hashes(output)
    mains = {n for n in before if n.endswith("_indirect_progressive.frag.spv")}
    guides = {"resident_rayquery.frag.spv", "resident_rayquery_normal.frag.spv"}
    changed = mains | guides | {"resident_spatial.comp.spv"}
    if (len(mains) != 4 or before.keys() != after.keys()
            or {n for n in before if before[n] != after[n]} != changed):
        raise ValueError("unexpected stable lighting shader variants")
    for name in changed:
        info = markers((output / "build" / name).read_bytes())
        if info.get("108") != dict(name="PR_PREVIEW_STABLE_LIGHTING_ABI", value=[1]):
            raise ValueError(f"missing stable lighting marker in {name}")
        if name in mains and info.get("109") != dict(name="PR_PREVIEW_AO_RADIUS_UM", value=[policy["ao_radius_um"]]):
            raise ValueError(f"incorrect compiled occlusion radius in {name}")
    save(output / "stable-seal.json", dict(manifest_sha256=digest(output / "source-manifest.json"), shaders=after))
    check_package(output)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    cli.add_argument("--ao-radius-m", type=float, default=.025,
                     help="World-space occlusion reach in metres; zero builds the unoccluded IBL control.")
    args = cli.parse_args()
    build(args.parent, args.output, args.ao_radius_m)


if __name__ == "__main__":
    main()
