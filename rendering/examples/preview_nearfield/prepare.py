"""Build the measured 720p local-diffuse preset from this checkout (Windows x64)."""
import argparse
from hashlib import sha256
from pathlib import Path
import os
import shutil
import subprocess
import sys


EXAMPLES = Path(__file__).resolve().parents[1]


def prepare(args):
    # Explicit tool paths also support an unpacked SDK without a global install.
    compiler = args.compiler
    if compiler is None:
        sdk = Path(os.environ.get("VULKAN_SDK", ".")) / "Bin"
        compiler = next((sdk / name for name in ("glslangValidator.exe", "glslang.exe")
                         if (sdk / name).is_file()), None)
        if compiler is None:
            compiler = shutil.which("glslangValidator") or shutil.which("glslang")
    if not compiler:
        raise ValueError("set VULKAN_SDK or pass --compiler")
    compiler = Path(compiler).resolve(strict=True)
    cmake = shutil.which("cmake")
    if not cmake:
        raise ValueError("CMake is required")
    flags = [f"-DGLSLANG_VALIDATOR={compiler}"]
    if args.vulkan_include:
        flags.append(f"-DVULKAN_INCLUDE_DIR={args.vulkan_include.resolve(strict=True)}")
    if args.vulkan_library:
        flags.append(f"-DVULKAN_LIBRARY={args.vulkan_library.resolve(strict=True)}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sparse, opaque, compact, local = (output / n for n in ("sparse", "opaque", "compact", "local"))

    def run(*command):
        subprocess.run([str(x) for x in command], check=True, cwd=EXAMPLES)

    def script(name, *parameters):
        run(sys.executable, "-B", EXAMPLES / name, *parameters)

    def compile_package(package):
        run(cmake, "-S", package / "source/_native", "-B", package / "build",
            "-G", "Visual Studio 17 2022", "-A", "x64", *flags)
        run(cmake, "--build", package / "build", "--config", "Release", "--parallel", "1")

    script("preview_sparse/build_source.py", "--rendering-root", EXAMPLES.parent,
           "--output", sparse, "--tile", "4", "--paths", "4", "--grid", "2",
           "--neighbors", "25", "--reconstruction-revision", "2", "--estimator", "visibility-ratio")
    compile_package(sparse)
    original_hash = sha256((sparse / "source-manifest.json").read_bytes()).hexdigest()
    script("preview_opaque/compose_sparse.py", "prepare", "--prepared", sparse,
           "--compiler", compiler, "--expected-manifest-sha256", original_hash, "--output", opaque)
    script("preview_compact/build_source.py", "--sparse-prepared", opaque, "--output", compact)
    compile_package(compact)
    script("preview_nearfield/build.py", "--parent", compact, "--compiler", compiler,
           "--radius-mm", "120", "--output", local)
    print(local / "source-manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="a new directory")
    for arg in ("compiler", "vulkan-include", "vulkan-library"):
        parser.add_argument("--" + arg, type=Path)
    prepare(parser.parse_args())
