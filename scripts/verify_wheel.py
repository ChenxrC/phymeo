"""Install a built wheel in a new native-pip environment and test outside the checkout."""

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def check_source_parity(wheel, root):
    """Reject stale, missing, extra or duplicate packaged Python source files."""
    source = root / "src"
    expected = {
        path.relative_to(source).as_posix(): path for path in (source / "phymeo").rglob("*.py")
    }
    with ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("phymeo/") and name.endswith(".py")
        ]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise SystemExit(
                "Wheel source file set differs from this checkout; rebuild before verification"
            )
        changed = [
            name for name, path in expected.items() if archive.read(name) != path.read_bytes()
        ]
        if changed:
            raise SystemExit(
                f"Wheel source differs from this checkout ({', '.join(changed)}); rebuild before verification"
            )
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheels = sorted(args.dist_dir.resolve().glob("phymeo-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one phymeo wheel, found {len(wheels)}")
    count = check_source_parity(wheels[0], root)
    print(f"Wheel source parity passed: {count} Python files match the checkout.", flush=True)
    with tempfile.TemporaryDirectory(prefix="phymeo-wheel-") as temporary:
        outside = Path(temporary).resolve()
        if outside.is_relative_to(root):
            raise SystemExit(
                "Set TEMP/TMP to a writable location outside the checkout for wheel validation"
            )
        environment = outside / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        def pip(*arguments):
            subprocess.run(
                [str(python), "-I", "-m", "pip", "--disable-pip-version-check", *arguments],
                cwd=outside,
                check=True,
            )

        pip("install", "--quiet", "--require-hashes", "-r", str(root / "requirements/runtime.lock"))
        pip("install", "--quiet", "--no-deps", str(wheels[0]))
        pip("check")
        subprocess.run(
            [str(python), "-I", str(root / "scripts/smoke_installed.py")], cwd=outside, check=True
        )
    print("Clean native-pip wheel installation and solver smoke passed.")


if __name__ == "__main__":
    main()
