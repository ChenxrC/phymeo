"""A stale or incomplete wheel must fail before installation smoke can mislead."""

import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest


def check():
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/verify_wheel.py"))
    function = namespace.get("check_source_parity")
    assert function is not None, "wheel verification must check source parity"
    return function


@pytest.mark.parametrize("case", ["matching", "stale", "missing", "extra"])
def test_wheel_code_must_match_current_source(tmp_path, case):
    source = tmp_path / "src/phymeo"
    source.mkdir(parents=True)
    (source / "__init__.py").write_bytes(b"VALUE = 2\n")
    (source / "solver.py").write_bytes(b"def step(): return 42\n")
    wheel = tmp_path / "example.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "phymeo/__init__.py", b"VALUE = 1\n" if case == "stale" else b"VALUE = 2\n"
        )
        if case != "missing":
            archive.writestr("phymeo/solver.py", b"def step(): return 42\n")
        if case == "extra":
            archive.writestr("phymeo/removed.py", b"OLD = True\n")
        archive.writestr("phymeo-0.1.dist-info/METADATA", "Name: phymeo\n")
    if case == "matching":
        assert check()(wheel, tmp_path) == 2
    else:
        with pytest.raises(SystemExit, match="rebuild"):
            check()(wheel, tmp_path)
