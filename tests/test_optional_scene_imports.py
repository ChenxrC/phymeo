"""The public compiler modules are importable without optional runtimes."""

import subprocess
import sys


def test_optional_runtimes_are_not_imported_by_public_compiler_modules():
    program = """
import importlib.abc
import sys

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'mujoco_warp', 'warp', 'tyx'}:
            raise AssertionError('unexpected optional import: ' + fullname)

sys.meta_path.insert(0, RejectOptional())
import phymeo.builders.scene_compiler
import phymeo.builders.mujoco_compiler
import phymeo.backends.mujoco_runtime
print('optional imports stayed lazy')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "optional imports stayed lazy"
