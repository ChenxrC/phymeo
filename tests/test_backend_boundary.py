"""Small static architecture guard regressions."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_public_tyx_imports",
    Path(__file__).resolve().parents[1] / "scripts/check_public_tyx_imports.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_root_exports_do_not_eagerly_load_optional_engines():
    source = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'tyx', 'mujoco', 'physolve_tyx', 'warp', 'newton', 'mujoco_warp'
        }:
            raise AssertionError('eager optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import phymeo
assert len(phymeo.__all__) == len(set(phymeo.__all__))
assert all(hasattr(phymeo, name) for name in phymeo.__all__)
assert {'PhySolveRuntime', 'PhySolvePayload', 'PhySolveInputError',
        'ResidentSceneBatch', 'MuJoCoTyxHeterogeneousBatch'} <= set(phymeo.__all__)
'''
    subprocess.run([sys.executable, "-c", source], check=True)


@pytest.mark.parametrize(
    "source, adapter",
    [
        ("import tyx as tx", False),
        ("import tyx.backends.cpu_cpp", True),
        ("from tyx._device_pool import Pool", True),
        ("from tyx import _LAZY_EXPORTS", True),
        ("from tyx import *", True),
        ("import tyx as tx\ntx.cuda._get_runtime()", True),
    ],
)
def test_private_or_misplaced_import_is_rejected(source, adapter):
    assert guard.violations(source, in_adapter=adapter)


def test_public_root_api_is_allowed_in_adapter():
    assert not guard.violations("import tyx as tx\ntx.kernel", in_adapter=True)


@pytest.mark.parametrize(
    "path,source",
    [
        ("backends/mujoco_task_workspace.py", "from tyx.torch import to_torch"),
        ("backends/mujoco_heterogeneous_io.py", "from tyx.torch import to_torch"),
        ("training/manipulation_env.py", "from tyx.torch import from_torch, stream_from_torch"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import from_torch, stream_from_torch"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import from_torch as borrow"),
        (
            "training/policy_response_diagnostics.py",
            "import tyx as tx\ntx.Runtime\ntx.CudaStream\ntx.use_runtime\ntx.CudaArray",
        ),
    ],
)
def test_declared_public_interop_consumers_are_allowed(path, source):
    assert not guard.violations(
        source,
        in_adapter=path.startswith("backends/"),
        relative_path=path,
    )


@pytest.mark.parametrize(
    "path,source",
    [
        ("training/arbitrary.py", "from tyx.torch import from_torch"),
        ("backends/mujoco_heterogeneous_io.py", "from tyx.torch import from_torch"),
        ("training/manipulation_env.py", "from tyx.torch import to_torch"),
        ("backends/arbitrary.py", "from tyx.torch import from_torch"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import to_torch"),
        ("backends/mujoco_task_workspace.py", "from tyx.torch import from_torch"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import *"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import _private"),
        ("training/mujoco_tyx_env.py", "from tyx.torch._internal import from_torch"),
        ("training/mujoco_tyx_env.py", "import tyx.torch"),
        ("training/mujoco_tyx_env.py", "from tyx import torch"),
        ("backends/mujoco_task_workspace.py", "from tyx import torch"),
        ("backends/mujoco_task_workspace.py", "import tyx as tx\ntx.torch.to_torch"),
        ("training/mujoco_tyx_env.py", "from tyx.torch import from_torch as ft\nft._private"),
        ("training/policy_response_diagnostics.py", "import tyx as tx\ntx.kernel"),
        ("training/policy_response_diagnostics.py", "import tyx as tx\ntx.cuda"),
        ("training/policy_response_diagnostics.py", "import tyx as tx\ntx.CudaArray._private"),
        ("training_other/mujoco_tyx_env.py", "from tyx.torch import from_torch"),
        ("training/mujoco_tyx_env.py.bak", "from tyx.torch import from_torch"),
        ("backends/example.py", "from tyx import CudaArray as Buffer\nBuffer._private"),
    ],
)
def test_interop_allowlist_preserves_module_symbol_and_private_boundaries(path, source):
    assert guard.violations(
        source,
        in_adapter=path.startswith("backends/"),
        relative_path=path,
    )
