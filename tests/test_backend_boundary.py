"""Small static architecture guard regressions."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_public_tyx_imports",
    Path(__file__).resolve().parents[1] / "scripts/check_public_tyx_imports.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


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
