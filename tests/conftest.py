"""Optional backends must execute when explicitly required, never silently skip."""

import importlib.util

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tyx-mode", default="interpret", choices=["interpret", "cpu_cpp", "cpu_llvm", "cuda"]
    )
    parser.addoption(
        "--require-tyx",
        action="store_true",
        help="Fail instead of skipping if the optional tyx package is absent.",
    )


@pytest.fixture
def tyx_mode(request):
    if importlib.util.find_spec("tyx") is None:
        if request.config.getoption("--require-tyx"):
            pytest.fail("--require-tyx was requested, but tyx is not installed")
        pytest.skip("optional tyx package is not installed")
    return request.config.getoption("--tyx-mode")
