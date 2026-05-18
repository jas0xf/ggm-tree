"""Test configuration: skip GPU-marked tests when no CUDA device is available."""
import pytest


def pytest_collection_modifyitems(config, items):
    requested = config.getoption("-m") or ""
    if "gpu" in requested:
        return
    try:
        import pycuda.autoinit  # noqa: F401
        return
    except Exception:
        skip_gpu = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
