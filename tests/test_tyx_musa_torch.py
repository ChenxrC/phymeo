"""Opt-in same-card MUSA copies. The CPU oracle runs after the measured interval."""

import os

import numpy as np
import pytest

from phymeo.backends.tyx_musa import MusaError, MusaRuntime


pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU"),
]


def test_torch_tyx_roundtrip_on_distinct_streams():
    import torch
    import torch_musa

    device = torch.device("musa", 0)
    producer = torch_musa.Stream(device=device)
    consumer = torch_musa.Stream(device=device)
    expected = np.arange(2, 14, dtype=np.float32).reshape(3, 4) + 10
    with MusaRuntime(0) as runtime:
        src = runtime.empty((3, 4), np.float32)
        backup = runtime.empty(src.shape, src.dtype)
        with torch_musa.stream(producer):
            base = torch.arange(16, device=device, dtype=torch.float32)
            tensor = base[2:14].reshape(3, 4)
            tensor.add_(10)
        with torch_musa.stream(consumer):
            output_owner = torch.full((16,), -7., device=device)
            out = output_owner[2:14].reshape(3, 4)
        before = runtime.report()
        src.copy_from_torch(tensor, stream=producer)
        backup.copy_from_device(src)
        assert backup.copy_to_torch(out, stream=consumer) is out
        independent = src.to_torch(stream=producer)
        src.close()
        backup.close()
        after = runtime.report()
        assert after["torch_to_tyx_bytes"] - before["torch_to_tyx_bytes"] == expected.nbytes
        assert after["tyx_to_tyx_bytes"] - before["tyx_to_tyx_bytes"] == expected.nbytes
        assert after["tyx_to_torch_bytes"] - before["tyx_to_torch_bytes"] == 2 * expected.nbytes
        assert after["device_to_device_bytes"] - before["device_to_device_bytes"] == 4 * expected.nbytes
        assert after["uploaded_bytes"] == before["uploaded_bytes"]
        assert after["downloaded_bytes"] == before["downloaded_bytes"]
        with pytest.raises(MusaError, match="closed"):
            src.copy_from_torch(tensor)
    # Caller-owned destination tensors survive both Tyx buffers and the stream.
    np.testing.assert_array_equal(independent.cpu().numpy(), expected)
    np.testing.assert_array_equal(output_owner.cpu().numpy(),
                                  np.r_[[-7., -7.], expected.ravel(), [-7., -7.]])


def test_torch_tyx_admission_and_empty():
    import torch
    import torch_musa

    device = torch.device("musa", 0)
    with MusaRuntime(0) as runtime:
        arr = runtime.empty((2, 3), np.float32)
        ok = torch.empty((2, 3), device=device)
        with pytest.raises(ValueError, match="same MUSA device"):
            arr.copy_from_torch(torch.empty((2, 3)))
        with pytest.raises(ValueError, match="shape and dtype"):
            arr.copy_from_torch(torch.empty((3, 2), device=device))
        with pytest.raises(ValueError, match="shape and dtype"):
            arr.copy_to_torch(torch.empty((2, 3), dtype=torch.int32, device=device))
        with pytest.raises(ValueError, match="contiguous"):
            arr.copy_from_torch(torch.empty((3, 2), device=device).T)
        with pytest.raises(ValueError, match="gradients"):
            arr.copy_to_torch(ok.requires_grad_())
        with pytest.raises(ValueError, match="stream"):
            arr.copy_from_torch(ok, stream=object())
        if torch_musa.device_count() > 1:
            with pytest.raises(ValueError, match="same MUSA device"):
                arr.copy_from_torch(torch.empty((2, 3), device="musa:1"))
            with pytest.raises(ValueError, match="stream"):
                arr.copy_from_torch(ok, stream=torch_musa.Stream(device=1))
        with MusaRuntime(0) as other:
            with pytest.raises(ValueError, match="same MUSA runtime"):
                other.empty(arr.shape, arr.dtype).copy_from_device(arr)
        assert runtime.report()["device_to_device_bytes"] == 0
        zero = runtime.empty((3, 0), np.int32)
        value = zero.to_torch()
        assert tuple(value.shape) == zero.shape and value.dtype == torch.int32 and value.device == device
        zero.copy_from_torch(value)
        assert runtime.report()["device_to_device_bytes"] == 0


def test_torch_owner_retained_until_failed_copy_drained(monkeypatch):
    import gc
    import weakref

    import torch
    import torch_musa  # noqa: F401 - register the private-use accelerator

    def injected_failure():
        raise MusaError("injected completion status failure")

    with MusaRuntime(0) as runtime:
        buf = runtime.empty((4,), np.float32)
        tensor = torch.zeros((4,), device="musa:0")
        owner = weakref.ref(tensor)
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_sync", injected_failure)
            with pytest.raises(MusaError, match="injected"):
                buf.copy_from_torch(tensor)
        del tensor
        gc.collect()
        assert owner() is not None
        assert runtime.report()["device_to_device_bytes"] == 0
        with pytest.raises(MusaError, match="failed previously"):
            buf.to_numpy()
        # Responsible cleanup performs the real stream wait before releasing.
        buf.close()
        gc.collect()
        assert owner() is None
