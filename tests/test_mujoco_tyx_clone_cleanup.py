"""Owned clone resources survive allocation failures without publishing state."""

from types import SimpleNamespace

import pytest
from test_mujoco_tyx_activation import encode, save

from phymeo.backends import _mujoco_tyx_runtime as implementation
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
XML = """<mujoco><option timestep=".001" gravity="0 0 0"/>
<worldbody><body><joint name="hinge" type="hinge" damping=".2" armature=".04"/>
<geom type="sphere" pos=".2 0 0" size=".06" mass=".5"/></body></worldbody>
<actuator><general joint="hinge" dyntype="filterexact" dynprm=".02"
gainprm="1"/></actuator></mujoco>"""


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("native_copy", "replacement", "retire", "cleanup_retry"))
def test_failed_clone_closes_owned_resources_and_preserves_source(mode, fault, monkeypatch, tmp_path):
    compilation = compile_mujoco_xml(XML, require_asset=False)
    runtime = MuJoCoRuntime(compilation, mode=mode, world_keys=(7, 91))
    clones, banks, closed = [], [], []
    record = {"mode": mode, "fault": fault}
    try:
        runtime.set_controls({7: MuJoCoControl([.6]), 91: MuJoCoControl([-.2])})
        runtime.step(.001)
        before = runtime.snapshot()
        checkpoint = runtime.checkpoint().to_bytes()
        record.update(before=before, checkpoint=checkpoint.hex())
        with runtime.clone() as reference:
            original_build = implementation.MuJoCoTyxRuntime._build_device
            original_copy = implementation.copy.copy
            copies = 0
            build_count = 0
            failed_close_once = False

            def build(owner, proposed):
                nonlocal build_count, failed_close_once
                assert owner is not runtime
                if not clones:
                    clones.append(owner)
                assert owner is clones[0]
                build_count += 1
                if fault == "replacement" and build_count == 2:
                    raise MemoryError("authored replacement allocation failure")
                bank = original_build(owner, proposed)
                banks.append(bank)
                close_bank = bank.close
                ordinal = len(banks)

                def close():
                    nonlocal failed_close_once
                    if ordinal == 1 and fault in ("retire", "cleanup_retry") and not failed_close_once:
                        failed_close_once = True
                        raise RuntimeError("authored first cleanup failure")
                    close_bank()
                    closed.append(id(bank))

                monkeypatch.setattr(bank, "close", close)
                return bank

            def copied(value):
                nonlocal copies
                copies += 1
                if fault in ("native_copy", "cleanup_retry") and copies == 2:
                    raise MemoryError("authored second native copy failure")
                return original_copy(value)

            with monkeypatch.context() as patch:
                patch.setattr(implementation.MuJoCoTyxRuntime, "_build_device", build)
                patch.setattr(implementation, "copy", SimpleNamespace(copy=copied))
                expected = BaseExceptionGroup if fault == "cleanup_retry" else (
                    RuntimeError if fault == "retire" else MemoryError
                )
                with pytest.raises(expected) as caught:
                    runtime.clone()
                record["exception"] = repr(caught.value)
                if fault == "cleanup_retry":
                    assert isinstance(caught.value.exceptions[0], MemoryError)
                    assert isinstance(caught.value.exceptions[1], BaseExceptionGroup)

            assert len(clones) == 1
            clone = clones[0]
            assert clone._closed and clone._device is None and not clone._data
            if fault == "cleanup_retry":
                assert runtime._checkpoint_retired == [clone]
                assert closed == []
            else:
                assert not runtime._checkpoint_retired
                assert set(closed) == {id(bank) for bank in banks}
            assert encode(runtime.snapshot()) == encode(before)
            assert runtime.checkpoint().to_bytes() == checkpoint
            runtime.step(.001)
            reference.step(.001)
            assert encode(runtime.snapshot()) == encode(reference.snapshot())
            with runtime.clone() as retry:
                assert encode(retry.snapshot()) == encode(runtime.snapshot())
            record.update(after_retry=runtime.snapshot(), source_counters=dict(runtime.counters),
                          built_banks=len(banks), closed_before_source_close=len(set(closed)))
        runtime.close()
        assert set(closed) == {id(bank) for bank in banks}
        assert not runtime._checkpoint_retired
        record["closed_after_source_close"] = len(set(closed))
    except BaseException as error:
        record["failure"] = repr(error)
        raise
    finally:
        runtime.close()
        save(tmp_path / "clone-cleanup.json.gz", record)
