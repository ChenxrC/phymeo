"""Generated-load delivery is explicit; user inputs stay strict."""

import numpy as np
import pytest
from test_deformable_contact_response import H, create, fixture, forces, manifest, saved

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("target", ("node", "wrench"))
def test_generated_subnormal_is_recorded_before_joint_publication(
    target, tyx_mode, tmp_path, monkeypatch
):
    item = fixture("E")
    with saved(tmp_path, "delivery-" + target, tyx_mode, case=manifest(item)) as evidence:
        with create((item,), tyx_mode) as batch:
            original = batch._engine.evaluate
            injected = {}

            def generated(*args):
                law, node, wrench = original(*args)
                array = (node if target == "node" else wrench).copy()
                index = int(np.flatnonzero(array.ravel() == 0)[0])
                array.view(np.uint32).ravel()[index] = 0x800000AC
                injected.update(index=index, array=array.copy())
                return (law, array, wrench) if target == "node" else (law, node, array)

            monkeypatch.setattr(batch._engine, "evaluate", generated)
            entry = batch.snapshot()[item["world"]]
            record = dict(entry=entry, inputs=forces((item,)))
            evidence["attempts"].append(record)
            receipt = batch.step(record["inputs"], H, expected_revision=batch.revision)
            record["returned"] = receipt
            record["after"] = batch.snapshot()[item["world"]]
            field = "node_force_delivery" if target == "node" else "rigid_wrench_delivery"
            delivery = receipt.worlds[item["world"]][field]
            index = injected["index"]
            assert delivery.raw.view(np.uint32).ravel()[index] == 0x800000AC
            assert delivery.delivered.view(np.uint32).ravel()[index] == 0x80000000
            assert delivery.subnormal_mask.ravel()[index]
            assert delivery.loss.ravel()[index] == -172 * float.fromhex("0x1p-149")
            assert abs(delivery.loss.ravel()[index]) < float.fromhex("0x1p-126")
            actual = (
                receipt.node_receipt.worlds[item["world"]].force
                if target == "node"
                else receipt.worlds[item["world"]]["uploaded_rigid_wrench"]
            )
            np.testing.assert_array_equal(
                actual.view(np.uint32), delivery.delivered.view(np.uint32)
            )
            assert record["after"].node.tick == record["after"].rigid.tick == entry.node.tick + 1
            record["injected_generated_aggregate"] = injected


@pytest.mark.parametrize("target", ("node", "wrench"))
def test_nonfinite_generated_load_rejects_without_partial_publication(
    target, tyx_mode, monkeypatch
):
    item = fixture("E")
    with create((item,), tyx_mode) as batch:
        before = batch.checkpoint()
        revision = batch.revision
        original = batch._engine.evaluate

        def broken(*args):
            law, node, wrench = original(*args)
            array = (node if target == "node" else wrench).copy()
            array.ravel()[0] = np.nan
            return (law, array, wrench) if target == "node" else (law, node, array)

        monkeypatch.setattr(batch._engine, "evaluate", broken)
        with pytest.raises(ValueError, match="must be finite"):
            batch.step(forces((item,)), H, expected_revision=revision)
        assert batch.revision == revision
        assert batch.checkpoint().content_sha256 == before.content_sha256


def test_external_subnormal_remains_rejected_before_contact_evaluation(tyx_mode, monkeypatch):
    item = fixture("E")
    with create((item,), tyx_mode) as batch:
        before = batch.checkpoint()

        def forbidden(*args):
            raise AssertionError("invalid external input reached the device contact engine")

        monkeypatch.setattr(batch._engine, "evaluate", forbidden)
        force = np.zeros((4, 3), np.float32)
        force.view(np.uint32)[0, 0] = 1
        with pytest.raises(ValueError, match="normal FP32 or zero"):
            batch.step({item["world"]: ph.StepInputs(force)}, H, expected_revision=batch.revision)
        assert batch.checkpoint().content_sha256 == before.content_sha256
