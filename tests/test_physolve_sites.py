"""Selected physical site projection: independent frame/time references, no rollout comparison."""

import numpy as np
import pytest
from test_physolve_measurements import HINGE
from test_physolve_runtime import CONTACT
from test_physolve_runtime import payload as payload_fixture

from phymeo import PhySolveRuntime
from phymeo.backends.physolve_initial import NativeInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.builders.native_assembly import NativeTemplate
from phymeo.sensors.body_site import BodySite

payload = payload_fixture


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_site_frames_solver_time_masked_reset(payload, nworld, monkeypatch):
    import mujoco as mj
    import tyx as tx

    record = compile_mujoco_xml(HINGE, require_asset=False)
    data = mj.MjData(record.fresh_native_model())
    data.qvel[:] = 2
    data.qfrc_applied[:] = 1
    data.xfrc_applied[1, 3] = 2  # World-X external couple; hinge constrains this axis.
    initial = NativeInitialState.from_data(record, data)
    template = NativeTemplate(record, initial, (), (("body", 1),), (), ())
    binding = template.bind_body_sites(
        (BodySite("offset", "body", (0.3, 0.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5)),)
    )
    alpha = 1 / (0.1 + 2 * 0.2**2)
    with PhySolveRuntime(
        record,
        payload=payload,
        nworld=nworld,
        initial_state=initial,
        body_observations=True,
        body_measurements=True,
    ) as runtime:
        plan = runtime.prepare_site_measurements(binding)
        selector = tx.Array(np.arange(nworld - 1, -1, -1, dtype=np.int32), device="cuda:0")
        count = tx.Array(np.asarray([nworld], np.int32), device="cuda:0")
        gate = tx.Array(np.zeros(1, np.int32), device="cuda:0")

        def sample():
            plan.sample(selector, count, gate, nworld)
            plan.storage.stream.synchronize()
            assert plan.errors.numpy()[0] == 0
            return plan.values.numpy().copy()

        baseline = sample()
        np.testing.assert_allclose(
            baseline[:, :3], np.tile([0.3 * alpha, 1.2, 9.81], (nworld, 1)), atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            baseline[:, 3:6], np.tile([0, 0, 2], (nworld, 1)), atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(baseline[:, 8], 19.62, atol=1e-6, rtol=0)
        # COM torque I*alpha, shifted by r=(.1,0,0) and F_y=m*.2*alpha.
        site_torque_z = (0.1 - 0.1 * 2 * 0.2) * alpha
        np.testing.assert_allclose(
            baseline[:, 9:12],
            np.tile([1.962, 2.0, site_torque_z], (nworld, 1)),
            atol=1e-6,
            rtol=0,
        )
        runtime.prepare_step_graph()
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket, steps=4, execution_mode="cuda_graph")
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        with monkeypatch.context() as patch:
            patch.setattr(
                tx, "empty", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hot allocation"))
            )
            patch.setattr(
                tx.Array,
                "numpy",
                lambda *a, **k: (_ for _ in ()).throw(AssertionError("hot snapshot")),
            )
            plan.sample(selector, count, gate, nworld)
        observed = sample()
        omega = 2 + 0.03 * alpha
        np.testing.assert_allclose(
            observed[:, :3],
            np.tile([0.3 * alpha, 0.3 * omega**2, 9.81], (nworld, 1)),
            atol=1e-6,
            rtol=0,
        )
        np.testing.assert_allclose(observed[:, 5], omega, atol=1e-6, rtol=0)
        selected = np.arange(nworld) % 2 == 0
        runtime.reset(selected)
        after = sample()
        reversed_mask = selected[::-1]
        np.testing.assert_array_equal(after[~reversed_mask], observed[~reversed_mask])
        np.testing.assert_allclose(after[reversed_mask], baseline[reversed_mask], atol=1e-6, rtol=0)
        ticket = runtime.begin_execution()
        runtime.enqueue_update(ticket)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        np.testing.assert_array_equal(sample(), after)
        with pytest.raises(ValueError, match="same-device"):
            plan.sample(
                tx.Array(np.arange(nworld, dtype=np.int64), device="cuda:0"), count, gate, nworld
            )
        plan.close()
        with pytest.raises(RuntimeError, match="closed"):
            plan.sample(selector, count, gate, nworld)


def test_body_site_identity_validation():
    with pytest.raises(ValueError, match="unit"):
        BodySite("bad", "body", rotation_xyzw=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        BodySite("bad", "body", position=(float("nan"), 0, 0))


@pytest.mark.cuda
def test_site_contact_specific_force_balance(payload):
    import tyx as tx

    from phymeo.builders.native_sites import NativeBodySites

    record = compile_mujoco_xml(CONTACT, require_asset=False)
    binding = NativeBodySites(record, (BodySite("ball", "ball"),), (1,))
    with PhySolveRuntime(
        record, payload=payload, nworld=8, body_measurements=True, contact_observations=True
    ) as runtime:
        plan = runtime.prepare_site_measurements(binding)
        selected = tx.Array(np.arange(8, dtype=np.int32), device="cuda:0")
        count = tx.Array(np.asarray([8], np.int32), device="cuda:0")
        gate = tx.Array(np.zeros(1, np.int32), device="cuda:0")
        runtime.prepare_step_graph()
        ticket = runtime.begin_execution()
        runtime.enqueue_step(ticket, steps=4, execution_mode="cuda_graph")
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        plan.sample(selected, count, gate, 8)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 0
        values = plan.values.numpy()
        storage, bank = runtime.storage_binding(), runtime.observation_bank
        total = int(storage.borrow("contact_count", bank).numpy()[0])
        worlds = storage.borrow("contact_world", bank).numpy()[:total, 0]
        forces = storage.borrow("contact_wrench", bank).numpy()[:total, :3]
        expected = np.zeros((8, 3), np.float32)
        np.add.at(expected, worlds, forces)
        assert np.all(expected[:, 2] > 0)
        # Unit mass, no external force: contact force equals specific acceleration.
        np.testing.assert_allclose(values[:, :3], expected, atol=1e-6, rtol=1e-6)
        # A free root has no parent reaction. Scale residual by physical force magnitude.
        scale = 1 + np.max(np.abs(expected), axis=1)
        assert np.all(np.max(np.abs(values[:, 6:12]), axis=1) <= 1e-6 * scale)


@pytest.mark.cuda
def test_site_binding_mismatch_and_prepare_cleanup_retention(payload, monkeypatch):
    from phymeo.backends._physolve_sites import PreparedSiteMeasurements
    from phymeo.builders.native_sites import NativeBodySites

    record = compile_mujoco_xml(HINGE, require_asset=False)
    site = BodySite("sensor", "body")
    binding = NativeBodySites(record, (site,), (1,))
    runtime = PhySolveRuntime(record, payload=payload, body_measurements=True)
    try:
        with pytest.raises(ValueError, match="compilation"):
            runtime.prepare_site_measurements(
                NativeBodySites(compile_mujoco_xml(HINGE, require_asset=False), (site,), (1,))
            )
        assert runtime._site_measurements == []
        with pytest.raises(ValueError, match="indices"):
            runtime.prepare_site_measurements(NativeBodySites(record, (site,), (-1,)))
        allocate = PreparedSiteMeasurements._array

        def fail_after_upload(plan, *args):
            allocate(plan, *args)
            raise MemoryError("site upload submitted")

        with monkeypatch.context() as patch:
            patch.setattr(PreparedSiteMeasurements, "_array", fail_after_upload)
            with pytest.raises(MemoryError, match="upload submitted"):
                runtime.prepare_site_measurements(binding)
        plan = runtime._site_measurements[-1]
        assert plan.resources[0].ptr
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime,
                "_synchronize_stream",
                lambda *a: (_ for _ in ()).throw(RuntimeError("site cleanup wait")),
            )
            with pytest.raises(RuntimeError, match="cleanup wait"):
                runtime.close()
        assert runtime._site_measurements[-1] is plan and plan.resources[0].ptr
        runtime.close()
        assert not plan.resources
    finally:
        runtime.close()
