"""Native comparison instrumentation, never imported by production runtimes.

Native reset below is a diagnostic implementation using host snapshots. It is
not an official MJWarp masked-reset API or a performance qualification.
"""

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

INTEGRATION = (
    "time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
    "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
)
OBSERVATIONS = (
    "qpos", "qvel", "qacc", "time", "ctrl", "qfrc_applied", "qfrc_actuator",
    "qfrc_constraint", "xpos", "xquat",
)
DIAGNOSTICS = (
    "qacc_warmstart", "qacc_smooth", "qfrc_smooth", "qfrc_bias", "qfrc_passive",
    "qM", "qLD", "qLDiagInv", "solver_niter", "ne", "nf", "nl", "nefc",
    "nacon", "ncollision",
)
CONTACT = ("geom", "worldid", "dim", "dist", "pos", "frame", "friction",
           "efc_address", "geomcollisionid")
EFC = ("type", "id", "J", "D", "aref", "force", "pos", "vel", "margin")


def payload_from_sources(path):
    from phymeo import PhySolvePayload

    path = Path(path)
    sources = json.loads(path.read_text(encoding="utf-8-sig"))
    frozen = json.loads((path.parent / "evidence/payload-baseline.json").read_text(
        encoding="utf-8-sig"))
    return PhySolvePayload(
        sources["native_host"], sources["cubins"], frozen["native_host"]["sha256"],
        tuple((row["name"], row["sha256"]) for row in frozen["cubins"]),
    )


def options(model):
    result = {}
    for name, value in vars(model.opt).items():
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            result[name] = dict(dtype=str(value.dtype), shape=list(value.shape),
                                values=value.tolist())
        elif isinstance(value, (bool, int, float, str)):
            result[name] = value
        else:
            result[name] = dict(unhandled_type=type(value).__name__)
    return result


class NativeRoute:
    def __init__(self, route, model, data, compute, stream, native=None, handle=None):
        self.route, self.model, self.data = route, model, data
        self.compute, self.stream = compute, stream
        self.native, self.handle = native, handle
        self.initial = {}

    def complete(self):
        if self.native is None:
            self.compute.synchronize_stream(self.stream)
        else:
            self.native._synchronize_runtime_stream(
                self.handle, int(self.compute.get_device("cuda:0").context),
                int(self.stream.handle),
            )

    def forward(self):
        if self.native is None:
            self.frontend.forward(self.model, self.data)
        else:
            self.native._forward_runtime(self.handle)
        self.complete()

    def step(self, force):
        self.data.qfrc_applied.assign(force)
        if self.native is None:
            self.frontend.step(self.model, self.data)
        else:
            self.native._step_runtime(self.handle)
        self.complete()

    def reset(self, mask=None):
        before = {k: getattr(self.data, k).numpy().copy() for k in OBSERVATIONS}
        for k, initial in self.initial.items():
            target = getattr(self.data, k)
            if not initial.size:
                continue
            value = initial.copy() if mask is None else target.numpy().copy()
            if mask is not None:
                value[mask] = initial[mask]
            target.assign(value)
        self.forward()
        # Match the snapshot publication contract explicitly; raw contact/EFC
        # below remain results of full native forward, not a published contact API.
        if mask is not None:
            for k in OBSERVATIONS:
                if k in INTEGRATION:
                    continue
                value = getattr(self.data, k).numpy().copy()
                value[~mask] = before[k][~mask]
                getattr(self.data, k).assign(value)
        self.complete()


class RuntimeRoute:
    def __init__(self, runtime):
        self.runtime = runtime
        self.model = runtime._model

    @property
    def data(self):
        return self.runtime._banks[self.runtime._active]

    def complete(self):
        self.runtime._complete()

    def forward(self):
        self.runtime.forward()

    def step(self, force):
        self.runtime.step(qfrc_applied=force)

    def reset(self, mask=None):
        self.runtime.reset(mask)


@contextmanager
def open_route(route, nworld, case, source_path):
    import mujoco

    case = Path(case)
    with ExitStack() as stack:
        if route == "runtime":
            from test_panda_collision_model import panda_connection_plan, panda_template

            template = panda_template()
            template = template.with_collision_exclusions(panda_connection_plan(template))
            assert template.compilation.source.model_binary == (case / "model.mjb").read_bytes()
            with np.load(case / "initial.npz") as frozen:
                for k, value in template.initial_state.fields:
                    np.testing.assert_array_equal(value, frozen[k])
            runtime = stack.enter_context(template.bind(nworld).create_runtime(
                payload=payload_from_sources(source_path), nconmax=128, njmax=256))
            stack.enter_context(runtime._scope())
            assert runtime._model.callback.control is None
            yield RuntimeRoute(runtime)
            return
        cpu = mujoco.MjModel.from_binary_path(str(case / "model.mjb"))
        cpu.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        initial = mujoco.MjData(cpu)
        with np.load(case / "initial.npz") as frozen:
            for k in INTEGRATION:
                if k == "time":
                    initial.time = float(frozen[k])
                else:
                    getattr(initial, k)[:] = frozen[k]
        mujoco.mj_forward(cpu, initial)
        if route == "warp":
            import mujoco_warp as frontend
            import warp as compute

            compute.init()
            stream = compute.Stream(device="cuda:0")
            stack.enter_context(compute.ScopedDevice("cuda:0"))
            stack.enter_context(compute.ScopedStream(stream))
            native = handle = None
        else:
            import physolve_tyx as frontend
            import tyx as compute

            assert route == "physolve"
            native = payload_from_sources(source_path).load()
            device = compute.get_device("cuda:0")
            stack.enter_context(compute.use_device(device))
            stream = compute.CudaStream(device=device.ordinal)
            stack.callback(stream.destroy)
            stack.enter_context(compute.use_stream(stream))
        model = frontend.put_model(cpu)
        model.opt.graph_conditional = True
        assert model.callback.control is None
        data = frontend.put_data(cpu, initial, nworld=nworld, nconmax=128, njmax=256)
        if route == "physolve":
            frontend.prepare_workspace(model, data)
            handle = native._create_runtime(model, data, "tyx", "tyx")
            stack.callback(native._close_runtime, handle, True)
            native._prepare_runtime(handle)
        instance = NativeRoute(route, model, data, compute, stream, native, handle)
        instance.frontend = frontend
        stack.callback(instance.complete)
        instance.forward()
        instance.initial = {k: getattr(data, k).numpy().copy() for k in INTEGRATION}
        yield instance


def snapshot(instance):
    instance.complete()
    data = instance.data
    assert int(data.nacon.numpy()[0]) <= data.naconmax
    assert int(data.ncollision.numpy()[0]) <= data.naconmax
    assert np.all(data.nefc.numpy() <= data.njmax)
    result = {k: getattr(data, k).numpy().copy()
              for k in dict.fromkeys(OBSERVATIONS + DIAGNOSTICS + INTEGRATION)}
    for parent, fields in (("contact", CONTACT), ("efc", EFC)):
        result.update({parent + "." + k: getattr(getattr(data, parent), k).numpy().copy()
                       for k in fields})
    for k in OBSERVATIONS:
        assert np.isfinite(result[k]).all(), k
    return result
