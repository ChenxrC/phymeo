"""Frozen physical impact cases; test-only models and host controller."""

from contextlib import contextmanager

import numpy as np
from _solver_comparison import INTEGRATION, RuntimeRoute, open_route, payload_from_sources

CASES = ("fall", "fall_half", "slider", "balance", "support", "panda_stress", "panda_motion")
STEPS = dict(fall=200, fall_half=400, slider=100, balance=500,
             support=4000, panda_stress=1000, panda_motion=1000)
DT = dict(fall=.001, fall_half=.0005, slider=.002, balance=.001,
          support=.001, panda_stress=.002, panda_motion=.002)
XML = {
    "fall": '''<mujoco><option timestep=".001" integrator="Euler" jacobian="dense"/>
<worldbody><body pos="0 0 2"><freejoint/><geom type="sphere" size=".1" mass="2"/>
</body></worldbody></mujoco>''',
    "slider": '''<mujoco><option timestep=".002" gravity="0 0 0" jacobian="dense"/>
<worldbody><body><joint type="slide" axis="1 0 0"/>
<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>''',
    "balance": '''<mujoco><option timestep=".001" jacobian="dense"/>
<worldbody><body><joint type="hinge" axis="0 1 0"/>
<inertial pos=".3 0 0" mass="2" diaginertia=".02 .02 .02"/>
<geom type="sphere" pos=".3 0 0" size=".1"/></body></worldbody></mujoco>''',
    "support": '''<mujoco><option timestep=".001" jacobian="dense"/>
<worldbody><geom name="ground" type="plane" size="0 0 1"/>
<body pos="0 0 .5"><freejoint/><geom name="ball" type="sphere" size=".1" mass="2"/>
</body></worldbody></mujoco>''',
}
XML["fall_half"] = XML["fall"].replace('.001', '.0005')


def compilation(name):
    if name.startswith("panda"):
        from test_panda_collision_model import panda_connection_plan, panda_template

        template = panda_template()
        template = template.with_collision_exclusions(panda_connection_plan(template))
        return template.compilation, dict(template.initial_state.fields)
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    record = compile_mujoco_xml(XML[name], require_asset=False)
    import mujoco

    data = mujoco.MjData(record.fresh_native_model())
    return record, {key: np.asarray(getattr(data, key)).copy() for key in INTEGRATION}


@contextmanager
def impact_route(route, nworld, name, case, sources):
    if route != "runtime":
        with open_route(route, nworld, case, sources) as runtime:
            yield runtime
        return
    from phymeo import PhySolveRuntime
    from phymeo.backends.physolve_initial import NativeInitialState

    record, _ = compilation(name)
    assert record.source.model_binary == (case / "model.mjb").read_bytes()
    with np.load(case / "initial.npz") as initial:
        state = NativeInitialState(record.source.model_sha256,
                                   tuple((k, initial[k]) for k in INTEGRATION))
    with PhySolveRuntime(record, payload=payload_from_sources(sources), nworld=nworld,
                         nconmax=128, njmax=256, initial_state=state) as runtime:
        with runtime._scope():
            assert runtime._model.callback.control is None
            yield RuntimeRoute(runtime)


def command(name, nworld, index, qpos, qvel, controls):
    force = np.broadcast_to(controls["force"][index], qvel.shape).copy()
    if name == "slider":
        force *= np.linspace(1, 2, nworld, dtype=np.float32)[:, None]
    elif name == "panda_stress":
        force *= np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
    elif name == "panda_motion":
        force += (controls["kp"] * (controls["target_q"][index] - qpos)
                  + controls["kd"] * (controls["target_v"][index] - qvel))
    return np.asarray(force, dtype=np.float32)


def observe(sim):
    sim.complete()
    data = sim.data
    assert 0 <= int(data.nacon.numpy()[0]) <= data.naconmax
    assert 0 <= int(data.ncollision.numpy()[0]) <= data.naconmax
    assert np.all((data.nefc.numpy() >= 0) & (data.nefc.numpy() <= data.njmax))
    state = {name: getattr(data, name).numpy().copy()
             for name in ("qpos", "qvel", "qacc", "qfrc_constraint", "time")}
    assert all(np.isfinite(value).all() for value in state.values())
    return state


def contacts(sim, nworld):
    data = sim.data
    count = int(data.nacon.numpy()[0])
    world = data.contact.worldid.numpy()[:count]
    dist = data.contact.dist.numpy()[:count]
    assert np.all((world >= 0) & (world < nworld)) and np.isfinite(dist).all()
    minima = np.zeros(nworld)
    np.minimum.at(minima, world, dist)
    return np.bincount(world, minlength=nworld), minima
