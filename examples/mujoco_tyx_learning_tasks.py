"""Frozen N4a task declarations; no task identity is embedded in the engine."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

UPDATES, ROLLOUT, LEARNING_WORLDS, HORIZON = 128, 32, 8, 256
SCALE_WORLDS = (1, 8, 32, 64, 128, 256)
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

SUPPORT_XML = """<mujoco model="t3-normal"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" solver="Newton"
 iterations="128" tolerance="1e-10" jacobian="dense"/>
<default><geom condim="1" solref=".02 1" solimp=".8 .9 .002 .5 2"/>
<joint armature=".01" damping=".04"/></default>
<worldbody><geom name="platform" type="box" size=".24 .18 .05"/>
<body name="root" pos="0 0 .109"><joint name="vertical" type="slide" axis="0 0 1" armature="0" damping="0"/>
<inertial mass="2.0" pos="0 0 0" fullinertia=".08 .10 .12 .002 -.001 .003"/>
<geom name="moving" type="box" size=".12 .08 .06"/></body></worldbody>
<actuator><motor joint="vertical"/></actuator></mujoco>"""


def _joint_xml():
    bodies = "".join(
        f'<body name="slider{i}" pos="{position} 0 0">'
        f'<joint name="j{i}" type="slide" axis="1 0 0" damping=".1" armature=".01"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia=".02 .025 .03"/>'
        '<geom type="box" size=".05 .04 .03" contype="0" conaffinity="0"/></body>'
        for i, position in enumerate(("-0.3", "0.3"))
    )
    motors = "".join(
        f'<motor joint="j{i}" gear="1" ctrllimited="true" ctrlrange="-2 2" '
        'forcelimited="true" forcerange="-2 2"/>'
        for i in range(2)
    )
    return (
        '<mujoco model="n4-j"><compiler angle="radian" fusestatic="false"/>'
        '<option timestep=".002" gravity="0 0 0" integrator="Euler"/>'
        f"<worldbody>{bodies}</worldbody><actuator>{motors}</actuator></mujoco>"
    )


def _branch_xml():
    def body(name, position, mass, joint, children=""):
        return (
            f'<body name="{name}" pos="{position}">{joint}'
            f'<inertial pos=".04 -.03 .02" mass="{mass}" '
            'fullinertia=".02 .025 .03 .001 -.0005 .0008"/>'
            '<geom type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>'
            f"{children}</body>"
        )

    def joint(index, axis, ref):
        return (
            f'<joint name="j{index}" type="hinge" axis="{axis}" ref="{ref}" '
            'armature=".04" damping=".12"/>'
        )

    children = body("left", ".3 .18 .02", ".8", joint(1, "1 0 0", "-.04")) + body(
        "right", "-.2 -.14 .1", "1.5", joint(2, "0 1 0", ".06")
    )
    root = body("shoulder", "0 0 .3", "2", joint(0, "0 0 1", ".03"), children)
    motors = "".join(
        f'<motor joint="j{i}" gear="1" ctrllimited="true" ctrlrange="-2 2" '
        'forcelimited="true" forcerange="-2 2"/>'
        for i in range(3)
    )
    return (
        '<mujoco model="n4-r"><compiler angle="radian" fusestatic="false"/>'
        '<option timestep=".001" gravity="0 0 0" integrator="Euler"/>'
        f"<worldbody>{root}</worldbody><actuator>{motors}</actuator></mujoco>"
    )


@dataclass(frozen=True)
class Task:
    name: str
    xml: str
    seeds: tuple
    h: float
    q_target: tuple
    q_scale: tuple
    v_scale: tuple
    q_lower: tuple
    q_upper: tuple
    v_bound: float
    action_scale: float

    @property
    def xml_sha256(self):
        return sha256(self.xml.encode()).hexdigest()

    def cost(self):
        from phymeo.training.mujoco_tyx_evaluation import QuadraticCost

        return QuadraticCost(self.q_target, self.q_scale, self.v_scale, 0.01)


TASKS = {
    "J": Task(
        "J",
        _joint_xml(),
        (14011, 14012, 14013),
        0.002,
        (0.0, 0.0),
        (0.2, 0.2),
        (1.0, 1.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        8.0,
        2.0,
    ),
    "S": Task(
        "S",
        SUPPORT_XML,
        (14111, 14112, 14113),
        0.001,
        (0.00075,),
        (0.0005,),
        (0.05,),
        (-0.002,),
        (0.004,),
        0.5,
        20.0,
    ),
    "R": Task(
        "R",
        _branch_xml(),
        (14211, 14212, 14213),
        0.001,
        (0.03, -0.04, 0.06),
        (0.25, 0.25, 0.25),
        (1.0, 1.0, 1.0),
        (-0.97, -1.04, -0.94),
        (1.03, 0.96, 1.06),
        8.0,
        2.0,
    ),
}


def task(name):
    if name not in TASKS:
        raise ValueError("unknown frozen N4a task")
    return TASKS[name]


def templates(name, *, held_out):
    item = task(name)
    rows = []
    if name == "J":
        for a in (0.04, 0.08) if held_out else (0.06, 0.10):
            for s, t in SIGNS:
                velocity = 0.03 if held_out else 0.04
                rows.append(((s * a, -s * a / 2), (t * velocity, -t * velocity / 2)))
    elif name == "S":
        magnitudes = (0.00005, 0.000175) if held_out else (0.00010, 0.00015)
        velocities = (0.003, 0.005) if held_out else (0.002, 0.004)
        for a, velocity in zip(magnitudes, velocities, strict=True):
            for s, t in SIGNS:
                rows.append(((item.q_target[0] + s * a,), (t * velocity,)))
    else:
        for a in (0.075, 0.125) if held_out else (0.10, 0.15):
            for s, t in SIGNS:
                q = tuple(
                    ref + s * offset
                    for ref, offset in zip(item.q_target, (a, -0.8 * a, 0.6 * a), strict=True)
                )
                v = tuple(
                    t * rate for rate in ((0.04, -0.03, 0.02) if held_out else (0.05, -0.04, 0.03))
                )
                rows.append((q, v))
    return tuple(rows)


def require_cuda():
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("N4a requires qualified CUDA Torch; no fallback")
    torch.cuda.set_device(0)
    return torch


def seed_all(seed):
    import random

    torch = require_cuda()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def make_environment(name, *, count=8, held_out=False, scale=False):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState, MuJoCoRuntime
    from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec, MuJoCoTyxVectorEnv

    item = task(name)
    if (
        type(count) is not int
        or count not in SCALE_WORLDS
        or (not scale and count != 8)
        or (held_out and (count != 8 or scale))
    ):
        raise ValueError("world count/split differs from frozen protocol")
    require_cuda()
    source = compile_mujoco_xml(item.xml, require_asset=False)
    model = source.fresh_native_model()
    width = len(item.q_target)
    if (model.nq, model.nv, model.nu) != (width, width, width):
        raise ValueError("compiled native task dimensions changed")
    if name == "S" and not np.isclose(model.body_invweight0[1, 0], 1 / 6, rtol=0, atol=1e-12):
        raise ValueError("support inverse-weight differs from predeclared native derivation")
    keys = tuple(5 + 36 * i for i in range(count))
    grid = templates(name, held_out=held_out)
    initial = {key: MuJoCoInitialState(*grid[i % 8]) for i, key in enumerate(keys)}
    cost = item.cost()
    spec = MuJoCoTyxTaskSpec(
        tuple(range(width)),
        tuple(range(width)),
        scale=tuple(1 / x for x in item.q_scale) + tuple(1 / x for x in item.v_scale),
        bias=tuple(-q / x for q, x in zip(item.q_target, item.q_scale, strict=True))
        + (0.0,) * width,
        target=item.q_target + (0.0,) * width,
        weight=tuple(1 / (width * x * x) for x in item.q_scale)
        + tuple(0.01 / (width * x * x) for x in item.v_scale),
        lower=item.q_lower + (-item.v_bound,) * width,
        upper=item.q_upper + (item.v_bound,) * width,
    )
    actions = MuJoCoTyxActionSpec(
        tuple(range(width)), scale=[item.action_scale] * width, bias=[0.0] * width
    )
    horizons = tuple(64 + 8 * (i % 8) for i in range(count)) if scale else (HORIZON,) * count
    with ExitStack() as owners:
        runtime = owners.enter_context(
            MuJoCoRuntime(source, mode="cuda", world_keys=keys, initial_states=initial)
        )
        env = owners.enter_context(
            MuJoCoTyxVectorEnv(runtime, spec, actions, h=item.h, horizons=horizons)
        )
        # Public example metadata; no physical state is read to construct policy observations.
        env.n4_cost = cost
        yield env
