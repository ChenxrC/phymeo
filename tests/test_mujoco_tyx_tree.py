"""Independent fixed-root MuJoCo-to-TYX rotational-tree cases (UNEXECUTED).

The constants/cases are declared before product execution. CPU MuJoCo 3.3.7
is the numerical oracle; analytic one-body checks do not reuse product arrays.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.001
# SI entries: hinge coordinates/rates are radians, prismatic are metres.
# M/J columns have the native corresponding mixed SI units; relative terms
# compare like entries only. These are engineering FP32 gates, not FP64 equality.
TOL = {
    "body_position": (2e-5, 3e-5),
    "body_rotation": (2e-5, 3e-5),
    "body_com": (2e-5, 3e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (2e-4, 3e-5),
    "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4),
    "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
    "integration_acceleration": (5e-4, 8e-4),
}
SHORT_Q_ATOL = 5e-5
SHORT_V_ATOL = 3e-4
TIME_ATOL = 2e-6


@dataclass(frozen=True)
class TreeCase:
    name: str
    xml: str
    offsets: tuple
    velocity: tuple
    wrench_body: str
    steps: int = 24


def _joint(name, *, kind="hinge", axis="0 1 0", anchor="0 0 0", ref="0", extra=""):
    return (
        f'<joint name="{name}" type="{kind}" axis="{axis}" pos="{anchor}" '
        f'ref="{ref}" armature=".04" damping=".12" {extra}/>'
    )


def _body(name, joint="", child="", *, pos=".1 .02 .04", quat="1 0 0 0",
          mass="1.1", com=".04 -.03 .02", inertia=".02 .025 .03 .001 -.0005 .0008"):
    return f'''<body name="{name}" pos="{pos}" quat="{quat}">{joint}
      <inertial mass="{mass}" pos="{com}" fullinertia="{inertia}"/>
      <geom name="{name}-visual" type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>
      {child}</body>'''


def _model(name, body, joints, *, gravity=".2 -.3 -9.81", gears=None, extra=""):
    gears = tuple(1 if index % 2 else 2 for index in range(len(joints))) if gears is None else gears
    motors = ''.join(
        f'<motor name="motor-{index}" joint="{joint}" gear="{gear}" '
        'ctrllimited="true" ctrlrange="-.4 .4" forcelimited="true" forcerange="-.35 .35"/>'
        for index, (joint, gear) in enumerate(zip(joints, gears, strict=True))
    )
    return f'''<mujoco model="{name}"><compiler angle="radian" fusestatic="false"/>
      <option timestep=".001" gravity="{gravity}" integrator="Euler"/>
      <worldbody>{body}</worldbody><actuator>{motors}</actuator>{extra}</mujoco>'''


def _chain(count):
    child = ''
    axes = ("0 1 0", "1 0 0", "0 0 1")
    for index in reversed(range(count)):
        joint = _joint(f"j{index}", axis=axes[index % 3], anchor=".01 -.02 .015",
                       ref=str(.025 * (index + 1)))
        child = _body(f"link{index}", joint, child,
                      quat=".9659258262890683 0 0 .2588190451025207")
    return _model(f"chain-{count}", child, tuple(f"j{i}" for i in range(count)))


PENDULUM_XML = '''<mujoco model="eccentric-pendulum"><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler"/>
<worldbody><body name="pendulum" pos=".1 -.08 .25">
<joint name="j0" type="hinge" axis="0 1 0" ref=".125" armature=".11" damping=".17"/>
<inertial mass="2" pos=".4 0 0" diaginertia=".03 .07 .08"/>
<geom name="shape" type="sphere" size=".05" contype="0" conaffinity="0"/>
</body></worldbody><actuator><motor name="motor-0" joint="j0" gear="2"
 ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>'''

DOUBLE_XML = _model(
    "double-pendulum",
    _body("upper", _joint("j0", ref=".05"),
          _body("lower", _joint("j1", axis="0 1 0", ref="-.07"),
                pos=".45 0 .03", com=".2 .01 -.03", mass=".8"),
          pos="0 0 .4", com=".22 -.01 .02", mass="1.4"),
    ("j0", "j1"),
)
MIXED_XML = _model(
    "mixed-same-body-rp",
    _body("carriage", _joint("j0", axis="0 0 1", anchor=".02 -.01 .03", ref=".08")
          + _joint("j1", kind="slide", axis="1 0 0", anchor="-.01 .02 0", ref="-.03"),
          _body("tip", _joint("j2", axis="1 0 0", anchor=".01 .01 -.02", ref=".04"), mass=".6")),
    ("j0", "j1", "j2"), gears=(2, -1, .5),
)
FORK_XML = _model(
    "shared-ancestor-fork",
    _body("shoulder", _joint("j0", axis="0 0 1", ref=".03"),
          _body("left", _joint("j1", axis="1 0 0", ref="-.04"), pos=".3 .18 .02", mass=".8")
          + _body("right", _joint("j2", axis="0 1 0", ref=".06"), pos="-.2 -.14 .1", mass="1.5"),
          pos="0 0 .3", mass="2"),
    ("j0", "j1", "j2"),
)
WELDED_XML = _model(
    "welded-tool-load",
    _body("shoulder", _joint("j0", ref=".04"),
          _body("wrist", _joint("j1", axis="0 0 1", anchor=".02 -.01 .01", ref="-.06"),
                _body("tool", pos=".14 -.07 .11", quat=".9238795325112867 .3826834323650898 0 0",
                      mass=".7", com=".04 .02 -.03"), mass=".9"), mass="1.6")
    + _body("fixed-marker", pos="-.2 .1 .4", mass=".4"),
    ("j0", "j1"),
)

CASES = (
    TreeCase("pendulum", PENDULUM_XML, (.295,), (-.6,), "pendulum"),
    TreeCase("double", DOUBLE_XML, (.21, -.28), (.7, -.5), "lower", 48),
    TreeCase("mixed", MIXED_XML, (.18, .035, -.22), (-.4, .12, .55), "tip", 48),
    TreeCase("fork", FORK_XML, (.2, -.17, .25), (.45, -.35, .5), "right"),
    TreeCase("seven", _chain(7), (.11, -.13, .15, -.17, .19, -.21, .23),
             (.2, -.25, .3, -.35, .4, -.45, .5), "link6"),
    TreeCase("welded", WELDED_XML, (.24, -.19), (.65, -.45), "tool", 48),
    TreeCase("eight", _chain(8), (.11, -.13, .15, -.17, .19, -.21, .23, -.25),
             (.2, -.25, .3, -.35, .4, -.45, .5, -.55), "link7"),
)
