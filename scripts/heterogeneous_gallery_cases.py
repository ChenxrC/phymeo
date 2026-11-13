"""Fixed G2b-A gallery declarations; no simulation or product import at module load.

Thirty-two genuinely different joint graphs and/or declared physical parameters.
Rendering may place each independent world in a tile; it must not change physics.
"""
from dataclasses import dataclass

H = 1 / 128
STEPS = 256
SHORT_STEPS = 12
DEVICE_PAYLOAD_LIMIT = 256 * 1024 * 1024


@dataclass(frozen=True)
class GalleryCase:
    index: int
    family: str
    name: str
    kinds: tuple[str, ...]
    parents: tuple[int, ...]

    @property
    def floating(self):
        return self.family.startswith("free")

    @property
    def world(self):
        return 11 + 13 * self.index

    @property
    def key(self):
        return f"gallery-{self.index:02d}-{self.name}"


# R=hinge, P=slide, B=ball. Parent -1 is fixed world or the true FREE root.
_GRAPHS = (
    ("fixed-scalar", (
        ("pendulum", "R", (-1,)),
        ("double", "RR", (-1, 0)),
        ("hinge-slider", "RP", (-1, 0)),
        ("slider-hinge", "PR", (-1, 0)),
        ("telescopic-elbow", "RPR", (-1, 0, 1)),
        ("hinge-fork", "RRR", (-1, 0, 0)),
        ("four-stage", "RPRP", (-1, 0, 1, 2)),
        ("eight-chain", "RRRRRRRR", (-1, 0, 1, 2, 3, 4, 5, 6)),
    )),
    ("fixed-ball", (
        ("spherical-rotor", "B", (-1,)),
        ("ball-hinge", "BR", (-1, 0)),
        ("hinge-ball", "RB", (-1, 0)),
        ("ball-slider", "BP", (-1, 0)),
        ("slider-ball", "PB", (-1, 0)),
        ("two-balls", "BB", (-1, 0)),
        ("ball-elbow-slide", "BRP", (-1, 0, 1)),
        ("spherical-fork", "RBP", (-1, 0, 0)),
    )),
    ("free-scalar", (
        ("space-hinge", "R", (-1,)),
        ("space-slider", "P", (-1,)),
        ("space-double", "RR", (-1, 0)),
        ("space-elbow", "RP", (-1, 0)),
        ("space-slider-elbow", "PR", (-1, 0)),
        ("space-two-sliders", "PP", (-1, 0)),
        ("space-fork", "RP", (-1, -1)),
        ("space-binary-seven", "RRPRPRP", (-1, 0, 0, 1, 1, 2, 2)),
    )),
    ("free-ball", (
        ("space-ball", "B", (-1,)),
        ("space-ball-hinge", "BR", (-1, 0)),
        ("space-hinge-ball", "RB", (-1, 0)),
        ("space-ball-slider", "BP", (-1, 0)),
        ("space-slider-ball", "PB", (-1, 0)),
        ("space-two-balls", "BB", (-1, 0)),
        ("space-ball-chain", "BRP", (-1, 0, 1)),
        ("space-double-ball-fork", "BB", (-1, -1)),
    )),
)
CASES = tuple(
    GalleryCase(8 * group + slot, family, name, tuple(kinds), parents)
    for group, (family, rows) in enumerate(_GRAPHS)
    for slot, (name, kinds, parents) in enumerate(rows)
)


def xml(case):
    """Generate only predeclared constants, never depend on simulation output."""
    axes = ("0 1 0", "1 0 0", "0 0 1", "1 2 -1")
    colors = (".2 .65 .95 1", ".95 .55 .2 1", ".35 .8 .4 1", ".75 .4 .95 1")

    def body(index):
        kind = case.kinds[index]
        length = .14 + .01 * ((case.index + index) % 5)
        mass = .5 + (case.index + index) % 5 / 16
        side = .07 if index % 2 else -.07
        pos = f"{length:.8g} {side:.8g} .04" if case.parents[index] >= 0 else "0 0 .4"
        scalar = "" if kind == "B" else f' axis="{axes[(case.index + index) % 4]}"'
        joint = f'<joint name="j{index}" type="{dict(R="hinge", P="slide", B="ball")[kind]}" damping=".12" armature=".04"{scalar}/>'
        children = "".join(body(child) for child, parent in enumerate(case.parents) if parent == index)
        return (
            f'<body name="link{index}" pos="{pos}">'
            + joint
            + f'<inertial pos="{length / 2:.8g} .015 -.01" mass="{mass:.8g}" fullinertia=".04 .055 .065 .002 -.001 .003"/>'
            + f'<geom name="link{index}-visual" type="capsule" fromto="0 0 0 {length:.8g} 0 0" size=".025" contype="0" conaffinity="0" rgba="{colors[case.index // 8]}"/>'
            + children + '</body>'
        )

    content = "".join(body(i) for i, parent in enumerate(case.parents) if parent == -1)
    if case.floating:
        content = (
            '<body name="free-base" pos="0 0 .8"><freejoint name="root"/>'
            '<inertial pos=".04 -.025 .03" mass="2" fullinertia=".13 .19 .23 .005 -.003 .004"/>'
            '<geom name="base-visual" type="box" size=".12 .08 .06" contype="0" conaffinity="0" rgba=".7 .75 .8 1"/>'
            + content + '</body>'
        )
    motors = []
    for index, kind in enumerate(case.kinds):
        gear = ".25 -.125 .0625" if kind == "B" else (".5" if index % 2 else "-.75")
        motors.append(f'<general name="motor{index}" joint="j{index}" gear="{gear}" dyntype="none" gaintype="fixed" gainprm="1" ctrllimited="true" ctrlrange="-.4 .4" forcelimited="true" forcerange="-.35 .35"/>')
    gravity = "0 0 0" if case.floating else ".2 -.3 -9.81"
    return (
        f'<mujoco model="{case.key}"><compiler angle="radian" fusestatic="false"/>'
        f'<option timestep=".0078125" gravity="{gravity}" integrator="Euler"/>'
        '<visual><global offwidth="480" offheight="360"/></visual>'
        f'<worldbody>{content}</worldbody><actuator>{"".join(motors)}</actuator></mujoco>'
    )
