"""Parent shared admission: full solids bound broad phase, never replace skin."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends._resident_floating_binding import compile_floating_tree_binding
from phymeo.backends._resident_floating_tree import compile_floating_tree_world
from phymeo.backends._resident_tree_binding import _shape_radius_upper, compile_tree_binding
from phymeo.backends._resident_tree_world import compile_tree_world
from phymeo.dynamics.floating_robot import FloatingRobotState

XML = """<robot name="curved_binding">
<link name="base"><inertial><mass value="2"/>
<inertia ixx="0.2" iyy="0.3" izz="0.4" ixy="0" ixz="0" iyz="0"/>
</inertial></link>
<link name="arm"><inertial><mass value="1"/>
<inertia ixx="0.05" iyy="0.075" izz="0.1" ixy="0" ixz="0" iyz="0"/>
</inertial><collision name="curved"><origin xyz="0.125 0 0" rpy="0 0.3 0"/>
<geometry><cylinder radius="0.125" length="0.5"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="8" velocity="2"/></joint>
</robot>"""


def setup_asset(tmp_path, floating):
    path = tmp_path / "curved.urdf"
    path.write_text(XML, encoding="utf-8")
    load = ph.load_floating_urdf if floating else ph.load_urdf
    asset = load(path, gravity=np.zeros(3), collapse_fixed_joints=False)
    gains = dict(kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.ones(1), delay_steps=0)
    if floating:
        state = FloatingRobotState([0, 0, 0, 0, 0, 0, 1, 0.125], np.zeros(7))
        world = ph.FloatingRobotTree(asset.model, state, **gains)
        spec = compile_floating_tree_world(world)
        return asset, spec, ph.ResidentFloatingTreeBinding, compile_floating_tree_binding, 1
    program = ph.DrivenRobotProgram(asset.model, **gains)
    world = ph.FixedRobotTree(program, program.initial_state(ph.RobotState([0.125], [0.0])))
    return asset, compile_tree_world(world), ph.ResidentTreeBinding, compile_tree_binding, 0


@pytest.mark.parametrize("floating", (False, True))
@pytest.mark.parametrize(
    "kind,dims",
    (
        ("cylinder", (1 / 8, 1 / 4)),
        ("cone", (3 / 16, 1 / 4)),
        ("ellipsoid", (1 / 4, 3 / 16, 1 / 8)),
    ),
)
def test_complete_solid_broad_bound_preserves_single_material(tmp_path, floating, kind, dims):
    _, spec, binding_type, compile_binding, group = setup_asset(tmp_path, floating)
    key = ph.CollisionKey(17, 23, 5)
    shape = ph.CollisionShape(key, kind, (71,), curved=ph.CurvedSolid(kind, dims, (0, 0.6, 0, 0.8)))
    binding = binding_type(key, 17, group, [[1 / 32, -1 / 64, 1 / 128]], topology=spec.topology)
    compiled = compile_binding(shape, binding, spec)
    bound = max(map(Fraction, dims)) if kind == "ellipsoid" else sum(map(Fraction, dims))
    assert Fraction(float(compiled.geometry[0])) >= bound
    assert float(compiled.geometry[0]) <= float(bound) + 2**-22
    assert shape.radius == 0 and shape.vertex_ids == (71,)
    assert compiled.points_m.shape == (1, 3)
    np.testing.assert_array_equal(compiled.points_m, binding.local_points_m)


def test_radius_addition_cannot_lose_tiny_positive_height_or_overflow():
    key = ph.CollisionKey(17, 23, 5)
    shape = ph.CollisionShape(
        key, "cylinder", (71,), curved=ph.CurvedSolid("cylinder", (1.0, 2**-100))
    )
    assert Fraction(float(_shape_radius_upper(shape))) >= Fraction(1) + Fraction(1, 2**100)
    huge = replace(shape, curved=ph.CurvedSolid("cylinder", (1e308, 1e308)))
    with pytest.raises(ValueError, match="finite"):
        _shape_radius_upper(huge)


@pytest.mark.parametrize("floating", (False, True))
def test_nonfolded_asset_keeps_original_orientation_and_source_proof(tmp_path, floating):
    asset, spec, binding_type, compile_binding, _ = setup_asset(tmp_path, floating)
    original = asset.collisions[0].shape
    key = ph.CollisionKey(17, 23, 5)
    binding = binding_type.from_asset(asset, original.key, key=key, world=17)
    shape = replace(original, key=key)
    assert shape.curved.orientation_xyzw != (0, 0, 0, 1)
    compile_binding(shape, binding, spec)
    changed = replace(shape, curved=replace(shape.curved, dimensions_m=(1 / 4, 1 / 4)))
    with pytest.raises(ValueError, match="source shape"):
        compile_binding(changed, binding, spec)


@pytest.mark.parametrize("floating", (False, True))
def test_folded_curved_source_refuses_unrepresented_site_rotation(tmp_path, floating):
    fixture = Path(__file__).parent / "fixtures/urdf/floating_fixed_probe.urdf"
    xml = fixture.read_text(encoding="utf-8")
    literal = '<box size="0.125 0.25 0.125"/>'
    assert xml.count(literal) == 1
    path = tmp_path / "folded-curved.urdf"
    path.write_text(
        xml.replace(literal, '<cylinder radius="0.125" length="0.5"/>'), encoding="utf-8"
    )
    asset = (ph.load_floating_urdf if floating else ph.load_urdf)(
        path, gravity=np.zeros(3), collapse_fixed_joints=True
    )
    collider = next(item for item in asset.collisions if item.shape.kind == "cylinder")
    binding_type = ph.ResidentFloatingTreeBinding if floating else ph.ResidentTreeBinding
    with pytest.raises(ValueError, match="folded curved source orientation"):
        binding_type.from_asset(asset, collider.shape.key, key=ph.CollisionKey(17, 23, 5), world=17)
