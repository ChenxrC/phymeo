"""A finite convex tool is attached to an actual reduced-coordinate link, with a material sheet."""

import json
from contextlib import ExitStack
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.core.scene import fingerprint

H = 1 / 128
RHO = 1 / 64
BOX = np.array(list(product((-0.25, 0.25), (-0.25, 0.25), (-0.125, 0.125))))
TOOL_IDS = (82, 13, 21, 35, 61, 56, 72, 3)


def case(
    kind,
    *,
    speed=0.0,
    shift=0.0,
    two_faces=False,
    swept=False,
    operator=None,
    friction=0.0,
    two_tools=False,
):
    from phymeo.solvers.cloth_tool_contact import LinkHullTool

    multiple = kind == "double"
    count = 2 if multiple else 1
    joint = "prismatic" if kind == "rail" else "revolute"
    pivot_z = 0.0 if swept else 5 / 32 + RHO
    robot = ph.RobotModel(
        np.array([-1, 0] if multiple else [-1]),
        (joint,) * count,
        np.array([[0.0, 0, pivot_z]] + ([[0, 0.375, 0]] if multiple else [])),
        np.tile([0.0, 0, 0, 1], (count, 1)),
        np.array(
            [[0.0, 0, 1]]
            if joint == "prismatic"
            else [[0.0, 1, 0]] + ([[0, 0, 1]] if multiple else [])
        ),
        np.full(count, 2.0),
        np.tile([0.125, 0, 0], (count, 1)),
        np.full((count, 3), 0.5),
        np.zeros(3),
    )
    center_x = 0 if kind == "rail" else 0.375
    x = np.array(
        [[center_x + shift, -1.0, 0], [center_x + shift + 1, 0, 0], [center_x + shift, 1, 0]]
        + ([[center_x + shift - 1.0, 0, 0]] if two_faces else [])
    )
    vertices = BOX + [center_x, 0, 0]
    if swept:
        x = np.array([[0.375, -0.25, 0.0], [0.625, -0.25, 0], [0.5, 0.25, 0]])
        x[:, 0] += shift
        vertices = BOX / 8 + [0, 0, 0.5]
    cloth = ph.ParticleModel(1 / np.array([1.0, 1.5, 2.0, 1.25][: len(x)]), np.zeros(3))
    scene = ph.SceneModel(
        (ph.DomainBinding("sheet", "particles", cloth), ph.DomainBinding("arm", "robot", robot))
    )
    faces = [ph.TrianglePrismSurface("right", "sheet", (0, 1, 2), half_thickness_m=RHO)]
    if two_faces:
        faces.append(ph.TrianglePrismSurface("left", "sheet", (0, 2, 3), half_thickness_m=RHO))
    edges = np.array([[0, 1], [1, 2], [2, 0]] + ([[2, 3], [3, 0]] if two_faces else []))
    rest = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
    material = ClothMaterial(ph.DistanceConstraints(edges, rest, np.full(len(edges), 0.25)))
    tool = LinkHullTool(
        "jaw",
        "arm",
        count - 1,
        vertices,
        ph.NormalLayer(1 / 16, 128, friction=friction),
        vertex_ids=TOOL_IDS,
    )
    tools = [tool]
    if two_tools:
        tools.append(
            LinkHullTool(
                "other", "arm", count - 1, vertices + [0, -4, 0], ph.NormalLayer(1 / 16, 128)
            )
        )
    prog = ph.CompliantClothContactProgram(
        scene, "sheet", material, faces, x, (), link_hull_tools=tools, link_hull_operator=operator
    )
    rs = ph.RobotState(np.zeros(count), np.full(count, float(speed)))
    state = ph.SceneState({"sheet": ph.ParticleState(x, np.zeros_like(x)), "arm": rs})
    inp = ph.SceneInputs({"sheet": ph.StepInputs.zeros(len(x)), "arm": ph.RobotInputs.zeros(count)})
    return prog, state, inp


def run(prog, state, inp, h=H):
    with prog.stepper(ph.StepSettings(substeps=1, record_substeps=True)) as runtime:
        return runtime.step(state, inp, h)


@pytest.mark.parametrize("kind,effort,torque", [("rail", 4.0, -0.0), ("wrist", -2.0, -1.5)])
def test_actual_articulated_tool_uses_material_patch_and_its_link_jacobian(
    kind, effort, torque, tmp_path
):
    prog, s, inputs = case(kind, speed=0.125, two_tools=True)
    obs = prog.observe_hull_tools(s)
    (pair,) = obs["contacts"]
    assert pair["tool"] == "jaw" and pair["face"] == "right" and len(pair["sites"]) == 4
    assert pair["middle_gap_m"] == pytest.approx(3 / 64) and pair["gap_m"] == pytest.approx(1 / 32)
    assert pair["force_n"] == pytest.approx(4)
    np.testing.assert_allclose(obs["efforts"]["arm"], [effort], atol=2e-12)
    np.testing.assert_allclose(pair["torque_about_com_nm"], [0, torque, 0], atol=2e-12)
    np.testing.assert_allclose(obs["nodal_force_n"][:, 2], [-1.75, -0.5, -1.75], atol=2e-12)
    # For the revolute asset J_v at COM contributes -0.5 N m in addition to its -1.5 N m moment.
    nodal_power = float(np.sum(obs["nodal_force_n"] * s.domains["sheet"].velocity))
    assert nodal_power + effort * 0.125 == pytest.approx(
        sum(row["power_w"] for row in obs["contacts"]), abs=2e-12
    )
    for site in pair["sites"]:
        assert np.linalg.norm(
            np.array(site["tool_point_m"]) - site["cloth_surface_point_m"]
        ) == pytest.approx(1 / 32)
        assert sum(v for _, v in site["cloth_weights"]) == pytest.approx(1)

    history = []
    initial = s
    for k in range(8):
        before = s
        o = run(prog, before, inputs)
        assert o.report.accepted, o.report.reason
        s = o.state
        d = o.report.substeps[0].diagnostics
        middle = d["hull_tools"]
        assert d["force_sample"] == "substep-midpoint"
        qeff = middle["efforts"]["arm"][0]
        inertia = 2.0 if kind == "rail" else 0.5 + 2 * (1 / 8) ** 2
        assert inertia * (s.domains["arm"].v[0] - before.domains["arm"].v[0]) == pytest.approx(
            H * qeff, abs=3e-12
        )
        assert s.domains["arm"].q[0] - before.domains["arm"].q[0] == pytest.approx(
            H * (s.domains["arm"].v[0] + before.domains["arm"].v[0]) / 2, abs=2e-13
        )
        mass = 1 / prog.cloth.inverse_mass[:, None]
        dp = mass * (s.domains["sheet"].velocity - before.domains["sheet"].velocity)
        np.testing.assert_allclose(
            dp, H * (middle["nodal_force_n"] + d["material_force_n"]), atol=3e-12
        )
        np.testing.assert_allclose(
            middle["nodal_force_n"].sum(0)
            + sum(np.array(p["force_on_tool_n"]) for p in middle["contacts"]),
            0,
            atol=3e-12,
        )
        history.append(
            dict(
                step=k + 1,
                q=s.domains["arm"].q.tolist(),
                v=s.domains["arm"].v.tolist(),
                effort=qeff,
                force=[p["force_n"] for p in middle["contacts"]],
                sites=[len(p["sites"]) for p in middle["contacts"]],
                tool_torque=[
                    np.asarray(p["torque_about_com_nm"]).tolist() for p in middle["contacts"]
                ],
                node_x=s.domains["sheet"].position.tolist(),
            )
        )
    assert s.domains["arm"].q[0] != initial.domains["arm"].q[0]
    assert np.ptp(s.domains["sheet"].position[:, 2]) > 1e-4
    (tmp_path / "articulated-tool.json").write_text(
        json.dumps(dict(asset=kind, dt=H, history=history), allow_nan=False), encoding="utf8"
    )


def test_offset_finite_edge_coating_and_two_shared_free_faces():
    prog, s, _ = case("rail", shift=0.25 + 3 / 128)
    obs = prog.observe_hull_tools(s)
    (row,) = obs["contacts"]
    d = np.hypot(3 / 128, 3 / 64)
    assert len(row["sites"]) == 2 and row["gap_m"] == pytest.approx(d - RHO)
    assert row["force_n"] == pytest.approx(128 * (1 / 16 + RHO - d))
    assert np.linalg.norm(row["force_on_tool_n"]) == pytest.approx(row["force_n"])

    full, state, inputs = case("rail", two_faces=True)
    state.domains["sheet"].velocity[:, 2] = [1 / 32, -1 / 16, -1 / 32, 1 / 16]
    for _ in range(6):
        out = run(full, state, inputs)
        assert out.report.accepted, out.report.reason
        d = out.report.substeps[0].diagnostics["hull_tools"]
        assert {c["face"] for c in d["contacts"]} == {"left", "right"}
        force = sum(c["force_on_tool_n"][2] for c in d["contacts"])
        assert 2 * (out.state.domains["arm"].v[0] - state.domains["arm"].v[0]) == pytest.approx(
            H * force, abs=2e-12
        )
        state = out.state
    assert np.ptp(state.domains["sheet"].position[:, 2]) > 0.001


def test_a_quarter_turn_passes_through_a_finite_face_while_all_three_force_samples_are_clear():
    prog, state, inputs = case("wrist", speed=2 * np.pi, swept=True)
    # A hull centered at (0,0,1/2) turns about y. Its center is (1/2,0,0) at pi/2.
    assert not prog.observe_hull_tools(state)["contacts"]
    nominal = {}
    for q in (np.pi, 2 * np.pi):
        trial = ph.SceneState(
            {
                "sheet": state.domains["sheet"],
                "arm": ph.RobotState(np.array([q]), np.array([2 * np.pi])),
            }
        )
        nominal[q] = prog.observe_hull_tools(trial)["contacts"]
    assert not any(nominal.values())
    original = fingerprint(state)
    out = run(prog, state, inputs, 1.0)
    assert not out.report.accepted and "clearance" in out.report.reason
    assert fingerprint(out.state) == original and fingerprint(state) == original
    far, other, inputs = case("wrist", speed=2 * np.pi, swept=True, shift=2.0)
    assert run(far, other, inputs, 1.0).report.accepted


def test_thick_core_rejection_and_clear_compound_motion_preserve_input():
    prog, s, inp = case("rail")
    # Middle-face separation is still positive. The material coating overlaps the tool.
    s.domains["arm"].q[0] = -5 / 128
    assert 0 < 3 / 64 - 5 / 128 < RHO
    with pytest.raises(ph.SolverFailure, match="thickness|clearance"):
        prog.observe_hull_tools(s)
    sig = fingerprint(s)
    assert not run(prog, s, inp).report.accepted and fingerprint(s) == sig
    multi, old, inp = case("double", speed=0.1, shift=3)
    sig = fingerprint(old)
    clear = run(multi, old, inp)
    assert clear.report.accepted, clear.report.reason
    assert fingerprint(old) == sig and fingerprint(clear.state) != sig


@pytest.mark.tyx
def test_midpoint_uses_one_native_material_batch_for_the_real_arm_and_sheet(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        prog, s, inp = case("rail", operator=layer)
        if tyx_mode == "cuda":
            import tyx

            prior = tyx.runtime_stats().cuda_kernel_launches
        out = run(prog, s, inp)
        assert out.report.accepted, out.report.reason
        middle = out.report.substeps[0].diagnostics["hull_tools"]
        assert layer.stats()["completed_launches"] == 2 and layer.stats()["sites"] == 8
        n = tyx.runtime_stats().cuda_kernel_launches - prior if tyx_mode == "cuda" else None
        if n is not None:
            assert n == 2
        assert out.state.domains["arm"].v[0] == pytest.approx(
            H * middle["efforts"]["arm"][0] / 2, abs=1e-12
        )
        (tmp_path / "tool-material-backend.json").write_text(
            json.dumps(
                dict(
                    mode=tyx_mode,
                    kernels=n,
                    layer=layer.stats(),
                    q=out.state.domains["arm"].q.tolist(),
                    v=out.state.domains["arm"].v.tolist(),
                    cloth=out.state.domains["sheet"].position.tolist(),
                    midpoint_effort=middle["efforts"]["arm"].tolist(),
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_tool_registration_owns_shape_material_ids_and_rejects_invalid_geometry():
    from phymeo.solvers.cloth_tool_contact import LinkHullTool

    vertices = BOX.copy()
    law = ph.NormalLayer(1 / 16, 128)
    tool = LinkHullTool("jaw", "arm", 0, vertices, law, vertex_ids=TOOL_IDS)
    vertices[0] = 99
    np.testing.assert_array_equal(tool.local_vertices[0], BOX[0])
    assert tool.vertex_ids == TOOL_IDS
    with pytest.raises(ValueError):
        LinkHullTool("bad", "arm", True, BOX, law)
    with pytest.raises(ValueError, match="match|distinct"):
        LinkHullTool("bad", "arm", 0, BOX, law, vertex_ids=(1, 2, 3, 4))
    collapsed = BOX.copy()
    collapsed[:, 2] = 0
    with pytest.raises(ValueError, match="convex"):
        LinkHullTool("flat", "arm", 0, collapsed, law)
