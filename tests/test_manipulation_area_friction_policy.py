"""Real manipulation VecEnv rollout with previous accepted area forces as actor input."""

import json
from contextlib import ExitStack
from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from experiments.area_pressure.moving_jaw_grasp import JAWS, H, _declarations
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh
from phymeo.training.area_friction_policy import FIELDS, ROLES, AreaFrictionPolicyInput
from phymeo.training.area_pressure_success import AreaPressureRoleBinding
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import Goal, Primitive, Scene, Target
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec


def _env(stack, *, mu=0.6):
    specs, worlds, assets = _declarations()
    model = specs[0]
    xml = ET.fromstring(model.compilation.source.primary_xml)
    for body, geom in (("object", "object-0"), ("left", "left-pad"), ("right", "right-pad")):
        xml.find(f".//body[@name='{body}']/geom").set("name", geom)
    # The two jaws and the table each own their finite area contact surface.
    table = ET.SubElement(xml.find("worldbody"), "body", name="table", pos="0 0 -.017")
    ET.SubElement(table, "geom", name="support", type="box", size=".04 .04 .005")
    jaws = assets[model.key]
    identity = replace(jaws[0].identity, rigid_body="table", rigid_material="table-top")
    face = RigidPressureMesh(
        ("sw", "se", "ne", "nw"),
        ((-0.04, -0.04, 0.005), (0.04, -0.04, 0.005), (0.04, 0.04, 0.005), (-0.04, 0.04, 0.005)),
        ((0, 1, 2), (0, 2, 3)),
        ("lower", "upper"),
    )
    asset = (*jaws, BodyAreaPressureAsset(identity, jaws[0].pressure_local, face))
    source = ET.tostring(xml, encoding="unicode")
    model = replace(
        model,
        compilation=compile_mujoco_xml(source, require_asset=False),
        area_pressure_pairs=tuple(
            (a.identity.compliant_body, a.identity.rigid_body) for a in asset
        ),
        area_friction_laws=tuple(
            replace(model.area_friction_laws[0], identity=a.identity, mu=mu) for a in asset
        ),
    )
    worlds = tuple(replace(w, model_key=model.key) for w in worlds)
    goal = Goal(
        "grasp", "object", "center", ("夹起方块", "抬起方块"), ("Lift the box", "Raise the box")
    )
    task = Scene(
        "fixed-gripper",
        "",
        source,
        (Primitive("box", (0.02, 0.02, 0.01)),),
        0.30,
        (0, 0, 0),
        (Target("center", "中央", "center", (0, 0, 0), 0),),
        (goal,),
        0,
        tuple(j + "-slide" for j in JAWS),
        tuple(j + "-close" for j in JAWS),
        ("left-pad", "right-pad"),
        timestep_s=H,
        support_top_m=-0.012,
    )
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch((model,), worlds, mode="cuda"))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {model.key: asset})
    keys = batch.world_keys
    loads = {41: 0.22, 7: 0.11}
    env = stack.enter_context(
        HeterogeneousManipulationEnv(
            batch,
            dict.fromkeys(keys, task),
            goals=dict.fromkeys(keys, goal),
            action_specs={
                k: MuJoCoTyxActionSpec(
                    (0, 1), (2,), scale=(1.0, 1.0, 0.0), bias=(0.0, 0.0, loads[k])
                )
                for k in keys
            },
            horizons={41: 19, 7: 100},
            area_scene=scene,
            area_roles={
                model.key: tuple(
                    AreaPressureRoleBinding(a.identity, "A", ROLES[i]) for i, a in enumerate(asset)
                )
            },
        )
    )
    return env


def _host(tensor):
    return tensor.detach().cpu().numpy()


@pytest.mark.cuda
def test_reader_cannot_restore_empty_cursor_from_different_friction_law():
    with ExitStack() as stack:
        first = _env(stack, mu=0.6)._new_reader().checkpoint()
    with ExitStack() as stack:
        env = _env(stack, mu=0.3)
        with pytest.raises(ValueError, match="cursor source binding differs"):
            env._new_reader().restore(first, frame=env.current_frame)


def test_mixed_role_projection_tracks_object_side_and_history():
    from types import SimpleNamespace

    import torch

    def role(name, side):
        return SimpleNamespace(role=name, object_side=side)

    binding = dict(
        world_keys=(9, 5, 8),
        source={"worlds": ((5, "normal", 1, 0), (9, "friction", 0, 0), (8, "native", 2, 0))},
        area_pair_names={"friction": (0, 1, 2), "normal": (0,), "native": ()},
        area_laws={"friction": (1,), "normal": None, "native": None},
    )
    encoder = AreaFrictionPolicyInput(
        binding,
        {
            "friction": (role("other", "A"), role("other", "B"), role("other", None)),
            "normal": (role("support", "A"),),
        },
        "cpu",
    )
    ids = torch.zeros(3, 4, 6, dtype=torch.int32)
    ids[0, :3, 0] = 1
    ids[0, :3, 1] = torch.arange(3)
    ids[1, 0, 0] = 1
    data = torch.zeros(3, 4, 23)
    data[0, :3, 8] = torch.tensor([2.0, 3.0, 4.0])
    data[0, :3, 15] = torch.tensor([1.0, 0.5, 7.0])
    meta = torch.zeros(3, 4, dtype=torch.int32)
    meta[:, 0] = torch.tensor([2, 1, 0])
    result = encoder.encode(dict(area_ids=ids, area_data=data, area_meta=meta))
    torch.testing.assert_close(result[0, 3], torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0, 0.5, 0.0, 0.0]))
    torch.testing.assert_close(result[1, 2], torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.count_nonzero(result[2]) == 0


@pytest.mark.cuda
def test_shared_actor_sees_load_release_and_selected_reset(tmp_path, monkeypatch):
    import mujoco
    import torch

    def reject_cpu(*args, **kwargs):
        pytest.fail("CUDA dynamics required")

    monkeypatch.setattr(mujoco, "mj_step", reject_cpu)
    with ExitStack() as stack:
        env = _env(stack)
        keys = env.world_keys
        reset_row = keys.index(41)
        other = keys.index(7)
        width = len(FIELDS)
        obs = env.observation
        assert env.binding["area_policy_input"][2] == list(FIELDS)
        # One shared feedback layer for both worlds. The reference jaw drive is
        # supplied separately; subtract the measured normal load and react to Ft,z.
        actor = torch.nn.Linear(1 + len(ROLES) * width, 3, bias=False, device=env.device)
        with torch.no_grad():
            actor.weight.zero_()
            for j in range(2):
                actor.weight[j, 0] = 1
                actor.weight[j, 1 + j * width + FIELDS.index("normal_load_n")] = -0.035
                actor.weight[j, 1 + j * width + FIELDS.index("tangent_z_n")] = 0.02
        history = []
        for tick in range(1, 36):
            contact = obs["area_object_contact"]
            assert contact.device == env.device and contact.dtype == torch.float32
            if tick == 1:
                assert torch.count_nonzero(contact[:, :, 2:]) == 0
            close = {41: 0.52, 7: 0.38}
            command = torch.tensor(
                [[close[k] if tick <= 23 else -1.45] for k in keys], device=env.device
            )
            actor_input = torch.cat((command, contact.flatten(1)), dim=1)
            # Record exactly what the policy consumes before the physics advance.
            with torch.no_grad():
                actions = actor(actor_input)
            transition = env.step(actions, world_keys=keys, expected_frame=env.current_frame)
            torch.testing.assert_close(transition.action, actions, atol=0, rtol=0)
            final = _host(transition.final_observation["area_object_contact"])
            returned = _host(transition.observation["area_object_contact"])
            if tick == 19:
                assert transition.truncated[reset_row] and not transition.truncated[other]
                assert np.all(final[reset_row, :2, 3:5] > 0)
                np.testing.assert_array_equal(returned[reset_row, :, 2:], 0)
                np.testing.assert_array_equal(returned[other], final[other])
            else:
                np.testing.assert_array_equal(final, returned)
            states = env.batch.integration_snapshot()
            for i, key in enumerate(keys):
                np.testing.assert_array_equal(final[i, :2, :3], 1)
                np.testing.assert_array_equal(final[i, 2, :3], 1)
                np.testing.assert_array_equal(final[i, 2, 3:], 0)
                np.testing.assert_array_equal(final[i, 3], 0)
                np.testing.assert_allclose(
                    final[i, :2, 4],
                    transition.success_frames[key].finger_normal_forces_n,
                    atol=2e-6,
                )
                if tick == 19 and i == reset_row:
                    continue  # Physics has already reset this owner's last_step.
                pairs = states[key].last_step.raw_contact_fields["area_friction"]
                for j, pair in enumerate(pairs):
                    assert pair["identity"].rigid_body == (*JAWS, "table")[j]
                    compare = (
                        bool(len(pair["sample_ids"])),
                        np.sum(pair["normal_force"]),
                        *np.asarray(pair["tangential_force"]).reshape(-1, 3).sum(axis=0),
                    )
                    np.testing.assert_allclose(final[i, j, 3:], compare, atol=2e-6, rtol=2e-6)
            if tick == 20:
                np.testing.assert_array_equal(_host(contact)[reset_row, :, 2:], 0)
                assert _host(contact)[other, :2, 4].min() > 0.1
            if tick == 18:
                # All other actor inputs held fixed: ablating the contact columns
                # changes both motor actions in each world.
                blank = actor_input.clone()
                blank[:, 1:] = 0
                assert torch.all(torch.abs(actor(blank)[:, :2] - actions[:, :2]) > 0.003)
                assert abs(final[0, 0, 4] - final[1, 0, 4]) > 0.03
                assert abs(final[0, 0, 7] - final[1, 0, 7]) > 0.005
            history.append(
                dict(
                    tick=tick,
                    actor_input=_host(actor_input).tolist(),
                    action=_host(actions).tolist(),
                    final=final.tolist(),
                    next_input=returned.tolist(),
                )
            )
            obs = transition.observation
        np.testing.assert_array_equal(returned[:, :2, 2], 1)
        np.testing.assert_array_equal(returned[:, :, 3:], 0)
        (tmp_path / "actor-trajectory.json").write_text(
            json.dumps(
                dict(world_keys=keys, roles=ROLES, fields=FIELDS, history=history), allow_nan=False
            ),
            encoding="utf-8",
        )
