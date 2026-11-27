from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def domains(monkeypatch):
    pytest.importorskip("mujoco")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    mod = import_module("_panda_pickup_rl_domains")
    xml = (examples / "checkpoints/panda-reach-factors-s4000/scene.xml").read_text(encoding="utf8")
    bank = examples / "checkpoints/panda-pickup-from-reach-s4000"
    return mod, xml, bank


def test_split_rng_and_source_rows_are_disjoint(domains):
    mod, xml, bank = domains
    splits = [mod.make_batch(xml, bank, s, 1, worlds=2) for s in ("train", "select", "blind")]
    meta = [s.manifest for s in splits]
    assert len({m["split_seed"] for m in meta}) == len({m["xml_sha256"] for m in meta}) == 3
    pools = [{(m["source_sha256"], i) for i in m["allowed_source_worlds"]} for m in meta]
    assert not (pools[0] & pools[1] or pools[1] & pools[2] or pools[0] & pools[2])
    for m in meta:
        assert m["valid_reset_count"] == len(m["rows"]) == 2
        assert all(row["source_world"] in m["allowed_source_worlds"] for row in m["rows"])
        assert sum(m["rejected_draws"].values()) == sum(row["attempts"] - 1 for row in m["rows"])
    again = mod.make_batch(xml, bank, "train", 1, worlds=2)
    later = mod.make_batch(xml, bank, "train", 1, worlds=2, batch=1)
    assert again.manifest == meta[0]
    assert later.xml == again.xml and later.manifest["resets_sha256"] != again.manifest["resets_sha256"]


def test_materials_scale_only_declared_parameters(domains):
    mod, xml, bank = domains
    base = mod.make_batch(xml, bank, "train", 0, worlds=1)
    alt = mod.make_batch(xml, bank, "train", 2, worlds=1)
    assert base.xml == xml and alt.xml != xml
    a, b = base.model, alt.model
    expected = alt.manifest["physics"]
    part = a.body("assembly_part").id
    for attr in ("body_parentid", "body_pos", "jnt_range", "geom_type", "geom_size", "geom_bodyid",
                 "geom_contype", "geom_conaffinity", "actuator_ctrlrange"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    np.testing.assert_array_equal(a.opt.gravity, b.opt.gravity)
    np.testing.assert_allclose(b.body_mass[part], a.body_mass[part] * expected["mass_factor"])
    np.testing.assert_allclose(b.body_inertia[part], a.body_inertia[part] * expected["mass_factor"])
    np.testing.assert_array_equal(a.geom_friction[:, 1:], b.geom_friction[:, 1:])
    expected_mu = a.geom_friction[:, 0].copy()
    for label, names in (("pad_slide_friction_factor", mod.PAD_NAMES), ("pin_slide_friction_factor", mod.PART_GEOMS)):
        expected_mu[[a.geom(n).id for n in names]] *= expected[label]
    np.testing.assert_allclose(b.geom_friction[:, 0], expected_mu)


def test_resets_preserve_pin_height_twist_and_controller_history(domains):
    mod, xml, bank = domains
    result = mod.make_batch(xml, bank, "train", 0, worlds=8)
    originals = mod.read_handoffs(bank / "factor-update.npz", result.model, expected_split="factor-update")
    geometry = mod._Geometry(result.model)
    for row, h in zip(result.manifest["rows"], result.handoffs, strict=True):
        src = originals[row["source_world"]]
        assert h.qpos[18] == src.qpos[18]
        np.testing.assert_array_equal(h.qvel[16:], src.qvel[16:])
        np.testing.assert_array_equal(h.bias, src.bias)
        assert np.linalg.norm(h.qpos[19:23]) == pytest.approx(1., abs=2e-15, rel=0)
        assert 0 < np.max(np.abs(h.qpos[:14] - src.qpos[:14])) <= mod.HALFWIDTH["arm_q_rad"]
        assert 0 < np.max(np.abs(h.qvel[:14] - src.qvel[:14])) <= mod.HALFWIDTH["arm_v_rad_s"] + 1e-7
        assert geometry.inspect(h)[0] is None
    src = result.handoffs[0]
    raised = mod.Handoff(src.qpos.copy(), src.qvel.copy(), src.target.copy(), src.bias.copy())
    raised.qpos[18] += .010
    assert geometry.inspect(raised)[0] is not None
    unsafe = mod.Handoff(src.qpos.copy(), src.qvel.copy(), src.target.copy(), src.bias.copy())
    unsafe.qvel[16] = .5
    assert geometry.inspect(unsafe)[0] == "initial_speed"
    with pytest.raises(ValueError, match="exact XML"):
        mod.make_batch(xml + " ", bank, "train", 0, worlds=1)
