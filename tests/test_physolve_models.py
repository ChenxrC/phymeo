"""Robot compilation fidelity; optional second asset uses explicit local sources."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phymeo.builders.mujoco_compiler import compile_mujoco


def robot_path(name):
    if name == "panda":
        import phymeo.builders
        return Path(phymeo.builders.__file__).parent / "panda_assets/panda.urdf"
    config = os.environ.get("PHYMEO_PHYSOLVE_SOURCES")
    if not config:
        pytest.skip("explicit robot asset source required")
    sources = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    return Path(sources["robot_assets"][name])


def native_fields(model):
    result = {}
    for prefix, owner in (("", model), ("opt.", model.opt)):
        for name in dir(owner):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                result[prefix + name] = value.copy()
            elif isinstance(value, (int, float, str, bytes)):
                result[prefix + name] = value
    return result


@pytest.mark.parametrize("name", ["panda", "fanuc"])
def test_robot_every_native_field_survives_compilation_snapshot(name):
    import mujoco
    source = robot_path(name)
    reference = native_fields(mujoco.MjModel.from_xml_path(str(source)))
    record = compile_mujoco(source, require_asset=False)
    compiled = native_fields(record.mj_model)
    restored = native_fields(record.fresh_native_model())
    assert restored.keys() == reference.keys()
    for field, value in reference.items():
        np.testing.assert_array_equal(compiled[field], value, err_msg=field)
        if field == "signature":
            # MuJoCo MJB omits the MjSpec compilation signature. It is an
            # identity token, not a physical field; a restored model has zero.
            assert restored[field] == 0
            continue
        # Binary snapshot should be exact, including mesh, collision, controls,
        # option scalars, name buffers and address tables, not just q dimensions.
        np.testing.assert_array_equal(restored[field], value, err_msg=field)
    assert record.mapping is not None
    q = record.qpos_to_q(record.native_initial_qpos)
    np.testing.assert_allclose(record.q_to_qpos(q), record.native_initial_qpos,
                               atol=2e-12, rtol=0)
    v = np.linspace(-.1, .1, len(q))
    np.testing.assert_allclose(record.qvel_to_v(q, record.v_to_qvel(q, v)), v,
                               atol=2e-12, rtol=0)
