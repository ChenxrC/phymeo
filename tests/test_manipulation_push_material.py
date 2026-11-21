"""Finite declared material admission, independent of push ranking or dynamics."""

from dataclasses import asdict

import numpy as np
import pytest
from test_manipulation_push_objective import propose
from test_manipulation_push_planner import model, planner, proposal

from phymeo.control.manipulation_push_material import NativePushMaterialFaces

TRI = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def faces(triangles=TRI, ids=("finite",), q=None):
    return NativePushMaterialFaces(
        np.array([0.1, 0.2], np.float32) if q is None else q, "a" * 64, ids, triangles
    )


@pytest.mark.parametrize(
    "point,normal,ok,reason",
    [
        ((0.2, 0.3, 0.0), (0, 0, 1), True, "declared_material_face"),
        ((0.2, 0.3, 0.01), (0, 0, 1), False, "material_point_off_declared_faces"),
        ((0.8, 0.8, 0.0), (0, 0, 1), False, "material_point_outside_finite_faces"),
        ((0.2, 0.3, 0.0), (0, 0, -1), False, "material_normal_mismatch"),
    ],
)
def test_finite_face_contract_rejects_plane_only_or_reversed_normal(point, normal, ok, reason):
    result = faces().classify(point, normal)
    assert result["eligible"] is ok and result["reason"] == reason
    assert result["face_ids"] == (("finite",) if ok else ())


def test_edge_collects_only_same_orientation_faces_in_stable_asset_order():
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
        ],
        dtype=float,
    )
    result = faces(triangles, ("z", "a", "reverse")).classify((0.5, 0.5, 0), (0, 0, 1))
    assert result["eligible"] and result["face_ids"] == ("z", "a")


def test_contract_transforms_with_arbitrary_triangle_pose():
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shift = np.array([0.34, -0.27, 1.8])
    moved = faces(TRI @ rotation.T + shift)
    for point in ((0.2, 0.3, 0.0), (0.8, 0.8, 0.0)):
        expected = faces().classify(point, (0, 0, 1))
        actual = moved.classify(np.asarray(point) @ rotation.T + shift, rotation @ [0.0, 0.0, 1.0])
        assert (actual["eligible"], actual["reason"], actual["face_ids"]) == (
            expected["eligible"],
            expected["reason"],
            expected["face_ids"],
        )


def test_frozen_binding_owns_q_and_geometry_and_rejects_stale_entry():
    q, triangles = np.array([0.1, 0.2], np.float32), TRI.copy()
    snapshot = faces(triangles, q=q)
    before = q.copy()
    q[0] += 0.1
    triangles += 3
    snapshot.validate_source(before)
    with pytest.raises(ValueError, match="q"):
        snapshot.validate_source(q)
    assert snapshot.classify((0.2, 0.3, 0.0), (0, 0, 1))["eligible"]
    with pytest.raises(ValueError):
        snapshot.triangles_world[0, 0, 0] = 1.0
    assert snapshot.asset_sha256 == "a" * 64
    assert snapshot.binding != faces(TRI + [0, 0, 0.1]).binding
    assert snapshot.binding != NativePushMaterialFaces(before, "b" * 64, ("finite",), TRI).binding


def test_degenerate_or_nonfinite_material_cannot_admit_candidates():
    for tri in (np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan)):
        with pytest.raises(ValueError):
            faces(tri)


def native_faces(q):
    return faces(
        np.array(
            [
                [[0.45, -0.05, 0.95], [0.45, -0.05, 1.05], [0.45, 0.05, 1.05]],
                [[0.45, -0.05, 0.95], [0.45, 0.05, 1.05], [0.45, 0.05, 0.95]],
            ]
        ),
        ("left0", "left1"),
        q,
    )


def native_candidate(index, *, inside):
    p = proposal(index)
    p.point_world = (0.45, 0.0 if inside else 0.5, 1.0)
    p.outward_normal_world = (-1.0, 0.0, 0.0)
    p.distance_ordinal = 0
    return p


def test_material_rejection_keeps_indices_and_never_spends_ik_on_omitted_face():
    m = model()
    misses, hits = native_candidate(4, inside=False), native_candidate(13, inside=True)
    result = planner(m).plan(m.qpos0, (misses, hits), material_faces=native_faces(m.qpos0))
    assert result.status == "sampled_reference_ready" and result.proposal.index == 13
    assert result.ik_calls == result.path_calls == 3
    evidence = next(a for a in result.attempts if a["kind"] == "material_admission")
    assert [r["index"] for r in evidence["proposals"]] == [4, 13]
    assert not evidence["proposals"][0]["eligible"]
    assert evidence["proposals"][1]["face_ids"] == ("left0", "left1")
    assert evidence["asset_sha256"] == "a" * 64


def test_all_material_misses_refuse_before_any_ik_or_path(monkeypatch):
    import phymeo.control.manipulation_push_planner as module

    def forbidden(*args, **kwargs):
        raise AssertionError("material miss consumed native solve budget")

    monkeypatch.setattr(module, "solve_native_point_ik", forbidden)
    monkeypatch.setattr(module, "audit_native_joint_path", forbidden)
    m = model()
    result = planner(m).plan(
        m.qpos0, (native_candidate(4, inside=False),), material_faces=native_faces(m.qpos0)
    )
    assert result.status == "no_material_qualified_proposal"
    assert result.proposal is None and result.ik_calls == result.path_calls == 0


def test_explicit_none_preserves_legacy_route_and_old_proposal_archive():
    from phymeo.control.manipulation_push_objective import complete_saved_push_proposal

    m = model()
    p = proposal()
    old, explicit = (
        planner(m).plan(m.qpos0, (p,)),
        planner(m).plan(m.qpos0, (p,), material_faces=None),
    )
    assert old.status == explicit.status and old.ik_calls == explicit.ik_calls
    assert all(a["kind"] != "material_admission" for a in old.attempts + explicit.attempts)
    saved = asdict(propose()[0])
    legacy = {
        k: v
        for k, v in saved.items()
        if k
        not in (
            "selection_policy",
            "selection_eligible",
            "selection_reason",
            "position_error_before_m",
            "position_error_after_m",
            "yaw_error_before_rad",
            "yaw_error_after_rad",
        )
    }
    assert complete_saved_push_proposal(legacy) == saved


def test_scene_adapter_uses_same_asset_digest_and_existing_world_transform():
    from test_pusht_release_material import _fixture

    from experiments.area_pressure.pusht_pose_replan import freeze_declared_material
    from phymeo.backends.mujoco_heterogeneous_area_scene import _asset_digest

    m, asset, state = _fixture()
    sha = _asset_digest({"fixture": (asset,)})
    frozen = freeze_declared_material(
        m, state["q"], asset, model_key="fixture", scene_asset_sha256=sha
    )
    frozen.validate_source(state["q"])
    np.testing.assert_array_equal(
        frozen.triangles_world, asset.rigid_local.world_vertices[asset.rigid_local.faces]
    )
    assert frozen.face_ids == asset.rigid_local.face_ids
    with pytest.raises(ValueError, match="asset"):
        freeze_declared_material(
            m, state["q"], asset, model_key="fixture", scene_asset_sha256="f" * 64
        )


def test_material_opt_in_cannot_change_B():
    from experiments.area_pressure.pusht_pose_replan import run

    with pytest.raises(ValueError, match="A"):
        run(*(["absent.json"] * 6), case="B", require_declared_material=True)


def test_A_cli_passes_material_opt_in_without_relabeling_ranking(monkeypatch):
    import sys

    from experiments.area_pressure import pusht_pose_replan as module

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return dict(cases={"A": dict(summary=dict(status="planning_rejected"))})

    monkeypatch.setattr(module, "run", fake_run)
    argv = ["pose"]
    for flag in ("summary", "release", "source", "audit", "checkpoint", "output"):
        argv += ["--" + flag, "unused.json"]
    argv += [
        "--case",
        "A",
        "--selection-policy",
        "joint-pose-balanced",
        "--require-declared-material",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        module.main()
    assert calls[0]["require_declared_material"] is True
    assert calls[0]["selection_policy"] == "joint-pose-balanced"


def test_saved_cases_with_and_without_material_gate_cannot_be_merged(tmp_path, monkeypatch):
    import json

    from test_pusht_pose_replan_comparison import _split_case_files

    reader, paths = _split_case_files(tmp_path)
    a = reader._read(paths["A"])
    a["material_admission"] = "declared-finite-faces-v1"
    paths["A"].write_text(json.dumps(a), encoding="utf-8")
    monkeypatch.setattr(reader, "compare_case", lambda path: dict(actual=dict(trajectory=[])))
    with pytest.raises(ValueError, match="material"):
        reader.compare_cases(paths["A"], paths["B"])


def test_scene_adapter_transports_finite_faces_without_changing_source_q():
    from test_pusht_release_material import _fixture

    from experiments.area_pressure.pusht_pose_replan import freeze_declared_material
    from phymeo.backends.mujoco_heterogeneous_area_scene import _asset_digest

    m, asset, state = _fixture()
    body = m.body("object").id
    m.body_pos[body] = (0.3, -0.2, 0.4)
    m.body_quat[body] = (2**-0.5, 0.0, 0.0, 2**-0.5)
    original = state["q"].copy()
    frozen = freeze_declared_material(
        m,
        state["q"],
        asset,
        model_key="fixture",
        scene_asset_sha256=_asset_digest({"fixture": (asset,)}),
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected = (asset.rigid_local.world_vertices @ rotation.T + [0.3, -0.2, 0.4])[
        asset.rigid_local.faces
    ]
    np.testing.assert_allclose(frozen.triangles_world, expected, atol=1e-14, rtol=0.0)
    np.testing.assert_array_equal(original, state["q"])
