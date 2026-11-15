"""Hand-vector tests for export only; no Newton/Warp import or physical oracle."""

import copy
import json

import numpy as np
import pytest

from experiments.area_pressure.newton_reference import (
    decode_octahedral_normals,
    export_unreduced_surface,
)

STAGES = (
    "broad_pairs",
    "sdf_pairs",
    "hydro_broad",
    "iso_l0",
    "iso_l1",
    "iso_l2",
    "iso_voxels",
    "hydro_faces",
    "rigid_contacts",
)


def fixture():
    raw = {
        "count": 1,
        "position_depth": [[1.0, 0.0, 0.0, -0.1]],
        "normal_oct": [[0.0, 0.0]],
        "shape_pairs": [[0, 1]],
        "contact_area": [2.0],
        "contact_pressure": [3.0],
        "contact_fingerprints": [7],
        "surface_vertices": [[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, 2.0, 0.0]],
        "surface_depth": [-0.1],
        "surface_pairs": [[0, 1]],
    }
    overflow = {
        "stages": [
            {
                "name": name,
                "count": int(name in ("hydro_faces", "rigid_contacts")),
                "capacity": 16384,
            }
            for name in STAGES
        ],
        "dropped": {
            "hydro_buffer": 0,
            "hydro_hashtable": 0,
            "global_buffer": 0,
            "global_hashtable": 0,
        },
    }
    return raw, overflow


def export(raw=None, overflow=None, **kwargs):
    initial, counters = fixture()
    return export_unreduced_surface(
        initial if raw is None else raw,
        points_world=[[1.0, 0.0, 0.0]],
        normals_world=[[0.0, 0.0, 1.0]],
        shape_a=0,
        shape_b=1,
        overflow=counters if overflow is None else overflow,
        **kwargs,
    )


def test_hand_force_moment_and_raw_area_not_visual_triangle_area():
    result = export()
    assert result["area_m2"] == 2.0  # Native triangle area is 3, clipped force area is 2.
    assert result["projected_area_m2"] == 2.0
    assert result["pressure_load_n"] == 6.0
    assert result["first_force_n"] == [0.0, 0.0, -6.0]
    assert result["second_force_n"] == [0.0, 0.0, 6.0]
    assert result["first_moment_nm"] == [0.0, 6.0, 0.0]
    assert result["second_moment_nm"] == [0.0, -6.0, 0.0]
    assert result["pressure_centroid_world"] == [1.0, 0.0, 0.0]
    assert result["cop_world"] is None
    assert result["raw"]["raw_ids"] == [1]
    assert result["quadrature"]["triangle_indices"] == [0]
    json.dumps(result, allow_nan=False)


def test_swapped_native_pair_preserves_declared_participant_sign():
    raw, counters = fixture()
    raw["shape_pairs"] = raw["surface_pairs"] = [[1, 0]]
    raw["normal_oct"] = [[1.0, 1.0]]
    result = export_unreduced_surface(
        raw,
        points_world=[[1, 0, 0]],
        normals_world=[[0, 0, -1]],
        shape_a=0,
        shape_b=1,
        overflow=counters,
    )
    assert result["first_force_n"] == [0.0, 0.0, -6.0]
    assert result["first_moment_nm"] == [0.0, 6.0, 0.0]


def test_rotation_and_translation_use_supplied_world_api_outputs():
    raw, counters = fixture()
    raw["surface_vertices"] = [[2, 1, 1], [2, 3, 1], [2, 2, 4]]
    result = export_unreduced_surface(
        raw,
        points_world=[[2, 2, 2]],
        normals_world=[[1, 0, 0]],
        shape_a=0,
        shape_b=1,
        overflow=counters,
        axis_world=(1, 0, 0),
    )
    assert result["first_force_n"] == [-6.0, 0.0, 0.0]
    assert result["first_moment_nm"] == [0.0, -12.0, 12.0]
    assert result["projected_area_m2"] == 2.0
    shifted = export_unreduced_surface(
        raw,
        points_world=[[2, 2, 2]],
        normals_world=[[1, 0, 0]],
        shape_a=0,
        shape_b=1,
        overflow=counters,
        origin_world=(2, 2, 2),
    )
    assert shifted["first_moment_nm"] == [0.0, 0.0, 0.0]


def test_nonpenetrating_face_is_saved_but_has_no_active_area_or_load():
    raw, counters = fixture()
    raw["position_depth"][0][3] = 0.0
    raw["surface_depth"] = [0.0]
    raw["contact_pressure"] = [0.0]
    result = export(raw, counters)
    assert result["unreduced_face_count"] == 1
    assert result["active_face_count"] == 0
    assert result["raw"]["contact_area"] == [2.0]
    assert result["area_m2"] == result["pressure_load_n"] == 0.0
    assert result["quadrature"]["points_world"] == []
    assert result["pressure_centroid_world"] is None


def test_empty_capture_keeps_real_zero_counts():
    raw, counters = fixture()
    raw = {name: (0 if name == "count" else []) for name in raw}
    for stage in counters["stages"]:
        stage["count"] = 0
    result = export_unreduced_surface(
        raw,
        points_world=[],
        normals_world=[],
        shape_a=0,
        shape_b=1,
        overflow=counters,
    )
    assert result["first_force_n"] == [0.0, 0.0, 0.0]
    assert result["unreduced_face_count"] == 0
    assert result["cop_world"] is None


@pytest.mark.parametrize(
    "encoded,expected",
    [
        ((0, 0), (0, 0, 1)),
        ((1, 0), (1, 0, 0)),
        ((0, -1), (0, -1, 0)),
        ((1, 1), (0, 0, -1)),
        ((0.5, 0), (2**-0.5, 0, 2**-0.5)),
    ],
)
def test_octahedral_normals_against_hand_vectors(encoded, expected):
    np.testing.assert_allclose(decode_octahedral_normals([encoded]), [expected], atol=1e-15)


@pytest.mark.parametrize(
    "field,value",
    [
        ("count", True),
        ("count", 16385),
        ("contact_area", [-1.0]),
        ("contact_pressure", [float("nan")]),
        ("contact_fingerprints", [True]),
        ("shape_pairs", [[0, 0]]),
        ("surface_pairs", [[1, 0]]),
        ("surface_depth", [-0.2]),
        ("normal_oct", [[2, 0]]),
        ("surface_vertices", [[0, 0, 0]]),
        ("position_depth", [[0, 0, 0]]),
    ],
)
def test_malformed_raw_data_rejected_without_mutation(field, value):
    raw, counters = fixture()
    raw[field] = value
    before = copy.deepcopy(raw)
    with pytest.raises(ValueError):
        export(raw, counters)
    assert repr(raw) == repr(before)


@pytest.mark.parametrize(
    "name", ["hydro_buffer", "hydro_hashtable", "global_buffer", "global_hashtable"]
)
def test_dropped_contacts_reject_even_when_counts_fit_capacity(name):
    raw, counters = fixture()
    counters["dropped"][name] = 1
    with pytest.raises(ValueError, match="overflow|dropped"):
        export(raw, counters)


def test_stage_overflow_and_missing_counter_cannot_pass():
    raw, counters = fixture()
    counters["stages"][0]["count"] = 16385
    with pytest.raises(ValueError, match="overflow"):
        export(raw, counters)
    _, counters = fixture()
    counters["stages"].pop()
    with pytest.raises(ValueError, match="counter|stage"):
        export(raw, counters)


def test_readonly_inputs_are_not_changed_and_output_does_not_alias():
    raw, counters = fixture()
    for key, value in tuple(raw.items()):
        if key != "count":
            raw[key] = np.array(value)
            raw[key].flags.writeable = False
    before = {key: value.tobytes() for key, value in raw.items() if key != "count"}
    result = export(raw, counters)
    result["raw"]["contact_area"][0] = 99.0
    assert all(raw[key].tobytes() == value for key, value in before.items())


def test_multiple_faces_sum_loads_and_keep_complete_raw_order():
    raw, counters = fixture()
    raw["count"] = 2
    raw["position_depth"].append([-1, 0, 0, -0.2])
    raw["normal_oct"].append([1, 1])
    raw["shape_pairs"].append([1, 0])
    raw["contact_area"].append(1.0)
    raw["contact_pressure"].append(2.0)
    raw["contact_fingerprints"].append(8)
    raw["surface_vertices"].extend([[-2, -1, 0], [0, -1, 0], [-1, 2, 0]])
    raw["surface_depth"].append(-0.2)
    raw["surface_pairs"].append([1, 0])
    for stage in counters["stages"]:
        if stage["name"] in ("hydro_faces", "rigid_contacts"):
            stage["count"] = 2
    result = export_unreduced_surface(
        raw,
        points_world=[[1, 0, 0], [-1, 0, 0]],
        normals_world=[[0, 0, 1], [0, 0, -1]],
        shape_a=0,
        shape_b=1,
        overflow=counters,
    )
    assert result["raw"]["raw_ids"] == [1, 2]
    assert result["raw"]["contact_fingerprints"] == [7, 8]
    assert result["quadrature"]["triangle_indices"] == [0, 1]
    assert result["area_m2"] == 3.0
    assert result["first_force_n"] == [0.0, 0.0, -8.0]
    assert result["first_moment_nm"] == [0.0, 4.0, 0.0]
    assert result["pressure_centroid_world"] == [0.5, 0.0, 0.0]


@pytest.mark.parametrize(
    "points,normals",
    [
        ([], [[0, 0, 1]]),
        ([[float("inf"), 0, 0]], [[0, 0, 1]]),
        ([[1, 0, 0]], [[0, 0, 2]]),
        ([[1, 0, 0]], [[0, float("nan"), 1]]),
    ],
)
def test_invalid_world_api_outputs_reject(points, normals):
    raw, counters = fixture()
    with pytest.raises(ValueError):
        export_unreduced_surface(
            raw,
            points_world=points,
            normals_world=normals,
            shape_a=0,
            shape_b=1,
            overflow=counters,
        )


def test_empty_payload_cannot_fill_a_nonempty_capture():
    raw, counters = fixture()
    raw["contact_area"] = []
    with pytest.raises(ValueError, match="shape"):
        export(raw, counters)


def test_nonpenetrating_pressure_and_mismatched_output_count_reject():
    raw, counters = fixture()
    raw["position_depth"][0][3] = 0.0
    raw["surface_depth"] = [0.0]
    with pytest.raises(ValueError, match="nonpenetrating pressure"):
        export(raw, counters)
    raw, counters = fixture()
    counters["stages"][-1]["count"] = 0
    with pytest.raises(ValueError, match="counter mismatch"):
        export(raw, counters)
