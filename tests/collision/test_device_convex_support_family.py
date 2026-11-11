"""COL1a tests first: frozen family literals and independent depth/witness checks."""

import gzip
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from contextlib import ExitStack, closing, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import test_device_convex_query as k0
import tyx as tx

import phymeo as ph


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).resolve().parents[1]/"reference_newton/rr/device_convex_family_reference.py"
    spec = importlib.util.spec_from_file_location("col1_independent_family_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW_PAIRS = ("SY", "SN", "SE", "CY", "CN", "CE", "BY", "BN", "BE", "HY", "HN", "HE", "YY", "YN", "YE", "NN", "NE", "EE")
family_x = family_y = family_z = None
KIND = dict(S="sphere", C="capsule", B="hull", H="hull", Y="cylinder", N="cone", E="ellipsoid")


@contextmanager
def saved(path, name, mode, inputs):
    files = (Path(__file__), Path(reference().__file__))
    record = dict(schema="COL1-family-query-v1", identity=name, mode=mode, inputs=inputs,
                  source=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
                  attempts=[], completed=False, failure=None)
    try:
        yield record
        record["completed"] = True
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with gzip.open(path/(name+".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(k0.encode(record), stream, allow_nan=False)


def shape_for(data, side, *, quaternion=None):
    code = data["kinds"][side]
    kwargs = dict(radius=data["radii"][side], revision=19+side)
    if code in "YNE":
        if quaternion is None:
            quaternion = (0, 0, 0, 1) if side == 0 else data["static_quaternion"]
        kwargs["curved"] = ph.CurvedSolid(KIND[code], data["dimensions"][side], tuple(quaternion))
    return ph.CollisionShape(ph.CollisionKey(*data["keys"][side]), KIND[code], data["ids"][side], **kwargs)


@contextmanager
def prepared(data, mode, path, **options):
    world = data["world"]
    tree = k0.ordinary_forest(anchor=data["centers"][0], orientation=data["body_quaternion"])
    with ph.TyxWorldBatch([(world, tree)], mode, cache_root=path) as batch:
        checkpoint = batch.snapshot()[world]
        shapes = (shape_for(data, 0), shape_for(data, 1))
        bindings = (
            ph.ResidentTreeBinding(shapes[0].key, world, 0, data["local"][0], topology=checkpoint.topology),
            ph.ResidentStaticBinding(shapes[1].key, ph.CollisionMotion(data["points"][1])),
        )
        with batch.prepare_collision(ph.CollisionModel(shapes), bindings) as broad:
            pairs = (tuple(shape.key for shape in shapes),)
            if data["requested_reverse"]:
                pairs = (tuple(reversed(pairs[0])),)
            options.setdefault("pairs", pairs)
            with closing(batch.prepare_convex_query(broad, **options)) as plan:
                yield batch, broad, plan


# Preserve the K0 wire checks locally; its old phase table intentionally stops at EPA.
def check_raw(result):
    assert isinstance(result.raw, Mapping) and set(k0.RAW_SHAPES) <= set(result.raw)
    n = len(result.rows)
    for name, tail in k0.RAW_SHAPES.items():
        value = result.raw[name]
        assert isinstance(value, np.ndarray) and value.shape[1:] == tail and len(value) >= n
        assert value.dtype == np.dtype(np.int32 if name in ("meta", "material_ids") else np.float32)
        assert not value.flags.writeable
        if value.dtype.kind == "f":
            assert np.isfinite(value).all()
    meta = result.raw["meta"][:n]
    for name, value in result.raw.items():
        assert np.all(value[n:] == (-1 if name == "material_ids" else 0))
    assert np.all((meta[:, 0] >= 1) & (meta[:, 0] <= 4))
    assert np.all(meta[:, 14:] == 0)
    assert np.all(meta[:, 12] == result.attempt_tag)
    np.testing.assert_array_equal(meta[:, 13], np.arange(n))
    assert isinstance(result.source, Mapping) and result.source
    assert {"attempt_delta", "broadphase_delta", "logical_pairs", "padded_pairs"} <= set(
        result.stats
    )
    assert result.stats["logical_pairs"] == n and result.stats["padded_pairs"] == len(
        result.raw["meta"]
    )
    check_wire_rows(result)


def check_wire_rows(result):
    status = ("UNWRITTEN", "RESOLVED", "UNKNOWN", "UNSUPPORTED", "RESOURCE_LIMIT")
    relation = ("UNDETERMINED", "SEPARATED", "OVERLAP", "CONTACT_BAND")
    reason = (
        "OK",
        "NEAR_ZERO",
        "NORMAL_NONUNIQUE",
        "CORE_CONTAINMENT",
        "GJK_LIMIT",
        "EPA_LIMIT",
        "DEGENERATE_SIMPLEX",
        "EPA_CAPACITY",
        "UNSUPPORTED_SHAPE",
        "SOURCE_PRECISION",
        "STALLED",
    )
    phase = ("NONE", "ANALYTIC_ROUND", "GJK_CORE", "GJK_POLYTOPE", "EPA", "GJK_SUPPORT")
    for index, row in enumerate(result.rows):
        meta = result.raw["meta"][index]
        assert (
            0 <= int(meta[1]) < len(relation)
            and 0 <= int(meta[2]) < len(reason)
            and 0 <= int(meta[3]) < len(phase)
        )
        assert np.all(meta[4:9] >= 0)
        assert row.status == status[int(meta[0])]
        assert row.relation == relation[int(meta[1])]
        assert row.reason == reason[int(meta[2])]
        assert row.phase == phase[int(meta[3])]
        assert 0 <= int(meta[9]) <= 7
        assert (row.gap_valid, row.normal_valid, row.witness_valid) == tuple(
            bool(meta[9] & flag) for flag in (1, 2, 4)
        )
        assert tuple(float(x) for x in result.raw["gap"][index]) == (
            row.gap_m,
            row.gap_lower_m,
            row.gap_upper_m,
            row.scale_m,
        )
        np.testing.assert_array_equal(row.normal, result.raw["normal"][index])
        assert tuple(int(x) for x in meta[4:9]) == (
            row.gjk_used,
            row.epa_used,
            row.support_calls,
            row.vertices_used,
            row.faces_used,
        )
        assert (
            row.gjk_used <= 64
            and row.epa_used <= 192
            and row.vertices_used <= 64
            and row.faces_used <= 128
        )
        for side, witness in enumerate((row.first, row.second)):
            count = int(meta[10 + side])
            assert 0 <= count <= 4 and witness.key == row.pair[side]
            assert np.all(result.raw["material_ids"][index, side, count:] == -1)
            assert np.all(result.raw["barycentric"][index, side, count:] == 0)
            if not row.witness_valid:
                assert np.all(result.raw["surface"][index, side] == 0) and np.all(
                    result.raw["skin_offset"][index, side] == 0
                )
            assert witness.valid == row.witness_valid
            assert tuple(int(x) for x in result.raw["material_ids"][index, side, :count]) == tuple(
                witness.vertex_ids
            )
            assert tuple(float(x) for x in result.raw["barycentric"][index, side, :count]) == tuple(
                witness.weights
            )
            np.testing.assert_array_equal(witness.offset_m, result.raw["skin_offset"][index, side])
            np.testing.assert_array_equal(
                witness.surface_relative_m, result.raw["surface"][index, side]
            )
        if not row.gap_valid:
            assert (row.gap_m, row.gap_lower_m, row.gap_upper_m) == (0, 0, 0)
        if not row.normal_valid:
            assert np.all(result.raw["normal"][index] == 0)
        if row.status == "RESOLVED":
            assert row.gap_valid and row.normal_valid and row.witness_valid
    for value in result.raw.values():
        with pytest.raises(ValueError):
            value.setflags(write=True)


def check_result(result, data):
    truth = reference().oracle(data)
    check_raw(result)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.pair == tuple(ph.CollisionKey(*key) for key in data["keys"])
    length = truth["scale_m"]
    assert abs(row.scale_m-length) <= 1e-4*length
    assert result.request_reversed == (data["requested_reverse"],)
    if data["phase"] == "touch" and row.status != "RESOLVED":
        assert row.relation != "SEPARATED"
        assert row.status in ("UNKNOWN", "UNSUPPORTED", "RESOURCE_LIMIT")
        return dict(truth=truth, status=row.status, reason=row.reason)
    assert row.status == "RESOLVED" and row.gap_valid and row.normal_valid and row.witness_valid
    assert abs(row.gap_m-truth["gap_m"]) <= 1e-4*length
    assert row.gap_lower_m <= row.gap_m <= row.gap_upper_m
    assert np.linalg.norm(np.asarray(row.normal)-truth["normal"]) <= 2e-3
    assert abs(np.linalg.norm(row.normal)-1) <= 2e-4
    assert row.relation == {"sep": "SEPARATED", "overlap": "OVERLAP", "touch": "CONTACT_BAND"}[data["phase"]]
    materials = tuple(reference().witness_check(data, side, witness, np.asarray(row.origin_m),
                      truth["normal"], 2e-4*length)
                      for side, witness in enumerate((row.first, row.second)))
    difference = np.asarray(row.second.surface_relative_m)-row.first.surface_relative_m
    assert np.linalg.norm(difference-row.gap_m*np.asarray(row.normal)) <= 2e-4*length
    return dict(truth=truth, materials=materials)


def check_source(result):
    src = result.source
    assert src["support_family_schema"] == "resident-convex-support-family-v1"
    n = len(src["shape_keys"])
    arrays = dict(declared_dimensions_m=(np.float64, (n, 3)),
                  uploaded_dimensions_m=(np.float32, (n, 3)),
                  dimension_quantization_error_m=(np.float64, (n, 3)),
                  declared_orientation_xyzw=(np.float64, (n, 4)),
                  declared_asset_rotation=(np.float64, (n, 3, 3)),
                  uploaded_asset_rotation=(np.float32, (n, 3, 3)),
                  asset_rotation_quantization_error=(np.float64, (n, 3, 3)),
                  effective_shape_rotation=(np.float32, (n, 3, 3)),
                  pair_origins_relative_m=(np.float32, (len(result.rows), 3)))
    for name, (dtype, shape) in arrays.items():
        value = src[name]
        assert value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    for a, b, error in (
        ("declared_dimensions_m", "uploaded_dimensions_m", "dimension_quantization_error_m"),
        ("declared_asset_rotation", "uploaded_asset_rotation", "asset_rotation_quantization_error"),
    ):
        np.testing.assert_array_equal(src[b], np.asarray(src[a], np.float32))
        np.testing.assert_array_equal(src[error], src[b].astype(float)-src[a])
    for i, row in enumerate(result.rows):
        world = src["world_keys"].index(row.pair[0].world)
        np.testing.assert_array_equal(row.origin_m, src["world_origins_m"][world]+src["pair_origins_relative_m"][i].astype(float))
    for i, shape in enumerate(src["model"].shapes):
        expected = np.zeros(3)
        if shape.curved is not None:
            expected[:len(shape.curved.dimensions_m)] = shape.curved.dimensions_m
            np.testing.assert_array_equal(src["declared_orientation_xyzw"][i], shape.curved.orientation_xyzw)
        np.testing.assert_array_equal(src["declared_dimensions_m"][i], expected)


@pytest.mark.tyx
@pytest.mark.parametrize("family", NEW_PAIRS)
@pytest.mark.parametrize("phase", ("sep", "touch", "overlap"))
def test_eighteen_new_family_three_states(family, phase, tyx_mode, tmp_path):
    data = reference().case(family, phase)
    with saved(tmp_path, family+"-"+phase, tyx_mode, data) as record:
        record["independent_oracle"] = reference().oracle(data)
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            before = batch.snapshot()
            result = k0.query_attempt(batch, plan, record, "query")
            record["checks"] = check_result(result, data)
            check_source(result)
            assert k0.canonical(before) == k0.canonical(batch.snapshot())


@pytest.mark.tyx
@pytest.mark.parametrize("family", NEW_PAIRS)
def test_rotated_reversed_queries(family, tyx_mode, tmp_path):
    data = reference().case(family, "sep", cycle=True)
    with saved(tmp_path, family+"-cycle-reverse", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "query")
            record["checks"] = check_result(result, data)
            check_source(result)
            np.testing.assert_allclose(result.source["effective_shape_rotation"],
                                       np.repeat(reference().rotation(reference().CYCLE_Q)[None], 2, axis=0),
                                       atol=2e-6, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("family", ("YE", "NE", "EH"))
def test_independent_oblique_support_planes(family, tyx_mode, tmp_path):
    data = reference().case(family, "sep", oblique=True)
    with saved(tmp_path, family+"-oblique", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "query")
            record["checks"] = check_result(result, data)


def _support_samples(count: tx.i32, kind: tx.i32, dims: tx.ndarray(tx.f32, 1),
                     directions: tx.ndarray(tx.f32, 2), output: tx.ndarray(tx.f32, 2)):
    for i in tx.range(count):
        output[i, 0] = family_x(kind, directions[i, 0], directions[i, 1], directions[i, 2], dims[0], dims[1], dims[2])
        output[i, 1] = family_y(kind, directions[i, 0], directions[i, 1], directions[i, 2], dims[0], dims[1], dims[2])
        output[i, 2] = family_z(kind, directions[i, 0], directions[i, 1], directions[i, 2], dims[0], dims[1], dims[2])


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("Y", "N", "E"))
@pytest.mark.parametrize("quaternion", ((0, 0, 0, 1), (0, .6, 0, .8)))
def test_support_function_values_and_ties(kind, quaternion, tyx_mode, tmp_path):
    global family_x, family_y, family_z
    from phymeo.backends._tyx_convex_support_family import curved_support_x, curved_support_y, curved_support_z
    family_x, family_y, family_z = curved_support_x, curved_support_y, curved_support_z
    ref = reference()
    R = ref.rotation(quaternion)
    world_directions = np.asarray(ref.DIRECTIONS, float)
    local_directions = world_directions@R
    local_directions = np.concatenate((local_directions, 7*local_directions)).astype(np.float32)
    dims = np.zeros(3, np.float32)
    declared = ref.primitive(kind)[2]
    dims[:len(declared)] = declared
    with saved(tmp_path, "support-"+kind+"-"+str(quaternion[1]), tyx_mode,
               dict(directions=world_directions, quaternion=quaternion, dimensions=declared)) as record:
        with ph.TyxWorldBatch([(17, k0.ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            with ExitStack() as resources:
                def array(value):
                    if tyx_mode == "cuda":
                        result = batch._tx.cuda_array(value)
                        resources.callback(result.free)
                        return result
                    return value.copy()
                d, dim, output = array(local_directions), array(dims), array(np.zeros_like(local_directions))
                batch._launch(tx.kernel(_support_samples), len(local_directions), {"Y": 4, "N": 5, "E": 6}[kind], dim, d, output)
                batch._sync()
                actual = output.to_numpy() if tyx_mode == "cuda" else output.copy()
        for direction, point in zip(local_directions.astype(float), actual.astype(float), strict=True):
            wanted = ref.support(kind, direction)
            assert abs(float(point@direction-wanted@direction)) <= 1e-4*max(1, ref.diameter(kind))*np.linalg.norm(direction)
            assert ref.set_residual(kind, point) <= 2e-4*max(1, ref.diameter(kind))
        np.testing.assert_allclose(actual[:13], actual[13:], atol=2e-4, rtol=0)
        record["actual_support"] = actual


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("Y", "N", "E"))
def test_curved_patch_remains_explicitly_unsupported(kind, tyx_mode, tmp_path):
    data = reference().case("S"+kind, "overlap")
    with saved(tmp_path, kind+"-no-patch", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path, site_capacity=8) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "query")
            record["checks"] = check_result(result, data)
            patch = result.rows[0].patch
            assert patch.status == "UNSUPPORTED" and not patch.sites
            assert patch.dimension == -1 and not patch.device_resolved and not patch.host_completed and not patch.skip_host_patch_search


@pytest.mark.tyx
@pytest.mark.parametrize("options", (dict(gjk_iterations=1), dict(epa_vertex_capacity=4)))
def test_original_iteration_and_capacity_limits_are_not_bypassed(options, tyx_mode, tmp_path):
    data = reference().case("YE", "sep", oblique=True) if "gjk_iterations" in options else reference().case("NE", "overlap")
    with saved(tmp_path, "resource-"+next(iter(options)), tyx_mode, dict(case=data, options=options)) as record:
        with prepared(data, tyx_mode, tmp_path, **options) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "bounded")
            check_raw(result)
            assert result.rows[0].status == "RESOURCE_LIMIT"
            assert result.rows[0].reason in ("GJK_LIMIT", "EPA_CAPACITY")
            assert not result.rows[0].witness_valid


@pytest.mark.tyx
def test_no_host_geometry_search_and_zero_pair_tail(tyx_mode, tmp_path, monkeypatch):
    data = reference().case("NE", "sep")
    with saved(tmp_path, "no-host-empty-tail", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            from phymeo.geometry import _convex_support
            def forbidden(*args, **kwargs):
                raise AssertionError("device family query called HOST support")
            monkeypatch.setattr(_convex_support._Body, "support", forbidden)
            assert k0.query_attempt(batch, plan, record, "nonempty").rows[0].status == "RESOLVED"
            with closing(batch.prepare_convex_query(broad, pairs=())) as empty:
                result = k0.query_attempt(batch, empty, record, "empty")
                assert result.rows == () and len(result.raw["meta"]) == 0
                check_raw(result)


@pytest.mark.parametrize("kind,dimensions,quaternion", (
    ("cylinder", (-1, 1), (0, 0, 0, 1)), ("cone", (1, 0), (0, 0, 0, 1)),
    ("ellipsoid", (1, 1, float("nan")), (0, 0, 0, 1)),
    ("cylinder", (1, 1), (0, 0, 0, 2)), ("cone", (1, 1), (0, 0, 0, 0)),
))
def test_invalid_curved_registration_is_not_a_separation(kind, dimensions, quaternion):
    with pytest.raises((ValueError, TypeError)):
        ph.CurvedSolid(kind, dimensions, quaternion)


def test_literal_matrix_and_all_negative_certificates():
    ref = reference()
    assert len(ref.PAIRS) == 28 and len(set(ref.PAIRS)) == 28
    assert ref.NEW_PAIRS == NEW_PAIRS
    for family in NEW_PAIRS:
        proof = ref.depth_certificate(ref.case(family, "overlap"))
        assert proof["strict_interior_overlap"]
        assert proof["lower_depth_m"] == proof["upper_depth_m"] == 1/64


@pytest.mark.tyx
def test_uploaded_dimension_loss_is_not_a_separation(tyx_mode, tmp_path):
    data = reference().case("SE", "sep")
    data["dimensions"] = ((), (1/4, 3/16, 2.0**-150))
    with saved(tmp_path, "lost-dimension", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "lost-dimension")
            assert result.rows[0].status == "UNSUPPORTED"
            assert result.rows[0].reason == "SOURCE_PRECISION"
            assert result.rows[0].relation != "SEPARATED"


@pytest.mark.tyx
def test_deep_coincident_centers_do_not_invent_a_unique_normal(tyx_mode, tmp_path):
    data = reference().case("EE", "overlap")
    data["centers"] = (np.zeros(3), np.zeros(3))
    data["points"] = (np.zeros((1, 3)), np.zeros((1, 3)))
    with saved(tmp_path, "deep-symmetric", tyx_mode, data) as record:
        with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "deep")
            check_raw(result)
            row = result.rows[0]
            assert row.status in ("UNKNOWN", "RESOURCE_LIMIT")
            assert not row.normal_valid and row.relation != "SEPARATED"

@pytest.mark.tyx
def test_exact_large_common_translation_preserves_relative_curved_support(tyx_mode, tmp_path):
    """All source words are exact FP32; small support offsets must stay relative."""
    translation = float(2**24)
    centers = (np.array((translation, 0.0, 0.0)), np.array((translation+4.0, 0.0, 0.0)))
    keys = (ph.CollisionKey(17, 11, 3), ph.CollisionKey(17, 29, 7))
    expected_gap, length = 29/8, 4.0
    inputs = dict(centers=centers, dimensions=(1/4, 1/4, 1/4), radius=1/8,
                  anchor=np.zeros(3), expected_gap_m=expected_gap, normal=(1, 0, 0))
    with saved(tmp_path, "large-common-local-translation", tyx_mode, inputs) as record:
        with ph.TyxWorldBatch([(17, k0.ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            state = batch.snapshot()[17]
            shapes = (ph.CollisionShape(keys[0], "ellipsoid", (101,), curved=ph.CurvedSolid("ellipsoid", (1/4, 1/4, 1/4))),
                      ph.CollisionShape(keys[1], "sphere", (201,), radius=1/8))
            bindings = (ph.ResidentTreeBinding(keys[0], 17, 0, centers[0][None], topology=state.topology),
                        ph.ResidentStaticBinding(keys[1], ph.CollisionMotion(centers[1][None])))
            with batch.prepare_collision(ph.CollisionModel(shapes), bindings) as broad:
                with closing(batch.prepare_convex_query(broad, pairs=(keys,))) as plan:
                    result = k0.query_attempt(batch, plan, record, "large-relative-source")
                    check_raw(result)
                    check_source(result)
                    np.testing.assert_array_equal(result.source["world_origins_m"], np.zeros((1, 3)))
                    np.testing.assert_array_equal(result.source["uploaded_local_points_m"], np.asarray(centers, np.float32))
                    row = result.rows[0]
                    assert row.status == "RESOLVED" and row.relation == "SEPARATED"
                    assert abs(row.scale_m-length) <= 1e-4*length
                    assert abs(row.gap_m-expected_gap) <= 1e-4*length
                    np.testing.assert_allclose(row.normal, (1, 0, 0), atol=2e-3, rtol=0)
                    for side, witness in enumerate((row.first, row.second)):
                        offset = np.array((1/4 if side == 0 else -1/8, 0.0, 0.0))
                        assert witness.vertex_ids == ((101,) if side == 0 else (201,))
                        assert abs(sum(witness.weights)-1) <= 64*2**-23
                        np.testing.assert_allclose(witness.offset_m, offset, atol=2e-4*length, rtol=0)
                        world_point = np.asarray(row.origin_m)+np.asarray(witness.surface_relative_m)
                        np.testing.assert_allclose(world_point, centers[side]+offset, atol=2e-4*length, rtol=0)
                    closure = np.asarray(row.second.surface_relative_m)-np.asarray(row.first.surface_relative_m)
                    np.testing.assert_allclose(closure, expected_gap*np.array((1, 0, 0)), atol=2e-4*length, rtol=0)
