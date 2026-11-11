"""COL1a tests first: frozen family literals and independent depth/witness checks."""

import gzip
import hashlib
import importlib.util
import json
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


def check_result(result, data):
    truth = reference().oracle(data)
    k0.check_raw(result)
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
                  effective_shape_rotation=(np.float32, (n, 3, 3)))
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
            k0.check_raw(result)
            assert result.rows[0].status == "RESOURCE_LIMIT"
            assert result.rows[0].reason in ("GJK_LIMIT", "EPA_CAPACITY")
            assert not result.rows[0].witness_valid


@pytest.mark.tyx
def test_no_host_geometry_search_and_zero_pair_tail(tyx_mode, tmp_path, monkeypatch):
    data = reference().case("NE", "sep")
    with prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
        from phymeo.geometry import _convex_support
        def forbidden(*args, **kwargs):
            raise AssertionError("device family query called HOST support")
        monkeypatch.setattr(_convex_support._Body, "support", forbidden)
        assert batch.convex_query(plan).rows[0].status == "RESOLVED"
        with closing(batch.prepare_convex_query(broad, pairs=())) as empty:
            result = batch.convex_query(empty)
            assert result.rows == () and len(result.raw["meta"]) == 0


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
            k0.check_raw(result)
            row = result.rows[0]
            assert row.status in ("UNKNOWN", "RESOURCE_LIMIT")
            assert not row.normal_valid and row.relation != "SEPARATED"
