"""Observe full cloth-face and particle-only Newton mesh contacts on the same frozen poses."""

import argparse
import json
import runpy
import warnings
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "mesh_triangle_cases.json").read_text("utf-8"))
SAMPLES = (
    "oblique-middle",
    "contained",
    "u-two-arms",
    "u-inside-vertices",
    "u-empty",
    "thin-cross",
    "thin-miss",
)


def mesh_data(asset):
    if asset == "u":
        u = runpy.run_path(str(HERE.parent / "rr" / "baseline.py"))["_u_geometry"]
        return u([1, 1, 1], False)
    base, idx = runpy.run_path(str(HERE.parent / "rr" / "mesh_round_poses.py"))["geometry"]("box")
    low, high = np.asarray(CATALOG["closed_boxes_m"][asset])
    return np.asarray((base + 1) * ((high - low) / 2) + low, dtype=np.float32), idx


def boxes(asset):
    return CATALOG["u_closed_pieces_m"] if asset == "u" else [CATALOG["closed_boxes_m"][asset]]


def exact_distance(point, asset):
    pieces = boxes(asset)
    rectangles = []
    if asset == "u":
        ring = ((-2, -2), (2, -2), (2, 2), (1, 2), (1, -1), (-1, -1), (-1, 2), (-2, 2))
        for p, q in zip(ring, ring[1:] + ring[:1], strict=True):
            rectangles.append(([*np.minimum(p, q), -0.5], [*np.maximum(p, q), 0.5]))
        for a, b in pieces:
            for z in (-0.5, 0.5):
                rectangles.append(([a[0], a[1], z], [b[0], b[1], z]))
    else:
        a, b = pieces[0]
        for i in range(3):
            for fixed in (a[i], b[i]):
                lo, hi = list(a), list(b)
                lo[i] = hi[i] = fixed
                rectangles.append((lo, hi))
    p = np.asarray(point)
    d = min(np.linalg.norm(p - np.clip(p, lo, hi)) for lo, hi in rectangles)
    interior = any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in pieces)
    return float(-d if interior else d)


def run_case(case, mesh, device, baked):
    import newton
    import warp as wp

    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for p in case["triangle_m"]:
        builder.add_particle(wp.vec3(*p), wp.vec3(0, 0, 0), 1.0, radius=0.0)
    builder.add_triangle(0, 1, 2)
    for opp, i, j in ((2, 0, 1), (0, 1, 2), (1, 2, 0)):
        builder.add_edge(opp, -1, i, j)
    builder.add_shape_mesh(
        -1, mesh=mesh, cfg=builder.ShapeConfig(gap=0.0, margin=0.0, has_particle_collision=True)
    )
    model = builder.finalize(device=device)
    assert (model.particle_count, model.tri_count, model.edge_count, model.shape_count) == (
        3,
        1,
        3,
        1,
    )
    assert newton.GeoType(int(model.shape_type.numpy()[0])).name == "MESH"
    state = model.state()
    q = state.particle_q.numpy().astype(float)
    np.testing.assert_allclose(q, case["triangle_m"], atol=1e-7, rtol=0)
    options = dict(
        broad_phase="nxn", soft_contact_gap=0.0, rigid_contact_max=1, max_triangle_pairs=32
    )
    read = runpy.run_path(str(HERE / "baseline.py"))["_read_soft"]

    def capture(pipeline):
        out = read(pipeline, state, q, str(device))
        for row in out["records"]:
            row["analytic_point_sdf_m"] = exact_distance(row["soft_point_m"], case["mesh"])
            row["gap_residual_m"] = row["gap_m"] - row["analytic_point_sdf_m"]
            row["rigid_boundary_error_m"] = abs(exact_distance(row["rigid_point_m"], case["mesh"]))
        return out

    particle = capture(newton.CollisionPipeline(model, **options))
    assert (
        particle["particle_candidates"],
        particle["edge_candidates"],
        particle["face_candidates"],
    ) == (3, 0, 0)
    clip = runpy.run_path(str(HERE.parents[1] / "collision" / "mesh_triangle_oracles.py"))[
        "clipped_fractions"
    ]
    result = dict(
        id=case["id"],
        relation=case["relation"],
        analytic_area_fractions=[str(x) for x in clip(case["triangle_m"], boxes(case["mesh"]))],
        particle_only=particle,
    )
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always")
        try:
            full = newton.CollisionPipeline(
                model, enable_rigid_soft_full_surface_contact=True, **options
            )
        except ValueError as exc:
            if baked or "signed-distance field" not in str(exc):
                raise
            result["full_construction_error"] = str(exc)
            return result
    if not baked:
        raise AssertionError("Unprovisioned mesh accepted by full-surface Newton pipeline")
    result["full_warnings"] = [
        str(w.message) for w in notes if "full_surface_contact" in str(w.message)
    ]
    result["full_surface"] = capture(full)
    assert (
        result["full_surface"]["edge_candidates"],
        result["full_surface"]["face_candidates"],
    ) == (3, 1)
    return result


def main():
    import sys

    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), required=True)
    parser.add_argument("--sdf", choices=("missing", "baked"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert version("newton") == "1.7.0"
    baked = args.sdf == "baked"
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    device = wp.get_device(args.device)
    assert str(device) == args.device and (not baked or device.is_cuda)
    cases = [c for c in CATALOG["cases"] if c["id"] in SAMPLES]
    state = dict(
        device=str(device),
        newton=version("newton"),
        warp=version("warp-lang"),
        mode=args.sdf,
        particle_radius_m=0,
        soft_gap_m=0,
        assets={},
        cases=[],
    )
    with (
        TemporaryDirectory(dir=args.out.parent, prefix="fresh-sdf-") as cache,
        wp.ScopedDevice(device),
    ):
        assets = {}
        for asset in sorted({c["mesh"] for c in cases}):
            vertices, indices = mesh_data(asset)
            mesh = newton.Mesh(vertices, indices, compute_inertia=False)
            assert mesh.sdf is None
            if baked:
                sdf = mesh.build_sdf(
                    device=device, max_resolution=64, margin=0.125, cache_dir=cache
                )
                wp.synchronize_device(device)
                tex = sdf.to_texture_kernel_data()
                assert (
                    mesh.sdf is sdf
                    and tex is not None
                    and str(tex.subgrid_start_slots.device) == args.device
                )
                state["assets"][asset] = dict(
                    face_count=len(indices) // 3,
                    max_resolution=64,
                    padding_m=0.125,
                    num_subgrids=int(tex.num_subgrids),
                    voxel_size_m=list(tex.voxel_size),
                )
            else:
                state["assets"][asset] = dict(face_count=len(indices) // 3, sdf=None)
            assets[asset] = mesh
        for case in cases:
            row = run_case(case, assets[case["mesh"]], device, baked)
            state["cases"].append(row)
            print(
                row["id"],
                "particles",
                row["particle_only"]["count"],
                "full",
                row.get("full_surface", {}).get("count", "construction rejected"),
                flush=True,
            )
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(state, out, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
