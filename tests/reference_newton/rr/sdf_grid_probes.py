"""Direct free-point reads from the distance and derivative function shipped with Newton 1.7.0."""

import argparse
import json
import runpy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import newton
import numpy as np
import warp as wp
from newton._src.geometry.sdf_texture import (
    SLOT_LINEAR,
    TextureSDFData,
    _locate_cell,
    texture_sample_sdf_grad,
)

HERE = Path(__file__).resolve().parent


@wp.kernel
def read_texture(
    texture: TextureSDFData,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec3),
    cells: wp.array2d(dtype=wp.int32),
):
    i = wp.tid()
    p = points[i]
    value, gradient = texture_sample_sdf_grad(texture, p)
    values[i] = value
    gradients[i] = gradient
    lo, hi = texture.sdf_box_lower, texture.sdf_box_upper
    clamped = wp.vec3(
        wp.clamp(p[0], lo[0], hi[0]), wp.clamp(p[1], lo[1], hi[1]), wp.clamp(p[2], lo[2], hi[2])
    )
    cell = _locate_cell(texture, wp.cw_mul(clamped - lo, texture.inv_sdf_dx))
    level = int(0)
    if cell.start_slot >= SLOT_LINEAR:
        level = 1
        cells[i, 1] = cell.x_base
        cells[i, 2] = cell.y_base
        cells[i, 3] = cell.z_base
    else:
        cells[i, 1] = cell.ix
        cells[i, 2] = cell.iy
        cells[i, 3] = cell.iz
    if wp.length(p - clamped) > 0.0:
        level = 2
    cells[i, 0] = level


def asset(kind):
    if kind == "u":
        return runpy.run_path(str(HERE / "baseline.py"))["_u_geometry"]([1, 1, 1], False)
    vertices, indices = runpy.run_path(str(HERE / "mesh_round_poses.py"))["geometry"]("box")
    half = (1 / 64, 0.5, 0.5) if kind == "thin" else (0.5, 0.375, 0.25)
    offset = (0.125, 0, 0) if kind == "thin" else (0, 0, 0)
    return np.asarray(vertices * half + offset, dtype=np.float32), indices


def main():
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert version("newton") == "1.7.0"
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda
    probes = json.loads((HERE / "sdf_grid_points.json").read_text("utf-8"))
    output = {}
    with (
        wp.ScopedDevice(device),
        TemporaryDirectory(dir=args.out.parent, prefix="free-sdf-") as cache,
    ):
        for kind, case in probes.items():
            vertices, indices = asset(kind)
            mesh = newton.Mesh(vertices, indices, compute_inertia=False)
            sdf = mesh.build_sdf(device=device, max_resolution=64, margin=0.125, cache_dir=cache)
            tex = sdf.to_texture_kernel_data()
            assert tex is not None and str(tex.subgrid_start_slots.device) == "cuda:0"
            host = np.asarray(case["points_m"], dtype=np.float32)
            step = 1 / 4096
            displacements = np.eye(3, dtype=np.float32) * step
            expanded = np.concatenate(
                [host, *(host + d for d in displacements), *(host - d for d in displacements)]
            )
            points = wp.array(expanded, dtype=wp.vec3, device=device)
            values = wp.empty(len(expanded), dtype=wp.float32, device=device)
            gradients = wp.empty(len(expanded), dtype=wp.vec3, device=device)
            cell_buffer = wp.empty((len(expanded), 4), dtype=wp.int32, device=device)
            wp.launch(
                read_texture,
                dim=len(expanded),
                inputs=[tex, points],
                outputs=[values, gradients, cell_buffer],
                device=device,
            )
            wp.synchronize_device(device)
            raw = values.numpy().reshape(7, len(host))
            vv, gg = raw[0], gradients.numpy()[: len(host)]
            fd = ((raw[1:4].astype(float) - raw[4:7].astype(float)) / (2 * step)).T
            cells = cell_buffer.numpy().reshape(7, len(host), 4)
            stable = ((cells[1:4] == cells[0]).all(axis=2) & (cells[4:7] == cells[0]).all(axis=2)).T
            assert np.isfinite(raw).all() and np.isfinite(gg).all() and np.isfinite(fd).all()
            lo, hi = np.asarray(list(tex.sdf_box_lower)), np.asarray(list(tex.sdf_box_upper))
            output[kind] = dict(
                shape_scale_xyz=[1, 1, 1],
                fine_voxel_size_m=list(tex.voxel_size),
                inv_fine_voxel_per_m=list(tex.inv_sdf_dx),
                fine_to_coarse=float(tex.fine_to_coarse),
                num_subgrids=int(tex.num_subgrids),
                native_box_m=[lo.tolist(), hi.tolist()],
                native_queries=[
                    dict(
                        position_m=p.tolist(),
                        distance_m=float(v),
                        gradient=g.tolist(),
                        normalized_direction=(g / np.linalg.norm(g)).tolist()
                        if np.linalg.norm(g) > 0
                        else None,
                        scalar_metric_gradient=(
                            (g * float(tex.fine_to_coarse) if cell[0] == 1 else g).tolist()
                            if cell[0] < 2
                            else None
                        ),
                        level=("fine", "coarse", "outside")[cell[0]],
                        native_cell_xyz=cell[1:].tolist(),
                        unchanged_cell_across_axis_difference=stencil.tolist(),
                        central_difference_per_m=derivative.tolist(),
                        central_step_m=step,
                        outside_texture_box=bool(np.any((p < lo) | (p > hi))),
                    )
                    for p, v, g, derivative, cell, stencil in zip(
                        host, vv, gg, fd, cells[0], stable, strict=True
                    )
                ],
            )
            print(kind, "free queries", len(host), "range", float(vv.min()), float(vv.max()))
    with args.out.open("x", encoding="utf-8") as fp:
        json.dump(
            dict(
                newton=version("newton"),
                warp=version("warp-lang"),
                device=str(device),
                interpolation="texture_sample_sdf_grad",
                assets=output,
            ),
            fp,
            allow_nan=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
