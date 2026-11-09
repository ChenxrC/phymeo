"""Pinned Newton 1.7.0 short reference for the four K0 first controls.

Import Newton only when executed. This does not import Phymeo, TYX, or the
K0 kernels, and it does not treat a Newton manifold as a Phymeo witness.
"""

import json
from pathlib import Path

import numpy as np

CASES = (
    ("K0-Q01-SS-sep", "SS", 0.02),
    ("K0-Q01-SS-overlap", "SS", -0.01),
    ("K0-Q01-CH-sep", "CH", 0.02),
    ("K0-Q01-CH-overlap", "CH", -0.01),
)
OUTPUT = Path(__file__).resolve().parents[3] / "artifacts" / "p65-generic-collision-20261107-01" / "k0-closeout" / "newton-short.json"


def hull_mesh():
    import newton

    ring = ((-0.5, -0.1), (0.5, -0.1), (0.6, 0.0), (0.3, 0.1), (-0.5, 0.1))
    vertices = np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)], dtype=np.float64)
    triangles = []
    top = [1, 3, 5, 7, 9]
    bottom = [8, 6, 4, 2, 0]
    for fan in (top, bottom):
        for i in range(1, 4):
            triangles.append((fan[0], fan[i], fan[i + 1]))
    for i in range(5):
        j = (i + 1) % 5
        triangles.append((2 * i, 2 * i + 1, 2 * j + 1))
        triangles.append((2 * i, 2 * j + 1, 2 * j))
    indices = np.array(triangles, dtype=np.int32).reshape(-1)
    return newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True)


def build(family, gap, device):
    import newton
    import warp as wp

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.default_shape_cfg
    cfg.gap = 0.0
    # Zero margin drops a 0.02 m surface gap from the AABB broad phase.
    # The margin only admits the pair; the recorded distance is Newton's
    # signed distance, not this margin.
    cfg.margin = 0.03
    cfg.is_hydroelastic = False
    shift = {"SS": 0.10 + gap, "CH": 0.15 + gap}[family]
    first = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), label="a")
    second = builder.add_body(xform=wp.transform((0.0, 0.0, shift), wp.quat_identity()), label="b")
    if family == "SS":
        builder.add_shape_sphere(first, radius=0.05, cfg=cfg, label="sphere_a")
        builder.add_shape_sphere(second, radius=0.05, cfg=cfg, label="sphere_b")
    else:
        # Newton capsules extend along local +Z. A +90 degree Y rotation
        # places that axis on the declared K0 +X segment of half-length 0.5.
        turn = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)
        builder.add_shape_capsule(first, xform=wp.transform((0.0, 0.0, 0.0), turn), radius=0.05, half_height=0.5, cfg=cfg, label="capsule")
        builder.add_shape_convex_hull(second, mesh=hull_mesh(), cfg=cfg, label="prism")
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=8, reduce_contacts=False, deterministic=True)
    return model, state, pipeline


def query(model, state, pipeline, device):
    import newton
    import warp as wp

    contacts = pipeline.contacts()
    gaps = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
    pipeline.collide(state, contacts)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=gaps)
    wp.synchronize_device(device)
    count = int(contacts.rigid_contact_count.numpy()[0])
    distance = gaps.numpy()[:count].astype(float).tolist() if count else []
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(float).tolist() if count else []
    margin0 = contacts.rigid_contact_margin0.numpy()[:count].astype(float).tolist() if count else []
    margin1 = contacts.rigid_contact_margin1.numpy()[:count].astype(float).tolist() if count else []
    return {
        "contact_count": count,
        "signed_distance_m": distance,
        "normal": normal,
        "margin0_m": margin0,
        "margin1_m": margin1,
        "shape_gap_m": model.shape_gap.numpy().astype(float).tolist(),
        "shape_margin_m": model.shape_margin.numpy().astype(float).tolist(),
    }


def summarize(row, declared):
    distances = row["signed_distance_m"]
    if not distances:
        chosen = None
        restored = None
        error = None
        normal = None
    else:
        index = min(range(len(distances)), key=lambda i: distances[i])
        chosen = distances[index]
        # Contact margin fields include the shape radius. Only the authored
        # shape margins were subtracted from Newton's signed distance.
        restored = chosen + float(sum(row["shape_margin_m"]))
        error = abs(restored - declared)
        normal = row["normal"][index]
    scale = 1.0
    row.update(
        declared_gap_m=declared,
        nearest_signed_distance_m=chosen,
        margin_restored_gap_m=None if chosen is None else restored,
        nearest_normal=normal,
        absolute_gap_error_m=error,
        within_1e_4_L=error is not None and error <= 1e-4 * scale,
    )
    return row


def run_device(device):
    rows = []
    for identity, family, gap in CASES:
        model, state, pipeline = build(family, gap, device)
        raw = query(model, state, pipeline, device)
        raw.update(identity=identity, family=family, device=device, collide_calls=1)
        rows.append(summarize(raw, gap))
    return rows


def main():
    import newton
    import warp as wp

    devices = ["cpu"]
    cuda_error = None
    try:
        if wp.is_cuda_available():
            devices.append("cuda")
    except Exception as error:
        cuda_error = f"{type(error).__name__}: {error}"
    report = {
        "schema": "k0-newton-short-v1",
        "newton_version": newton.__version__,
        "newton_file": newton.__file__,
        "identities": [item[0] for item in CASES],
        "cutoff": "shape gap is 0 and each shape margin is 0.03 m, only so the 0.02 m separation survives Newton's AABB broad phase. gjk_separation_cutoff stays active. Newton returns a signed distance with both shape margins removed; margin_restored_gap_m adds those two shape margins back. The contact margin fields also contain the radii and are not added.",
        "cuda_probe_error": cuda_error,
        "devices": [],
    }
    for device in devices:
        try:
            rows = run_device(device)
        except Exception as error:
            report["devices"].append({"device": device, "error": f"{type(error).__name__}: {error}"})
            continue
        report["devices"].append({
            "device": device,
            "collide_calls": len(rows),
            "within_1e_4_L": sum(row["within_1e_4_L"] for row in rows),
            "cases": rows,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)
    for block in report["devices"]:
        print(block["device"], block.get("error") or f"{block['within_1e_4_L']}/{block['collide_calls']}")
        for row in block.get("cases", []):
            print(" ", row["identity"], "raw", row["nearest_signed_distance_m"], "restored", row["margin_restored_gap_m"], "err", row["absolute_gap_error_m"], "n", row["contact_count"])


if __name__ == "__main__":
    main()
