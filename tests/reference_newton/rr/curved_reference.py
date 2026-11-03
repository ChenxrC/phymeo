"""Phymeo's static native curved solids at the catalog's prescribed poses."""

import numpy as np


def _shape(ph, key, asset, node):
    kind = asset["kind"]
    if kind == "plane":
        return ph.CollisionShape(key, kind, (node,), normal=asset.get("normal", (0, 0, 1)))
    if kind == "sphere":
        return ph.CollisionShape(key, kind, (node,), asset["radius_m"])
    if kind == "ellipsoid":
        dims = asset["radii_xyz_m"]
    elif kind in ("cylinder", "cone"):
        dims = (asset["radius_m"], asset["half_height_m"])
    else:
        raise ValueError(f"Unknown curved reference shape {kind}")
    solid = ph.CurvedSolid(kind, dims, asset.get("local_quat_xyzw", (0, 0, 0, 1)))
    return ph.CollisionShape(key, kind, (node,), curved=solid)


def _check_boundary(shape, offset, tolerance):
    from phymeo.dynamics.rotation import quaternion_matrix

    offset = np.asarray(offset)
    if shape.kind == "sphere":
        assert abs(np.linalg.norm(offset) - shape.radius) <= tolerance, offset
    elif shape.kind == "plane":
        n = np.asarray(shape.normal)
        assert abs(offset @ (n / np.linalg.norm(n))) <= tolerance, offset
    else:
        rotation = quaternion_matrix(np.asarray([shape.curved.orientation_xyzw]))[0]
        x, y, z = rotation.T @ offset
        dims = np.asarray(shape.curved.dimensions_m)
        if shape.kind == "ellipsoid":
            assert abs(np.linalg.norm([x, y, z] / dims) - 1) <= tolerance / min(dims)
        else:
            r, h = dims
            radial = np.hypot(x, y)
            bound = r if shape.kind == "cylinder" else r * (h - z) / (2 * h)
            cap = abs(abs(z) - h) if shape.kind == "cylinder" else abs(z + h)
            assert abs(z) <= h + tolerance and radial <= bound + tolerance
            assert min(abs(radial - bound), cap) <= tolerance


def phymeo_curved(case, inputs, assets, device):
    import phymeo as ph

    if device != "cpu" or inputs["query"] != "sampled_poses":
        raise ValueError("The native curved reference measures static HOST poses")
    keys = [ph.CollisionKey(0, i + 1, i) for i in range(2)]
    if inputs.get("reverse_registration"):
        keys.reverse()
    shapes = [
        _shape(ph, key, asset, i + 1)
        for i, (key, asset) in enumerate(zip(keys, assets, strict=True))
    ]
    model = ph.CollisionModel(shapes)
    paths = np.asarray([inputs["a_path_m"], inputs["b_path_m"]], dtype=float)
    tol = case["tolerances"]["length_m"]
    samples = []
    with ph.CollisionWorkspace(model) as work:
        for t in inputs["sample_fractions"]:
            pose = paths[:, 0] + t * (paths[:, 1] - paths[:, 0])
            motions = {
                key: ph.CollisionMotion([origin]) for key, origin in zip(keys, pose, strict=True)
            }
            # Retain the actual separated witnesses too. Broad-phase slack does not inflate either surface.
            policy = ph.CollisionPolicy(margin=float(np.linalg.norm(pose[0] - pose[1])))
            result = ph.detect(model, motions, policy, work)
            assert result.status == "ok" and work.last_result is result
            (pair,) = result.pairs
            assert pair.candidate and pair.family == "RR" and pair.sweep is None
            assert pair.status in ("overlap", "touching", "separated")
            assert {pair.first, pair.second} == set(keys)
            trace, witness = pair.curved, pair.witness
            owner = next(i for i, shape in enumerate(shapes) if shape.curved is not None)
            assert trace.owner == keys[owner] and witness is not None
            # These authored poses have a well-defined separating/penetration normal.
            assert witness.normal is not None
            normal = np.asarray(witness.normal)
            supports = [witness.first_point, witness.second_point]
            offsets = [witness.first_offset, witness.second_offset]
            material = [witness.first_weights, witness.second_weights]
            if keys[0] != pair.first:
                supports.reverse()
                offsets.reverse()
                material.reverse()
                normal = -normal
            for i, shape in enumerate(shapes):
                assert material[i] == ((shape.vertex_ids[0], 1.0),)
                _check_boundary(shape, offsets[i], tol)
                np.testing.assert_allclose(pose[i] + offsets[i], supports[i], rtol=0, atol=tol)
            np.testing.assert_allclose(
                np.subtract(supports[1], supports[0]), witness.gap_m * normal, rtol=0, atol=tol
            )
            samples.append(
                dict(
                    fraction=t,
                    prescribed_time_s=t * inputs["duration_s"],
                    origins_world_m=pose.tolist(),
                    status=pair.status,
                    contact=pair.status != "separated",
                    gap_m=witness.gap_m,
                    normal_a_to_b=normal.tolist(),
                    a_support_world_m=supports[0],
                    b_support_world_m=supports[1],
                    a_material_id=material[0][0][0],
                    a_offset_m=offsets[0],
                    b_material_id=material[1][0][0],
                    b_offset_m=offsets[1],
                    curved_shape_key=(trace.owner.world, trace.owner.body, trace.owner.shape),
                    curved_features=trace.features,
                    curved_locus=trace.locus,
                    curved_points_m=trace.points_m,
                )
            )
        return dict(
            backend=work.stats()["backend"], shape_types=[s.kind for s in shapes], samples=samples
        )
