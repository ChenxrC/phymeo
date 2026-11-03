"""Original sparse/dense sphere rows and two geometrically independent contact holdouts."""

import json
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np
from g4_geometry_cases import copies, frozen

import phymeo as ph

NAMES = ("sparse65", "sparse65-one", "dense34", "dense34-split")


def sphere_case(name, world):
    base = "sparse65" if name.startswith("sparse65") else "dense34"
    mod, start = frozen(base)
    groups, frames = copies(mod, start, (world,))
    model = groups[0][1]
    paths = frames[0][world]
    if name == "sparse65-one":
        key = ph.CollisionKey(world, 64, 0)
        paths[key] = ph.CollisionMotion([[0, 3 / 16, 0]])
    elif name == "dense34-split":
        paths = {
            k: ph.CollisionMotion(p.start + [0, 4, 0]) if k.body % 2 else p
            for k, p in paths.items()
        }
    elif name not in ("sparse65", "dense34"):
        raise ValueError(name)
    return model, paths


def check_spheres(model, paths, buffer):
    assert buffer.status == "ok" and not buffer.diagnostics
    expected = []
    for i, a in enumerate(model.shapes):
        for b in model.shapes[i + 1 :]:
            x, y = paths[a.key].start[0], paths[b.key].start[0]
            distance = float(np.linalg.norm(y - x))
            if distance <= a.radius + b.radius:
                expected.append((a, b, x, y, distance))
    assert (
        len(expected)
        == buffer.required_candidates
        == buffer.required_contacts
        == len(buffer.contacts)
    )
    for (a, b, x, y, distance), row in zip(expected, buffer.contacts, strict=True):
        assert (row.first, row.second) == (a.key, b.key)
        normal = (y - x) / distance
        assert (
            row.status == "overlap"
            and abs(row.witness.gap_m - (distance - a.radius - b.radius)) < 2e-12
        )
        np.testing.assert_allclose(row.witness.normal, normal, atol=1e-12)
        np.testing.assert_allclose(row.witness.first_point, x + a.radius * normal, atol=1e-12)
        np.testing.assert_allclose(row.witness.second_point, y - b.radius * normal, atol=1e-12)
    doc = asdict(replace(buffer, backend="canonical"))
    return sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
