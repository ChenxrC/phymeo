"""Pinned, complete classic Panda visual declarations, with no device or physics.

panda_visual_assets(asset_dir) returns immutable (meshes, materials, instances).
It verifies the exact published manifest, all prepared meshes, source archives
and license notices before returning any resource. The optional directory
defaults to the wheel's package-local resources, or rendering/assets/panda in a
source checkout. Explicit overrides retain the same pinned integrity checks.

Mesh positions already include each COLLADA visual-scene matrix in the authored
URDF link's metre frame. Instances attach to panda_link0..7, panda_hand and both
finger link names. Only the right finger adds the URDF visual Rz(pi). IDs 100..158
are stable per source part, not per-link labels. This asset set has no renderer
capacity exemption: the complete robot contains 134,912 surface triangles (134,924 source indices; 12 exact zero-area indices archived).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .assets import Material, Mesh
from .scene import Instance, Pose

_MANIFEST_SHA256 = "d6c5f952ec5e293403d37962a5a462a70fb1359307479c87e744dfb80b4846e4"
_MAX_MANIFEST = 512 * 1024
_MAX_RESOURCE = 8 * 1024 * 1024


def _verified(path: Path, expected: str, limit: int) -> bytes:
    if not path.is_file():
        raise ValueError(f"missing regular asset file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"asset exceeds bounded file size: {path}")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"asset SHA256 digest mismatch: {path}")
    return data


def _resource(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("manifest resource must be a contained relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("manifest resource escapes asset directory")
    return resolved


def _default_asset_dir() -> Path:
    """Resolve stable filesystem resources without extraction or SDK imports.

    pip installs the data namespace beside this module. The explicit source
    fallback only applies to the src/ layout; a damaged installed bundle never
    falls back to a repository or a cache discovered elsewhere on the machine.
    """
    package = Path(__file__).resolve().parent
    installed = package / "_panda_assets"
    if installed.exists() or package.parent.name != "src":
        return installed
    return package.parent.parent / "assets" / "panda"


def panda_visual_assets(asset_dir=None) -> tuple[tuple[Mesh, ...], tuple[Material, ...], tuple[Instance, ...]]:
    """Load the fixed manufacturer's prepared visual set, or reject corruption.

    No download, simplification, proxy substitution, renderer import, cache or
    physical state mutation occurs. Material colors and roughness are a declared
    Phong-to-PBR approximation; the manifest preserves the original parameters.
    The six source CAD line segments are archived but cannot be submitted through
    the current triangle-only mesh contract.
    """
    if asset_dir is None:
        asset_dir = _default_asset_dir()
    root = Path(asset_dir).resolve()
    manifest = json.loads(_verified(root / "manifest.json", _MANIFEST_SHA256, _MAX_MANIFEST))
    for row in manifest["licenses"] + manifest["visual_placement_sources"]:
        _verified(_resource(root, row["file"]), row["sha256"], _MAX_RESOURCE)
    for row in manifest["sources"]:
        _verified(_resource(root, row["archive_file"]), row["archive_sha256"], _MAX_RESOURCE)
    meshes = []
    for row in manifest["meshes"]:
        path = _resource(root, row["file"])
        _verified(path, row["sha256"], _MAX_RESOURCE)
        meshes.append(Mesh(row["id"], path.as_uri()))
    materials = tuple(
        Material(row["id"], tuple(row["base_color"]), row["metallic"], row["roughness"])
        for row in manifest["materials"]
    )
    instances = tuple(
        Instance(row["id"], row["mesh_id"], row["material_id"], row["node_id"],
                 Pose(tuple(row["position_m"]), tuple(row["orientation_xyzw"])))
        for row in manifest["instances"]
    )
    return tuple(meshes), materials, instances
