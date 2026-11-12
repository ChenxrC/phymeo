"""Owned cold display geometry; no solver state or renderer dependency."""

from dataclasses import dataclass

import numpy as np


def immutable_array(values, dtype, width):
    array = np.asarray(values, dtype=dtype).reshape(-1, width)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("visual buffer contains nonfinite values")
    # Bytes backing prevents consumers from re-enabling writeability.
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class VisualMesh:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray | None = None

    def __post_init__(self):
        vertices = immutable_array(self.vertices, np.float32, 3)
        triangles = immutable_array(self.triangles, np.int32, 3)
        if (
            not len(vertices)
            or not len(triangles)
            or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))
        ):
            raise ValueError("visual mesh requires valid vertex/triangle buffers")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        if self.uvs is not None:
            uvs = immutable_array(self.uvs, np.float32, 2)
            if len(uvs) != len(vertices):
                raise ValueError("visual UVs must share the exported vertex index space")
            object.__setattr__(self, "uvs", uvs)


@dataclass(frozen=True)
class VisualInstance:
    name: str
    body: str | None
    kind: str
    size: tuple
    position: tuple
    rotation_xyzw: tuple
    rgba: tuple
    mesh: str | None = None
    source_role: str = "native_geom"
    visible: bool = True
    material: tuple = ()


@dataclass(frozen=True)
class VisualScene:
    instances: tuple[VisualInstance, ...]
    meshes: tuple[VisualMesh, ...] = ()
    contract_version: int = 1

    def __post_init__(self):
        names = [item.name for item in self.instances]
        meshes = [item.name for item in self.meshes]
        if len(names) != len(set(names)) or len(meshes) != len(set(meshes)):
            raise ValueError("visual identities must be unique")
        for item in self.instances:
            if item.mesh is not None and item.mesh not in meshes:
                raise ValueError("visual instance references an unknown mesh")
