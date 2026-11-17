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
class VisualTexture:
    """Owned compiled texels; projection is explicit, never inferred from shape."""

    name: str
    kind: str
    pixels: np.ndarray
    source_name: str = ""
    native_color_space: int = 0

    def __post_init__(self):
        values = np.asarray(self.pixels)
        if self.kind not in {"2d", "cube", "skybox"}:
            raise ValueError("unknown visual texture projection")
        if (
            values.dtype != np.uint8
            or values.ndim != 3
            or values.shape[2] not in (1, 3, 4)
            or min(values.shape) <= 0
        ):
            raise ValueError("visual texture requires nonempty HWC uint8 texels")
        object.__setattr__(
            self, "pixels", np.frombuffer(values.tobytes(), np.uint8).reshape(values.shape)
        )


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
    textures: tuple[VisualTexture, ...] = ()

    def __post_init__(self):
        names = [item.name for item in self.instances]
        meshes = [item.name for item in self.meshes]
        textures = [item.name for item in self.textures]
        if len(textures) != len(set(textures)):
            raise ValueError("visual texture identities must be unique")
        if self.textures and self.contract_version != 2:
            raise ValueError("visual textures require contract v2")
        if len(names) != len(set(names)) or len(meshes) != len(set(meshes)):
            raise ValueError("visual identities must be unique")
        for item in self.instances:
            if item.mesh is not None and item.mesh not in meshes:
                raise ValueError("visual instance references an unknown mesh")
            for _, texture in dict(item.material).get("texture_bindings", ()):
                if texture not in textures:
                    raise ValueError("visual material references an unknown texture")
