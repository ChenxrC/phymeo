"""Static visual resources shared by scene instances."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Mesh:
    """Visual mesh URI, authored in metres in its local coordinate frame."""

    id: str
    uri: str


@dataclass(frozen=True, slots=True)
class Material:
    """Opaque metallic-roughness material; base color is linear RGB."""

    id: str
    base_color: tuple[float, float, float]
    metallic: float
    roughness: float
