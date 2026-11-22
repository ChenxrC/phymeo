"""Static visual resources shared by scene instances."""

from dataclasses import dataclass
from numbers import Real


def _validate_text(value: object, field: str, asset: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{asset}.{field} must be a string")
    if not value.strip():
        raise ValueError(f"{asset}.{field} must not be blank")


def _validate_unit_interval(value: object, field: str, asset: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{asset}.{field} must be a real number, not bool")
    # This rejects NaN and infinities as well as out-of-range finite values,
    # without a float conversion that could overflow for a large integer.
    if not 0 <= value <= 1:
        raise ValueError(f"{asset}.{field} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class Mesh:
    """Visual mesh URI, authored in metres in its local coordinate frame.

    Identity and URI are nonblank strings retained verbatim. Construction
    declares a resource; it does not resolve the URI or access the resource.
    """

    id: str
    uri: str

    def __post_init__(self) -> None:
        asset = f"Mesh {self.id!r}"
        _validate_text(self.id, "id", asset)
        _validate_text(self.uri, "uri", asset)


@dataclass(frozen=True, slots=True)
class Material:
    """Opaque metallic-roughness material; base color is linear RGB.

    Color is an immutable three-tuple. Color components, metallic and
    roughness are finite real numbers in [0, 1], excluding bool. Validation
    preserves values without clamping, float coercion or color conversion.
    """

    id: str
    base_color: tuple[float, float, float]
    metallic: float
    roughness: float

    def __post_init__(self) -> None:
        asset = f"Material {self.id!r}"
        _validate_text(self.id, "id", asset)
        if not isinstance(self.base_color, tuple):
            raise TypeError(f"{asset}.base_color must be an immutable tuple")
        if len(self.base_color) != 3:
            raise ValueError(f"{asset}.base_color must have exactly 3 components")
        for index, component in enumerate(self.base_color):
            _validate_unit_interval(component, f"base_color[{index}]", asset)
        _validate_unit_interval(self.metallic, "metallic", asset)
        _validate_unit_interval(self.roughness, "roughness", asset)
