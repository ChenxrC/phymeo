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


@dataclass(frozen=True, slots=True)
class TextureMap:
    """Opaque RGB image pinned by its original file bytes.

    The declaration does not read the image or import Pillow. Sampling is repeat
    with bilinear filtering, with (0, 0) at the top left as in glTF. This first
    contract has no alpha blending, UV transform or sampler override. Color maps
    use sRGB; normal and roughness maps interpret the stored bytes as linear data.
    """

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_text(self.uri, "uri", "TextureMap")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.sha256)):
            raise ValueError("TextureMap.sha256 must be a lowercase SHA256 hex digest")


@dataclass(frozen=True, slots=True)
class TextureChannel:
    """One linear data channel of an image, on the mesh's primary UV set."""

    image: TextureMap
    channel: str = "r"

    def __post_init__(self) -> None:
        if not isinstance(self.image, TextureMap):
            raise TypeError("TextureChannel.image must be a TextureMap")
        if self.channel not in ("r", "g", "b"):
            raise ValueError("TextureChannel.channel must be r, g or b")


@dataclass(frozen=True, slots=True)
class TexturedMaterial(Material):
    """UV maps: sRGB color, linear roughness and linear tangent-space normals.

    normal_map uses glTF +Y, normalize(2*c-1), with fixed strength one. It changes
    shading only. Backends must explicitly support it; unsupported inputs fail.
    The optional field is last to preserve existing positional construction.
    """

    base_color_map: TextureMap | None = None
    roughness_map: TextureChannel | None = None
    normal_map: TextureMap | None = None

    def __post_init__(self) -> None:
        Material.__post_init__(self)
        if self.normal_map is not None and not isinstance(self.normal_map, TextureMap):
            raise TypeError(f"Material {self.id!r}.normal_map must be a TextureMap")
        if self.base_color_map is None and self.roughness_map is None and self.normal_map is None:
            raise TypeError(f"Material {self.id!r} needs a color, roughness or normal texture")
        if self.base_color_map is not None and not isinstance(self.base_color_map, TextureMap):
            raise TypeError(f"Material {self.id!r}.base_color_map must be a TextureMap")
        if self.roughness_map is not None and not isinstance(self.roughness_map, TextureChannel):
            raise TypeError(f"Material {self.id!r}.roughness_map must be a TextureChannel")
