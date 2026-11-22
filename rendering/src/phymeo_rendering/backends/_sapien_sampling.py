"""Create a local, explicitly seeded derivative of the installed reference RT pack."""
from pathlib import Path
from shutil import copytree


def make_seeded_pack(original: Path, destination: Path, seed: int) -> Path:
    source = (original / "camera.rgen").read_text(encoding="utf-8")
    changes = {
        "uint magic = 0x9e3779b9;": f"uint magic = 0x{seed:08x}u;",
        "ray.seed = seed;": """ray.seed = seed;
  // A randomized shift of the primary Halton sequence, decorrelated from
  // the path RNG. No draw from the path sequence is consumed by the shift.
  uint pixelIndex = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
  uint shiftState = tea(pixelIndex, magic ^ 0x7f4a7c15u);
  float shiftX = rnd(shiftState);
  float shiftY = rnd(shiftState);
  vec2 pixelShift = vec2(shiftX, shiftY);""",
        "vec2 r = halton2d(frameCount * spp + i, 2, 3);":
            "vec2 r = fract(halton2d(frameCount * spp + i, 2, 3) + pixelShift);",
    }
    for anchor, replacement in changes.items():
        if source.count(anchor) != 1:
            raise ValueError("installed RT sampling code differs from the supported shader")
        source = source.replace(anchor, replacement)
    copytree(original, destination)
    (destination / "camera.rgen").write_text(source, encoding="utf-8")
    return destination
