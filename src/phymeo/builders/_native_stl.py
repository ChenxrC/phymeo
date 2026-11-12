"""Cold ASCII STL encoding for the native compiler's binary-only decoder."""

import math
import struct


def binary_stl(content):
    """Preserve binary bytes; encode ASCII triangles without changing topology."""
    # A binary header may begin with 'solid'. Size/count takes precedence.
    if len(content) >= 84:
        count = struct.unpack_from("<I", content, 80)[0]
        if len(content) == 84 + 50 * count:
            return content
    if not content.lstrip().startswith(b"solid"):
        return content  # Let the native decoder diagnose other invalid binary files.
    try:
        lines = [line.split() for line in content.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise ValueError("invalid ASCII STL encoding") from exc
    if len(lines) < 2 or lines[0][0] != "solid" or lines[-1][0] != "endsolid":
        raise ValueError("ASCII STL requires solid/endsolid")
    facets = lines[1:-1]
    if not facets or len(facets) % 7:
        raise ValueError("ASCII STL requires complete triangular facets")
    output = bytearray(b"PHYMEO ASCII STL".ljust(80, b"\0"))
    output.extend(struct.pack("<I", len(facets) // 7))
    for i in range(0, len(facets), 7):
        normal, begin, a, b, c, end, close = facets[i : i + 7]
        if (
            normal[:2] != ["facet", "normal"] or len(normal) != 5
            or begin != ["outer", "loop"] or end != ["endloop"] or close != ["endfacet"]
            or any(len(v) != 4 or v[0] != "vertex" for v in (a, b, c))
        ):
            raise ValueError("invalid ASCII STL facet grammar")
        try:
            values = tuple(float(x) for x in (*normal[2:], *a[1:], *b[1:], *c[1:]))
            if not all(math.isfinite(x) for x in values):
                raise ValueError("nonfinite coordinate")
            output.extend(struct.pack("<12fH", *values, 0))
        except (ValueError, OverflowError) as exc:
            raise ValueError("ASCII STL requires finite float32 coordinates") from exc
    return bytes(output)
