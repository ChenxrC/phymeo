"""Metadata-only bounded resident friction scratch, part of the owner ledger."""

FRICTION_FIELDS = (
    "sizes",
    "mass",
    "inputs",
    "steps",
    "samples",
    "ja",
    "jb",
    "matrices",
    "vectors",
    "contact",
    "meta",
    "scalars",
    "trace_meta",
    "trace",
)


MAX_LIVE_DOF = 8


def friction_width(nv):
    """Preserve the six-column ABI for existing models."""
    if not 1 <= nv <= MAX_LIVE_DOF:
        raise ValueError("live area-friction capacity: 1..8 DOF")
    return 6 if nv <= 6 else MAX_LIVE_DOF


def scratch_layout(nworld, nv):
    width = friction_width(nv)
    if type(nworld) is not int or not 1 <= nworld <= 32:
        raise ValueError("live area friction requires 1..32 worlds per bucket")
    w = nworld
    return {
        "sizes": ((w, 4), "int32"),
        "mass": ((w, width, width), "float32"),
        "inputs": ((w, 3, width), "float32"),
        "steps": ((w,), "float32"),
        "samples": ((w, 128, 10), "float32"),
        "ja": ((w, 128, 3, width), "float32"),
        "jb": ((w, 128, 3, width), "float32"),
        "matrices": ((w, 4, width, width), "float32"),
        "vectors": ((w, 8, width), "float32"),
        "contact": ((w, 128, 17), "float32"),
        "meta": ((w, 5), "int32"),
        "scalars": ((w, 3), "float32"),
        "trace_meta": ((w, 290, 4), "int32"),
        "trace": ((w, 290, 3 + 2 * width), "float32"),
        "pack_status": ((w,), "int32"),
    }
