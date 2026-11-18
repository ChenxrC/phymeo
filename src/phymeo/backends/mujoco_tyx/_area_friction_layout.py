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


def scratch_layout(nworld):
    if type(nworld) is not int or not 1 <= nworld <= 32:
        raise ValueError("live area friction requires 1..32 worlds per bucket")
    w = nworld
    return {
        "sizes": ((w, 4), "int32"),
        "mass": ((w, 6, 6), "float32"),
        "inputs": ((w, 3, 6), "float32"),
        "steps": ((w,), "float32"),
        "samples": ((w, 128, 10), "float32"),
        "ja": ((w, 128, 3, 6), "float32"),
        "jb": ((w, 128, 3, 6), "float32"),
        "matrices": ((w, 4, 6, 6), "float32"),
        "vectors": ((w, 8, 6), "float32"),
        "contact": ((w, 128, 17), "float32"),
        "meta": ((w, 5), "int32"),
        "scalars": ((w, 3), "float32"),
        "trace_meta": ((w, 290, 4), "int32"),
        "trace": ((w, 290, 15), "float32"),
        "pack_status": ((w,), "int32"),
    }
