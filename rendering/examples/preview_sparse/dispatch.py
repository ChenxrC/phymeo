"""Explicit private opt-in: ordinary DLLs never resolve sparse ABI symbols."""
import ctypes as ct


def progressive_setter(device):
    spatial=bool(getattr(device,"_spatial_requested",False))
    settling=bool(getattr(device,"_settling_requested",False))
    sparse=bool(getattr(device,"_preview_sparse_requested",False))
    if sparse:
        if not spatial or settling:
            raise ValueError("sparse requires its exclusive spatial transport")
        version=device._lib.pr_preview_sparse_abi_version
        version.argtypes=[];version.restype=ct.c_uint32
        if version()!=1:
            raise ValueError("private sparse ABI differs")
        return device._lib.pr_preview_sparse_set
    if settling:
        return device._lib.pr_resident_set_settling
    if spatial:
        return device._lib.pr_resident_set_spatial
    return device._lib.pr_resident_set_progressive
