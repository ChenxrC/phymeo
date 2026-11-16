"""Versioned borrowed storage metadata, independent of a solver or training API."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class RuntimeField:
    name: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    frame: str
    index_space: str
    sample: str
    writable: bool = False


class RuntimeStorage:
    """Retain the owner; borrowed arrays are invalid after owner close/replacement.

    Borrow once when preparing consumer operations. A returned device array is
    not a safe external handle: the consumer must call require_valid at its own
    admission boundary, and may only write the prepared candidate bank. Raw
    aliases cannot be revoked by Python. No array copy or device ownership is
    transferred by this object. All accesses use the canonical stream, and the
    consumer must prevent concurrent owner close for the entire borrowed use.
    """

    contract_version = 1
    write_precondition = "caller-validated finite values and physical coordinate constraints"
    post_execution_validation = "reject publication; does not suppress intermediate native substeps"

    def __init__(self, *, owner, generation, fields, banks, stream, device):
        self._owner = owner
        self.generation = generation
        self.fields = MappingProxyType({field.name: field for field in fields})
        self._banks = tuple(MappingProxyType(dict(bank)) for bank in banks)
        self.stream, self.device = stream, device

    def require_valid(self):
        if self._owner.storage_generation != self.generation:
            raise RuntimeError("runtime storage binding is stale or closed")

    def borrow(self, field, bank, *, write=False):
        self.require_valid()
        if type(bank) is not int or not 0 <= bank < len(self._banks):
            raise ValueError("invalid storage bank")
        if write and (not self.fields[field].writable or
                      self._owner.writable_bank != bank):
            raise RuntimeError("field writes require the prepared candidate bank")
        return self._banks[bank][field]

    def _release(self):
        # Called only after the native owner has completed teardown successfully.
        self._banks = ()
        self.stream = None
