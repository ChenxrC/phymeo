"""Bounded column export of detached MuJoCo contact records.

The runtime already detaches contact arrays from native MjData. Recursively
applying dataclasses.asdict adds a NumPy deepcopy for every array of every
contact. This module copies numeric fields directly into one column per field,
without calling deepcopy or executing physics. No contact rows are filtered.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .mujoco_runtime import MuJoCoContact

CONTACT_COLUMN_SCHEMA = "phymeo-mujoco-contact-columns-v1"
# Preserve all native runtime fields, including inactive rows and cone Hessians.
_LAYOUT = (
    ("index", "i8", ()),
    ("geometry_ids", "i8", (2,)),
    ("flex_ids", "i8", (2,)),
    ("element_ids", "i8", (2,)),
    ("vertex_ids", "i8", (2,)),
    ("dimension", "i8", ()),
    ("distance", "f8", ()),
    ("position", "f8", (3,)),
    ("frame", "f8", (3, 3)),
    ("include_margin", "f8", ()),
    ("friction", "f8", (5,)),
    ("solref", "f8", (2,)),
    ("solref_friction", "f8", (2,)),
    ("solimp", "f8", (5,)),
    ("regularized_mu", "f8", ()),
    ("cone_hessian", "f8", (36,)),
    ("excluded", "i8", ()),
    ("constraint_address", "i8", ()),
    ("wrench_contact_frame", "f8", (6,)),
)
CONTACT_ROW_BYTES = sum(
    np.dtype(dtype).itemsize * int(np.prod(shape, dtype=np.int64)) for _, dtype, shape in _LAYOUT
)
_ID_PAIRS = frozenset({"geometry_ids", "flex_ids", "element_ids", "vertex_ids"})


@dataclass(frozen=True, slots=True)
class MuJoCoContactColumns:
    """A detached, immutable numeric payload created by contacts_to_columns.

    Arrays retain independent immutable byte buffers. Holding this packet does
    not keep a native model, MjData, runtime, or source contact alive. Native
    solver/endpoint phase, world, tick, and time belong in the caller's envelope;
    export the two phases separately instead of concatenating unlabeled rows.
    """

    count: int
    arrays: Mapping[str, np.ndarray]
    schema: str = CONTACT_COLUMN_SCHEMA

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self.arrays.values())

    def contact(self, index):
        """Reconstruct one record without copying array data or losing types."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("contact index must be an integer")
        if not 0 <= index < self.count:
            raise IndexError("contact index outside column payload")
        values = {}
        for name, dtype, shape in _LAYOUT:
            value = self.arrays[name][index]
            if name in _ID_PAIRS:
                value = tuple(int(item) for item in value)
            elif not shape:
                value = int(value) if dtype == "i8" else float(value)
            values[name] = value
        return MuJoCoContact(**values)


def contacts_to_columns(contacts, *, max_bytes=64 * 1024**2):
    """Export all 19 contact fields without recursive NumPy deepcopy.

    max_bytes caps the final numeric payload, not total process RSS. While
    freezing a column, its staging allocation and byte copy coexist, so peak
    numeric allocation is at most payload bytes plus the largest column.
    Caller-owned contacts, Python objects, and allocator overhead are excluded.
    An over-budget batch fails before allocating any output columns.

    Native float64/int64 values and shapes are required; no narrowing, finite
    filtering, row truncation, or solver-parameter changes are performed. This
    also preserves NaN bit patterns in inactive cone Hessian fields. A packet
    is published only after every field has passed validation and been copied.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(contacts, Sequence):
        raise TypeError("contacts must be a finite sequence of MuJoCoContact records")
    count = len(contacts)
    if count * CONTACT_ROW_BYTES > max_bytes:
        raise ValueError("contact column payload exceeds max_bytes; no rows were truncated")
    records = tuple(contacts)
    if len(records) != count:
        raise ValueError("contact sequence changed while exporting")
    if any(not isinstance(record, MuJoCoContact) for record in records):
        raise TypeError("contacts must contain MuJoCoContact records")
    columns = {}
    for name, dtype, shape in _LAYOUT:
        expected_dtype = np.dtype(dtype)
        column = np.empty((count, *shape), dtype=expected_dtype)
        for index, record in enumerate(records):
            value = np.asarray(getattr(record, name))
            if value.shape != shape or value.dtype != expected_dtype:
                raise ValueError(
                    f"contact {index} field {name} must have dtype {expected_dtype} "
                    f"and shape {shape}, got {value.dtype} and {value.shape}"
                )
            column[index] = value
        # Setting WRITEABLE=False on an owning ndarray is reversible. Immutable
        # bytes backing also prevents consumers from re-enabling writes.
        frozen = np.frombuffer(column.tobytes(order="C"), dtype=expected_dtype).reshape(
            column.shape
        )
        columns[name] = frozen
        del column
    return MuJoCoContactColumns(count, MappingProxyType(columns))
