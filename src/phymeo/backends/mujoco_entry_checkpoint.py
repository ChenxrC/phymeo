"""Raw accepted entry records at explicit reset/clone/checkpoint boundaries.

These helpers never construct contact data from HOST receipts. Receipts are
only consistency checks on bytes copied from the accepted device bank.
"""

from collections.abc import Mapping

import numpy as np

from .mujoco_tyx._entry_storage import entry_layout

ENTRY_SCHEMA = "mujoco-tyx-accepted-entry-v1"


def _require(condition, message):
    if not condition:
        raise ValueError("MuJoCo TYX entry: " + message)


def _entry_binding(runtime):
    layout = entry_layout(
        runtime._plan,
        len(runtime.world_keys),
        (runtime._contact_capacity, runtime._constraint_capacity),
    )
    binding = {
        "schema": ENTRY_SCHEMA,
        "phase": "accepted-step-input",
        "clock": "input-high, input-compensation, output-high, output-compensation",
        "meta": ("valid", "input_tick", "output_tick", "detected", "active", "overflow"),
        "contact_ids": ("valid", "active", "geom1", "geom2", "body1", "body2", "feature", "condim"),
        "contact_data": ("distance", "point3", "frame-rows9", "contact-wrench6", "include_margin"),
        "layout": {key: (shape, np.dtype(dtype).str) for key, (shape, dtype) in layout.items()},
    }
    if runtime._area_pressure_pairs:
        from .mujoco_area_pressure_device import AREA_SCHEMA

        binding["area_schema"] = AREA_SCHEMA
        binding["area_measure"] = "IEEE754-binary64-little-endian-low32-high32"
    return binding


def _accepted_entry(device):
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        return {name: device._get(banks[device.active]) for name, banks in device.entry.items()}


def _empty_entry(runtime, ticks):
    layout = entry_layout(
        runtime._plan,
        len(runtime.world_keys),
        (runtime._contact_capacity, runtime._constraint_capacity),
    )
    result = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in layout.items()}
    # A legacy checkpoint has no historical entry, but its next input tick is known.
    result["entry_meta"][:, 2] = [ticks[key] for key in runtime.world_keys]
    return result


def _validate_entry(runtime, record, ticks, receipts, clock):
    layout = entry_layout(
        runtime._plan,
        len(runtime.world_keys),
        (runtime._contact_capacity, runtime._constraint_capacity),
    )
    _require(isinstance(record, Mapping) and set(record) == set(layout), "record fields")
    for name, (shape, dtype) in layout.items():
        value = record[name]
        _require(
            isinstance(value, np.ndarray)
            and value.shape == shape
            and value.dtype == np.dtype(dtype)
            and np.isfinite(value).all(),
            "record layout: " + name,
        )
    nq, nv, na = runtime._model.nq, runtime._model.nv, runtime._model.na
    for row, key in enumerate(runtime.world_keys):
        meta = record["entry_meta"][row]
        valid, before, after, count, active, overflow = map(int, meta)
        _require(valid in (0, 1) and after == ticks[key], "validity/output tick")
        _require(overflow == 0, "accepted overflow")
        if not valid:
            _require(not np.any(meta[[0, 1, 3, 4, 5]]), "invalid record metadata")
            _require(
                all(
                    not np.any(value[row]) for name, value in record.items() if name != "entry_meta"
                ),
                "invalid record contains historical data",
            )
            continue
        receipt = receipts[key]
        _require(receipt is not None and before >= 0 and before + 1 == after, "input tick")
        _require(receipt.tick == after, "receipt tick")
        _require(0 <= active <= count <= runtime._contact_capacity, "contact counts")
        _require(len(receipt.contacts) == count, "receipt contact count")
        stamp = record["entry_clock"][row]
        _require(
            np.array_equal(stamp[2:], clock[row])
            and float(stamp[0]) - float(stamp[1]) == receipt.input_time
            and float(stamp[2]) - float(stamp[3]) == receipt.output_time,
            "input/output clock binding",
        )
        for name, width, expected in (
            ("entry_q", nq, receipt.input_q),
            ("entry_v", nv, receipt.input_v),
        ):
            _require(
                record[name][row, :width].tobytes() == np.asarray(expected, np.float32).tobytes()
                and not np.any(record[name][row, width:]),
                "native input binding: " + name,
            )
        act = record["entry_act"][row]
        if na:
            _require(
                act[:na].tobytes() == np.asarray(receipt.actuation.input_act, np.float32).tobytes(),
                "input activation binding",
            )
        _require(not np.any(act[na:]), "activation padding")
        ids, data = record["entry_contact_ids"][row], record["entry_contact_data"][row]
        _require(not np.any(ids[count:]) and not np.any(data[count:]), "contact padding")
        _require(np.all(ids[:count, 0] == 1), "contact validity")
        _require(np.all(np.isin(ids[:count, 1], (0, 1))), "contact active flag")
        _require(int(np.sum(ids[:count, 1])) == active, "active count")
        for ci, contact in enumerate(receipt.contacts):
            geom1, geom2 = contact.geometry_ids
            bodies = tuple(int(v) for v in runtime._model.geom_bodyid[[geom1, geom2]])
            expected_ids = (
                1,
                int(contact.active),
                geom1,
                geom2,
                *bodies,
                int(getattr(contact, "feature_index", 0)),
                contact.dimension,
            )
            _require(np.array_equal(ids[ci], expected_ids), "contact identity")
            expected_data = np.concatenate(
                (
                    [contact.distance],
                    contact.position,
                    np.asarray(contact.frame).reshape(-1),
                    contact.wrench_contact_frame,
                    [contact.include_margin],
                )
            ).astype(np.float32)
            _require(data[ci].tobytes() == expected_data.tobytes(), "contact device/receipt bytes")
    from .mujoco_area_pressure_device import validate_entries

    validate_entries(runtime, record, receipts)
    return record


def _write_entry(device, record):
    # Allowed only at explicit state restoration boundaries, not per-step packing.
    with device.tx.use_runtime(device.runtime):
        for name, value in record.items():
            device._put(device.entry[name][device.active], value)
        device.runtime.sync()
        device.counts["synchronizations"] += 1
    actual = _accepted_entry(device)
    _require(set(actual) == set(record), "restored record fields")
    for name, value in actual.items():
        expected = record[name]
        _require(
            value.shape == expected.shape
            and value.dtype == expected.dtype
            and value.tobytes() == expected.tobytes(),
            "restored record bytes: " + name,
        )


def _pack_entry(record):
    """Bound each serialized array without changing the resident entry layout."""
    from .mujoco_checkpoint import MAX_ARRAY_ITEMS

    packed = {}
    for name, value in record.items():
        if value.size <= MAX_ARRAY_ITEMS:
            packed[name] = value
        else:
            flat = value.reshape(-1)
            packed[name] = {
                "schema": "mujoco-tyx-entry-chunks-v1",
                "shape": value.shape,
                "dtype": value.dtype.str,
                "chunks": tuple(
                    flat[start : start + MAX_ARRAY_ITEMS]
                    for start in range(0, value.size, MAX_ARRAY_ITEMS)
                ),
            }
    return packed


def _unpack_entry(runtime, record):
    """Validate the bounded extension before allocating reconstructed HOST arrays."""
    from .mujoco_checkpoint import MAX_ARRAY_ITEMS

    layout = entry_layout(
        runtime._plan,
        len(runtime.world_keys),
        (runtime._contact_capacity, runtime._constraint_capacity),
    )
    _require(isinstance(record, Mapping) and set(record) == set(layout), "record fields")
    decoded = {}
    for name, (shape, dtype) in layout.items():
        value = record[name]
        if isinstance(value, np.ndarray):
            _require(
                value.shape == shape
                and value.dtype == np.dtype(dtype)
                and np.isfinite(value).all(),
                "entry array layout or data",
            )
            decoded[name] = value
            continue
        _require(
            isinstance(value, Mapping) and set(value) == {"schema", "shape", "dtype", "chunks"},
            "entry chunk fields",
        )
        size = 1
        for dimension in shape:
            size *= dimension
        expected_dtype = np.dtype(dtype)
        _require(
            size > MAX_ARRAY_ITEMS
            and value["schema"] == "mujoco-tyx-entry-chunks-v1"
            and type(value["shape"]) is tuple
            and all(type(dimension) is int for dimension in value["shape"])
            and value["shape"] == shape
            and value["dtype"] == expected_dtype.str,
            "entry chunk layout",
        )
        chunks = value["chunks"]
        count = (size + MAX_ARRAY_ITEMS - 1) // MAX_ARRAY_ITEMS
        _require(type(chunks) is tuple and len(chunks) == count, "entry chunk count")
        for index, chunk in enumerate(chunks):
            length = min(MAX_ARRAY_ITEMS, size - index * MAX_ARRAY_ITEMS)
            _require(
                isinstance(chunk, np.ndarray)
                and chunk.shape == (length,)
                and chunk.dtype == expected_dtype
                and chunk.nbytes == length * expected_dtype.itemsize
                and np.isfinite(chunk).all(),
                "entry chunk extent or data",
            )
        decoded[name] = np.concatenate(chunks).reshape(shape)
    return decoded
