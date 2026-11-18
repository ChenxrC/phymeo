"""Device area layout, receipt consistency and copy-owned batch export."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .mujoco_device_controls import _arrays, _overlap, _require

AREA_FIELDS = ("area_meta", "area_ids", "area_data", "area_measure_words")
COPY_ENTRY_FIELDS = ("entry_q", "entry_v", "entry_clock", "entry_meta") + AREA_FIELDS
AREA_SCHEMA = "mujoco-tyx-area-device-entry-v1"


def area_layout(plan, nworld, capacities):
    if not hasattr(plan, "area_pair_names"):
        return {}
    cap = max(1, capacities[1])
    return {
        "area_meta": ((nworld, 4), np.dtype("int32")),
        "area_ids": ((nworld, cap, 6), np.dtype("int32")),
        "area_data": (
            (nworld, cap, 23 if getattr(plan, "area_friction_laws", ()) else 15),
            np.dtype("float32"),
        ),
        "area_measure_words": ((nworld, cap, 2), np.dtype("int32")),
    }


def validate_row(record, index, receipt):
    """Readback validation only; never produce the accepted device record."""
    if record["area_data"].shape[-1] == 23:
        from .mujoco_area_friction_live import validate_device_row

        return validate_device_row(record, index, receipt)
    meta, ids, data, words = (record[k][index] for k in AREA_FIELDS)
    if receipt is None:
        _require(
            not any(np.any(v) for v in (meta, ids, data, words)),
            "invalid area record contains history",
        )
        return
    pairs = receipt.raw_contact_fields["area_pressure"]
    count = sum(len(p["normal_force"]) for p in pairs)
    active = sum(int(np.count_nonzero(p["normal_force"] > 0)) for p in pairs)
    _require(np.array_equal(meta, (1, count, active, 0)), "area entry metadata")
    _require(
        count <= len(ids)
        and not np.any(ids[count:])
        and not np.any(data[count:])
        and not np.any(words[count:]),
        "area entry padding/capacity",
    )
    cursor = 0
    for pi, pair in enumerate(pairs):
        n = len(pair["normal_force"])
        sl = slice(cursor, cursor + n)
        expected_ids = np.zeros((n, 6), np.int32)
        expected_ids[:, 0] = 1
        expected_ids[:, 1] = pi
        expected_ids[:, 2] = np.arange(cursor, cursor + n)
        expected_ids[:, 3:5] = pair["body_ids"]
        _require(np.array_equal(ids[sl, :5], expected_ids[:, :5]), "area device identities")
        expected = np.column_stack(
            (
                pair["points"],
                pair["normals"],
                pair["stiffness"],
                pair["surrogate_gap"],
                pair["normal_force"],
            )
        ).astype(np.float32)
        _require(np.array_equal(data[sl, :9], expected), "area device input/load bytes")
        exact_words = np.asarray(pair["dA"], dtype="<f8").view("<i4").reshape(n, 2)
        _require(np.array_equal(words[sl], exact_words), "area exact measure words")
        _require(
            np.allclose(data[sl, 9:12], pair["force_on_a"], rtol=3e-5, atol=2e-6),
            "area device world force",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            point, force = expected[:, :3], data[sl, 9:12]
            left = point[:, [1, 2, 0]] * force[:, [2, 0, 1]]
            right = point[:, [2, 0, 1]] * force[:, [1, 2, 0]]
        moment_valid = np.all(
            np.isfinite(left)
            & np.isfinite(right)
            & (np.abs(left) <= 2.0**125)
            & (np.abs(right) <= 2.0**125),
            axis=1,
        )
        _require(np.array_equal(ids[sl, 5], moment_valid.astype(np.int32)), "area moment validity")
        _require(not np.any(data[sl, 12:15][~moment_valid]), "unavailable moment sentinel")
        # Check this FP32 representation, not an unrelated FP64 cancellation
        # target that would narrow the already accepted physical domain.
        l64, r64 = left[moment_valid].astype(np.float64), right[moment_valid].astype(np.float64)
        bound = (
            8 * np.finfo(np.float32).eps * (np.abs(l64) + np.abs(r64)) + np.finfo(np.float32).tiny
        )
        _require(
            np.all(np.abs(data[sl, 12:15][moment_valid] - (l64 - r64)) <= bound),
            "area FP32 moment representation",
        )
        cursor += n


def validate_entries(runtime, record, receipts):
    if runtime._area_pressure_pairs:
        for i, key in enumerate(runtime.world_keys):
            validate_row(record, i, receipts[key])


def copy_layout(runtime):
    from .mujoco_tyx._entry_storage import entry_layout

    _require(bool(runtime._area_pressure_pairs), "runtime has no area declaration")
    layout = entry_layout(
        runtime._plan,
        len(runtime.world_keys),
        (runtime._contact_capacity, runtime._constraint_capacity),
    )
    result = {name: layout[name] for name in COPY_ENTRY_FIELDS}
    for name in ("q", "v"):
        result["endpoint_" + name] = layout["entry_" + name]
    result["endpoint_clock"] = ((len(runtime.world_keys), 2), np.dtype("float32"))
    return MappingProxyType(result)


def copy_accepted(runtime, outputs, expected_revision):
    """Synchronous copy; preflight writes nothing, copy failure returns no frame.

    On a transfer failure caller destinations may be partial. Physics is never
    changed. Callers must consume outputs only after successful return.
    """
    with runtime._lock:
        runtime._check_open()
        _require(
            type(expected_revision) is int and expected_revision == runtime._checkpoint_revision,
            "stale area observation revision",
        )
        layout = copy_layout(runtime)
        _require(isinstance(outputs, Mapping), "area output fields")
        outputs = dict(outputs)
        _require(set(outputs) == set(layout), "area output fields")
        device, mode = runtime._device, runtime._mode
        tx, ordinal = device.tx, int(device.runtime.device)
        forbidden, seen = [], set()
        owners = [
            device,
            runtime._control_packet,
            *runtime._control_retired,
            *runtime._checkpoint_retired,
        ]
        while owners:
            owner = owners.pop()
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            forbidden.extend(_arrays(owner, tx, mode))
            owners.append(getattr(owner, "_device", None))
            owners.extend(getattr(owner, "_checkpoint_retired", ()))
            owners.extend(getattr(owner, "_control_retired", ()))
        values = []
        for name, (shape, dtype) in layout.items():
            value = outputs[name]
            if mode == "interpret":
                _require(
                    isinstance(value, np.ndarray)
                    and value.dtype == dtype
                    and value.shape == shape
                    and value.flags.c_contiguous
                    and value.flags.writeable,
                    "area output array layout",
                )
            else:
                _require(
                    isinstance(value, tx.CudaArray)
                    and value.shape == shape
                    and np.dtype(value.host_dtype) == dtype
                    and value.device_ordinal == ordinal,
                    "area output device layout",
                )
                value._ensure_alive()
            _require(
                not any(_overlap(value, old, mode) for old in forbidden + values),
                "area outputs alias owned or supplied storage",
            )
            values.append(value)
        # Catalog is identity-only HOST metadata, never numerical output producer.
        catalogs = {}
        for key in runtime.world_keys:
            receipt = runtime._receipts[key]
            catalogs[key] = (
                ()
                if receipt is None
                else tuple(
                    MappingProxyType(
                        {
                            name: pair[name]
                            for name in ("identity", "stamp", "sample_ids", "source_ids")
                        }
                    )
                    for pair in receipt.raw_contact_fields[
                        "area_friction" if runtime._area_friction_laws else "area_pressure"
                    ]
                )
            )
        frame = MappingProxyType(
            dict(
                schema="mujoco-tyx-area-device-copy-v2"
                if runtime._area_friction_laws
                else "mujoco-tyx-area-device-copy-v1",
                revision=expected_revision,
                world_keys=runtime.world_keys,
                pair_names=runtime._plan.area_pair_names,
                catalogs=MappingProxyType(catalogs),
                measure_encoding="little-endian IEEE754 binary64: low32,high32",
                phase="accepted-step-input; endpoint fields separate",
                world_moment="area_ids[...,5] validity; zero sentinel when unavailable",
            )
        )
        sources = {name: device.entry[name][device.active] for name in COPY_ENTRY_FIELDS}
        sources.update(
            endpoint_q=device.q[device.active],
            endpoint_v=device.v[device.active],
            endpoint_clock=device.time[device.active],
        )
        with tx.use_runtime(device.runtime):
            device.runtime.sync()
            device.counts["synchronizations"] += 1
            for name, target in outputs.items():
                if mode == "interpret":
                    np.copyto(target, sources[name])
                else:
                    target.copy_from_device(sources[name])
            device.runtime.sync()
            device.counts["synchronizations"] += 1
        runtime._extra_counts["area_copy_bytes"] += sum(v.nbytes for v in outputs.values())
        runtime._extra_counts["area_copies"] += 1
        return frame
