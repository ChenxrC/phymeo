"""Exclusive wall-clock attribution for a single-threaded live area-friction experiment.

Hooks exist only inside the diagnostic step. The ordinary Scene.step is the
throughput reference. Fenced kernel stages include Python dispatch and waiting.
"""

from collections import Counter
from contextlib import ExitStack
from functools import wraps
from time import perf_counter_ns
from unittest.mock import patch

from phymeo.backends import (
    _tyx_area_friction_entry,
    _tyx_area_friction_kernels,
    mujoco_area_friction_live,
    mujoco_area_friction_state,
    mujoco_area_pressure,
    mujoco_area_pressure_device,
    mujoco_area_pressure_scene,
    mujoco_heterogeneous_area_scene,
)
from phymeo.backends.mujoco_tyx import _entry_storage


def measure_scene_step(scene, h, *, detailed):
    """Return the accepted mapping, duration, exclusive partitions and call counts."""
    batch = scene.batch
    revision = batch.checkpoint_revision
    if not detailed:
        start = perf_counter_ns()
        accepted = scene.step(h, expected_revision=revision)
        return accepted, {"scene_ns": perf_counter_ns() - start}

    costs, calls = Counter(), Counter()
    child_ns = 0
    active_device = None
    depth = 0
    with ExitStack() as hooks:

        def bind(
            owner, method, category, *, kernel=False, advance=None, entry=False, extra_get=False
        ):
            original = getattr(owner, method)

            @wraps(original)
            def timed(*args, **kwargs):
                nonlocal active_device, child_ns, depth
                # Pose collection and the candidate download already include
                # their D2H calls. Only the additional entry arrays remain.
                if extra_get and depth:
                    return original(*args, **kwargs)
                parent_children, previous_device = child_ns, active_device
                if advance is not None:
                    active_device = advance
                if kernel:
                    # Pre-existing mechanics/upload work belongs to the parent.
                    active_device.runtime.sync()
                child_ns = 0
                depth += 1
                start = perf_counter_ns()
                try:
                    result = original(*args, **kwargs)
                    if kernel:
                        active_device.runtime.sync()
                    if entry:
                        device = args[0]
                        with device.tx.use_runtime(device.runtime):
                            device.runtime.sync()
                    return result
                finally:
                    elapsed = perf_counter_ns() - start
                    costs[category] += elapsed - child_ns
                    calls[category] += 1
                    child_ns = parent_children + elapsed
                    active_device = previous_device
                    depth -= 1

            hooks.enter_context(patch.object(owner, method, timed))

        bind(mujoco_heterogeneous_area_scene, "_geometry_requests", "poses_and_requests")
        bind(mujoco_area_pressure_scene, "_build_geometry_input", "mesh_validation")
        bind(mujoco_area_pressure_scene, "_intersect_checked", "intersection")
        bind(mujoco_area_pressure, "pack", "surface_packet")
        bind(_tyx_area_friction_entry, "pack_entry", "cuda_row_assembly", kernel=True)
        bind(_tyx_area_friction_kernels, "solve_batch", "cuda_friction_solve", kernel=True)
        bind(_entry_storage, "prepare_entry", "device_entry", entry=True)
        bind(mujoco_area_friction_live, "publish", "force_publication")
        bind(mujoco_area_pressure_device, "validate_row", "receipt_checks")
        bind(mujoco_area_friction_state, "validate_friction_receipt", "receipt_checks")
        for bucket in batch._buckets:
            device = bucket._device
            bind(device, "advance", "device_advance_other", advance=device)
            bind(device, "download", "candidate_download")
            bind(device, "_get", "candidate_download", extra_get=True)

        start = perf_counter_ns()
        accepted = scene.step(h, expected_revision=revision)
        elapsed = perf_counter_ns() - start
    costs["transaction_other"] = elapsed - sum(costs.values())
    if min(costs.values()) < 0 or depth or sum(costs.values()) != elapsed:
        raise RuntimeError("timing partitions do not cover one accepted scene step")
    return accepted, {"scene_ns": elapsed, "exclusive_ns": dict(costs), "calls": dict(calls)}
