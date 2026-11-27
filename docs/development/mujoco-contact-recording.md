# Recording MuJoCo contacts without recursive copies

`MuJoCoRuntime` already detaches its public contact arrays from native `MjData`.
Calling `dataclasses.asdict(contact)` then recursively deep-copies every NumPy
field again. At high contact counts this creates many independent array
allocations for each world, phase, and substep.

Use `phymeo.backends.mujoco_recording.contacts_to_columns` for numeric recording.
It creates one immutable column per field and never calls `deepcopy`. The CPU
runtime, integration, collision geometry, controls, and contact law are unchanged.
This is an opt-in recording API; existing consumers are not silently redirected.

```python
import numpy as np
from phymeo.backends.mujoco_recording import contacts_to_columns

state = runtime.snapshot()[world]
solver = contacts_to_columns(state.last_step.contacts, max_bytes=16 * 1024**2)
endpoint = contacts_to_columns(state.contacts, max_bytes=16 * 1024**2)

# Store these separately with world/tick/time and their original phase labels.
# Compression and durable commit are the recorder's responsibility.
np.savez_compressed("solver.npz", **solver.arrays)
np.savez_compressed("endpoint.npz", **endpoint.arrays)
metadata = {
    "schema": solver.schema,
    "world": state.world,
    "tick": state.tick,
    "solver_phase": state.last_step.contact_phase,
    "step_input_time": state.last_step.input_time,
    "step_output_time": state.last_step.output_time,
    "endpoint_phase": state.contact_phase,
    "endpoint_time": state.contact_time,
}
```

The 19 fields preserve the `MuJoCoContact` schema, including IDs, distance,
position/frame, material and solver parameters, inactive constraint rows, the
36-entry cone Hessian, and the measured contact-frame wrench. Solver and endpoint
rows must not be conflated. The exporter neither filters nonfinite values nor
reinterprets them as valid physics; existing physical validation still applies.
NaN bit patterns in the diagnostic Hessian are retained.

Each row occupies 664 numeric bytes. `max_bytes` limits the final numeric
payload and rejects oversized input before output allocation; it never truncates
rows. During freezing, the largest staging column can coexist with its byte
copy. The limit is not a process-RSS cap and excludes input records and Python
object overhead. A caller collecting multiple substeps needs its own aggregate
budget, bounded queue, and backpressure.

All columns own immutable byte buffers, so the packet survives further steps,
reset, and runtime closure without retaining native objects. Consumers cannot
re-enable array writes. `packet.contact(i)` reconstructs a typed row using views
of those immutable buffers. Use `np.load(..., allow_pickle=False)` when reading
the array archive; preserve and validate metadata separately.

## Incident evidence and scope

A captured Windows process fault occurred while `dataclasses.asdict` invoked
`numpy.ndarray.__deepcopy__` on a contact's `cone_hessian`. The Python frames were
recovered from the dump using the matching CPython 3.11 layout. The source array
had valid shape `(36,)`, float64 elements, stride 8, and read-only bytes-backed
ownership. The native exception was an execute access violation at address zero.
This identifies the active recording path, not the earlier cause of any native
memory corruption. Stack candidates alone are not treated as a proven unwind.

This API removes that recursive-copy path for adopters and bounds its numeric
export. It does **not** establish that a NumPy, allocator, or MuJoCo defect caused
the incident, reproduce the original access violation, or prove a crash cure.
It does not authorize replaying a stopped learning run from older random state.
Further native fault investigation and a bounded integration run remain necessary
before claiming long-running stability or end-to-end training speedup.

## Validation

- 19 new checks cover all fields and types, immutable detached ownership,
  budget rejection before allocation, malformed schemas, nonfinite diagnostic
  bit patterns, concurrent export, and actual native step/reset/close lifetime.
- 20 related CPU runtime checks pass. One existing invalid-mode test expects
  `NotImplementedError`, while the existing API raises `ValueError`; that test
  also fails against the unmodified pre-change runtime and is not changed here.
- An offline comparison of 40 saved native substeps covered 61,468 contact rows
  across both phases and 1,167,892 field-row values. All dtype, shape, and raw-byte
  comparisons passed, and source-file identity was unchanged. No new learning
  samples or optimizer updates were produced by that comparison.

Run the new checks with `pytest tests/test_mujoco_recording.py`. The native
lifetime check requires the pinned optional MuJoCo dependency.
