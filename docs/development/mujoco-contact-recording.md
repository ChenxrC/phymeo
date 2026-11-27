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

## Integration validation and commonality

A subsequent fixed four-worker recording check processed 557,278 captured contact
rows across eight worlds. The new exporter, a bounded archival writer, and the
reader preserved every field byte-for-byte. The legacy `asdict` path also completed
that check without an access violation. Thus this is not a deterministic crash
reproduction or proof of a native-memory repair.

For this schema, the legacy conversion makes eight additional ndarray deep-copy
calls (544 array payload bytes) per contact. That implies 4,458,224 such calls in
the fixed workload. The exporter bypasses recursive deep-copy dispatch; it still
allocates and copies its output columns. Export-only thread timings are not
end-to-end policy-training throughput.

One separate native diagnostic used the original 1,939-geometry / 1,930-mesh M16
scene, eight independent worlds, four worker threads, and four fixed zero-action
decisions: 1,280 world integration steps, or 0.16 simulated seconds per world.
All 1,651,868 captured contacts retained their solver/endpoint phase and all 19
fields through archival readback. No access violation or physical limit failure
occurred, and the child exited normally under first-chance fault supervision.
The diagnostic used the CPython debug allocator, not full native page-heap or
ASAN instrumentation. It neither sampled a policy nor performed PPO, replayed
the failed policy trajectory, or established long-run reliability. Project scene
fixtures and detailed run evidence are maintained outside this repository.

The general issue is repeated serialization of already detached array-backed
records. Lower-volume control receipts can exhibit the same copying pattern.
Do not mechanically remove `deepcopy` from optimizer rollback, checkpoints, or
mutable configuration isolation. `SuccessHistory` serialization, for example,
primarily handles scalar and tuple fields and has different costs.

The default runtime path still materializes individual contacts and freezes their arrays;
`step` and `snapshot` still copy native state and preserve distinct contact
phases. These are remaining performance costs, not newly proven faults. The
captured immutable source array and per-instance runtime locks do not establish
a shared-mutable-array race. Similar access-violation codes in other backends
must not be treated as evidence of the same root cause.

## Direct native columns and reusable storage (CPU, opt-in)

The default `contact_storage="records"` behavior above remains unchanged. For a
high-frequency recorder, enable native column capture instead:

```python
from phymeo.backends.mujoco_runtime import MuJoCoRuntime
from phymeo.backends.mujoco_contact_pool import retained_contact_bytes

runtime = MuJoCoRuntime(
    compilation,
    world_keys=(0,),
    contact_storage="columns",
    contact_pool_bytes=128 * 1024**2,
    contact_block_rows=4096,
)
runtime.step(runtime.timestep)
state = runtime.snapshot()[0]
solver, endpoint = state.last_step.contacts, state.contacts
chunks = (solver, endpoint)
queue_charge = retained_contact_bytes(chunks)
# Submit each chunk's .arrays directly to a bounded writer, together with
# world/tick, input/output times, and each original contact phase label.
# Retain chunks/arrays until the writer is done. Do not concatenate them merely
# to produce a single archive; independent .npy members can represent chunks.
```

This path copies batched native fields straight into reusable column slabs. It
does not first construct one Python contact and eight frozen arrays per row.
`mj_contactForce` still fills the actual six-component wrench for every contact.
No native contact pointer is exported. All 19 fields, int64/float64 representation,
inactive Hessian bit patterns, finite-array checks, and solver/endpoint semantics
are retained. Scalar validation remains the caller's responsibility, as in the
default runtime. Step/snapshot still copy native `MjData`; this change does not
remove integration isolation or change the solver, controls, or physics.

Each published array is read-only. NumPy retains an exporter that owns a slab
lease: slices, `np.asarray`, ndarray views and `memoryview` keep it alive even
after the packet is discarded. A slab is reusable only after every such export
is released. Captures use nonoverlapping rows within a slab. Reset and close do
not invalidate retained records; clones get independent pools. Private storage
and exporters must not be modified. This is an ownership contract, not protection
against deliberate private-attribute or raw-pointer writes.

`packet[i]` lazily reconstructs a typed contact for compatibility. Hot code should
read columns, for example `packet.arrays["distance"]`, rather than iterate rows.
`packet.detach()` makes independent immutable bytes-backed columns when a record
must outlive the pool's practical retention budget; it deliberately copies.

The pool cap covers allocated numeric slab capacity, including pinned and free
slabs. It is not an RSS limit; native state, Python objects, short-lived vector
validation/index arrays and compression buffers are separate. The allocator
grows on demand within this cap, reuses released slabs, and never truncates rows.
`retained_contact_bytes(chunks)` counts unique pinned allocations including unused
slab rows, suitable for conservative queue admission. Charge non-contact arrays
and metadata as well. A full pool raises `ContactPoolCapacityError`; in a runtime
step this follows the existing `PhysicsStepFailure` publication contract.
Bound the producer's retained chunk window and apply writer backpressure before
starting more work. There is no automatic replay or eviction of live records.

The new CPU mode is explicit; GPU adapters are not silently redirected or
qualified by these checks. Avoid recursively serializing the packet with
`dataclasses.asdict`: consume its numeric arrays and a separate metadata envelope.

### Direct-column validation

22 new ownership/schema/publication checks and 39 related CPU/runtime/export
checks passed. The one previously documented baseline exception-type test was
excluded; no GPU qualification is claimed.

One paired diagnostic compared default records followed by `contacts_to_columns`
with direct pooled columns. Both used the same original M16 scene, eight worlds,
four threads, four zero-action decisions, the same chunk archive writer, and
1,280 world physics steps per variant. Neither sampled a policy or updated PPO.
Each archive contained 1,651,868 contacts. All 19 fields, phase labels, state
records and physical decision results matched byte-for-byte across variants;
both supervised processes exited normally without an access violation.

The sum of four batch **wall** intervals (physics, capture, packing and durable
writer commit, excluding subsequent verification reads) was 211.79 s for records
and 36.30 s for pooled columns: 5.83x on this single fixed diagnostic. Including
initialization, verification reads and closure, elapsed time was 250.24 s versus
63.66 s. These are not statistical benchmarks, policy-training throughput, or
evidence that the original native fault is repaired. Both used the Python debug
allocator and the same compression settings.

Pools allocated 160 slabs across eight worlds and reused released slabs during
the run. Per-world peak numeric capacity was about 44.1–54.5 MiB against a
128 MiB cap. Writer admission peaked at two records / 109.5 MiB against its
two-record / 256 MiB limits. The writer charge counts pinned slab capacity,
including unused rows, and is not total process memory. Reusable capacity costs
some headroom in exchange for avoiding repeated payload allocation. Short
zero-action diagnostics do not establish random-policy or long-run stability.
