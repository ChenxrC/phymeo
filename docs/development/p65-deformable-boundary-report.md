# D2 / M2-Q deformable boundary source — author handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED** in this author worktree.
The parent owns the sole execution slot and all shared integration edits. No
project import, collection, numerical probe, test, formatter, lint, build,
installation or profiler was run by this author. Git whitespace inspection is
not a Python or TYX qualification result.

Base: `c45a01740d131ed6bebfabe2285169e67f296bd7`.
Branch: `codex/p65-deformable-query-b-20261112`.
Accepted D1 producer behavior is reused, not reimplemented or requalified here.
The design basis is [M2 producer design](p65-m2-producer-bank-design.md), and the
accepted producer boundary is recorded in
[Batch A](p65-three-track-batch-a-20261112.md).

## 1. Exact scope and ownership

Six new author paths:

| Path | Responsibility |
| --- | --- |
| `src/phymeo/backends/_deformable_boundary_source.py` | Explicit open material declarations, registration, scoped adapter, current-source proof |
| `src/phymeo/backends/_tyx_deformable_boundary.py` | Accepted node coordinate/tick gather only |
| `tests/collision/test_deformable_boundary_query.py` | Actual deforming trajectories, independent candidate/distance checks, stable identities |
| `tests/collision/test_deformable_boundary_protocol.py` | Admission, resource limits, source corruption, late publication and cleanup |
| `tests/installed/check_deformable_boundary.py` | Standalone installed public consumer and complete saved evidence |
| This report | Frozen interface, cost, cases, limits and handoff |

The parent owns the D1 public prepare/runtime/lease hooks, the generic boundary
query's four optional adapter hooks and publication ordering, and public exports.
This author does not modify any existing integrator, distance projection kernel,
BVH kernel, pair packing kernel, finite triangle-distance kernel or force solver.

This slice reaches **persistent actual deformation → current accepted-device
geometry → device BVH candidates → finite sphere/capsule–triangle distance**.
It does not apply a contact force, change a physical endpoint, integrate a rigid
partner, solve friction or produce a continuous collision certificate. It is not
the completion of the roadmap's rigid–deformable response chain.

## 2. Public declarations and truthful identities

`DeformableBoundarySurface(key, surface_key, group=1, mask=INT_MAX, revision=0)`
names one already registered `DeformableSurface` in `key.world`. Its original
surface key, facet IDs, oriented stable point IDs and native row mapping are
retained. A surface may be open: no closed `CollisionShape(kind="mesh")` claim
or artificial closing facets are introduced.

`DeformableBoundaryModel(shapes, excluded=())` contains those surfaces plus
existing static `CollisionShape` sphere/capsule declarations. Static rounds
require exactly their matching `ResidentStaticBinding` with a fixed explicit
`CollisionMotion`; moving paths and orientation installation are rejected.
Surface bindings come from the producer and are not supplied as fake FK bodies.
Shape keys are sorted deterministically. Masks, explicit exclusions and existing
same-body filtering remain complete. Duplicate keys, duplicate source-surface
aliases, unknown worlds/surfaces and mixed static/dynamic ownership of one body
are rejected.

The parent public entry is:

```python
plan = batch.prepare_boundary_query(model, static_bindings,
                                    limits=BoundaryDistanceLimits(),
                                    memory_limit_bytes=64 * 1024 * 1024)
result = plan.query(margin=margin)
plan.require_current(result)
```

The exported class names requested from the parent are
`DeformableBoundarySurface` and `DeformableBoundaryModel`; the adapter is an
internal `DeformableBoundaryAdapter(batch)`.

The existing distance packet's `facet_id` remains a **local facet ordinal**.
`result.source["surfaces"][shape_key]` supplies `surface_key`, `facet_ids`,
`oriented_point_ids`, `native_indices`, `gather_node_indices` and `point_ids`.
Every test resolves each packet ordinal back to its actual original material
facet. The ordinal is neither a new globally stable ID nor a replacement for the
original facet ID. Shared-edge candidates remain separate facets.

## 3. Device path and the four adapter hooks

Registration resolves topology and static coordinates only. Dynamic `local`
entries are zero placeholders; no current HOST vertex snapshot is uploaded.
The adapter borrows the actual D1 accepted `x` buffers and all world `tick`
buffers in one valid `borrow_vertices(expected_revision=...)` scope.

The only new kernel, `gather_deformable_points`, directly copies each selected
native FP32 node coordinate and its exact point interval, or copies an authored
static outward interval. It gathers the actual accepted tick for every producer
world, including worlds without a shape. Original leaf/round AABB, device refit,
stackless traversal, full candidate prepass, pair-local pack and 18-candidate
finite-distance calculation then run unchanged.

The optional shared-plan hooks are:

1. `gather(plan, view, ledger) -> (outputs, raw, ticks)`;
2. `source_prepass_bytes(plan) -> int`;
3. `empty_source(plan, view) -> (outputs, raw, ticks)`;
4. `source_record(plan, raw, ticks, view) -> immutable mapping`.

Existing registration, borrow, validate, logical-tick and require-current adapter
methods bind the real producer owner/runtime/topology, state revision, active
bank, world order and material registration. No rigid pose is invented.

Actual source-bank HOST readback is retained solely for independent publication
validation. Gathered dynamic points and exact intervals must match the indexed
accepted FP32 bank byte for byte. Static points retain their original conversion
delta. Actual device ticks must equal the current public accepted snapshot's
ticks; a HOST tick never replaces the device return. Pair material error is
therefore computed against the current gathered geometry, not the initial shape.
All original local scale, interval and material-error gates remain unchanged.

Empty models have no owned native query buffers, kernel calls or groups, but
read and bind every actual world tick. The empty path is not evidence of a
zero-cost source observation.

## 4. Resource accounting, limits and lifetime

Definitions: `S` registered shapes, `V` total concatenated material points,
`W` all producer worlds; for each eligible surface group, `F` facets and `R`
round shapes. `C` is the complete actual candidate count from the device prepass.

| Owned device payload | Bytes |
| --- | ---: |
| Persistent shape metadata, int32 `S×6` | `24S` |
| Persistent radius/outward radius, FP32 `S×2` | `8S` |
| Persistent node indices, int32 `V` | `4V` |
| Persistent static local coordinates, FP32 `V×3` | `12V` |
| Persistent local point intervals, FP32 `V×6` | `24V` |
| **Persistent subtotal P** | **`32S + 40V`** |
| Gathered current points and intervals | `36V` |
| Gathered shape stamps | `16S` |
| Gathered full-world ticks | `4W` |
| **Gather subtotal G, nonempty model** | **`36V + 16S + 4W`** |
| Original one-group BVH/topology/bounds/output scratch | `40F + 48(2F−1) + (52 + 4ceil(F/30))R` |
| Original all-candidate pack/narrow scratch | `1060C` |

The complete owned-native peak is `P + G + sum(group scratch) + 1060C`.
Empty models have zero P/G/native scratch. All registered topology limits are
checked before owned native allocation: 256 shapes, 8192 packed points, and
original hard group bounds of 32 groups, 512 vertices, 128 facets and 64 round
segments per group. D1 independently restricts total registered facets to 128 per
world. The original 8192-candidate hard limit and caller capacities use the full
prepass result. Equality with a capacity is accepted; one less rejects the whole
query without publishing a truncated result.

The query's existing `memory_limit_bytes` contract is deliberately unchanged:
it is an **owned native peak cap**, exposed as `native_owned_limit`. D1's own
device-plus-HOST-array reservation remains a separate cap. The queried source
uses borrowed positions of dynamic-surface worlds and all W ticks:
`borrowed_device_bytes = 12 sum(N_dynamic_world) + 4W`. These are real borrowed
payloads, not another native allocation.

Complete bank validation reads `12 sum(N_dynamic_world)` HOST bytes and material
deltas use `24V` FP64 bytes. They are named separately as
`actual_bank_host_readback_bytes` and `material_delta_host_bytes`; the parent
review distinguishes actual readback accounting from the pre-execution size
table. Existing cumulative `host_downloaded_bytes`, native upload/download
counters, `host_constant_bytes` and `packet_array_bytes` remain present. These
fields do not turn the native cap into a claim about all HOST memory.

The old `last_result` and `last_attempt`, new raw prefix, downloaded source bank,
new immutable packet and callers' retained old packets can coexist. HOST array
copies, Python object overhead, caller history, runtime/compiler allocations and
process RSS are not bounded by the query's native cap. This is an accounting
boundary, not a measured bandwidth or throughput result.

The generic plan owns its persistent/query buffers and a dependency lease on the
producer runtime. It does not own or close the physical banks. Query resources
must close normally, final same-bank checks must pass, and the D1 loan's exit
synchronization must succeed before `_last_result`, layouts or sequence publish.
A known pre-ledger pending close retains its owner and makes the producer
close-only. An actual opaque release exception retains the original UNKNOWN
owner and cannot be blindly retried; this uses the accepted shared resource
ledger, not a new D2 recovery scheme. Scoped views are protocol read-only, not
hardware write protection against hostile Python code.

## 5. Predeclared cases and evidence

The two new test files contain **31 manually enumerated parameter nodes per
backend** (8 query and 23 protocol nodes), before collection. This is an authored
count, not a test result. All cases use the original D1 `h=1/128`, one substep,
200 maximum ordered sweeps and `2e-6` material residual tolerance.

The positive geometry cases include the original three-point compliant edge,
four-point two-face patch, and fixed-node patch. Each advances 12 real steps;
maximum actual edge-length and actual device leaf-bound changes over that path
must both exceed the original `1e-6` deformation gate. Relaxation is allowed:
the final unloaded strain is not required to exceed that gate. The W2 case
uses sparse worlds 7/31, source row/facet/world/registration reversal, true
shared-edge facets, near sphere, finite capsule and far sphere, and retains
original IDs. Restore and partial reset are followed by real subsequent steps.

Independent tests use the existing external reference's complete finite AABB
enumeration and exact finite-distance oracle, supplied with the actual accepted
HOST snapshot and authored static material. Product packed coordinates never
define the expected geometry. The installed consumer carries its own finite
segment/triangle convex-feature minimization, using no repository helper import.

Protocol cases cover full native cap/cap−1, actual candidate cap/cap−1, first
launch and completed narrow/readback/packet failures, actual tick/point readback
corruption, wrong owner/registration/epoch/bank, expiration/reentrant mutation,
missing/duplicate/unknown material declarations, moving static source rejection,
complete masks/exclusions, empty full-world ticks, no current HOST geometry
upload, both pending-cleanup entry points, and exit-sync failure with/without a
simultaneous consumer error. Late failures retain the prior query publication.
The shared UNKNOWN ledger behavior remains reused; a new opaque-release test
is not advertised as having run.

The standalone consumer is fixed at 16 attempted/returned physical batch steps,
32 worldsteps and 19 complete queries per backend: initial query, 12-step path,
checkpoint A/restore/B, partial reset and two successors. It records complete
requests, immediately returned physical/query receipts, raw distance packets,
independent witnesses before comparisons, source/stamp/identity/resource data,
failure prefixes, and actual module provenance. No HOST projection or contact
force is permitted. The output is an exclusive fresh directory with
`summary.json` and a gzip JSONL stream with logical SHA/length/record count.

## 6. Author commits and parent validation entry

- `82b7e4dd`: tests-first query and protocol declarations.
- `7b01eb00`: the two new source modules.
- `ebb22b33`: current-publication device tick binding and corruption controls.
- `c1583fdc`: standalone installed consumer, pending cleanup and source identity
  tests, and additional complete cap evidence.

The parent may apply style-only changes and reviewed shared hooks. Those later
tested hashes must be recorded separately from these unexecuted author hashes.
The parent specifically requested freezing product/tests/consumer after this
handoff; only this report was subsequently authored.

Parent execution entry points, **not executed here**:

```text
pytest tests/collision/test_deformable_boundary_query.py tests/collision/test_deformable_boundary_protocol.py
python check_deformable_boundary.py --mode interpret --expected-install-root <wheel-site> --output <fresh-absolute-directory>
python check_deformable_boundary.py --mode cuda --expected-install-root <wheel-site> --output <another-fresh-absolute-directory>
```

Backend/profile options, serial guards, source bindings and exact pytest flags
remain the parent's established execution contract. Source compilation, both
actual backends, complete saved-evidence review and the new installed package are
pending. No learning, performance, Newton speed comparison, physical contact
response, deformable–deformable collision, bending or CCD acceptance follows
from this author handoff.
