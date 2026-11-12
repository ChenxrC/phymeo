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

## 7. Pre-validation interpretation correction: conservative capsule status

The parent's first interpretation run reached the original square-patch capsule
rows and returned `CONDITIONING`, with the two initial sphere rows resolved.
The declared capsule axis `(.375,.375,.25) → (.625,.625,.25)` is parallel to the
square's shared diagonal. The unchanged finite-distance kernel only recognizes
the small edge-axis cross branch as structurally parallel when its products are
individually zero. This diagonal has nonzero products which cancel, so the
existing conservative status is intentional. The selected candidate can still
have the correct distance; that does not turn the row into a resolved certificate.
The original failed run remains evidence and is not rewritten as a pass.

The parent authorized a narrow change to this report and the new D2 query test
only. No fixture, coordinate, trajectory, threshold, original reference helper,
producer or collision kernel changes. The local D2 assertion now requires:

- all original three-point-patch sphere/capsule rows remain `RESOLVED`;
- square-patch sphere rows remain `RESOLVED`;
- only the fixed square-patch capsule family may return `CONDITIONING` or
  `RESOLVED`; at initial tick zero its two shared-diagonal rows must explicitly
  be `CONDITIONING`;
- `DEGENERATE`, absent selected candidates and missing candidate groups remain
  failures for these fixed nondegenerate cases.

Every row, including `CONDITIONING`, still evaluates the independent low-level
reference oracle from accepted source geometry. Its original `1e-4 L` distance
and gap gates, `2e-4 L` witness gates, `8u` barycentric gate and complete finite
candidate ledger are retained. Each emitted candidate is independently checked
against its actual triangle/axis material address; the selected witness must be
finite and selected from the complete emitted minimum. Unresolved rows must
publish no normal, a zero raw normal vector and `normal_defined=0`. The prior
reference helper's unresolved early return is not used. Oracle inputs and exact
reference distance/witness records are saved before assertions.

Expected surface/round group membership and the entire same-world pair list are
now independently constructed from the model declarations, ownership, masks and
explicit exclusions. Output groups cannot define the oracle's case set. Filtered
empty groups, same-family exclusions, finite far pairs and all candidate row
indices are checked against that declared coverage.

This is an author-only test correction, not an execution result. The 12-step,
permutation, restoration and filtering cases remain intact. A successful future
run can qualify correct source and conservative-result propagation; it cannot
claim that every square-patch capsule row is resolved or suitable for generating
contact force. The parent owns the corresponding installed-consumer adjustment
and all subsequent serial verification.

## 8. Parent source validation (2026-11-12)

The first raw run `20-d2-interpret` failed on the documented parallel-state
expectation. Run `27-d2-interpret-final` then exposed an independent test error:
the fixed-point square measured only its nearly stationary first edge. The
deformation assertion now examines every original declared material edge at
every step, with the unchanged `> 1e-6` gate. No initial state, motion, force,
distance kernel or tolerance was changed; both failures remain in evidence.

Final complete source runs: `29-d2-interpret-final` **31 passed / 7.56 s** and
`30-d2-cuda` **31 passed / 8.10 s**, each RAW_EXIT0. They cover both new files,
all original multi-frame fixtures, declared membership, conservative statuses,
ownership/resource failures and restore/reset continuation. The installed
consumer incorporates the same status and all-edge interpretation, plus the
parent's pre-import provenance, complete group/round membership and selected
reset-state checks. Its execution is still pending at this checkpoint.

Evidence: `artifacts/p65-three-track-b-20261112-01`; source snapshots, inputs,
actual returns and all first failures are retained. This qualifies the D2 source
query connection only; unresolved capsule rows cannot be used for contact force.

## 9. Parent installed qualification

Unified product `0321c5a4`, 522 source/wheel/installed identical members. Runs
`39/40` passed interpret/CUDA: each 16 batch steps / 32 worldsteps, 19 checked
queries, 102 RESOLVED and 12 CONDITIONING rows, zero HOST projection calls.
Original trajectories, gates and all-edge deformation checks are retained.
Each complete trace has 133 records; `47-close-installed` independently checked
its original logical bytes and archived the compressed evidence. The CLI requires
an explicit absolute installation root (the `--installed-root` alias also allows
the outside bootstrap to forward it unambiguously).

D2/M2-Q query connection is accepted. Static round geometry, open material
surfaces and conservative unresolved rows remain explicit. Motion of another
device owner, contact law/force, CCD and continuous load are D3/D4 work.
