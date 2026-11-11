# Resident participant registration and common response

## ROB1b parent qualification — 2026-11-11

**ROB1b_R01_INTERPRET_CUDA_QUALIFIED.** Product `08042d21` adds the shared transaction service and public `ResidentSceneBatch` API. The complete new two-file group passed **11 interpret tests / 53.59 s** and **11 CUDA tests / 32.79 s**, RAW_EXIT=0, on each backend's first functional run. Parent review also found and repaired stale-source validation blocking resource release; its new regression passed in both groups. Initial import/format failures are retained and final static checks passed.

Four continuous R01 variants each run 40 steps, covering W1/W2/W4, sparse identities, reversed registration and a common rigid transform: 320 logical-world steps per backend, with two independently moving bodies per world. Tests independently check the radial spring law, initial 2 N opposing forces, actual nonzero incoming velocities, M/J/b, complete device diagnostics, momentum and state/tick history. Each storage performs one entry mechanics stage, one solve and one finalize per step. This is common-entry known contact loading with block-diagonal participant mass, not an implicit unknown-force cross-tree QP.

Late participant rejection retains every participant in its logical world while an unaffected world advances. Late receipt exceptions retain the entire batch and previously submitted controls. Tests cover actual retry, shared site capacity, global candidate preflight, complete-world restore and replay, selected reset, source/owner/CAS failures, and close after source invalidation. Ten full gzip evidence payloads per backend and all 15 input file bindings are indexed by `artifacts/p65-parity-n2-rob1b-col1a-20261111-01/rob1b-evidence-index.json`; this index performs file I/O only, with physical assertions belonging to the original tests.

Current execution domain: ordinary floating trees, <=8 participants and <=33 physical bodies per world, explicit HOST full allowed-pair candidates and HOST patch sampling. All actual mechanics, J projection and integration remain TYX. Only R01 and the stated transaction cases are qualified; R02 arm/free-box, R03 two arms/tool, R04 mixed graph families, late restore upload/check faults and ROB1d package consumption remain later nodes. No broad general-robot or device-patch qualification is inferred. Earlier registration and authoring records below are history.

## ROB1a historical qualification

Earlier parent status (2026-11-11): **HOST_QUALIFIED_REGISTRATION**. The complete
focused file passed **33 tests in 1.13 s, RAW_EXIT=0** (run09). All three new Python
files passed final Ruff/format checks; `b6a6534d` contains the formatting-only
normalization. Source snapshots, raw logs and the source/evidence receipt are in
`artifacts/p65-env-refresh-20261111-01/n1-rob-binding-acceptance-receipt.json`.
The accepted scope is immutable registration/layout/source preflight only: no
multi-participant device dynamics, shared step or ROB1b acceptance is claimed.
No fresh wheel was built for this increment. ROB1b is the next dependent node.
The original authoring handoff below is retained as history.

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. No project import,
collection, numerical probe, pytest, static check, formatter, compiler or device
execution was performed by this author. Parent alone owns validation.

Base: `a33a09ef911ddd974ba1ff999b93a3b76c9b0cc7`.
Contract: [resident multi-participant plan](../superpowers/plans/2026-11-11-resident-multi-participant.md).
Tests-first checkpoint: `741f2726321e0f9030d5ea249c1914d85294b286`; that commit
contains only the new tests, before either product module existed. A subsequent
source-read-driven test adds the original from_asset shape/graph proof.

## Delivered API and actual boundary

Import from `phymeo.backends.resident_scene`; the parent owns future root exports.

- `ResidentParticipantKey(world, participant)`: immutable ordered nonnegative
  i32 keys, compatible with CollisionKey device identity.
- `ResidentParticipant(key, system, body_ids)`: one of the existing four exact
  declaration types, compiled through the original static compiler. Every full
  physical body has an explicit immutable logical body ID.
- `ResidentParticipantBinding(participant, binding)`: owns the existing binding
  data and retains original from_asset model/shape/graph proof. No identity is
  inferred from a group's integer or a collision body's numeric value.
- `ResidentParticipantLayout(participants, *, static_body_ids=())`: canonical
  world/participant order; per-family storage rows; separate independent nq/nv
  and full physical-body offsets; physical nq/nv retained separately; bidirectional
  body registry and explicit distinct static owners. Maximum eight participants
  per world. Empty layouts reject; empty geometry registrations are supported.
- `layout.bind_geometry(model, bindings)`: complete immutable geometry preflight;
  exact shape membership, world/family/local body, original topology/source,
  static one-pose ownership and original shape/vertex limits. It returns
  `ResidentParticipantGeometry` with original compiled binding data.
- `require_current`: checks current declaration/model/graph/initial-state bytes;
  geometry also binds its exact layout owner and original model/binding sources.

There is deliberately no `ResidentSceneBatch`, `step`, force result, device bank,
contact query or accepted-world stub. Geometry registration does not validate
material-law coverage, filtering, candidate/site budgets or publish state:
those belong to ROB1b's actual contact plan and parent shared service.

## Dimensions and ownership

The literal mixed-layout test registers fixed N2, zero-child free root,
affine N2/K1 and floating affine n2/K1. Canonical independent configuration
offsets are (0,2,9,10), velocity offsets (0,2,8,9), physical body offsets
(0,2,3,5), totals (18,16,8). Floating affine retains physical nq9/nv8/body3
while its independent nq8/nv7 remain the state address counts. Body slots are
world-local differences of these explicit offsets, not participant IDs.

Compiled arrays reuse original bytes-backed readonly storage. No input model
or caller array is modified. The public `system` is an original declaration
reference, **not a promise that its nested model is immutable**. The owned
compiled snapshot remains unchanged if that external source is mutated; a later
`require_current` refuses the stale registration. Static revalidation presently
reuses the original initial-state compiler and is not proposed as a per-step
performance optimization. Any later source-validation cache must retain the
same source checks; this patch does not add one.

Topology, original physical model and affine graph tokens remain separate.
Source comparison also retains array dtype/shape/raw bytes, original q/v,
control gains/caps, delay history and iteration budget. Geometry copies original
static pose arrays and wrapper material arrays; no caller can re-enable writes
on the owned NumPy storage using its public flags.

## Tests authored, not run

The single new test file covers four-family dimensions, sparse worlds and
canonical order, repeated local IDs across different worlds, independent storage
rows, multiple same-world participants, body forward/reverse maps, per-world
8-to-9 capacity rejection, duplicate/missing/invalid IDs and static collisions.
It includes external state/model/graph mutation, direct wrong topology/family,
wrong owner/world/body, exact geometry membership, explicit static one-pose
ownership, source revision changes, immutable compiled arrays, and original
from_asset graph/shape provenance. The latter prevents copying a wrapper while
accidentally dropping its source proof. No product dynamics define an oracle.

These are HOST registration tests and do not claim interpretation/CUDA physical
qualification, new world scale, contact law correctness or full ROB1 completion.

## Exact scope and parent handoff

Only four new tracked paths:
1. `tests/test_resident_participant_layout.py`
2. `src/phymeo/backends/_resident_participant_layout.py`
3. `src/phymeo/backends/resident_scene.py`
4. `docs/development/p65-resident-multi-participant-report.md`

No old source, export, binding, test, plan or fixture was changed. The unrelated
untracked `docs/development/mujoco-tyx-t1-review.md` remains untouched.
The source/working hashes and commit IDs are delivered separately, so the report
does not attempt a recursive self-hash.

After source review and a fresh entrypoint-aware execution guard, the parent may
run the existing environment's Ruff check/format-check on exactly the three new
Python files, followed by the one complete new test file:

```text
ruff check --no-cache src/phymeo/backends/_resident_participant_layout.py src/phymeo/backends/resident_scene.py tests/test_resident_participant_layout.py
ruff format --check --no-cache src/phymeo/backends/_resident_participant_layout.py src/phymeo/backends/resident_scene.py tests/test_resident_participant_layout.py
python -B -m pytest tests/test_resident_participant_layout.py -q -x --tb=short -p no:cacheprovider
```

These are pending commands, not results. Existing PYTHONPATH/pin/environment and
unique serial execution rules apply. No old green suite or G1 rerun is required
for this pure registration increment. ROB1b still needs the parent shared batch
service, source-aware collision adapter and world-atomic candidate publication.


## ROB1b first authoring increment — 2026-11-11

Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**.
The parent alone owns all project execution. No import, collection, numerical
probe, pytest, lint, formatter, compile, build or install was run in this lane.
The prior ROB1a HOST qualification remains distinct; this section adds no pass.

Base: `63614090`. Tests were authored and committed first as
`dda12400fd7c9d0eb3c90ab47d97b29b36ba9cd4`; source checkpoint is
`95984bc4c0bba158b0a0e50da736ef52f1f25392`.
The final report-only commit does not change those source/test bytes.

### Executable scope authored

`ResidentSceneBatch` is additive and uses the parent's new
`_resident_batch_transaction.ResidentBatchTransaction` service. The child does
not copy that service or read/write its private batch banks. Registration/layout
continues to describe all four existing tree families. **ROB1b execution admits
only ordinary FloatingRobotTree**, with at most 33 physical bodies in a complete
world and the original maximum eight participants. R01 positives exercise two
zero-child free roots; articulated/affine qualification is still pending.

The scene offers held typed controls, free step/contact step, complete world
snapshot, expected-epoch restore, partial-world-set reset, statistics and close.
A world checkpoint retains all original participant checkpoint fields, including
control/history/local origin, and its scene owner plus registry fingerprint.
Partial participant restore, foreign owner and stale epoch are rejected before
physical mutation. `static_body_ids=()` is an explicit optional constructor
registration, reusing ROB1a's static-owner preflight.

The first contact route is explicitly `host_patch` with conservative enumeration
of **all allowed pairs**, not device BVH or device patch. Candidate capacity is
checked against that complete domain before creating the plan. Materials exactly
cover its filtered canonical pairs. Same-body, mask/exclusion, static-static and
cross-world filtering remain explicit; equal local body indices in different
participants never suppress a cross-participant pair. Optional same-participant
pairs still require actual non-adjacent physical bodies.

Body-local material points remain the registered original values. Current
position, rotation, COM and twists come from the shared service's actual device
FK packet; the adapter does not evaluate HOST FK/M/J. The existing finite patch,
normal/friction law and work transport are reused without changing their gates.
One world-wide body-wrench array is sliced by ROB1a's explicit body offsets, then
the parent projects each participant's full Jv/Jw and solves its original block.
This is an entry-force explicit coupling, not a cross-tree implicit contact QP.

### Atomic publication and evidence

The new `_tyx_participant_transaction.py` kernel has one lane per logical world.
It scans that world's explicit storage rows; if any original status is nonzero
or geometry flag is not one, it sets only still-successful members to status 6.
Existing failures keep their original code. Different lanes write disjoint rows.
The existing floating finalize alone retains root/state/applied/FK/twist/energy/
ticks and chooses per-row acceptance; no integration or state copy is duplicated.

All participant receipts, world frames/sites, transported material/application
work, statuses and world checkpoints are constructed before the parent's single
bank/epoch commit. Scene-side publication is assignment-only under the same
owner lock. A site-capacity failure rejects every participant of that world;
candidate preflight overflow or a late receipt exception publishes no partial
batch. A prior successful held-control submission remains held through failure.

Receipts preserve the actual entry M/b/Jv/Jw/input external, original complete
checkpoints/geometry, FP64 law wrench, submitted FP32 wrench, device projected
generalized load and full original FloatingTreeDiagnostic. The FP32 wrench field
is the exact host value submitted through the agreed conversion, not a new
readback claim. Raw matrix/response diagnostics retain the actual device fields.
HOST sampling, readback and assembly are explicit costs; no zero-readback or
throughput claim is made.

### Frozen tests authored, unexecuted

Two full test files contain six test functions and ten authored parameter
instances per backend; these counts have not been collected or executed.

- R01: W1, transformed/reversed W1, transformed W2 and reversed W4; each is 40
  uninterrupted steps with the original nonzero incoming velocities. Literal
  first-step forces are (-2,+2) N and accelerations (-1,+2) m/s^2. An independent
  radial spring plus semi-implicit advance supplies the full-window oracle.
- Every participant's q/v, angular motion, mass/J, generalized force, original
  controls and tick are checked; momentum and world isolation are checked.
  Storage entry/solve/finalize counters must each increase once per physical step,
  independently of participant/world count.
- The first or last world's last participant is rejected at the named internal
  pre-finalize seam; its entire world is retained, the unrelated world advances,
  and a real retry subsequently succeeds.
- A late fully-constructed-receipt exception preserves all old publications,
  bank/epoch/state and previously committed controls, followed by a real retry.
- Candidate capacity rejects globally at prepare; world site capacity rejects
  only its complete world. Full world restore/CAS, repeated continuation, partial
  reset, stale receipt, wrong owner and bad last control are covered.

Evidence files preserve complete authored inputs, all step states/receipts,
diagnostics, counters, byte-bearing immutable fingerprints and failures using a
lossless array dtype/shape/raw-byte codec in gzip JSON. This is not a saved-audit
result. The tests do **not** cover R02/R03/R04, the complete ROB1c multi-arm/tool
matrix, device_patch, old bundle scale, or late restore upload/check injection.

### Child file manifest and pending parent validation

Exactly six changed paths relative to the base; layout and its old tests remain
unchanged. The two new source modules and two new tests plus resident_scene have
these source-checkpoint SHA256 values:

| Path | SHA256 |
|---|---|
| `src/phymeo/backends/resident_scene.py` | `14668dc2bb86bad2df55d2dc1963f929e9795fc1bac0b307815e89dfbd1a0e3d` |
| `src/phymeo/backends/_resident_participant_contacts.py` | `25617667fed98cd7d08cf17a104b633269ae0a55ecfaf34aa389a32761492c95` |
| `src/phymeo/backends/_tyx_participant_transaction.py` | `6bc16093a25b1edfc8ab624530ec8d333e58deb99204aebf2d934b32b7928e34` |
| `tests/scale/test_resident_multi_participant.py` | `7709d89934ebeead9c206b8fd9f8c2f5643342352adb689d40a7312ef002f6df` |
| `tests/scale/test_resident_multi_participant_transactions.py` | `45412c436fc00ac263cf13923c810814f04bf1dc1a57d0dbd47482cadc02e19a` |

The sixth path is this append-only report. Parent owns the shared service, root
exports, integration and every actual check. After source integration and its
fresh global execution guard, pending full new tests are:

```text
python -B -m pytest tests/scale/test_resident_multi_participant.py tests/scale/test_resident_multi_participant_transactions.py --tyx-mode=interpret -q -x --tb=short -p no:cacheprovider
python -B -m pytest tests/scale/test_resident_multi_participant.py tests/scale/test_resident_multi_participant_transactions.py --tyx-mode=cuda -q -x --tb=short -p no:cacheprovider
```

Parent should first run exact changed-Python Ruff/format checks including its
shared service/export seam. No old full green suite is requested. The original
untracked T1 review remains untouched. Two read-only authoring observations about
the parent service (dict input versus pair iterable, and stats after close) were
reported and corrected by parent before execution; no runtime failure is claimed.
