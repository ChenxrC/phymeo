# ROB1a participant registration and layout

Current parent status (2026-11-11): **HOST_QUALIFIED_REGISTRATION**. The complete
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
