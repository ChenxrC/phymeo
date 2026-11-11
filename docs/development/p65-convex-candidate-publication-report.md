# COL same-candidate separation publication

Current status: **NOT_ACCEPTED / EXPERIMENT_STOPPED**. See the actual parent validation section below. Historical authoring baseline: `847b8ee3`.
Parent alone owns all project execution. This is one bounded hypothesis after
[the actual six captures](p65-convex-query-trace-findings-20261111.md) and
[the validated affine reducer](p65-convex-affine-reconstruction-report.md).
The four Y normal entries pass on this experimental source, but the wider regression rejects the change; details are recorded below.

## Product scope and manual argument

Only `src/phymeo/backends/_tyx_convex_query_kernels.py` changes.
For the generic GJK branch, the current retained simplex has a closest q and
the original normalized weights/material IDs. The next existing support call
uses direction -q and computes the support-plane lower bound for that q.

After observing duplicate support, but before appending/reducing, apply the
original positive-separation depth test to this same candidate. If it passes,
publish normal -q/|q|, depth |q| (with original round-core skin adjustment),
the current lower bound, and the same retained simplex/weights. No point,
index or weight is replaced on this accepting iteration. Otherwise append
and reduce as before; its outgoing q is tested by a later support call, still
inside the original budget. Remove the old post-reduction plane/duplicate
success branch. A duplicate alone cannot authorize a wider plane residual.

The existing near-origin threshold/round-core unsupported/EPA route is retained
after reduction. SUPPORT_BALL is unchanged and retains its existing priority.
The first iteration has no retained candidate and cannot take the new early
exit. The original 8e-7 depth tolerance, 0.002 external normal gate, all
material/witness gates and 64 GJK /192 EPA /64 vertices /128 faces remain.
There is no extra support call, hidden pass, shape-specific branch, HOST
fallback or resurrected strict FW/normal gate.

Cross-direction lower/upper bounds were not intrinsically an invalid distance
interval. This change removes their use to authorize an outgoing candidate's
normal while publishing a different set of supporting evidence. Even a
same-direction small depth width does not guarantee the fixed normal accuracy:
for exact convex geometry at positive distance, a bound is proportional to
sqrt((upper-lower)/upper), not just absolute width. Stored FP32 q and real
weighted witnesses are also not identical by algebra alone. The independent
original normal and q/material/witness checks remain required; failures stay
failures. No promise is made that this single hypothesis fixes Y.

## Trace without a layout change

The existing trace validator accepts a zero selected mask. On the new
same-candidate decision2 exit:

- entry/exit q, squared norm, simplex vertices, weights and material slots
  are unchanged;
- selected count and pre-reduction count equal the retained count;
- selected mask=0, appended=0, all subset Gram/weight/comparison fields are
  zero because this iteration performs no subset enumeration;
- duplicate records the actual comparison, but never bypasses the plane gate;
- F21/F22 contain the actual candidate distance and distance-minus-bound
  used by this accepting branch.

No wire widths, enum values, allocation, accounting, source or policy binding
change. The old protocol prose limited F21/F22 validity to reduction; parent
owns the corresponding narrow documentation clarification. This report and
the new tests explicitly distinguish mask0 acceptance from missing trace data.
The original decision3 value remains reserved for historical captures.

## Tests-first matrix and stop condition

NEW `tests/collision/test_device_convex_candidate_publication.py` declares
**6 TYX nodes per backend, UNCOLLECTED**:

1. Three complete separated queries: BB finite face, cyclic/reversed BB face,
   oblique BE. Existing independent literals and geometry/material oracle are
   retained. Ordinary vs diagnostic geometry is compared; actual trace checks
   current q/weights/IDs and both published witnesses, with unchanged gates.
2. Two exact BB budget boundaries: budget1 honestly exhausts with no published
   witness; budget2 repeats the support and accepts the same retained candidate.
   Both count actual support calls and preserve the accepted physical state.
3. One independent validator negative node starts from one real accepted BB
   trace, then rejects changed outgoing q, changed outgoing weights, and a
   duplicate flag paired with a plane width outside the original gate.
   These mutations are explicitly validator forgeries, not physical fixtures.

The q-to-bilateral witness and material-position checks remain
2e-4*original scale; stored weights remain nonnegative with sum defect at most
64*2^-23. No current production geometry is used as the expected geometry.
Full inputs, ordinary output, actual trace and validator-forgery arrays are
saved in the existing compact format, including exact dtype/shape/raw bytes.

Parent first runs the six original affected entries (four Y plus the two
repaired HN), then these new nodes and the affected query/patch/trace matrix.
All old inputs and six-entry identities remain untouched. Changed iteration
counts can truthfully expose a resource limit; they do not justify raising a
cap or suppressing an old negative. If this bounded hypothesis does not reduce
the remaining obstacle, retain its evidence and stop rather than adding a
second speculative algorithm in this grant.

## Authoring handoff

Tests-first checkpoint: `6bb2d435`. The implementation adds the current-
candidate exit before append/reduction, removes the old post-reduction
plane/duplicate success branch, and moves duplicate's zero initialization to
the surrounding iteration scope so both guarded blocks have a defined scalar.
No new allocation, helper, launch, support evaluation or trace field is added.

The positive-separation eligibility condition is the finite-value complement
of the unchanged near-origin gate: d>1e-5, or d>8e-7 for the existing non-full-
surface round-core branch. It does not move near-origin cases into separation.
On success the retained simplex is already compacted from the previous
iteration, so its indices identify the same vertices and weights consumed by
the unchanged final material coalescing and bilateral witness publication.

The actual kernel text was compared with the baseline plus exactly these
three scoped replacements. All remaining kernel text, including support-ball,
Gram/reducer arithmetic, clamping, EPA and final witnesses, is unchanged.
`git diff --check` passed. No import, collection, numerical probe, pytest,
lint, formatter, compile or build was run in this lane.
Status: **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**.

## Actual parent validation and bounded rejection (2026-11-11)

**NOT_ACCEPTED / EXPERIMENT_STOPPED.** The original six affected entries and
the new local tests pass on both backends, but the changed stop policy causes
nine resource-limit regressions in the wider interpret matrix. This branch
is retained as an experiment; it is not authorized for integration into the
parent product, and the parent's accepted algorithm remains unchanged.

Authoring product checkpoint:
`93a8d950dc8e153c217a25a38c714f6377be9fd4`.
Style-only tested checkpoint:
`b4f632e9aa1f583509bddf91286e9c804fb000e2`.
The actual run receipts identify the tested kernel SHA256 as
`4d75b7c68f86090bae4fbdd248beeccc4b9e290716124002e29def5545de5492`
and the new test SHA256 as
`8c9b00a4e5cb4152b01faadabfef29978d4d81f7825bf3684134a58b34a3d1b2`.
These are the formatted tested bytes, not a retroactive claim about execution
of the original authoring checkpoint.

Canonical evidence root:
`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-parity-n4b-rob2a-col-stop-20261111-01`.
Each run below retains `final.json`, `RAW_EXIT.txt`, `stdout.log`,
`stderr.log`, its actual source snapshots and original basetemp evidence.
This report update read the existing receipts and logs only; it did not rerun
any query, numerical analysis, test or source audit.

| Original run directory | Actual result | pytest duration | RAW_EXIT |
| --- | --- | --- | --- |
| 24-col-six-interpret | original six: 6 passed | 11.79 s | 0 |
| 25-col-six-cuda | original six: 6 passed | 19.97 s | 0 |
| 26-col-candidate-interpret | new candidate tests: 6 passed | 4.03 s | 0 |
| 27-col-candidate-cuda | new candidate tests: 6 passed | 18.49 s | 0 |
| 28-col-matrix-interpret | 273 passed, 9 failed, 12 deselected | 130.04 s | 1 |

The original six consist of the four Y source-layout entries and the two HN
entries. Their success establishes the result on those fixed inputs only.
The 12 deselections in run28 are those six entries already tested in run24
and six unchanged HOST-only parameter nodes; they are not concealed failures.
No complete CUDA affected matrix was started after the interpret regression.

The nine actual failures all return `RESOURCE_LIMIT` where the unchanged
query/patch qualification requires `RESOLVED`:

- `test_k0_family_gap_witness[K0-Q01-CB-sep]`;
- `test_k0_family_gap_witness[K0-Q01-CB-overlap]`;
- `test_k0_family_gap_witness[K0-Q01-BH-sep]`;
- `test_k0_frame_and_identity[K0-Q03-CB-reverse]`;
- `test_k0_frame_and_identity[K0-Q03-CB-cycle]`;
- `test_k0_frame_and_identity[K0-Q03-CB-tilt]`;
- `test_k0_frame_and_identity[K0-Q03-CB-permute]`;
- `test_k0_frame_and_identity[K0-Q03-CB-cycle_reverse_ids]`;
- `test_affected_k1_identity[cb_cycle_reverse_ids_segment]`.

The first eight are in `tests/collision/test_device_convex_query.py`; the
last is in `tests/collision/test_device_convex_patch_qualification.py`.
Original physical inputs, expected geometry, resource caps and numerical
gates remain unchanged. Passing four Y entries does not outweigh regression
of already supported finite/round-core families.

### Source-only interpretation, not a new measured diagnosis

In the tested kernel, lines769-803 detect duplicate support and require the
retained candidate's depth-plane gap to pass. Lines805-806 skip adding a
duplicate point, while lines1019-1065 reduce/compact the retained simplex and
update the search direction. The former post-reduction
`distance-bound <= 8e-7 or duplicate != 0` success exit was removed.

Consequently, when the retained candidate misses the plane gate and a repeated
support cannot enlarge the simplex, the new path has no duplicate-only
success exit. A persistent unchanged reduction can consume the original
iteration budget. This is a concrete control-flow difference and a plausible
regression mechanism, not a claim that each of the nine saved failures has
been individually traced to that exact sequence. No new trace or numerical
probe was authorized to turn this hypothesis into evidence in this batch.
The overlap label can also occur on a positive-distance round-core problem
after skin subtraction; it does not by itself identify an EPA failure.

The one-hypothesis stop condition is now exercised: preserve this experiment
and its failed results, do not enlarge capacities or relax gates, do not
special-case Y/CB/BH, and do not try another algorithm in this grant. Any later
repair requires a separate decision covering both candidate consistency and
the existing duplicate/stagnation behavior.

Only this report changes after validation. Product and test bytes remain at
the tested checkpoint. Status: **NOT_ACCEPTED / AUTHORING_SLOT_RELEASED**.
