# Bounded convex query failure diagnostics

Status: **DIAGNOSTIC_PROTOCOL_QUALIFIED / GEOMETRY_FAILURES_REMAIN**.
Parent tested source `e7c247c1`: run27 interpret 18 passed / 11.08 s
(14 TYX plus 4 HOST), run28 CUDA 14 passed / 39.89 s (4 already qualified HOST
nodes deselected). Runs29–34 captured all six predeclared CUDA inputs; all
returned RAW_EXIT0 and exact same-state ordinary/diagnostic geometry equality.
No geometry repair is claimed. The canonical root is
`artifacts/p65-parity-n3-rob1c-col-trace-20261111-01`.
See [recorded decisions and bounded next hypotheses](p65-convex-query-trace-findings-20261111.md):
the four Y cases exit on distance before normal accuracy; the two HN cases
repeat an identical affine-simplex reduction state through the original budget.
The original 0.002 normal gate and all resource caps remain unchanged.

The changed kernel ABI/default path also passed the frozen affected old
query/patch/source regression: run43 interpret **18 passed / 10.33 s**, run44
CUDA **18 passed / 61.32 s**. No complete old geometry matrix was rerun.
`col-evidence-index.json` SHA256
`fc683830e6777aee9a4fa99b31706e41c031982e42f97f1aaee698e546a17abb`
binds the ten actual runs, raw exits/logs, before/after input snapshots and
42 saved gzip files (six new complete captures plus 36 regression records).
Execution input drift is zero. Protocol assertions run in the tests themselves;
the index is file I/O only, not a second geometry verification.

The following interface and wire contract was frozen before execution.
Baseline: ed6f6d847c3d64abb13ab28f5f7042832c5b2f3e (product 0ea0fee2).
This node records actual device GJK execution; it does not repair GJK, change
the 0.002 independent normal gate, or increase any existing resource limit.

## API and publication

`plan.query_with_diagnostics(pairs=(...))` returns an immutable
`ConvexQueryTrace`, schema `phymeo-convex-gjk-trace-v1`.
Its `query` is the ordinary, unmodified `ConvexQueryResult` schema.
The explicit selection contains 1..8 unique canonical pairs, in caller order,
each present in this attempt's full query membership. Selection occurs before
results exist; RESOLVED rows may be selected because an external oracle fails.

`last_diagnostics` and `last_result` publish together only after complete
query and trace validation and temporary resource cleanup.
`require_current_diagnostics(trace)` binds the creating plan, immutable policy,
selection, actual shape/pair rows and ordinary source/epoch/frame validation.
Normal `query()` retains its original result and numeric decisions. It does
not allocate or read diagnostic buffers and does not overwrite the previous
diagnostic. A retained trace remains evidence for its original entry epoch,
not a claim about a later accepted state.

The diagnostic adds a single clear launch and exactly three temporary buffers:
float32 [D,H,416], int32 [D,H,96], int32 [D,2]. D <= 8, H is the original
GJK budget <= 64. Maximum device allocation is 1,048,640 bytes. The original
memory_limit_bytes covers resident plus temporary storage before allocation.
All allocation, clear launch, upload, readback and D2H bytes use existing plan
counters. Temporaries are freed in finally on every path. The trace also records
its precise owned layout, readback bytes and clear/transfer counts; immutable
host copies survive. No EPA history or extra support evaluations are added.
The disabled kernel receives existing arrays of matching dtype/rank as
never-accessed placeholders. Off/on geometry equality is tested; no claim of
compiler instruction or bit-level equivalence is made without execution.

## Integer wire: [selected pair, GJK iteration, 96]

All slots start at zero. Written iterations form a contiguous prefix. No-GJK
analytic or source-rejected rows have an entirely zero history; their final
status remains in query.raw.meta. Material slots are indices into original
uploaded vertex arrays, not compact contact identities.

| Indices | Actual values |
| --- | --- |
| 0..3 | written=1, attempt tag, full query row, zero-based iteration |
| 4..8 | entry simplex count, exit simplex count, duplicate (-1 when not evaluated), selected count, selected subset mask |
| 9 | decision: 1 continue, 2 plane exit, 3 duplicate exit, 4 enter EPA, 5 round-core unsupported, 6 no usable subset, 7 SUPPORT_BALL, 8 GJK budget exhausted |
| 10..15 | actual status, reason, phase, need_epa, GJK used, support calls |
| 16..19 | support A/B uploaded slots and original material IDs |
| 20..23 | appended-support flag, simplex count before reduction, full_surface flag, reserved zero |
| 24..31 | entry simplex material slots, 4x2 (only entry count valid) |
| 32..35 | entry indices, 4 (only entry count valid) |
| 36..43 | exit simplex material slots, 4x2 (only exit count valid) |
| 44..47 | exit indices, 4 (only exit count valid) |
| 48..92 | masks 1..15, three integers each: subset count, usable after raw-weight gate, comparison flags |
| 93..95 | reserved zero |

Comparison flags: bit0 actual better branch, bit1 actual equal-square stable
comparison was evaluated. A zero subset count means not visited. An unusable
subset has no candidate q/square and no comparison; its raw Gram and weights
remain available. The selected subset mask describes the pre-compaction
simplex formed by entry vertices plus the newly appended support, if any.
Decision 8 is written only after the original outer loop sets GJK_LIMIT.
Final query meta additionally records later EPA/SUPPORT_BALL publication.

## Float wire: [selected pair, GJK iteration, 416]

These are copies of actual FP32 variables/arrays, except two explicitly marked
diagnostic expressions that never feed a solver branch. Unknown/uncomputed
fields remain zero, with integer flags indicating validity.

| Indices | Values and frame |
| --- | --- |
| 0..3 | entry closest q and squared norm |
| 4..7 | exit closest q and squared norm (the actual variables, including unchanged q on SUPPORT_BALL) |
| 8..10 | actual support search direction |
| 11 | actual support-plane lower bound, normalized by query scale |
| 12..17 | actual bilateral support scratch, A then B; new full-surface pairs are pair-relative metres, old cores preserve their existing world-origin-relative representation |
| 18..20 | actual normalized Minkowski support w |
| 21..22 | chosen closest distance and distance-minus-bound; valid when selected count > 0 and regular reduction ran |
| 23 | diagnostic-only entry q dot (entry q - w), not an actual termination gate |
| 24..29 | actual curved support local directions A/B; only a curved side computes these |
| 30..31 | actual query scale, reserved zero |
| 32..67 | entry raw vertices 4x9: normalized w, support A metres, support B metres |
| 68..71 | entry selected weights |
| 72..107 | exit raw vertices 4x9 |
| 108..111 | exit selected weights |
| 112..411 | 15 subsets x20 fields as below |
| 412..415 | actual common kernel anchor ox/oy/oz and normalized rolling radius |

Each subset float block:
0 ee, 1 ef, 2 ff, 3 re, 4 rf, 5 actual triangle denominator (count=3 only),
6 actual tetra determinant (count=4 only), 7..10 raw weights before clamping,
11..13 candidate q, 14 squared norm, 15 diagnostic-only stable
(q-bestq) dot (q+bestq), 16 previous best square, 17..19 previous best q.
The ordinary stable tie branch expression remains unchanged. The diagnostic
copy in field15 is not substituted for that branch. Counts 1/2 do not invent
a triangle determinant. The complete simplex plus these values identifies all
Gram inputs; no HOST reconstruction is represented as an actual FP32 solve.

The original source, accepted ticks/active bank, shape revision, effective
rotation and pair origin remain in trace.query.source. trace.pair_rows and
shape_rows bind each history row to that exact result. Trace policy stores
GJK/EPA budgets, V/F limits, field widths and selected-pair limit.

## Fixed runner and protocol matrix

The new reference runner selects exactly:
Y-ordinary, Y-affine, Y-floating, Y-floating_affine, sep-HN,
rotated-reversed-HN. Y repeats the original four sparse worlds and first real
nonzero-velocity step, selecting world17 before query; all four states and
results remain saved. HN repeats the original fixed sep/cycle inputs.
All accepted states are saved before external geometry checks. The existing
independent literal oracle supplies expected geometry; product algorithms are
not an oracle. A saved diagnostic does not certify that geometry passed.
Each invocation writes one fresh gzip with dtype/shape/raw bytes, full query,
trace, inputs, independent oracle and source hashes. It never reruns the matrix.

Tests-first inventory was **14 TYX protocol nodes per backend +4 HOST
validation nodes**; actual execution counts appear above:
2 off/on geometry paths, 1 eight-pair selection, 1 bad/nonmember selection,
2 exact/short memory budgets, 3 late allocation/read/construction faults with
retry, 3 owner/stale/binding faults, 1 zero-GJK analytic result, 1 closed plan.
The four HOST nodes reject empty, duplicate, ninth and malformed selection.
The six fixed runner inputs are separate diagnostic captures, not six new
geometry passes. Existing accepted full matrices are not requested again.

Stop condition: any diagnostic failure to bind source/raw data or default-path
geometry change must be fixed before interpreting the six snapshots. The
resulting evidence may justify one later generic fix, which needs a separate
parent decision. This grant contains no algorithm or tolerance changes.

## Authoring completion and bounded validation handoff

The implementation records all original GJK assignments without substituting
trace values into the original algorithm. The kernel diff is additive except
for four explicit trace parameters. Default placeholders have matching ranks
and scalar trace_count=0, so no diagnostic array is accessed.
The helper validates full-surface anchor and scale against the same query's
actual gathered origin and raw scale; original material slots and IDs are
checked against that immutable source. Nonfinite *intermediate* arithmetic is
retained rather than rejected as missing evidence. k0.encode recursively tags
nonfinite values while bytes_hex retains every original FP32 bit; a HOST
assertion checks this serialization with allow_nan=False.

Each fixed runner entry now performs one ordinary query then one opt-in query
at the **same accepted state**, with no intervening physical step. It saves both
complete results. Every source/geometry field and raw byte must agree, except
the explicitly identified query.attempt_tag and raw.meta[:,12]. Pure statistics
counters are retained in both records and excluded from geometry equality
because diagnostic launches/transfers are additional real work. A mismatch is
saved as a failure, not normalized away. This pairing is necessary validation
of the new instrumented kernel, not a repeat of the old full matrix.

Parent-only proposed order: the four HOST validation nodes, 14 interpret
protocol nodes, 14 CUDA protocol nodes, then the six named fixed-input captures
in the parent-selected failing backends. Parent may capture both modes for a
specific original failure when needed to compare actual arithmetic; no full
COL matrix is requested. Fresh outputs and original failed evidence remain.
No imports, collection, functional tests, numerical probes, lint, formatter,
build or profile were executed by this authoring lane.
