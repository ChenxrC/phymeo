# COL3 installed public dispatch consumer

Status: CODE_READY_TESTS_PENDING / UNEXECUTED / AUTHORING_SLOT_RELEASED.

NEW `tests/installed/check_triangle_dispatch.py` is a standalone, source-external consumer. Copy the file outside every checkout and invoke the existing installed profile:

```text
python <outside>/check_triangle_dispatch.py --mode interpret|cuda --installed-root <absolute-site> --output <fresh-absolute-directory>
```

It never edits sys.path or imports repository tests/examples. Preflight rejects checkout entries, an already imported Phymeo, or checkout cwd/script/install paths. Every loaded Phymeo module/file and package search path must remain under the bound installed root, with file SHA256 recorded. Dependency locations remain the already qualified profile; this does not install or alter them.

## Frozen bounded workload

Exactly **four aggregate queries per backend**, not a repeated source matrix:

1. Main workspace: original snapshot.
2. Independent workspace: same snapshot, establishing a genuine foreign owner for the owner guard.
3. Main workspace: new snapshot inside an outer transaction, then an authored outer abort.
4. Main workspace: the same new snapshot as a normal successful successor.

Each aggregate has worlds 7/4096 and repeated local body/shape/material IDs. Each world contains a closed 12-triangle cube mesh; a sphere or finite capsule is above the two shared bottom facets. Ten genuinely distant facets per group must be absent from the actual same-call BVH bitset. Each query has two groups/four finite rows; each backend therefore performs eight COL2 group queries and records sixteen rows, including the aborted but actually evaluated prefix.

The second snapshot moves only the original bottom four mesh vertices upward by 1/16, keeping the top four fixed. It is a real coordinate deformation, not merely a changed label. All registrations/revisions/IDs, round sources and margin remain unchanged. Exact independent distances change from 1/4 to 3/16; radius is 1/8. The original finite-axis endpoint, projected material point, downward normal and shared-facet ownership are checked directly from fixed coordinates. The original gates remain distance `1e-4 L`, witness `2e-4 L`, normal `2e-3`, with `L=sqrt(8)` for these selected original bottom triangles. No absolute translation inflates L; world 4096 is translated by `(2^20,-2^20,4)`.

The public entry is exclusively `CollisionWorkspace.boundary_distances` and `require_boundary_distances`. No manual candidate array is supplied to the narrow query, and no product nearest-point helper is used as the oracle. The complete original facet table, dense-to-material map, actual candidate bitset, BVH visit counters, all COL2 raw fields/source/upload/conversion records and original filtered pair table are retained. The result remains threshold-limited boundary distance, not solid classification, global nearest outside the band, CCD, force sites or a resident vertex loan.

Negative/transaction assertions are bounded and advance no extra geometry: changed-input binding, genuine wrong owner, uncommitted current rejection, superseded current rejection; outer abort retains the original public aggregate and its bytes, while the next normal query publishes only one increment in public sequence. Ordinary `last_result` remains untouched. Attempted private launches/counters are not rolled back or hidden.

## Evidence and failure retention

`triangle-dispatch.jsonl.gz` stores complete typed array bytes/shape/dtype, dataclass fields, original inputs, every returned aggregate before its checks, the accepted/staged/retained sequence, installation provenance and actual counters. `summary.json` contains status, fixed scope, gates and logical/compressed SHA256/byte counts. No field is deleted to make a comparison pass.

All independently owned workspaces are closed even when a sibling close fails. Cleanup exceptions are collected with the original test exception as a BaseExceptionGroup; a log-write error also retains the original rather than masking it. Main finalization attempts provenance, journal close and summary independently. A missing failed internal packet is never fabricated: the consumer retains the full returned prefix, requests, public previous aggregate, counters and error. Public source-suite late-read/UNKNOWN_RELEASE controls remain in their already authored dedicated suite.

This commit was checked only by source reading and `git diff --check`. No import, collection, native call, project computation, formatter, lint, pytest, build or installed execution occurred in the author lane. Parent review, sole serial execution and final wheel/source parity are still required before an installed acceptance claim.