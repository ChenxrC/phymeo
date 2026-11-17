# I3d: heterogeneous current-state area geometry

2026-11-17. codex/exp starts at 79dba4f0; dev is unchanged. Root owns shared implementation and the sole serial execution slot; original agents author tests and review without execution.

## Scope

Connect each declared model's current accepted TYX state to full body-local pressure assets, HOST finite-surface intersection, then the existing whole-batch area solver. Assets are owned immutable copies keyed by model key, including models sharing one physical bucket. Each current area bucket performs one fresh TYX FK. Native-only worlds continue through their original solver in the same transaction.

Generate all geometry and admit all packets before preparing physics. Publish all states, forces and the geometry receipt together. Failure leaves accepted banks, prior receipt and geometry publication intact; FK scratch and diagnostic counters may change. Keep declared empty contacts. Do not cache leaf owners across reset/restore. External batch revisions invalidate the scene's historical geometry.

Add a scene checkpoint envelope binding every model key and complete local asset digest to the original heterogeneous capsule. Asset mismatch is rejected before physical restore. Selective reset and restore clear geometry after successful revision change; empty reset preserves it. Retain original capacities, pressure law, friction restrictions, HOST geometry and TYX dynamics. No GPU clipping, FFT material coupling or new training-success semantics are claimed.

## Implementation

1. Extract shared current geometry preparation from the single-model scene, and shared locked pack/solve from the heterogeneous batch.
2. Add MuJoCoTyxHeterogeneousAreaPressureScene and an immutable scene checkpoint envelope. Exact area model/pair coverage is mandatory.
3. Add focused interpret/CUDA tests: same-bucket different assets, native/slide/hinge and independent single-scene agreement; continuous current geometry and entry identities; selective reset/restore and stale revisions; geometry/packet/late physics failure and retry; checkpoint asset binding.
4. Run affected single-scene protocol and heterogeneous transaction seams only, review source, record actual results, and commit implementation and documentation in small scopes.

## Validation and stopping rule

Use the existing Python/CUDA environment, a fresh entrypoint-aware guard before each serial project command, and fixed short trajectories. Preserve original failures. Diagnose a failed invariant before changing implementation; do not adjust physical parameters or tolerances to force passing. Once focused and affected checks pass, do not rerun unchanged accepted groups. No installs, remote CI, push or dev edits.

## Next boundary

The next independent capability is device finite-surface generation and total-step cost accounting. This batch completes automatic geometry for the declared full-pressure/finite-rigid asset domain, not arbitrary mesh collision, CCD, general soft-soft response or end-to-end device collision.

## Completed

New interpret 4 passed (21.78 s), CUDA 4 passed (14.34 s); affected single-scene protocol 16 passed (1.99 s), area transaction compatibility 2 passed (4.27 s). Four Python files pass final Ruff and format-check. No functional failures or parameter/tolerance changes. Tests cover five unordered worlds, three physical buckets, shared-bucket distinct assets, native contact, independent scalar-force/rotation oracles, late failures and retry, and owner replacement across boundaries. The layer-exterior world remains an empty-contact control in this short window; no new entry/unload trajectory claim. Original source and test reviewers found no blockers. See [report](../../development/area-pressure-heterogeneous-scene-report-20261117.md).
