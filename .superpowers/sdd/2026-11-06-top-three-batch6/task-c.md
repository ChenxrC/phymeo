# C2 phase-transfer C0 brief

Status: TEST_AND_PROFILE_AUTHORING_ONLY; parent owns all project execution.
Plan: docs/superpowers/plans/2026-11-06-top-three-batch6.md.
Reuse original rd author/tree; clean ordinary fast-forward supplied parent commit; no reset.

NEW only:
- scripts/profile_material_phase_transfer.py
- tests/scale/material_phase_transfer_cases.py
- tests/scale/test_material_phase_transfer_profile.py
- docs/benchmarks/p65-material-phase-transfer.md

First tests/protocol commit then profile implementation in own separate commit after parent valid RED. Never modify existing TyxNormalLayer/TyxMultifaceContactBatch, old profiling tools or physical inputs. No MaterialForceFrame now. No new engine until 15% gate. Source-specific facts: each actual phase page128 uploads6656B/downloads2560B with one kernel/sync; phases entry/midpoint/endpoint physically dependent cannot combine. WorldContactEngine already once-sync per phase chain, not scope here.

Freeze W1/W32 ×2/8 faces and heterogeneous W3留出 from accepted physics constructors; exact inputs/ordered world IDs/phase site identities before any output. Include independent 0/127/128/129-row law protocol; these arithmetic rows are not physical trajectory results. Require actual >128 pages in measured dense task; if physical 32x8 unexpectedly doesn't reach it, report instead of tuning output. Boundary rows finite FP32 and independent closed-form forces incl normal, zero, separation, friction/damping, varied weights.

Profile contract should use --mode cuda/interpret --worlds --faces --expected-head --out; optional named heterogeneous case. All selection/budgets validate before creating runtime. Save full13-column inputs/5-column results, each phase S/P and actual counter deltas, full physical results/accepted states/reports (only predeclared exact timing field normalization for rebuild parity). Separate cold, bounded warmups, fixed formal windows. Bound duration/bytes in protocol before execution, not hundreds of duplicate old windows. Native error/unsupported capability not skip/fallback.

Measure wall categories disjointly on caller thread: full step, material-layer upload/launch/sync/download/HOSTpack/scatter-publication; don't sum parked worker intervals as wall time. Capture instrumentation full outputs and compare a separate no-instrumentation same-input run, preserve both actual counts. Existing W4x8 0.60/162.36ms is prior evidence not new claim. Gate: actual removable phase transfer/sync overhead / full task wall >=15% in a dense real case. Cold compile not gate evidence. If gate fails, report honest C0 completion/no default change.

Protocol tests may inject clocks/recorded calls to verify non-overlap, page counts/tail/empty, rejected capacity/late failure and retry metadata; do not mirror implementation assertions. Exact RED node imports new profile interface in a guarded test? Prefer existing imported module surface absent only if substantive API missing; missing script import itself is not valid behavioral RED. Author first an independent protocol/schema validator then first behavior test if needed; parent decides valid gap.

After TESTS_READY report first, no Python/import/pytest/collection/lint/format/profile/build. Subsequent profile-only code also no project execution. No helpers. Report baseline commands and snapshot lists; parent queues. No implementation of optimization, defaults or physics changes.
