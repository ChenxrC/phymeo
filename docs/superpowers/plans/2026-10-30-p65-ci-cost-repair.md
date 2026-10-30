# P6.5 CI cost repair implementation plan

> Use the existing original implementer for this directed delivery repair. Keep one numerical/index owner and do not repeat old Task1, builder or whole-stage review.

Spec: [exact arithmetic equivalence](../specs/2026-10-30-p65-ci-cost-repair-design.md). Physical authority remains the [thick friction specification](../specs/2026-10-29-p65-thick-friction-design.md). Validation policy: [affected consumers](../../development/p65-validation-policy.md).

## Global constraints

Keep30-minute workflow byte-identical; no omitted test, shorter window, changed fixture/oracle/tolerance/iteration, dependency or physics change. Single production file _prism_exact.py plus new mathematical tests; fail-closed canonical/exact behavior. Three-run paired observations only, not900/P95. Preserve all historical failed/cancelled/interrupted evidence. Baseline127cd28, product307digestcd6d77917dee8af09f13a430f4fdd4a5b744461874364a142aff0d8e6722d84b.

## Task 1: exact arithmetic cost reduction

1. Record baseline Git/diff, unchanged workflow hash and source/tests manifest. Read the spec and existing exact arithmetic consumers. Preserve profile69618 completed exit0; do not poll/restart it.
2. Write tests/test_prism_exact.py: independent exact math/invalid inputs/cross-field/zero/canonical/bounds/float controls and deterministic redundant-root-work RED. Run only new tests, record initial GREEN separately from true RED.
3. Implement rational/canonical fast paths in _prism_exact.py. Use no global memoization, new dependencies or geometry/solver changes. Run new tests and the frozen baseline differential artifact, inspect exact tuple/bound/exception parity.
4. Run identical profile under a new artifact tag and compare exact state fingerprints and CCD/replay counts; capture raw reports/fingerprints and apply R5 only to typed SceneSubstepReport.timings when requiring complete canonical report equality in fresh old/new observations. The original profile raw report fingerprint remains historical, not a cross-run equality gate. Run3old/3new unprofiled A-step observations in fresh sequential processes; no production file swapping or git checkout/reset. If gain is marginal, report before expanding scope.
5. Freeze source and run host once: tests/test_prism_exact.py tests/test_thick_patch_geometry.py tests/test_thick_contact_mapping.py tests/test_thick_robot_contact.py tests/test_thick_robot_transaction.py. Then actual tests/test_thick_robot_tyx.py --tyx-mode interpret --require-tyx; then cuda, serial. Existing window assertions and every late fault/retry remain. Fresh checked basetemps. Capture current actual record outputs. Reuse the same run for independent JSON audits; no duplicate physical run just for reporting.
6. Ruff/format changed files, publictyx boundary, source smoke as applicable. Commit logical production/test repair and final report; report exact manifest/commands/raw exits/limitations/0remainingprocesses/clean tree. No push, new assistant or reviewer. Write .superpowers/sdd/2026-10-29-p65-thick-friction/task-1-ci-cost-report.md; update original ledger with current ownership/status. Parent resumes delivery.

## Task 2: parent verification and delivery

Main reviews mathematical equivalence and scopes, audits raw outputs and snapshot hashes, completes new-source packaging parity, updates delivery records without rewriting5150c3e cancellation, pushes dedicatedbranch and verifies remote exactSHA/newCI+Documentation. Existing fullmatrix30min is final cost gate. No new physics contract until delivery closes.

## Preflight rulings

| Pair/task | Producer/consumer | Finding and ruling |
| --- | --- | --- |
| Task1 self | Tests, optimized scalar, same native consumers | Exact coefficient/bound/exception equality is binding; speed never changes physics. Cost RED is separate from functional initial GREEN. |
| Task1/Task2 | Frozen new source/evidence to parent wheel/CI | Parent waits for release; no concurrent numerical/index use, no old wheel represented as new source. |
| Existing stage/new repair | Previous unique review and failedCI | Same delivery repair, not repeat implementation. Main verifies this one-file arithmetic diff; no second whole-stage review. |
| Performance policy/user | Formal900 vs CI cost diagnosis | User requests shorter CI, not new solver benchmark. Bounded paired observations + actual30minCI answer that question; no formal P95 claim. |

## Progress

- [x] User chose original30minutes; completed baseline profile establishes redundant exact arithmetic hotspot.
- [x] Task1 optimized source0065200, exact equivalence/R5 reports, host96 and actualinterpret9/CUDA9 plus224actualrecord audits complete; numerical/index released with final report.
- [ ] Task2 new-source package and accurate fullCI success.

R5 observation correction: original rawreport fingerprint contains perf_counter timings and is not cross-run invariant. Preserve it and allrawreports; for planned3old/3new require exact canonical report/input/state/history/CCD equality excluding only the typed SceneSubstepReport.timings field, per specR5. No originalprofile rerun or production/runtime change.
