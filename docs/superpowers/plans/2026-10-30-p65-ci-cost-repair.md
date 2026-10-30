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

父检查点：308源码及93证据hash无差异，20成功rawexit核对；新源码packaging0065200-a两step exit0，wheel153字节相同。Task2仅剩专用分支push和准确新CI/Documentation；现未勾完成。

## Task 3: second attempt, exact interval duplicate-work repair

Read specR6 first; it explicitly supersedes only the first attempt's one-file scope for this attempt. Baseline a478828aa6ab8c59dc310cd4639e6f7a55704bec, source/tests/scripts308 digest9bbdeeeba487184c3a95fd96b0d20ae928ce1554c550beb2c5a0d598b79c745d. First attempt remains locally complete but7CPU timeouts leave fullCI delivery open. No new physical capability.

**Files:** Modify only src/phymeo/geometry/_patch_interval.py (mul and equal-q prismatic endpoints); create tests/test_patch_interval.py. Reports/artifacts under artifacts/p65-ci-interval-optimization and .superpowers/sdd/2026-10-29-p65-thick-friction/task-3-ci-interval-report.md. Original old tests and all other production unchanged.

**Interfaces:** Existing tuple-of-two-Fraction interval helpers and PatchCurve fields/methods remain byte-value equivalent. No new public API/cache. Source consumers robot_patch_sweep, prism_patch_manifold/sweep and finite/thick solvers receive unchanged complete exact intervals.

- [ ] Verify Git clean, final attempt1 manifest/workflow and no numerical process; freeze original _patch_interval.py bytes from a478828 with exact hash. Write before-edit impact/test list. Do not rerun ended profiles/CI or old baselines.
- [ ] Write tests/test_patch_interval.py with independent products: (2,2)*(-3,5)=(-6,10); (-2,-2)*(-3,5)=(-10,6); (-2,3)*(-5,7)=(-15,21); (0,0)*huge mixed=(0,0), and left/right/two-singleton variations, reversed endpoints, huge/subnormal Fractions, exact output types and zero-divisor errors. Add counting controls for duplicate multiplication and _prismatic calls; capture original functional GREEN separately from genuine cost RED before code.
- [ ] Add curve controls: sameconvertedq uses one endpoint evaluation, differentq two, noncoordinate prismatic norm encloses exact values, shared ancestors/fixed frames, rotor bounds/range rejection unchanged. at(t) including interiorFraction and outside[0,1] retains current exact tuples. No mutation of caller q/model/collider.
- [ ] Implement only those two pure optimizations. Run new math tests and frozen-old differential before any native validation. Compare helper tuples/errors and full curve outputs across independent deterministic cases; do not weaken prior oracle.
- [ ] Collect one new sameA profile and3old/3new fresh sequential observations (a478828 oldmodule injected before any phymeo consumer import). Require full R5 canonical report/input/state/history/CCD equality; record rawtimings/fingerprints. This is bounded CI-cost observation, notP95/900. Report if marginal or any equality fails before expanding scope.
- [ ] Freeze final source and run host once with fresh basetemp: tests/test_patch_interval.py tests/test_prism_exact.py tests/test_thick_patch_geometry.py tests/test_thick_contact_mapping.py tests/test_thick_robot_contact.py tests/test_thick_robot_transaction.py tests/test_finite_patch_geometry.py tests/test_finite_robot_contact.py tests/test_finite_robot_physics.py tests/test_finite_robot_transaction.py. Confirm collection, originalwindows/halving/rebuilds and all latefaults; preserve rawexit and versioned manifests. This broader set is required because the pure interval helper is shared.
- [ ] Then actual tests/test_thick_robot_tyx.py tests/test_finite_robot_tyx.py --tyx-mode interpret --require-tyx, then sameCUDA; each serial with fresh basetemp, no skip/fallback. Native transactions/FP32/rotor consumers included. Use the same physical runs for actualrecord audit; never repeat just to generate reports.
- [ ] Ruff/format changed files, publicboundary and source smoke as applicable. Commit product/tests and final report, give exact before/after manifests, commands/results, original failures and limitations, resource/process0 and clean/indexrelease. No helpers/reviewer/push. Do not change parent pendingCI boxes tocomplete.

## Task 4: parent interval-repair delivery

- [ ] Read the original implementer's final report, compare narrow math/constructor diff withR6, inspect rawexits, differential inputs/outputs, full report parity and affected consumer evidence. No second whole-stage fresh review or duplicate test run.
- [ ] Fresh build and outsidecheckout nativepip smoke with source byte-parity at the new snapshot; no oldCPP or wheel mislabeled as newhash. Update versioned docs, push dedicatedbranch, verify remote fullSHA and exactnewCI/Documentation plus8CPU/examples/required success under30minutes. Only then close this thickness slice and proceed next production contract.

## R6 preflight / review focus

| Risk | Owning check / ruling |
| --- | --- |
| Negative singleton or reversed endpoints select wrong product | Task3 independent sign table and four-corner frozen differential; always min/max, no ordering assumption |
| Invalid/large input behavior changes through scalar shortcut | Task3 zero division, huge/subnormal Fractions, numeric types and exception differential |
| Sameq endpoint sharing leaks mutable state | Endpoints are immutable Fraction tuples; Task3 caller immutability and distinct-object fresh-workspace checks; no shared cache |
| Noncoordinate prismatic or rotor intervals silently shrink | Complete rational ends/at/curvature/plane differential plus all finite host/actualbackend consumers |
| Evidence reuse hides shared-helper regression | Final full finite+thick affected selection atnewhash, twoactualbackends, rawexit; unrelated oldlong/900 excluded with import rationale |

Ruling: user already chose30minute optimization and authorized continuingP6.5. Parent resolves this bounded shared-pure-geometry extension from existing profile evidence; no new user decision. All older frozen physical requirementsR1-R4 and observerR5 remain. Original implementer alone owns production/numerical/index after dispatch; parent only reads until release. Existing builder and whole-stage reviewer are finished and must not be repeated.
