# Directed CI interval cost repair

Spec authority: docs/superpowers/specs/2026-10-30-p65-ci-cost-repair-design.md R6. The parent-only Task4 is context, not an instruction to package/push.

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
