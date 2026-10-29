# Task1 review fixes I1/I2 — execution/evidence ledger

Review-fix implementation and directed verification COMPLETE. Baseline a2ba474; I2e8c821d; I1/R4 product33ec9ea52dc15ee5b8050adb0a73d1591aea11f5. Unique review0C/2I/0M; main owns finding closure, no second review. Final source/tests/scripts digestcd6d77917dee8af09f13a430f4fdd4a5b744461874364a142aff0d8e6722d84b (307files), source-onlydf8c2ad9698f3f4c958b75ad9eee4135162fac1be490e56dd29cfce3df1f8d94 (153files). Actualinterpret10passed134.13s/CUDA10passed327.19s rawexit0 on this exact source; focused host7 and host-tail17 rawexit0. No process remains; numerical/production/index released to main with this final report commit. No helper/reviewer rerun/push.

## Dependency and impact before edits

- I1 changes only held g_gen in `_current`, using initial selected SAT axis and its complete real patch support vertex set, translated by current legal all-prismatic FK. Preserve exact finite point/segment/polygon membership, the initial held side, and actual entire-patch distance fallback outside that feature. New helper belongs in existing `_thick_contact_geometry.py`; `_current` consumes it. Public SAT/manifold, ports, J0/Jpre/J1, native reaction points, laws, CCD and budgets stay unchanged.
- I1 consumers: position update/metrics and held metrics rechecked in terminal. Exact edge-side automatic collision and corner-side automatic collision must converge from zero; independent reviewer p=.25 candidate stays a separate lawful native control. Boundary feature-membership controls prevent infinite-plane ghosts. Cap A/B/material, R2 grazing, C partial exit/reentry, birth ownership, selected-pre and late rollback are direct regressions.
- I2 changes only new independent mapping ledger require: retain supplied-request aggregation/coefficient guard and moment/power; add independently expected J.T multiplied by actual world impulses, accumulated per node, plus independent actual resultant. Preserve original speed/length/tol scales and do not divide by impulse magnitude. Both held and terminal replay call this ledger by node domain. No old shared module is modified.
- I2 exact controls: original 2^-36 coefficient perturbation ×2^20 impulse (valid control remains GREEN), plus equal/opposite perturbations on nodes0/2 giving zero resultant/moment/power but nonzero weighted node error. Shared-node aggregation and zero-impulse coefficient guard included.

## Exact intended selections

Host RED before product edits:
- tests/test_thick_robot_contact.py::test_review_side_feature_automatic (edge/corner)
- tests/test_thick_contact_mapping.py::test_review_resultant_and_weighted_mapping (translation/canceling)
- initially GREEN: ::test_review_valid_large_impulse_and_shared_nodes and ::test_review_independent_legal_side_candidate

Focused host GREEN/regressions:
- tests/test_thick_contact_mapping.py (all small independent ledger/law controls)
- tests/test_thick_patch_geometry.py (all small exact SAT/CCD controls)
- tests/test_thick_robot_contact.py::test_review_side_feature_automatic
- ::test_review_selected_feature_finite_fallback
- ::test_review_independent_legal_side_candidate
- ::test_two_jaws_static_and_low_friction_single_step
- ::test_open_impact_full_three_corner_oracle
- ::test_distance_material_pre_ledger_and_terminal_post
- ::test_anchor_birth_never_copies_held_capacity
- ::test_exact_grazing_reentry_prefers_patch_face_axis
- ::test_partial_overlap_exit_and_input_driven_reentry (one prescribed h.25 C, not full repeated/halved windows)
- tests/test_thick_robot_transaction.py::test_late_failure_rollback_and_retry_fresh (mapping/CCD/close at substeps2/3; six exact parameter cases)
- ::test_best_does_not_hide_execution_or_unknown_failure
- ::test_terminal_only_replay_rechecks_complete_selected_pre_evidence
- ::test_anchor_capacity_and_geometry_rebuild_budget_fail_closed (three exact controls)

Actual interpret then CUDA, strictly serial, --require-tyx:
- tests/test_thick_robot_tyx.py::test_review_actual_side_feature (edge/corner)
- ::test_actual_two_jaw_support_and_slip (mu1/.25 cap normal controls)
- ::test_actual_three_corner_impact_and_material (both material variants, full mapping phases)
- ::test_actual_partial_overlap_lifecycle (one C h.25 finite lifecycle)
- ::test_actual_failure_and_all_trials_count (actual FP32, every trial/resource close, second-substep fault/retry)
- tests/test_terminal_impulse_tyx.py::test_actual_terminal_analytic_and_original_material_ledger_only_accepts_pre (both variants, unchanged direct native consumer)

No host60, old full matrices, unaffected repeated A/C halved windows, old long material/normal suites, CPP/wheel/900. Final source differences and reused1fcd receipts remain explicitly versioned. Any expanded necessary selection will be justified before running. Main owns original finding closure and later CPP/build/wheel, not a second review.

## RED and first repair evidence

`red-02.txt`: five expected assertion failures (I2 translation/canceling/shared canceling accepted; I1 edge/corner no automatic pressure after two sweeps), three initially GREEN controls, 1.57s, raw exit1. `red-01` was interrupted before the bounded response assertion was introduced; it is not a completed budget result. The original a2ba474 product is the RED source; `red-attempt-manifest-01.json` preserves the earlier test byte version.

`green-01.txt`: 12 passed/1 failed87.72s, raw exit1. All11 mapping tests and edge automatic GREEN; corner still rejects after512 due point support membership using rounded port coordinates. Its owned B/s ideal point has exact equal X/Z, whereas port evaluation differs one ULP; first repair source captured in `first-repair-manifest-01.json`. Remaining later nodeids were not executed due -x. Next repair evaluates exact projection from owned B/s/current stored prism, gates rounded-port representation with original b, and includes both signed gap bounds. No epsilon membership, normal snap, new budget or state/load replacement.

`green-02`:16 passed/6.37s raw exit0, focused-manifest-02. Added actual side feature parameterization for identical automatic inputs. Expanding host selection to existing oblique static and rejected-candidate birth regressions: exact current owned-point predicate is shared by those non-axis/anchor discovery consumers. No long window expansion needed; cap finite face and C lifecycle directly exercise existing face/interior/outside paths. Next regression/native source is regression-manifest-03 (format-only delta from focused manifest plus two native tests).

I2 completed unit: all11 mapping tests GREEN in green-02, including original large-force defect and zero-resultant/nonzero-node-error control. Staging only the new mapping ledger/test for a focused commit; I1 WIP remains uncommitted while serial host regressions run.

## Regression diagnostic checkpoint

`host-03` regression-manifest-03 digest9c2c998ecefa7fe85022f878ec0c054a43ce68892fc45c2aed55bd3559740a2a:31passed/1failed59.04s raw exit1, stopped at existing `test_anchor_birth_never_copies_held_capacity`; all later host and transaction selections unexecuted. C finite-face boundary exact-owned position moves outward by a tiny current unitnormal component from stored-node roundoff, while actual port position remains exactly on finite boundary. Corner has the reverse: exact-owned point exactly on support point, rounded port projection is −1/36028797018963968 offpoint. `birth-trace-01` seven-sweep exit0/resourcesclosed; `boundary-probe-02` exact fractions exit0/resourcesclosed. boundary-probe-01 was an observer syntax error, never product evidence. Original clipped material gives the same C outward position, so changing owned to ideal-clipped does not resolve it. Main notified for semantic ruling; no epsilon inclusion or union of representations applied. No law/tol/budget change. No native/backend regression started.

I2 focused commit e8c821d. Original mapping probe closure copy changes only new-manifest validation and expects SolverFailure on original delta; valid input remains accepted with four zero residuals, translation defect4.57763671875e-5 rejected. mapping-original-closure-01 raw exit0, no native resources. Original reviewer probe preserved untouched.

## R4 main ruling before further tests

Main explicitly resolves g_gen discrete query to actual FP64 public-port position converted exactly to Q/Fraction; current selected finite support feature remains exact and initial-side identity fixed. Reject actual/ideal union, epsilon, snapping, or case-dependent representation. Preserve independent actual-v-current-owned error<=original b and signed gap interval enclosing both actual/owned when actual is inside. Outside actual projection strictly uses finite-distance/inactive. All ideal full-prism SAT/manifold/CCD/extreme-velocity acceptance stays unchanged.

The new diagonal square-corner input remains unchanged in helper and prior failed512 log; its general automatic-convergence assertion is withdrawn under the frozen absence of global convergence guarantee, not relabeled success. Replace that assertion with bounded exact actual-outside/ideal-inside finite-inactive regression. It remains a documented candidate reachability limitation within supported input domain. No case-specific production gate.

Before testing, freeze separate exact point-dimensional normal control: same triangle ((.5,0,-1),(.5,0,1),(1.5,0,0)), uniform initial velocity(-.25,0,0), h=.25, mu0, rho=.125; triangular patch ((.5,0,0),(-.5,0,-.5),(-.5,0,.5)) has unique +X support vertex(.5,0,0). Exact clipped row B=(.5,.5,0), s0; effective normal response .5, so independent held impulse .5 along+X, each first-two node impulse .25 and final vx0, third vx−.25, robotY stays0, terminalworld0. Exact normal projection is one point and stays Z0 under pure X update; no two-component rounded diagonal cancellation required. Host/native automatic parameter `point` uses this independent oracle; original `corner` preserved only finite-membership/limitation controls.

Next bounded RED targets actual-only diagonal membership before changing predicate; then R4 production change and focused edge/point/birth/fallback/diagonal controls. Already-passed host03 pure mapping/geometry and A/B fixtures retain explicit version evidence; only g_gen membership changes and affects boundary branch. Planned host tail and transactions remain outstanding.

R4 RED1failed/.55s exit1 before actual predicate; r4-green04 7passed13.10s exit0 at r4-manifest04. Added explicit one-support-vertex/one B=(.5,.5,0) assertion plus independent totalP=.5 oracle; host-tail05 reruns automatic edge/point with that additional observation and all previously unexecuted tail/6fault controls. Exact command is host-tail-05.ps1; snapshot final-fix-manifest05. Host03 pure mapping/geometry unchanged source; A2mu/B/material queries lie strictly inside wide patch face, so actual/ideal predicate both true and both signed interval values unchanged. Its passes remain source-labeled, not rerun or called final full suite. C boundary/birth explicitly retested on actual predicate; R2/lifecycle/oblique/candidate hook pending host tail.

I1 completed focused implementation checkpoint: R4 affected7 passed; subsequent tail has independently passed automatic edge/point, exact R2 reentry, C full h.25 lifecycle, oblique static and rejected-trial anchor preservation on final-fix05 source. Six late-fault/retry and remaining fail-closed controls still executing; no completion claim. Ruff/format6files and public boundary153 pass; diffcheck clean. version-delta05 verifies307files,153source; exactly3newproduct/3newtest changed from1fcd, no added/removed module. Source-only sortedmap SHA256df8c2ad9698f3f4c958b75ad9eee4135162fac1be490e56dd29cfce3df1f8d94. Commit R4 geometry/test/spec now; following numerical result report will retain this source manifest despite Git HEAD advancement.

Host-tail05 completed17passed433.27s rawexit0 on exact final-fix05 source (product33ec9ea; I2e8c821d). All6 selected late mapping/CCD/close faults at second/last substep rolled back and retried against immutable fresh-program state/history oracle; remaining selected-pre/fingerprint/capacity/rebuild fail-closed controls passed. No wholehost60/longwindows rerun claim. Same-run C13record audit exit0 max analytic stateerror8.595465450511597e-10<=original1e-8, fullhistory present, physics/force/CCD/finite lifecycle passed.

Original side probe closure05 changes only source-manifest gate/reference, preserves original fixture and independent p=.25 control: rawexit0, resourcesclosed. Automatic edge response separately passes from zero; original probe after-two-sweeps now generates pressure. Direct new smoke entry exit0, no optional runtime import; no wheel/C++ claim. Ruff/format6 and publicboundary153 exit0. Starting actualinterpret05 exact10-case selection saved in native-interpret-05.ps1, --tyx-mode interpret --require-tyx, fresh checked basetemp; source/pin preflight verifies307file hashes and existing99f6df7c. No actualbackend pass claimed yet; CUDA not started.

Actualinterpret05 completed10passed134.13s rawexit0, no skip/fallback, exact product33ec9ea source. Same-run13C actual state/fullhistory records independently audited: max analytic error1.8835067748801393e-6<=original2e-5; exactFP32 promotions, per-step original physics/mapping/force/CCD/lifecycle all passed rawexit0. Native trial/count/late terminal fault and old2terminal consumers passed. Starting CUDA05 identical10selection and source/pin preflight, --tyx-mode cuda --require-tyx, fresh checked basetemp; no CUDA result yet.

Reuse accounting is explicit: host03's31passes consist of11mapping+11standalonegeometry+4canonicalA/B/material (26 unaffected passes reused), plus five side-feature checks. The five side checks are not wholesale reused: edge/fallback/legal are retested under R4, and the diagonal automatic success on the temporary ideal-predicate source is not a final-source success claim. Its earlier actual-predicate failed512 plus bounded R4 inactive control remain the limitation evidence. The new exact point normal control passes host and interpret; CUDA remains running. No old source1fcd long-window result is relabeled final-fix05; direct finite lifecycle has fresh same-run records and no duplicated halving/rebuild window.

## Final directed verification and release

- CUDA05:10passed327.19s, rawexit0, no skip/fallback. Same actual source/pin preflight as interpret; product33ec9ea. CUDA same-run13C state/fullhistory records audited independently, exactFP32 promotions, max analytic stateerror1.8835067748801393e-6<=original2e-5, original per-step physics/weighted-map/force/CCD and finite exit/reentry passed. Host same-run13 error8.595465450511597e-10<=1e-8; interpret13 error1.8835067748801393e-6<=2e-5. These39 observed records are copied to `artifacts/p65-thick-friction-review-fixes/actual-lifecycle-{numpy,interpret,cuda}-05.jsonl`; no expected-only sample is counted. No new rebuild/halving comparison claim is made.
- `verification-summary-06.json`/`final-evidence-check-06.txt/.exit`:307current source/tests/scripts hashes verified, exact changed set3new production modules +3new testfiles, no new/removed module. Saved command scripts `host-tail-05.ps1`, `native-interpret-05.ps1`, `native-cuda-05.ps1`; original probe closure copies, read-only audit scripts and per-run raw stdout/exits are retained. `evidence-receipts-06.json` hashes73flat evidence files, excluding itself and its active validator log/exit. The latter exit0 is separately retained. Old interrupted RED1/observer error/failed firstrepair/birth regression are not erased or relabeled passing.
- Original I1 edge now solved automatically fromzero, exact point-dimensional independent normal control also solves; both actual backends verify actual promoted states. Side original candidate probe still allzero residuals and certifiedCCD; original I2valid probe allzero and unchanged2^-36×2^20 defect rejected. Zero-resultant/nonzero-node-error and shared aggregation controls pass. Both held and terminal replay consumers, late faults, original pre identity, resources and finite budgets verified by declared selections. Parent closes findings by reading these receipts, no new reviewer.
- R4 limitation remains: the preserved diagonal square-corner input can fall into actual-outside/ideal-inside finite-point inactivity; earlier512failure is recorded at its exact firstrepair snapshot, final R4 bounded current-input inactivity is verified. No assertion of general point convergence, no union/epsilon/snap/case-specific gate. The withdrawn added convergence assertion is disclosed; frozen A/B/C and original edge expectations were not weakened.
- Reuse/omission:26unaffected host03passes versioned as above; old1fcd long A/C halving/rebuild windows retain their source labels. Finite boundary C and both A/B/material actual phase consumers were freshly covered; no host60, old full, long unaffected windows,900, C++/build/wheel, CI or push by implementer. Main owns remaining engineering and original finding closure.
- Final process inventory `final-process-check-06.json`:remaining_python_processes0, empty list after native sessions99862/65222 and host32901 completed; all resource assertions passed. Final report commit contains only3stage reports, then clean worktree verified; numerical/production/index ownership released to main. No outstanding implementation/verification process or tool blocker.
