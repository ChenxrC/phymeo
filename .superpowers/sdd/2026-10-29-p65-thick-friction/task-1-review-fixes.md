# Task1 review fixes I1/I2 — execution/evidence ledger

Baseline: a2ba4745bc750ac858a61965221cd893561ee97f; product1fcd0e7; frozen spec unchanged. Unique review 0C/2I/0M, accepted by main. Implementer exclusively owns production/numerical/index. No numerical process active at this checkpoint; RED/first repair results below; no tool blocker, no helper/reviewer/push.

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
