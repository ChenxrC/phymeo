# B2-M serial validation result

Status: **HOST_VERIFIED_NATIVE_REPRESENTATION_UNRESOLVED**.
B2-M is **not accepted for native/composite delivery**. This releases the executable
slot only; it does not complete B2 or waive its outstanding native/physical gates.

## Frozen versions and execution provenance

- Code-authoring baseline: `1108af389dde69506dd6f76c3b0902ac008ac695`.
- Scoped style/evidence-capture checkpoint: `3094c1e`.
- Product snapshot for the accepted host run and rejected native run:
  `72bdbcb8c86f65c6cd5be8c13bed2123254fa647`.
- No product source/test change followed native failure or diagnosis. A pure
  file-hash comparison confirmed the current 19 Python files exactly match the
  complete passing host run's frozen manifest.
- Worktree: C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo;
  branch codex/p65-bending-module. Old B1, distance, shared runtime/contact source,
  tests, exports, workflows and pins remain unchanged.
- Host: original .venv-py311, Python3.11.15, numpy2.4.6, pytest9.1.1.
  Native: original .venv, Python3.12.14, numpy2.5.4, pytest9.1.1,
  tyx-compute0.1.0a0 at commit99f6df7c0fd8fdeef77f97df83914da27b64b7ee.
  Both actual phymeo imports were this worktree's src/phymeo/__init__.py.

Evidence root: `artifacts/p65-b2-validation-20261030-181949` in this worktree.
Every numbered command has a JSON receipt with expanded interpreter/arguments,
start/end time, environment, HEAD and RAW_EXIT, a complete raw log, and its exact
19-file SHA256 manifest. Original code and the first failing test snapshot are
also archived. Final product, installed distribution metadata and installed tyx
source manifests are retained. The final evidence-file manifest and release
receipt are written after this report commit to avoid self-referential hashes.

All execution was serial under the parent's sole grant. PYTHONPATH was explicitly
this child/src; PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
PYTHONHASHSEED=0 and OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS thread counts=1.
Pytest used -B, -vv, -x, -p no:cacheprovider and a distinct checked-absent absolute
basetemp within the new evidence directory. No old basetemp deletion, dependency
installation, remote action, build, profile, helper or reviewer occurred.

## Exact command groups and raw outcomes

The serial launcher is evidence-root/run.ps1. Full command arguments and absolute
basetemps are in each correspondingly named JSON, rather than a reconstructed
shell command. Frozen selection is also in frozen-grouping.txt.

| Receipt | Operation/source | RAW_EXIT | Observed result |
| --- | --- | --- | --- |
| 01-host-environment | original1108af3 host import/provenance | 0 | Child src imported |
| 02-native-environment | original1108af3 native metadata command | 1 | Command asked distribution `tyx`; installed name is `tyx-compute` |
| 03-native-environment-correct-distribution | same product, corrected metadata query | 0 | Child import and exact pinned native distribution verified |
| 04-first-post-code-ruff | original19-file authored snapshot | 1 | 17 style findings:15 safe import/unused fixes,2 assigned lambdas |
| 05-first-post-code-format-check | original authored snapshot | 1 | All19 files required formatting |
| 06-safe-lint-fixes | scoped new files only | 0 | 15 findings fixed |
| 07-format-allowlist | scoped19 files only | 0 | 19 reformatted |
| 08-lint-after-style | prepared3094c1e content | 0 | All checks passed |
| 09-format-check-after-style | prepared3094c1e content | 0 | 19 already formatted |
| 10-first-post-code-host-full |3094c1e|1|75 collected;40 passed,1 failed; fail-fast at stale message expectation |
| 11-hidden-mutation-diagnostic | diagnostic assertion correction only |0|1 collected/passed in1.66s |
| 12-host-full-after-diagnostic-fix |72bdbcb|0|75 collected/passed in602.45s |
| 13-first-actual-interpret-full |72bdbcb|1|29 collected;5 passed,1 failed in1.37s; dependent cases stopped |
| 14-diagnose-native-budget |same72bdbcb, one unchanged B first sweep observed|1|Original cumulative budget rejection reproduced; exact rows saved |
| 15-offline-saved-budget-breakdown |saved14 data, no physics run|0|Exact component breakdown and first exceeding row |
| 16-offline-nearest-fp32-proof |saved14 operands, no physics run|0|Nearest-output proof for fixed observed operands |
| 17-audit-saved-host-evidence |saved12 data, no physics imports/run|0|24 complete windows,640 accepted saved steps, rebuild hashes matched |

The initial artifact setup had a Git archive --output argument-shape error before
any executable validation; it was corrected without changing source or deleting
a prior directory. The native metadata command failure was an evidence-command
name error, not a native solver failure; installed METADATA supplied the correct
name. Both original outcomes remain saved.

The first host failure was a stale test message, not a missed rejection. Altered
final position already failed at the new `compute boundary original mismatch`
gate before the older `final` message. Only the regex expectation changed in
72bdbcb; the independent mutation, rejection and all numerical/physical gates
were unchanged. Parent inspected and confirmed that classification. This is
post-code verification, not a fabricated historical TDD RED cycle.

Host full selection used the six new host files plus these four unchanged nodes:

- tests/test_material_ledger.py::test_angular_loss_is_reported_as_material_discrete_contribution
- tests/test_initial_robot_material.py::test_fixed_angular_budget_rejects_coarse_rotating_material
- tests/test_load_replay.py::test_p_zero_p_uses_original_time_state_and_inputs
- tests/test_load_replay.py::test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial

Those four nodes expand to7 cases; the six new files supplied68 cases, total75.
Direct source inspection found the last node has no tyx_mode and defaults to
numpy. Parent explicitly ruled it belongs once in this host group, with no native
relabeling or replacement. consumer-grouping-ruling.txt retains that ruling.

The interpret full selection was both new native files plus:

- tests/test_tyx_material_path.py::test_actual_rotating_edge_keeps_original_discrete_loss
- tests/test_tyx_participants.py::test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction
- tests/test_tyx_participants.py::test_node_position_units_reconstruct_only_once_and_reject_bad_state

It used --tyx-mode interpret --require-tyx. The equivalent full CUDA group and
one bounded cpu_cpp first-sweep smoke were **not run**, because native failure
stopped dependent work. No native/composite window or later direct-consumer
acceptance is inferred from the five passing primitive tests.

## Accepted host evidence from the same run

All frozen A windows T=1/4 with h=1/64,1/128,1/256, and B windows T=1/8 with
h=1/128,1/256,1/512 passed with two fresh rebuilds. Both signed node4 perturbations,
mass factors1/2 and2, proper coordinate rotation/translation, disconnected copies,
standalone external p-zero-p/once-only terminal controls, complete trace forgeries,
continuous-domain controls, later-substep rollback/retry and NUM005 host consumers
passed under their original equations and limits.

The read-only saved-data audit indexes 24 complete window invocations and640
accepted step records in host-same-run-evidence-audit.json. Full state/history and
material/physical diagnostics hash identically across matching rebuilds. There
are also four standalone p-zero-p/terminal evidence files, covering two rebuilds.
Maximum whole-window positive energy sum: **0.0J**. Maximum whole-window angular
L1 sum: **0.0005624681672712902N*m*s**, below the frozen1e-3 ceiling. No extra
physics was sampled to populate these report values. This is declared test
coverage, not a measured line/branch coverage percentage.

## Native rejection and exact root-cause separation

The first five actual interpret primitive tests passed, including the literal
45-field first-sweep oracle and positive launch count, prohibited-host-fallback
control, late native-row rollback, conversion/cut rejection and explicit empty
block with no invented launch. The sixth test's original six-node B case at
h=1/128 rejects in its first private complete sweep before publication.

The first exceeding point in the frozen accumulated per-node/component gate is
bending row1 (the second hinge), node3, y component:

| Normalized term | Value |
| --- | ---: |
| Actual projection/storage error |2.3420806820402713e-5|
| Independent gradient uncertainty/error contribution |1.890040772992456e-10|
| Ideal constitutive equation correction contribution |4.9449411074473403e-8|
| Conversion |0|
| Lambda update/conversion |0|
| Total |2.3470445235554487e-5|
| Original tau |2e-5|

After the final row the same coordinate's bound is2.9275224522847993e-5.
Its arithmetic-only projection part is2.9225424779765093e-5, exactly
33694620709861/1152921504606846976. The remaining terms are gradient
3.245400768840405e-10 and equation4.9475203006015826e-8; conversion and lambda
remain zero. The maximum normalized row equation is only6.511180967005247e-8.
These values come from exact saved binary operands and independent interval
algebra; offline-budget-breakdown.json gives every row. No term was averaged,
cancelled, dropped or charged in different units.

nearest-fixed-row-proof.json compares each saved exact ideal_after with adjacent
FP32 representable values using rational distances. Every one of the seven
dominant node3/y outputs is already nearest. Thus their minimum possible sum for
**those fixed observed operands** is the same2.9225424779765093e-5 >2e-5. Two other
row/components are nonnearest, with avoidable normalized excesses only
4.26204627146376e-11 and8.520228966801824e-11; neither changes the dominant bound.
This rejects a simple fixed-update FMA/order cleanup as a remedy. It does **not**
prove every possible FP32 implementation impossible: a changed whole solve can
change the operands. No broad impossibility or convergence claim is made.

Parent ruled to preserve product72bdbcb, all original physical budgets, h/Vref,
precision/publication and fixtures. No compensated/local-coordinate or higher-
precision representation, revised trace, changed accumulation, tolerance increase
or scene-specific repair is approved or implemented. A representation proposal
is separate documentation-only work after this execution-slot release.

## Remaining gates and release scope

Full interpret composite acceptance, all full CUDA cases, the three actual
consumer nodes per mode and the bounded actual cpu_cpp bending smoke remain
pending. B2 remains unaccepted for native/composite delivery. Host75 is retained
at its exact source version; it is not repeated merely for reporting. NUM005,
old nonsmooth soft-distance cusp derivative limitations, joint original-input
contact work, selected-pre identity dispatch and all full P6.5 gates remain open.

The final report commit changes only the two authorized reports. A fresh process
inventory and clean Git status are captured in execution-slot-release.json before
formal handoff. The slot is released so the independent S2 geometry validation
can proceed. Subsequent lane work is documentation-only until a new explicit
execution/implementation grant. Remote CI remains user-deferred.

---

# Historical code-authoring checkpoint (retained unchanged below)

# B2-M code-authoring checkpoint

Status: **CODE_READY_TESTS_PENDING**. This is an authored implementation and test
checkpoint, not a verified physical/native capability claim. First post-code
execution has not occurred. No historical TDD RED or passing result is claimed.

Contract: parent checkpoint `17b7afa9d2f46acb12ba4e7dceaf0cf01f63405a`, including
R1-R4 in `docs/superpowers/specs/2026-10-30-p65-material-self-gripper-integration-design.md`.
Normal merge in this lane: `848f3d2`; existing B1 files were preserved.
Source checkpoint: `d82bc01`; tests/audit checkpoint:
`dd81cc6eb9ccadbb35c58f64334f87178449526f`. Conversion-binding checkpoint: `df01584582e223d8021fc12e93306e5b5a1acc35` records explicit prediction/compliance/reconstruction boundaries and alpha launch counts. The final document checkpoint includes
only reports. The conversion-binding checkpoint also includes a hand-derived precision correction to the independent decimal
pi fixture; no test outcome motivated that correction.

Worktree: `C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo`.
Branch: `codex/p65-bending-module`. Only the 11 new source modules, 8 new test
files and two authorized reports are lane changes. No old source, old tests,
exports, dependency pins, workflow, shared spec/plan or parent ledger was edited.

## Execution status

No Python/project import, collection, pytest/unittest, numerical probe, lint,
formatter, compile/build/install/profile, remote action or helper/reviewer ran.
Only source/document reads, file writes and Git were used. The parent now owns
the G2-P execution slot; this lane has used **no executable-validation slot**.

- Collected tests: **NOT COLLECTED**.
- Test/static/native command and RAW_EXIT: **NOT RUN / NOT AVAILABLE**.
- Coverage, actual mode execution, native launch receipts: **NOT MEASURED**.
- Numeric first failure: **NONE OBSERVED; NO EXECUTION**.
- Source/test hashes below are file-I/O provenance, not test evidence.
- Git clean status is reported after the final lane commit; no source swapping,
  reset, push, CI, dependency install or another lane's cleanup occurred.

## Authored behavior and independent mathematics

`ClothMaterial` owns both concrete configurations, preserves explicit empty
blocks and declares distance then bending. The host program and participant
predict once, zero both multiplier vectors per workspace/trial, compute a whole
private sweep, audit it, then publish both lambda vectors and positions together.
Velocity is reconstructed once. All-dynamic FP64 storage and explicit budgets
are required; old fixed-node and distance-only APIs remain untouched.

The native owner uses one TyxBendingMath engine, the old ordered distance kernel
and the new actual ordered FP32 bending kernel. Bending computes its own scaled
normals, atan2, analytic gradient, XPBD update, post-angle and 45-field rows.
There is no host solver fallback. Native status failures discard private arrays.
Per-block launch/completion counts, node identities, actual before/after positions,
compute conversions, lambda conversions and real parameter arrays are bound.
Raw FP32 rows and promoted outputs are checked independently. These are authored
requirements; compilation and execution on interpret/CUDA/cpu_cpp remain pending.

The ideal audit uses exact binary-rational input coordinates, E=e dot e,
D=det(e,u,v), A=(e dot u)(e dot v)-E(u dot v), and
Theta=atan2(-sqrt(E)*D,A). Rational sqrt enclosures, exact quadrant decisions,
two atan half-angle reductions, alternating-series remainders and independent
Machin pi bounds replace floating atan2 as the ideal oracle. Independent raw
normal gradient expressions bound the constitutive equations and elastic energy.
Normalized enclosure widths are limited to tau/64; residuals retain original tau.
No Fraction(float atan2) is treated as the exact ideal angle.

Domain proof uses exact affine-coordinate polynomial construction, exact
Bernstein restriction and deterministic dyadic subdivision. Every face satisfies
all six strict component-scale area inequalities. Rectangle angle bounds certify
both principal and error branches over each accepted cell. Ambiguous cut/common
root/strict-boundary cells subdivide or return typed unknown, which rejects.
Default sqrt_bits256, angle_terms64, max_domain_intervals4096 and
max_domain_depth32 are fingerprinted. The limits apply per named segment proof.
Every projection (including a distance row), compute conversion, and final
physical affine initial-to-final segment is certified. Solver paths are not
physical time and these certificates are not collision CCD or an ODE proof.

Path budgets accumulate absolute displacement errors per physical node/component
and equation/lambda errors per element across the complete sweep sequence, never
average over sweeps. They include gradient, original parameter, conversion,
actual update and reconstruction discrepancies in the existing h*Vref units.
The physical ledger retains original-input work, actual energy and unmodified
NUM005 angular defect. Ordered kinetic work, material impulse work, numerical
work discrepancy, local moments and the discrete angular mechanism are exposed;
that mechanism is never subtracted to make angular acceptance pass.

R1 standalone external receipts bind owned held/terminal requests, duration,
original inputs, selected pre-state, final state and the explicit external purpose.
The audit independently recomputes W_held and full aggregate terminal kinetic
work and checks represented loads/response. Raw energy, external transfer and
unexplained defect remain separate. Internal-boundary use of the external token
rejects. R2 physical windows carry zero additional held/terminal impulses. R3
window totals sum each step's positive energy upper bound and absolute angular
component/domain contributions; negative steps or opposite moments cannot cancel.

## Authored acceptance coverage

- Unit-hinge first sweep: literal gradient, delta_lambda=1/56, stated coordinates,
  independently written post-angle, K=3/49, initial U=1/4. This first sweep is
  explicitly nonconverged and is not mislabeled an accepted time step.
- A release windows: T=1/4, h=1/64,1/128,1/256, two fresh rebuilds.
- B asymmetric six-node mixed strip: exact frozen order, masses, rest values,
  velocities and constant original force; T=1/8, h=1/128,1/256,1/512, two rebuilds.
- Both modes retain host tau1e-8/native tau2e-5, max512 position iterations,
  per-step AND window ceilings1e-3J/1e-3N*m*s. These values were not tuned.
- Perturbations +/-2^-12, mass factors1/2,2, proper signed-coordinate rotation
  and translation, plus two disconnected copies in one physical domain.
- Separate B p-zero-p held replay at h=1/128 from the same original state,
  terminal node2 impulse once, complete state/history/trace equality, fresh retry.
- Independent one-particle external-power control retains raw gain J+J^2/2,
  W_held=J*(1+J), unexplained=-J^2/2; unaccounted or missing/stale receipt rejects.
- Forged row families, identity/policy/input bindings, missing/reordered blocks,
  conversion and launch evidence, hidden final mutation, interior degeneration,
  branch/cut, proof exhaustion, late private sweep and later-substep velocity,
  audit/diagnostics/close failures, and complete outer rollback/retry.
- Native tests require actual launches and explicitly forbid B1 host fallback.
  The new first-sweep native test is also the bounded future cpu_cpp smoke.

The source-level test names below are **authored functions, not collected nodeids**;
parametrized expansion and actual collection counts remain unmeasured.

### tests/test_bending_bounds.py

- `test_sqrt_enclosures_are_rational_proofs_not_float_roots` — UNEXECUTED.
- `test_pi_machin_and_quarter_angle_enclose_independent_decimal_brackets` — UNEXECUTED.
- `test_explicit_atan_quadrants` — UNEXECUTED.
- `test_zero_cut_and_unknown_rectangles_are_explicit` — UNEXECUTED.
- `test_unit_flat_hinge_literal_gradient_and_energy` — UNEXECUTED.
- `test_oblique_energy_gradient_has_independent_directional_derivative` — UNEXECUTED.
- `test_permutations_obey_axis_and_face_orientation_independently` — UNEXECUTED.
- `test_degeneracy_and_rest_error_cut_rejected` — UNEXECUTED.
- `test_proof_policy_validation_and_resource_width_is_visible` — UNEXECUTED.

### tests/test_bending_domain.py

- `test_bernstein_certificate_bounds_exact_polynomial` — UNEXECUTED.
- `test_translation_segment_has_complete_strict_partition` — UNEXECUTED.
- `test_nontrivial_fold_segment_is_certified` — UNEXECUTED.
- `test_valid_endpoints_do_not_certify_interior_area_collapse` — UNEXECUTED.
- `test_branch_cut_and_error_branch_reject_even_without_degeneracy` — UNEXECUTED.
- `test_fixed_budget_exhaustion_is_unknown_and_fingerprinted` — UNEXECUTED.
- `test_conversion_domain_loss_cannot_be_hidden_by_valid_original_nodes` — UNEXECUTED.
- `test_distance_row_must_certify_every_affected_hinge` — UNEXECUTED.

### tests/test_composite_cloth.py

- `test_owned_material_empty_blocks_and_positive_mass_scope` — UNEXECUTED.
- `test_first_unit_hinge_sweep_matches_literal_nonconverged_equations` — UNEXECUTED.
- `test_complete_order_owned_trace_both_residual_families_and_one_reconstruction` — UNEXECUTED.
- `test_participant_binds_material_identity_and_rejects_unrecorded_load` — UNEXECUTED.
- `test_empty_material_uses_public_runtime_and_complete_empty_observations` — UNEXECUTED.

### tests/test_composite_material_path.py

- `test_each_bending_observation_family_is_independently_checked` — UNEXECUTED.
- `test_missing_frames_and_wrong_material_or_input_binding_rejected` — UNEXECUTED.
- `test_conversion_launch_and_final_hidden_state_cannot_be_forged` — UNEXECUTED.
- `test_altered_model_parameters_and_proof_limits_do_not_rebind_existing_receipt` — UNEXECUTED.
- `test_ordered_work_includes_numerical_discrepancy_without_erasing_num005` — UNEXECUTED.
- `test_prediction_reconstruction_and_alpha_compute_boundaries_are_bound` — UNEXECUTED.

### tests/test_composite_material_physics.py

- `test_frozen_release_windows_two_fresh_rebuilds` — UNEXECUTED.
- `test_strip_perturbations_and_proper_transform_keep_original_window_budgets` — UNEXECUTED.
- `test_two_disconnected_copies_share_one_owner_without_double_application` — UNEXECUTED.
- `test_standalone_p_zero_p_terminal_once_external_work_and_fresh_retry` — UNEXECUTED.
- `test_one_particle_independent_accounted_and_unaccounted_energy_control` — UNEXECUTED.
- `test_window_positive_parts_and_component_absolute_values_forbid_cancellation` — UNEXECUTED.
- `test_physical_frozen_force_moment_and_power_match_independent_energy_derivative` — UNEXECUTED.

### tests/test_composite_material_transaction.py

- `test_late_bending_failure_discards_distance_prefix_and_both_lambdas` — UNEXECUTED.
- `test_later_substep_failure_rolls_back_full_outer_state_history_and_retry` — UNEXECUTED.
- `test_independent_domain_unknown_rejects_before_private_publication` — UNEXECUTED.
- `test_hidden_mutation_after_complete_solve_cannot_be_reported_as_acceptance` — UNEXECUTED.

### tests/test_tyx_bending.py

- `test_actual_bending_first_sweep_and_launch_receipt` — UNEXECUTED.
- `test_native_path_cannot_call_host_bending_implementation` — UNEXECUTED.
- `test_native_late_row_failure_preserves_all_caller_arrays` — UNEXECUTED.
- `test_native_conversion_induced_degeneracy_and_cut_do_not_publish` — UNEXECUTED.
- `test_empty_native_bending_is_explicit_and_does_not_invent_launches` — UNEXECUTED.

### tests/test_tyx_composite_material.py

- `test_actual_composite_complete_operation_receipts_and_storage` — UNEXECUTED.
- `test_actual_native_frozen_windows_two_rebuilds` — UNEXECUTED.
- `test_actual_native_perturbation_windows_original_budgets` — UNEXECUTED.
- `test_actual_native_external_replay_and_terminal_retry` — UNEXECUTED.
- `test_late_native_launch_failure_discards_complete_sweep_and_retry_is_fresh` — UNEXECUTED.
- `test_native_participant_rejects_unrecorded_position_load` — UNEXECUTED.
- `test_actual_native_late_substep_failure_preserves_outer_state_and_retry` — UNEXECUTED.
- `test_actual_native_two_disconnected_copies_no_double_application` — UNEXECUTED.

## Exact source/test file manifest (SHA256, authored bytes)

Hashing here is pure file I/O; no Python or source evaluation was used. These
hashes cover the checkpoint working files, including native platform line endings.
The final Git commit identifies canonical repository contents.

| File | SHA256 |
| --- | --- |
| src/phymeo/constraints/cloth_material.py | f36339baaef82cfbccd70f3d90faa99f075a8f61fcad6be74c7f55d52a408f00 |
| src/phymeo/constraints/material_trace.py | 0ea4eaf43dde1769127006d1ea2d3c82a4a6aca452d6cee23956666c6b864e5f |
| src/phymeo/solvers/composite_cloth.py | 9af00659f684323fda88c9910dc85e46d5f586166e4bf82022960d92d1268125 |
| src/phymeo/coupling/composite_cloth.py | f499d859a168b5a6629e0f3ae3263d48065cd680d15556000762cc38f29621a7 |
| src/phymeo/backends/tyx_bending.py | de27ce8143131a46059c18d8f02d5b975bf4ee278bf202f30080f92a8a2170ce |
| src/phymeo/backends/_tyx_bending_kernels.py | 8caa70e1e3a004ff73cefcb21828725649ddf857c6750a3e038763cb16188ed2 |
| src/phymeo/backends/tyx_composite_cloth.py | f8dbcce3631a6bdb4c01660abf151d680631bf284d912d7e0fc6ae5f506f3eb4 |
| src/phymeo/validation/bending_bounds.py | b93e46e1eea7312b061ab1771bb6959cf275c4089548d9eda101e723e17851df |
| src/phymeo/validation/bending_domain.py | 441a3eebd88d64d5c6fe5ebec18e95e0e677aa1e48f1e02c06a36d461e18a9f2 |
| src/phymeo/validation/composite_material_path.py | 319e461d36d24ce716f3f3b1140ed4c550c54bf4775bde2019db7b519003a6df |
| src/phymeo/validation/composite_material_energy.py | 8b499ce38667bbd891a424e84db55404bfd2953ef39183477f961660b97223c3 |
| tests/test_bending_bounds.py | 697c1b87bfb33285d52dfba55cff1f7980dab87b04d2437fd591440518f5ca51 |
| tests/test_bending_domain.py | 2a39bd01ef769f0e42210de369caac3267b06bc8de1776429a0f6fca87c35107 |
| tests/test_composite_cloth.py | 4674779b1f3ef7c35752714c449ef5f98b9bbb08a3f49e303b2a65d574db0b75 |
| tests/test_composite_material_path.py | bbb372a549b52b4cb3a1929833afd7161bb8bc3a1262224c02fae20e2c680437 |
| tests/test_composite_material_physics.py | d874fdcf7dd7e969a406cb8551406631a49f6c10bcf6b1a4ae2989ea7c4e5f64 |
| tests/test_composite_material_transaction.py | 3d12ec06cdbd25da3d4459af62ff31d5267254b9872884eee011ed8b20c9bbce |
| tests/test_tyx_bending.py | d9e0ba24f2c0e11a92bf5532f8d1d70c6c16a640156a82026aaa2f79cd53e9de |
| tests/test_tyx_composite_material.py | 95cdad84d71d4de5f485b869016f33cb976ce62dcd1ccde542fd349928254dd0 |

## Pending verification and limitations

The code has only been text-inspected. Syntax, imports, style, runtime schema
cloning, proof cost, all frozen windows and the real kernel frontend/backend
boundary are unverified. No claim is made that the fine native grids fit the
unchanged cumulative h*Vref budget; a genuine failure must be retained and sent
to the parent, not fixed with larger tolerances or altered fixtures.

The participant terminal operation validates unchanged configuration and recorded
impulses, while its material path/physical pre-audit stays bound to the selected
pre-state. The independent standalone external audit covers terminal energy.
Future parent joint-contact dispatch must audit whole-system original-input work,
reject external-purpose tokens for internal contact, and bind the same material
identity to the live owner. No legacy dispatch gate was removed here.

At an old soft-distance coincidence with positive rest length the conservative
energy derivative is nonsmooth; the optional frozen-force diagnostic explicitly
reports undefined-distance-cusp. It does not fabricate a unique derivative.
The old deterministic distance path and exact endpoint potential remain intact;
all frozen B2 fixtures use smooth distance configurations. Parent should retain
this explicit limitation when advertising force/power diagnostics.

A later sole grant must run focused ruff/format, the six host files, then actual
interpret and CUDA for the two native files and their physical cases with
--require-tyx; one bounded cpu_cpp first-sweep smoke follows serially. Freeze
source/tests/dependencies/import path, use fresh checked basetemp, retain full raw
logs/exits and the first real failure, then report actual collection/coverage and
clean/process0 release. No environment or test command has been run yet.

Direct existing consumers selected by the frozen proposal (future execution):

- tests/test_material_ledger.py::test_angular_loss_is_reported_as_material_discrete_contribution
- tests/test_tyx_material_path.py::test_actual_rotating_edge_keeps_original_discrete_loss
- tests/test_initial_robot_material.py::test_fixed_angular_budget_rejects_coarse_rotating_material
- tests/test_tyx_participants.py::test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction
- tests/test_tyx_participants.py::test_node_position_units_reconstruct_only_once_and_reject_bad_state
- tests/test_load_replay.py::test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial
- tests/test_load_replay.py::test_p_zero_p_uses_original_time_state_and_inputs

No B1/S1/G1 suite repeat is requested. B2/external-contact dispatch, complete
folded/self-contact cloth, S2-F, mesh compliance calibration, nonlinear continuum
convergence, general robot support and final P6.5 acceptance remain parent gates.
Remote CI stays user-deferred. This lane releases code only and waits for the
parent's explicit sole execution grant.


