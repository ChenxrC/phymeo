# Task 2 S2-L response implementation and serial validation report

## Serial validation and current delivery

Status: LOCAL_VALIDATION_COMPLETE; parent acceptance remains pending. The sole execution slot is released only by the final clean/process receipt saved with this report. Tested production/test snapshot: bcf4c1d343df2ee0a15be9f62f1bd1e50ca087d4. All subsequent repository changes are these two reports.

The parent grant d806f28b53505cb045c9c1a346f8e62ed81fe27f authorized serial validation. The initial fifteen authored working hashes/Git blobs were saved at 7f68bdea32cc816352a5cfbd839c1eec532d3b65, then parent 174e2ab19d17652448109dad953e91c4654d2cea was merged normally as ee6370a991bfac90ca8e12cc34c7c438ff995da2 without conflicts. All fifteen authored files stayed unchanged at that merge. The adapter source and two tests matched accepted 721b5c162f1f7e42dd618c70378bba098af0698e byte hashes and Git blobs. No later dependency merge or shared-source edit occurred.

All commands use the child src explicitly. Environment preflights assert eight imported-module paths. Executions set PYTHONDONTWRITEBYTECODE=1, disable plugin autoload, set all configured thread counts to 1 and disable pytest cache; each pytest group has a checked-absent absolute basetemp. Host used the existing original .venv-py311: Python3.11.15, NumPy2.4.6, pytest 9.1.1, Ruff0.17.2. Actual used the existing .venv: Python3.12.14, NumPy2.5.4, pytest 9.1.1, Ruff0.17.2, tyx-compute0.1.0a0 at pinned 99f6df7c0fd8fdeef77f97df83914da27b64b7ee. No installation was performed.

| Required selection | Saved run | Observed outcome |
| --- | --- | --- |
| Scoped Ruff on all thirteen new Python files | 21-current-ruff-check | RAW_EXIT0 |
| Scoped format check on the same thirteen | 22-current-format-check | RAW_EXIT0 |
| Complete five host files together | 18-host-face-index-fix | 136 passed, 2193.93 s, RAW_EXIT0 |
| Complete actual file, interpret, --require-tyx | 27-interpret-first-executed | 80 passed, 1565.79 s, RAW_EXIT0 |
| Complete actual file, cuda, --require-tyx | 32-cuda-first | 80 passed, 1749.03 s, RAW_EXIT0 |
| Exact bounded terminal node, cpu_cpp, --require-tyx | 33-cpp-bounded-first | 1 passed, 11.37 s, RAW_EXIT0 |

The full host selection is test_cloth_self_response.py, test_cloth_self_anchors.py, test_cloth_self_mapping.py, test_cloth_self_physics.py and test_cloth_self_transaction.py. Both full actual selections use test_cloth_self_response_tyx.py; C++ uses only ::test_actual_bounded_cpp_terminal_control. Every pytest command used -q -x --tb=short -p no:cacheprovider. There were no skips or fallback substitutions. C++ proves only the specified bounded one-step terminal control, not all windows or backend lifetime/TYX003 closure.

### Original outcomes and repairs

All original post-code failures, commands, source snapshots and RAW_EXIT files are retained. They are not historical TDD RED. First Ruff run 03 found 25 issues; first format run 04 found 13 files. Style commit 6ea6c760e3984c865b8b0c7407831a90137937a7 sorted 19 imports, renamed two ambiguous LDL locals and replaced four assigned lambdas by local functions. Formatting-only ASTs matched for all 13; import-normalized ASTs also matched for all 13. The local rename/lambda transformations are separately preserved and are not mislabeled exact-AST equality.

Host run 14 reached 40 passes, then found a fixture using a generic StateRegistry where the public participant-owned registry was required. Commit 8b86b10 fixes only the new fixture to use base.registry. Host run 16 again reached 40 passes, then exposed tuple-based topology indexing that selected a scalar. Commit bcf4c1d uses triangles[list(pair)] to retain both oriented face rows. The original failure receipts and small diffs remain. The complete host run 18 and every later physical selection passed on the same resulting snapshot. No physical inputs, oracle equations, h, windows, proof resources, tolerances or physical gates changed.

Native preflight 23 failed in the observer's distribution metadata lookup (tyx versus installed tyx-compute); corrected preflight 25 passed without product changes. File-only observer 19 failed at final grouping because empty first-step owner evidence combined distinct B cases. The preserved v2 was not executed. V3 binds pairs from authored collection order before output comparison; it passed host run 28. Observer 30 then found that pytest's truncated native C directory names retain the variant and restart local numbering. V4 derives exactly that naming from the authored variant names and pytest 9.1.1 _mk_tmp behavior; no physics or source changed. Original observer failures and all versions remain preserved.

Launches 26, 29 and 34 stopped at broad process guards before execution and have no fabricated RAW_EXIT. Foreign processes were identified, never killed or treated as Phymeo validation; their results are not claimed. The initial native-preflight approval-review rejection, based on a stale host-running condition, is also retained alongside the already-finished host receipt and subsequent allowed command. These incidents did not cause physical resampling. The last identified foreign process ended naturally; the fresh global zero snapshot at 2026-10-31T05:00:26.7314786+08:00 is retained in 34-process-ended-global-zero.json before audit 35.

### Same-run saved evidence

Evidence root: C:/Users/38347/AppData/Local/Temp/phymeo-p65-s2-response-7f68bde-serial. Raw logs, exact commands, source manifests, exits, before-source snapshots, failed attempts, observer versions and process receipts remain there. The final same-snapshot audit checks 328 before/after/current source/test/config hashes for every listed passing selection and observer, plus all 13 authored SHA/Git blobs and the three accepted adapter files. Its source and adapter mismatch counts are zero.

| Saved physical group | Audit run/version | Windows / steps / exact rebuild pairs | Maximum positive energy / absolute angular window sum |
| --- | --- | --- | --- |
| Host | 28 / v3 (RAW_EXIT0) | 78 / 1434 / 39 | 0 J / 3.3306690738754696E-16 N*m*s |
| Interpret | 31 / v4 (RAW_EXIT0) | 79 / 1435 / 39 | 0 J / 5.9604644775390625E-08 N*m*s |
| CUDA | 35 / v4 (RAW_EXIT0) | 79 / 1435 / 39 | 0 J / 5.9604644775390625E-08 N*m*s |
| Bounded C++ | 36 / v4 (RAW_EXIT0) | 1 / 1 / 0 | 0 J / 0 N*m*s |

The saved-data observer checks complete actual state/history, original inputs, requested/represented/actual phase transfers, selected-pre binding, path/CCD receipts, independent R2 phase evidence and native FP32/launch/upload/download receipts. It compares the entire stable serialized payload for each prescribed pair, excluding only the exact typed SceneSubstepReport.timings field. Pair membership is fixed by authored function/case and ordinal block/stride before any output comparison; each pair contains exactly two distinct resolved physical directories. Aliases are deduplicated. This avoids vacuous singleton or output-selected rebuild equality.

All saved full-window energy and angular sums stay within their original noncancelling budgets: host 1e-8 and actual 1e-5. Capacity enclosure and complete phase gates retain the original 1e-8/2e-5 tolerances. No extra numerical run was used to generate this report.

To preserve full evidence within disk limits, the parent authorized lossless compression of closed per-step JSON only. Each file retains its original path, byte length, SHA/name sidecar, compressed SHA and independently streamed decompression byte/SHA verification in a .storage.json receipt. Atomic publication of the compressed file and receipt precedes removal of the verified redundant original. Raw logs, exits, commands, manifests, source snapshots and last-attempt.json remain uncompressed. The saved audits reconstruct and verify the exact original bytes. The first 352 sidecar filenames were independently rechecked; the later filename/regex corrections and original safe-stop evidence are retained.

### Exact scope of the ideal-map capacity enclosure

For actual stored B, L=(B M^-1 B^T)^-1 B M^-1 and p=L ell, the selected-address discrepancy is E=B_star^T-B^T, enclosed by |E|<=Gamma. With nodal delivery uncertainty |delta|<=e, the ideal-map certification witness is defined in the stored-map gauge by p_star=p+L delta-L E p_star. Its remainder is r_star=(I-B^T L)(ell+delta-E p_star). G=|L|Gamma and strict gamma=||G||inf<1 give the uniform contraction bound and the propagated pressure/remainder enclosures used by _ideal_capacity_error. Unknown or exhausted enclosure rejects.

This p_star is a stored-map gauge certification witness. It is not asserted to be the ideal-map mass-weighted pseudoinverse, a unique measured physical pressure decomposition, or a state correction. Gamma encloses exact-versus-stored J only at the plan's selected material addresses. Differences at all complete clipping-extremum addresses are separately checked by _extreme_velocity, the ideal velocity bounds and _residual_speed over both maps. The Gamma contraction alone does not certify every ideal extremum or a nonlinear held-position sensitivity. The parent's hand derivation of these identities is file-only mathematical evidence, not an additional numerical test or independent acceptance of the whole response implementation.

### Remaining scope and release

The delivered claim is the frozen nonincident unique-exposed-ray, constant-normal loaded held domain with None/legacy Distance, including the full A/B/C controls and strict gates above. Full incident folds/S2-F, arbitrary rotating loaded continuation, typed B2/v2 coupling and overall P6.5 integration remain open. Blanket shared-vertex exclusion or rejecting every folded case does not satisfy full folded-cloth contact. No general-tree energy or arbitrary convergence claim is made. Remote CI is user-deferred and unpassed. Accepted LG69/S1/G2/B2/adapter groups were not repeated; no helper/reviewer/push/remote or profiling occurred.

Final report/source manifests, evidence inventory and exact final clean/process 0 release receipt are returned out of band to avoid self-referential hashes. Parent acceptance is a later decision after independent inspection of these saved bytes.

## Current tested source identities

All thirteen entries below were shared by host 136, interpret 80, CUDA 80 and bounded C++ 1. Current report hashes and final report commit are returned in final-source-report-15.json and the final delivery message, avoiding a self-referential hash.

| Path | Working-file SHA256 | Git blob |
| --- | --- | --- |
| src/phymeo/constraints/cloth_self_response.py | 4fdf824385070b1757b0b1d81070d22d5eaf00b5a3ded246dd3e698ab29bf5f2 | edfbf98aedc11c6a7094c2f31513a1e04e1130f4 |
| src/phymeo/geometry/cloth_self_anchors.py | 0cf01577f8be05bdfa0043427a414de9a759ec6f8e0ef3876df3b828784f314b | 348d932bb4f316925a52c9f991516964f8a438d5 |
| src/phymeo/validation/cloth_self_mapping.py | 2f02cda0859bf209e6312eb659e477d0ba95f6a34c29aed82b3776a5b99d6cfc | 7d620583dfeaff2d87d4832abbf0aed0790beead |
| src/phymeo/validation/cloth_self_response.py | d2b1e84780d606b86961d519019a7b4ac32f3bef39ffb4e0888521100658e32e | 6ba3abd01d9dfb2205f6eacde7fb748b58800834 |
| src/phymeo/solvers/cloth_self_contact.py | 83ec8c0d2558b33f88c28cd8b3476933097ccddfcb944e5b4785f74e106eb3f3 | 85c8d3544498be0d3fd45ac9a39d22f784506184 |
| src/phymeo/solvers/_cloth_self_contact_workspace.py | d57e38fe0fa5a4ba299956e1a0d5fec0c00e5ad87817de53e5d43d4d5bb886da | 06fd76b8d6c144f3b725d9ec262bd6b6ae234cb4 |
| src/phymeo/solvers/_cloth_self_contact_replay.py | 9e6e67edd695d036a7288af9085f5aa68bf41a93d5ea24e974bd673efc0bf9bf | dc9f33dfaaf8599d8390cea0b088ed66e2fe4d78 |
| tests/test_cloth_self_response.py | 085b7e6aebb8357bc9682c88eccf07e36e88373576de1edc987b23bd05132527 | 4a5406724e5e9a2dcb7d73cf88f81897c2c96d2d |
| tests/test_cloth_self_anchors.py | 6b60059154cab12bdc73d6c907921badfd40dd83997b08cc43054a1bf2390c20 | 7446b0e6a31067fe749743dfcb9ac8efe5e1c0c9 |
| tests/test_cloth_self_mapping.py | 5f3025ce1b8974f37aace1ad23c5fcd252032ff4b3b2e32832d01b928469a6ac | a9558ab3c9a754e1da63c236850a39588ab65c79 |
| tests/test_cloth_self_physics.py | 92f634ee597cc0e18a37b0dfcb4b9a3c5c392b69984a62129b839d840e8cd374 | 7225de466b5d0024a61b3a3c70bcfafb3d6b6dff |
| tests/test_cloth_self_transaction.py | d2c34ccccccc79f3bf2a4b974fb5a03af71da7b96490cc8fa9f93b8416a2cbaa | 8266dd20a5c99d54543016d1d71fc695dc0cd2c1 |
| tests/test_cloth_self_response_tyx.py | bbcab1159d662371c78850215c961bad9353d4fa53ac4c7f76e3a97e56cf708d | e8eb56370e9c8190dc1197c635284985e8d36f18 |

---

## Preserved authoring record: 7f68bde

Historical authoring status: CODE_READY_TESTS_PENDING; AUTHORING_SLOT_RELEASED. The following original hashes, text-discovered names and execution/dependency/release statements describe the earlier unexecuted7f68bde checkpoint. Current tested-source identities and serial outcomes are above. No historical TDD RED is inferred.

Worktree: C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo
Branch: codex/p65-self-contact-module
Starting clean lineage: a075d1826bb8ba29edb08a91460940184e0346e7
Frozen contract: parent d55182ea99b8c8a079acb4ea6ca4afa9833fa234; pre-execution R1 and R2, with R2 committed at174e2ab19d17652448109dad953e91c4654d2cea.
Code checkpoints: c93e3dd96ef87956545750cdd72fd92e5f1bef0b and b22b79e03e92a75ea7d09d3c8d63bda1d2a54e6e.

The parent-owned PointParticipantProgram dependency remains absent from this authoring lineage. Its public source was read in the parent integration tree, never copied, changed or merged. Parent reports tested adapter721b5c162f1f7e42dd618c70378bba098af0698e/report5c0a170bdc75b24e2c3c210e660f047c1dd2a032; those are parent-delivered facts, not execution by this lane. Parent must arrange the dependency snapshot before granting validation.

## Scope and decisions

Exactly fifteen NEW paths: the thirteen source/test files below plus this report and docs/development/p65-s2-response-report.md. No old S1/LG/source/test/export/report/dependency/workflow file changed. No helper/reviewer/subagent, merge/reset, push or remote CI.

Public replay route A, one physical cloth base, full shared-node response, all maximal ties, owned material extrema/counts/offset bits/generations, exact finite support and complete actual two-normal CCD. Charged held paths require a whole-path constant-normal positive-area proof and retained finite support. Terminal birth may leave rotating outgoing velocity; subsequent rotating loaded continuation is not certified.

R1 exact positive LDL and strict full-matrix kappa_inf<10^12; raw asymmetry ceiling1e-12 remains. R2 exact stored-map Gram and mass-weighted p_eff witness are separated from the requested proposal and actual native nodal receipt. Held native prediction uncertainty, terminal actual momentum, ideal-map outward bounds and residual resultant/couple/work/extreme response are all retained. Capacity uses safe lower normal and upper tangent bounds over the complete admitted interval. No state correction, pressure-uniqueness claim, zero-load capacity invention or external-work credit for internal self contact.

All rational A/B/C controls, full h/windows, two fresh rebuilds, rigid covariance and exchanged/cyclic identities are authored unchanged. Host1e-8 and actual1e-5 energy/angular gates are per step and noncancelling whole window; phase/map tolerances1e-8/2e-5, penetration0 and anchor2^-40. The lane report contains complete equations and R2 adversaries. No fixtures/gates were fitted to outcomes.

Independent enclosure and failure paths remain fail-closed. Constant-root transport follows the actual full-path proof, never endpoint-normal equality alone. Full incident S2-F/fold geometry, arbitrary rotating loaded continuation, typed B2/v2 coupling and stage acceptance remain required future work. Dense exact rank/capacity algebra is deliberately bounded by configured rows but has not been timed; no performance/native/convergence claim is made.

## Source SHA256 and Git blob identity

These are file-I/O hashes of the unexecuted authoring snapshot. Git normalizes line endings according to repository policy; both working-file SHA256 and committed blob IDs are given. No tested-source manifest or RAW_EXIT exists yet.

| Path | Working-file SHA256 | Git blob |
| --- | --- | --- |
| src/phymeo/constraints/cloth_self_response.py | df21107f8047606fa2dd87f04d6521b4cc4cd99f0c13e599bf2206a44f18a7b9 | e72b33e841e05b8cdade477cb4abaf4184eb4ad5 |
| src/phymeo/geometry/cloth_self_anchors.py | 166a644e3b3de91c02fb6d929ec0867da8d688248bed12f8fcc907821ad6719c | 1e2bf76d770f227d1959ffbacbbf7761d1e5525d |
| src/phymeo/validation/cloth_self_mapping.py | a29d15a42f53460cef3499a45e4b3c74142da13a1c7fcf238ba1b66d1a252e50 | c828b36e54b799f15f9c256e6caf511c9b725cdf |
| src/phymeo/validation/cloth_self_response.py | 8e000e7b7143f3a71315ff4efa888ebe51844cae9315a47576610eb8ffc742a2 | e8e5d246d0049d8614bda563317420c92b0e9088 |
| src/phymeo/solvers/cloth_self_contact.py | 45d2883f9a620e955f0c85915a6c2f2dbe891aa9a9c49a78279481e83dbef474 | 7c0ddc989a4bba1b657b737656e056e90e8a9914 |
| src/phymeo/solvers/_cloth_self_contact_workspace.py | 54536a068de24b86d4bf630ef69383a6760836824246230d45afd1590ead2ca2 | b96b0583841014f9971589a8b12b0bd77600a1a0 |
| src/phymeo/solvers/_cloth_self_contact_replay.py | 771cd7fdc9db82c4c34613e5c6f7bf9896f5fa57bcca003d17ecbe7868ad815a | eaaf3aae93204cb6f8905fa92ac1f563da66a52a |
| tests/test_cloth_self_response.py | 66dac35ecc9f8b0c2a9578b1e0a5fdf4370a10fe5c41dcb9277b97763b6fbba7 | 572d8dbdc318a766ef5cb4a040d0706961169d4d |
| tests/test_cloth_self_anchors.py | 083734303924f6d05ce7ed6932009fb0dd7c782062046d3906217476d1ffffbc | 725fe79349acefa4c11791897a2750351e3f584e |
| tests/test_cloth_self_mapping.py | dfd67dae5c10e98c84dca7df1d00d67f156b81e25173bc07fb8adef1aed5c373 | 3f2d27f19f7c5984a0f1244e4498a29c5d1fbc24 |
| tests/test_cloth_self_physics.py | 9f0d8f01c2215fcd43a2cf14615fc46380813b7fd90e555c1c3c3c2a470e79a9 | dde47239100f2580e4be50491e4ed069220d7280 |
| tests/test_cloth_self_transaction.py | 593c9076b663101e7728b8bedf32e593124584004349f230b47eb8185c1c4439 | d2273d05f078419276b01e023ee0fb30411c7a6f |
| tests/test_cloth_self_response_tyx.py | d7e6bde5403e34c57ee37ad1076338e310d6d9221a9069850be3747be2c9ce24 | 9c608e5fcee1ad85aa5924d9f363fb59a9aa8cae |

Lane report SHA256: b50f0bb55a79acc513af8160bfba105222740fe0ce518f2dc10dffefe86a19e0.
The final report commit and this report own SHA256 are returned out-of-band to avoid a self-referential hash.

## Authored, uncollected test function nodes

The following names come from text inspection only. Parameter expansions have not been collected, and no pass/fail/TDD RED is asserted. Later execution must use all five host files, then the complete actual file under explicit --require-tyx with interpret/CUDA, plus the named bounded C++ control under the parent serial grant. Existing accepted LG69/S1/G2 suites are not scheduled for repetition.

- tests/test_cloth_self_response.py::test_uniform_offset_tangent_coefficient_and_conditional_disk
- tests/test_cloth_self_response.py::test_shared_middle_face_has_negative_cross_response
- tests/test_cloth_self_response.py::test_proposal_never_applies_base
- tests/test_cloth_self_response.py::test_owned_capacity_is_not_duplicated_or_reused_between_phases
- tests/test_cloth_self_response.py::test_exact_full_rank_and_strict_condition_boundary
- tests/test_cloth_self_response.py::test_asymmetry_is_retained_and_rejected
- tests/test_cloth_self_response.py::test_unsupported_laws_reject
- tests/test_cloth_self_response.py::test_owned_plan_cannot_be_unfrozen_or_alias_caller
- tests/test_cloth_self_response.py::test_phase_data_rejects_shape_precision_and_nonfinite
- tests/test_cloth_self_anchors.py::test_complete_extrema_owned_dyadic_counts_and_bits
- tests/test_cloth_self_anchors.py::test_canonical_exchange_and_cyclic_node_order_preserve_owners
- tests/test_cloth_self_anchors.py::test_loaded_noncollinear_point_tie_rejects_without_first_axis
- tests/test_cloth_self_anchors.py::test_duplicate_pair_and_reversed_identity_reject
- tests/test_cloth_self_anchors.py::test_unknown_and_exhausted_clipping_do_not_allocate_capacity
- tests/test_cloth_self_anchors.py::test_fixed_normal_support_continuation_and_rotating_negative
- tests/test_cloth_self_anchors.py::test_interior_area_collapse_is_not_endpoint_proof
- tests/test_cloth_self_mapping.py::test_offset_map_independent_b_oracle_and_aggregate_work
- tests/test_cloth_self_mapping.py::test_omitted_normal_derivative_is_caught_independently
- tests/test_cloth_self_mapping.py::test_actual_terminal_double_application_rejected
- tests/test_cloth_self_mapping.py::test_separated_endpoint_couple_is_not_identically_zero
- tests/test_cloth_self_mapping.py::test_actual_mapping_receipt_forgery_rejects
- tests/test_cloth_self_mapping.py::test_r2_exact_on_range_delivery_and_off_range_residual
- tests/test_cloth_self_mapping.py::test_r2_small_nodal_uncertainty_amplifies_beyond_phase_budget
- tests/test_cloth_self_mapping.py::test_r2_changed_actual_delivery_cannot_reuse_unchanged_proposal_witness
- tests/test_cloth_self_mapping.py::test_r2_exact_held_subtraction_and_native_prediction_uncertainty
- tests/test_cloth_self_mapping.py::test_r2_open_birth_changed_held_phase_changes_only_its_owned_interval
- tests/test_cloth_self_mapping.py::test_r2_open_birth_cannot_reuse_witness_after_one_held_receipt_changes
- tests/test_cloth_self_physics.py::test_a_sustained_pressure_full_window
- tests/test_cloth_self_physics.py::test_b_forcefree_terminal_stick_slip
- tests/test_cloth_self_physics.py::test_c_shared_base_cross_response
- tests/test_cloth_self_transaction.py::test_late_failure_preserves_original_and_retry_equals_fresh
- tests/test_cloth_self_transaction.py::test_later_frame_substep_rolls_back_earlier_state_and_history
- tests/test_cloth_self_transaction.py::test_selected_pre_binding_rejects_each_forged_evidence_field
- tests/test_cloth_self_transaction.py::test_r5_normalizes_only_precise_report_timings
- tests/test_cloth_self_transaction.py::test_unknown_geometry_and_material_evidence_cannot_publish
- tests/test_cloth_self_transaction.py::test_failed_attempt_receipts_are_owned_without_physical_history_mutation
- tests/test_cloth_self_transaction.py::test_topology_and_actual_phase_receipts_are_independently_bound
- tests/test_cloth_self_response_tyx.py::test_actual_a_full_pressure_window
- tests/test_cloth_self_response_tyx.py::test_actual_b_terminal_stick_slip
- tests/test_cloth_self_response_tyx.py::test_actual_c_shared_base
- tests/test_cloth_self_response_tyx.py::test_actual_bounded_cpp_terminal_control
- tests/test_cloth_self_response_tyx.py::test_actual_terminal_failure_preserves_original_and_retries

## Same-run evidence and release

Physical tests save full actual state/history/replay/geometry/native receipts and SHA256 sidecars before asserting. Failed attempts retain immutable original state/inputs, intended plans/requests and any available actual runtime receipt; an interrupted native operation is not relabeled complete. Retry/fresh comparisons are explicit transaction controls. R5 normalizes only exact SceneSubstepReport.timings, never arbitrary timings keys.

Ready for the parent to inspect the clean authoring snapshot, supply its adapter dependency, and grant the sole serial validation slot. All executable checks, numerical success and final integration acceptance remain pending. No execution resources or processes were started by this authoring lane.
