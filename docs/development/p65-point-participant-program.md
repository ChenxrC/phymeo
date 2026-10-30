# Public particle participant program

Status: LOCAL_SCOPED_VALIDATION_PASSED on source721b5c162f1f7e42dd618c70378bba098af0698e. Host17, actual interpret4, CUDA4 and bounded cpu_cpp1 passed in their first complete numerical runs. Parent has audited original exits, all363 source/test/script hashes across the four modes and56 evidence files. Final report/clean/process0 release follows the documentation checkpoint. This accepts the public adapter scope only.

## Scope

PointParticipantProgram(scene,participants,*,registry=None,history_fields=()) provides the standard SceneProgram stages for a robot-free particles-only scene through public Participant methods. A terminal request batch validates all domains/ports before mutation and forwards one aggregate application per nonempty domain. It does not solve contact, calculate particle motion or replace native velocity reconstruction. Borrowed factories survive adapter/replay close; temporary children are closed on partial construction, failure and normal completion. Failed candidates are poisoned and cannot publish. Declared pass-through history is owned, signature-bound and copied exactly.

No existing source, exports, tests, dependencies or workflows changed. This adds only runtime/point_participant_program.py, two matching host/actual test files and this report. A public import from the module is the first API; no root __init__ export yet. S2 owns separate contact sources and must explicitly reject composite/v2 until typed material/contact integration is supplied.

## Independent authored controls

The two-node control uses inverse masses(1/2,1/4), initial x(0,1), vx(1/4,-1/8), force(1/2,0), h1/4. Three terminal rows aggregate to ell(3/4,1/2); the free base produces x(5/64,31/32), pre-v(5/16,-1/8), final-v(11/16,0). With one Distance edge rest1/compliance1/16, alpha1 and delta-lambda1/16 give x(3/64,63/64), pre-v(3/16,-1/16), final-v(9/16,1/16). These are handwritten expectations from the original equations, not producer output. Full kinetic identity uses aggregated ell before squaring.

Other controls cover p-zero-p original-input/history reuse, two-domain batching, all-request prevalidation, poisoned double application, partial construction, mismatched family/backend/role/history/model, stale bindings, and failures on the second outer substep in prediction/material/reconstruction/terminal/diagnostics/close with retry=fresh. Native controls require actual mode and FP32 output/launch receipts. One bounded Distance+terminal node is designated for cpu_cpp; no lifecycle-stress or TYX003 claim.

## Original serial selection

After the current B2 owner formally releases, parent will grant this adapter's static/host selection then actual interpret/CUDA and bounded cpu_cpp, with exact snapshot/current-src/environment/raw-exit evidence and fresh basetemps. No test count is claimed until collection and no first GREEN/RED is fabricated. First style/test failures and repair versions must be kept. Any need to alter an old shared file requires a narrow original-form parent ruling.

This supplies a runtime seam only. Full S2 response, B2 typed contact ledger, lawful incident folding, general robot coupling and the complete P6.5 stage remain unfinished.

## Authored file hashes
| File | SHA256 |
| --- | --- |
| src/phymeo/runtime/point_participant_program.py | a19efbbeb23dcdb928a7b1cde309d16df541727a2386b9d3cc2af2871d24d319 |
| tests/test_point_participant_program.py | 62322ab0d403525bbb7a41ac15d07091f11eef31b6f4399eac4eeb520e823ae5 |
| tests/test_tyx_point_participant_program.py | 11fb834963af9a87d3e33b09a5356f32ec53449dea124c116282dc22a7168412 |

## Authored test functions, not collected nodes

- tests/test_point_participant_program.py::test_terminal_duplicate_rows_match_independent_particle_equations
- tests/test_point_participant_program.py::test_replay_p_zero_p_retains_original_history_and_borrowed_factory
- tests/test_point_participant_program.py::test_all_request_validation_precedes_any_native_mutation_and_poison
- tests/test_point_participant_program.py::test_one_terminal_batch_per_domain_and_repeated_call_rejected
- tests/test_point_participant_program.py::test_partial_construction_closes_preceding_participant
- tests/test_point_participant_program.py::test_late_second_substep_failure_rolls_back_and_retry_is_fresh
- tests/test_point_participant_program.py::test_mismatched_factory_contract_rejects_before_workspace
- tests/test_point_participant_program.py::test_binding_mutation_and_history_schema_are_not_silently_rebound
- tests/test_tyx_point_participant_program.py::test_actual_point_adapter_bound_stages_and_terminal_receipts
- tests/test_tyx_point_participant_program.py::test_actual_point_adapter_terminal_failure_retries_from_original
- tests/test_tyx_point_participant_program.py::test_actual_point_adapter_bounded_distance_terminal

## Final local acceptance (2026-10-31)

Evidence root: artifacts/p65-point-adapter-local-20261031-a. Authoringba8568a and initial source snapshots are preserved; tested style/source721b5c162f1f7e42dd618c70378bba098af0698e changes only import spacing/format. All three complete ASTs are identical. First lint found two I001 import-format findings; initial format check reported three files. Runs03/04 exited1, scoped05/06 corrections and final08/09 checks exited0. No physical/transaction implementation, formula, oracle, tolerance or dependency changed. These are post-code checks, not a fabricated TDD cycle.

| Original run | Result | RAW_EXIT |
| --- | --- | --- |
| 10 complete host | 17 passed /0.63s | 0 |
| 11 complete actual interpret | 4 passed /0.62s | 0 |
| 12 complete actual CUDA | 4 passed /1.10s | 0 |
| 13 bounded cpu_cpp distance/terminal | 1 passed /11.32s | 0 |

The complete actual files require --require-tyx, and assert native mode, FP32 publication and completed-launch/terminal receipts; CUDA also checks upload/download counters. No skip/fallback. These small runtime seams are not full S2 physical response or a long C++ lifecycle claim. Pytest wall times are not performance benchmarks.

Environment01/02 verified current integration src imports. Host uses existing Python3.11.15/NumPy2.4.6; actual uses Python3.12.14/NumPy2.5.4 and tyx_compute0.1.0a0 pinned99f6df7c0fd8fdeef77f97df83914da27b64b7ee. One-thread, no plugin autoload/cache/bytecode, fresh checked-absent absolute basetemp per run. Every command, output, before/after manifest and RAW_EXIT is retained.

14-final-evidence-audit.json independently reads original logs/receipts and matches363 source/test/script entries across all modes and current source, with zero mismatch;56 saved files are hashed. No additional physical run was made for the audit. Physical receipts and fault/retry evidence are the same pytest assertions, not invented extra saved trajectories.

Before taking the slot, unrelated Python3.13 QA appeared after B2's formal process0 release; it was left untouched and then naturally disappeared. 00-slot-start.json records fresh global process0. A failed JavaScript tool-cell parse during style-command preparation executed no shell or project code; it did not alter first static/numerical outcomes.

Final working hashes:
- runtime/point_participant_program.py: b5ded24790e1ef9cc5c08c4d7d1b1f710171e5fe212da21969c0d4b8fa8cda39
- tests/test_point_participant_program.py: 0cbf98082e874ac39207ec66e5127e7f5b29076ae194ed0163786265672de429
- tests/test_tyx_point_participant_program.py: 3a6dba6a73674826584a7f1a6c374ab2557e1303cf8717ea7227e83b62da53c2

The adapter is available for the response lane's later clean integration and sole serial grant. B2 internal-contact material accounting, S2 response/lawful fold/general robot, final integrated review/build/wheel and complete P6.5 remain open. Remote CI remains deferred/unpassed.