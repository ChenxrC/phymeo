# Public particle participant program

Status: CODE_READY_TESTS_PENDING. Parent code/test authoring under the frozen S2 response design; every import, lint/format, collection, host/native test and compile is UNEXECUTED. B2 R7 exclusively owns execution. This checkpoint does not add a physics pass or native support claim.

## Scope

PointParticipantProgram(scene,participants,*,registry=None,history_fields=()) provides the standard SceneProgram stages for a robot-free particles-only scene through public Participant methods. A terminal request batch validates all domains/ports before mutation and forwards one aggregate application per nonempty domain. It does not solve contact, calculate particle motion or replace native velocity reconstruction. Borrowed factories survive adapter/replay close; temporary children are closed on partial construction, failure and normal completion. Failed candidates are poisoned and cannot publish. Declared pass-through history is owned, signature-bound and copied exactly.

No existing source, exports, tests, dependencies or workflows changed. This adds only runtime/point_participant_program.py, two matching host/actual test files and this report. A public import from the module is the first API; no root __init__ export yet. S2 owns separate contact sources and must explicitly reject composite/v2 until typed material/contact integration is supplied.

## Independent authored controls

The two-node control uses inverse masses(1/2,1/4), initial x(0,1), vx(1/4,-1/8), force(1/2,0), h1/4. Three terminal rows aggregate to ell(3/4,1/2); the free base produces x(5/64,31/32), pre-v(5/16,-1/8), final-v(11/16,0). With one Distance edge rest1/compliance1/16, alpha1 and delta-lambda1/16 give x(3/64,63/64), pre-v(3/16,-1/16), final-v(9/16,1/16). These are handwritten expectations from the original equations, not producer output. Full kinetic identity uses aggregated ell before squaring.

Other controls cover p-zero-p original-input/history reuse, two-domain batching, all-request prevalidation, poisoned double application, partial construction, mismatched family/backend/role/history/model, stale bindings, and failures on the second outer substep in prediction/material/reconstruction/terminal/diagnostics/close with retry=fresh. Native controls require actual mode and FP32 output/launch receipts. One bounded Distance+terminal node is designated for cpu_cpp; no lifecycle-stress or TYX003 claim.

## Later serial selection

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
