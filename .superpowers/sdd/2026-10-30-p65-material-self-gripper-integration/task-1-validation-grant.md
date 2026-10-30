# Task 1 B2-M sole serial validation grant

This supersedes only the execution prohibition in the original code-authoring brief. Parent issues the actual grant to the original p65_bending_module after G2-P process0/clean release. Starting code HEAD: 1108af389dde69506dd6f76c3b0902ac008ac695; worktree C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo. Do not merge parent G2 code or rewrite the implementation. S2 waits; parent performs only read/Git/document file IO during B2 execution.

## Scope and provenance

Keep the existing eleven new source, eight test and two report files as the only editable product scope. Use original repository .venv-py311 for host and existing .venv for actual tyx; no installs or pin changes. Every executable operation is serial, including import checks, collection, lint, formatter, native compilation and the bounded C++ smoke. Explicit PYTHONPATH must point to this B2 tree's src; record actual imports, interpreter/dependencies and tyx pin99f6df7c0fd8fdeef77f97df83914da27b64b7ee. Set one-thread environment, PYTHONDONTWRITEBYTECODE=1, disable pytest plugin autoload/cache, and use a distinct checked-absent absolute basetemp for each run. Preserve expanded commands, frozen source/test manifests, raw logs and RAW_EXIT.

Retain the first post-code result; code-first authorization never implies historical TDD RED. Lint/format only the nineteen allowlisted Python files. Fix scoped defects against frozen R1-R3 equations and original budgets; preserve all failed versions. No weakening of fixtures, proof limits, windows, repeats, tolerances or energy/angular gates. Shared findings require original-form evidence and a parent ruling before old-file changes. No helpers/reviewer/push/remote CI/build/install/profile/old full suites.

## Frozen execution selection

1. Environment/import provenance, focused ruff check and format check; preserve first style outcome.
2. Full six host files: tests/test_bending_bounds.py, tests/test_bending_domain.py, tests/test_composite_cloth.py, tests/test_composite_material_path.py, tests/test_composite_material_physics.py, tests/test_composite_material_transaction.py.
3. Full tests/test_tyx_bending.py and tests/test_tyx_composite_material.py in actual interpret, then actual cuda, each with --tyx-mode <mode> --require-tyx. All frozen A/B windows, three h values, two rebuilds, perturbations, native traces and failure/rollback controls remain required.
4. Direct host consumers:
   - tests/test_material_ledger.py::test_angular_loss_is_reported_as_material_discrete_contribution
   - tests/test_initial_robot_material.py::test_fixed_angular_budget_rejects_coarse_rotating_material
   - tests/test_load_replay.py::test_p_zero_p_uses_original_time_state_and_inputs
   - tests/test_load_replay.py::test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial
5. Direct actual consumers in interpret then cuda:
   - tests/test_tyx_material_path.py::test_actual_rotating_edge_keeps_original_discrete_loss
   - tests/test_tyx_participants.py::test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction
   - tests/test_tyx_participants.py::test_node_position_units_reconstruct_only_once_and_reject_bad_state
6. One bounded actual C++ smoke: tests/test_tyx_bending.py::test_actual_bending_first_sweep_and_launch_receipt --tyx-mode cpu_cpp --require-tyx. One projection/45-field trace/real launch; no lifetime or stress claim and no TYX003 closure.

Read decorators/fixtures and freeze the exact grouping before executing consumers. If these legacy node names or mode scopes differ, report the exact read-only finding before inventing replacements. Steps 4-5 may be grouped with matching host/actual commands when valid, once per required mode. A real failure stops dependent work for diagnosis; record raw evidence and use systematic debugging, no blind retries.

## Report and release

Audit saved same-run state/history/trace/window data without extra physical sampling for reporting. Final exact source/evidence receipts, actual pass/exit/launch/FP32 results, full mathematical and scope limitations, small commits, clean tree and fresh process0 are required before formal release. Preserve NUM005 and unsupported nonsmooth distance-cusp derivatives explicitly. External transfer receipts apply only to standalone externally powered controls; they cannot excuse internal contact energy. Parent retains all shared ledger/contact/history dispatch ownership. S2 receives no slot until B2 releases.

Parent classification ruling after read-only verification: test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial has no tyx_mode fixture and calls mixed_replay whose default is numpy. Execute it once as host, not once per native flag. The unchanged seven consumers are therefore four host plus three mode-bound actual nodes per interpret/CUDA. No replacement or relaxed coverage; new actual suites and bounded cpu_cpp remain required.
