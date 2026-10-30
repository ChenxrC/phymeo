# Finite parallel-jaw module (G1)

Status: **G1_LOCAL_TESTS_PASSED**. The parent executed this foundation serially after S1's clean/process-zero release. This is model/input verification, not gripper load-bearing acceptance.

## Model and support boundary

The public submodule phymeo.builders.parallel_jaw_gripper creates a real 3-link coordinate-prismatic RobotModel: massive +Z carriage, +Y left finger, -Y right finger, two finite RobotContactPatch rectangles. All three efforts are finite requests. Scalar finger mass broadcasts to both fingers; a two-vector supports asymmetric masses. Configuration inputs are owned; the returned initial RobotState is mutable and independent.

Positive finger q closes the jaws. opening_m(q)=2*half_gap-q1-q2 is a signed diagnostic, not a stop constraint. gripper_inputs rejects out-of-limit effort; negative finger effort opens. Gravity/contact remain runtime computations. No control prediction is published as q/v; no cloth is pinned, welded or moved by this builder.

G2 must assemble public material/contact runtime and finite closing/lift control with actual dynamic cloth, then demonstrate independently audited high-mu hold and low-mu/insufficient-force slip. Durations, halved steps, geometry/mass variation, actual backends and rollback remain necessary. Full folded-cloth acceptance also requires B2/S2. General articulated thickness support remains separate.

## Versioned local evidence

- Worktree: C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo; branch codex/p65-parallel-modules.
- Original authored G1 code checkpoint: 88631ae16b6a3c3625ebc42b66e2d995fdf471b2.
- First verification head: 48fdd379ce3a3a27bead61cc871242825331ee2b, after B1/S1 integration. The G1 Python files were unchanged from the authored checkpoint.
- First full G1 pytest: **28 passed in 0.34s, RAW_EXIT=0**. Initial lint also passed. Initial format check exited1 and requested layout changes in the two G1 Python files.
- Applied formatting only; the complete source and test ASTs matched the original with no exceptions. Formulas, values, limits, selection and independent expectations did not change. This is post-code verification, not a historical TDD RED.
- Final tested source: **a16bf4a38223af102698a7d94ecd67d82b6378da**. Final lint/format both exit0; complete final pytest **28 passed in 0.29s, RAW_EXIT=0**. All four manifest input hashes stayed unchanged after the run.
- CPython3.11.15, NumPy2.4.6, pytest9.1.1, Ruff0.17.2; host NumPy FP64 only. phymeo and the builder import paths were asserted under this integration worktree's src.
- No native physics, profile, dependency install, old full suite, build or remote CI ran. No shared source, physics law, workflow or dependency pin changed.

Evidence: artifacts/p65-g1-local-20261030-a. Initial manifest/logs/results are 00-initial-manifest.json, 00-environment.log, 01-ruff-check.log, 02-ruff-format-check.log, 03-first-pytest.log and 04-initial-results.json. The format application and complete AST equality proof are 05-format-apply.log and 06-format-ast-audit.log. Final manifest/logs/results are 07-final-manifest.json, 08-final-environment.log, 09-final-ruff-check.log, 10-final-ruff-format-check.log, 11-final-pytest.log and 12-final-results.json. Commands and raw exits are preserved in every log. Parent release/evidence hashes are in 13-release-receipt.json after the report commit.

Both test runs used the existing C:/Users/38347/Documents/ChatGPT/phymeo/.venv-py311/Scripts/python.exe with PYTHONPATH explicitly set to this worktree's src, PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and all OMP/OpenBLAS/MKL/NumExpr/VECLIB/Rayon thread counts1. Both basetemps were checked absent before execution; neither was reused or cleared.

Final test arguments:

```text
-m pytest tests/test_parallel_jaw_gripper.py -q -p no:cacheprovider --basetemp C:\Users\38347\.codex\worktrees\p65-parallel-integration\phymeo\artifacts\p65-g1-local-20261030-a\11-final-pytest-temp
```

Both static commands named only src/phymeo/builders/parallel_jaw_gripper.py and tests/test_parallel_jaw_gripper.py: python -m ruff check --no-cache, and python -m ruff format --check --no-cache.

| Tested input | SHA256 of working bytes |
| --- | --- |
| src/phymeo/builders/parallel_jaw_gripper.py | 340243b728ac893dced31acf4dc8769e7905229dc0922eadd315b99431fbc941 |
| tests/test_parallel_jaw_gripper.py | 1c074cd5ec70752d41ffe008f0336324b1e022cd9c8efb2c03a6cba49fadd2f8 |
| pyproject.toml | 9de728c631c14dbafe50beee0e9bcf5a2c07e03e8a40c57bba073d85225a1d14 |
| tests/conftest.py | 979dc68d6d47b0016ae1f2e2765592c0914e773352fa246efd8953617af20614 |

Independent checks covered shared-carriage FK/Jacobian, finite patch dimensions, symmetric/asymmetric masses, bounded signed efforts, source state/input isolation, invalid configuration and signed opening without a hidden stop. They establish no hold/slip or composite-material result.

Contract: ../superpowers/specs/2026-10-30-p65-parallel-cloth-modules-design.md.
