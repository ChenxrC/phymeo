# G2-P local finite-grasp acceptance

Status: LOCAL_SCOPED_VALIDATION_COMPLETE. Tested source e9ab075fd078c32d9695899d2a1cfff33d6b8ccd; initial authored checkpoint e584b5605ea4719e9d0ea7d590f96fabec50fc74. Parent owns this lane. No old solver/runtime/law/participant/pin/history source was changed.

## Actual results

| Selection | Result | Pytest wall time | Saved accepted runs / steps |
| --- | --- | --- | --- |
| Host: gripper cloth and transaction files | 29 passed, RAW_EXIT0 | 1138.14s | 24 / 164 |
| Actual interpret: complete native file, --require-tyx | 14 passed, RAW_EXIT0 | 728.75s | 20 / 84 |
| Actual CUDA: complete native file, --require-tyx | 14 passed, RAW_EXIT0 | 862.92s | 20 / 84 |

Sessions9415,71841,86307 ended naturally, each finish receipt records process0. Their logs, exits, expanded commands, environment and starting manifests remain in artifacts/p65-g2-local-20261030-a as09-host01,12-interpret01,15-cuda01. No suite was restarted. These durations are validation timings, not performance or cross-backend speed measurements.

Host used the existing Python3.11.15/NumPy2.4.6 environment. Actual modes used Python3.12.14/NumPy2.5.4 and tyx_compute0.1.0a0 pinned to99f6df7c0fd8fdeef77f97df83914da27b64b7ee. Environment receipts verified actual current-worktree imports. Each command used a fresh checked-absent basetemp, one-thread settings, disabled plugin autoload/cache and actual native requirements.

The first numerical results were green. Initial ruff/format style failures are retained; unused import and an equivalent local-helper style correction preceded formatting. Format AST parity and final static checks passed. The pre-execution PowerShell automatic-variable authoring error was corrected before project import/collection and is not a numerical RED. No formula, oracle, tolerance, budget or window was relaxed.

## Physical coverage and independent evidence

The additive runner configures public model/runtime APIs and stores complete actual results, typed lossless state/history bytes, diagnostics, raw timing and provenance. It contains no expected-solution oracle or state repair. Independent Fraction equations live in tests.

A covers true finite-force high-friction support and low-friction sliding with shared carriage mass; B starts open, closes, captures through both normal phases, applies terminal friction once and hoists; C executes active DistanceConstraints. Host A/B windows retain T=1, h=1/8 and1/16 and two fresh rebuilds. Actual modes retain T=1/2, both h/rebuilds, real FP32 publication and launches. C retains h=1/4 and1/8 controls. Mass/effort scaling, changed finite patch, vertical translation, declaration order, force/work/momentum/energy, complete CCD/history, immutable inputs and serialization remain covered.

Late prediction/path/terminal/mapping/CCD/close and construction faults test complete state/history/input rollback and retry equal to a fresh frame. Capture uses a real nonzero terminal load. Additional fault/retry simulations are represented by original pytest evidence, not added to the saved-trajectory counts.

Parent read-only audit16-final-saved-evidence-audit.json checked all327 source/test/script files unchanged across all three manifests and94 evidence hashes. Canonical ordered source-manifest SHA256:99c9f95e1e07ceb74881db989ab5b4dc6d0d13c600a6c2c4f575daad703b07e7. This manifest encoding is distinct from historical rolling digests.

All64 same-run records contain332 accepted steps. Two independent rebuilds agree bitwise on full state and substep/history: eight groups per mode, with74 host and38/38 actual rows compared. Each actual mode has168 FP32 pre/final state receipts and69936 completed launches across saved trajectories. Every accepted step has exactly one terminal application and two complete certified CCD partitions. Original host/native tolerances1e-8/2e-5 and energy/angular budgets1e-8/1e-5 are unchanged.

Maximum positive energy upper bound is6.938893903907228e-18 J on host and1.7166712488833187e-11 J in both actual modes. Saved symmetric cases have zero material angular L1 defect; this does not close NUM005. Audit16-audit-saved-records.py/log/exit completed0 using only saved files, no project imports or additional physics.

## Scope and remaining delivery

This validates planar coordinate-prismatic finite grasp with distance material. It does not validate B2 composite bending, self-contacting/folded cloth, arbitrary robot joints, final mesh calibration or full P6.5. B2-M and S2-LG code are ready in isolated trees and await their serial validation. Shared contact/material dispatch, S2 response and lawful incident folding remain parent-owned work.

One new integrated review and local build/source-external wheel belong to the completed integrated Task4. No new C++ claim is made here; original thickness C++ evidence remains versioned and TYX003 stays open. Remote CI remains user-deferred and unpassed. No remote operation or extra physical run was used for this report.
