# Task 1 implementation report

Branch `codex/p6-5-handoff`; implementation commit `700fd32` (`feat: certify affine convex feature clearance over bounded intervals`). Binding spec/brief read; no subagent/reviewer spawned. Parent owns the single final fresh review, final wheel, formal measurements, progress docs and push.

Implemented only these five product/test files:

- `src/phymeo/geometry/linear_sweep.py` (225 lines): owned FP64 input validation, immutable validated report, Fraction trajectories/weights/threshold, synchronous endpoint projection proof, declared closest failure handling, finite vertex/axis candidates, left-first bounded traversal, complete pending partitions.
- `src/phymeo/__init__.py`: two public exports.
- `tests/test_linear_sweep.py`: analytic plate/independent rod paths, required safe subdivision, exact ULP attacks, subnormal threshold, depth 52 non-dyadic contact, validation/report/exception/transform/degeneracy coverage and selective source smoke.
- `tests/test_linear_sweep_transaction.py`: thin verifier consumes actual before/after native endpoints; two substeps; accepts safe multi-domain motion; rejects real final violation/unresolved; restores state/history/inputs after final rejection or close failure; retry equals fresh (unsafe retries remain rejected). Reuses the existing test-side GeometryProgram only, without changing old tests or runtime.
- `scripts/smoke_installed.py`: crossing/safe/public-export helper, called from installed main. Only the helper was exercised from source; no source run is represented as installed-wheel proof.

No old closest/static certificate/solver/runtime/pin/tolerance changes. No implementation of TOI, curved FK CCD, contact lifecycle, reaction, or production stepping policy. tyx generates native trajectories; certificate is exact host arithmetic.

## RED/GREEN evidence

All commands used repository `.venv/Scripts/python.exe` from `C:\Users\38347\Documents\ChatGPT\phymeo`; each `--basetemp=artifacts/...` path was new and inside the repository.

1. `-m pytest tests/test_linear_sweep.py -q --basetemp=artifacts/p65-red-api-1`: **2 failed**, explicit test-body assertions `missing public sweep API`; no collection error. After minimal single-interval proof/export, same target with `p65-green-api-1`: **2 passed**.
2. Same target with `p65-red-subdivision-1`: **2 failed, 8 passed**, specifically quarter-time crossing returned unresolved instead of violation and safe turning rod returned unresolved instead of certified. After DFS, `p65-green-subdivision-1`: **10 passed**. Normal budgets stayed 128/20.
3. Boundary extension with `p65-boundary-1`: **58 passed**. These defensive tests passed existing exact code immediately; do not count them as original missing-feature RED.
4. `-m pytest tests/test_linear_sweep_transaction.py -m 'not tyx' -q --basetemp=artifacts/p65-transaction-first-1`: **4 passed, 2 deselected**. These initial integration tests passed immediately. The later reject-removal mutation below demonstrates they really detect a missing integration rejection.
5. Geometry target with `p65-red-smoke-1`: **1 failed, 58 passed**, explicit `missing sweep smoke helper`. An edit briefly placed the trailing unknown-error test block after the new smoke function; a read caught and corrected this before GREEN. No helper/collection error was counted as RED. After adding helper, both new files with `-m 'not tyx'` and `p65-green-smoke-1`: **63 passed, 2 deselected**.
6. Scoped negative controls ran through a Python subprocess driver, temporarily mutating one owned source/test file at a time under `try/finally`, then restoring exact original contents:
   - `-m pytest tests/test_linear_sweep.py::test_subnormal_clearance_is_not_squared_in_float -q --basetemp=artifacts/p65-negative-underflow-1`: **1 failed** when threshold was changed from `Fraction(clearance_m)**2` to `Fraction(clearance_m*clearance_m)`; false certified result detected.
   - `-m pytest tests/test_linear_sweep.py::test_rational_midpoint_not_rounded_geometry_witness -q --basetemp=artifacts/p65-negative-rounded-1`: **2 failed** when `_weighted` rounded rational vertices before witness reconstruction; false violation and missed strict violation detected.
   - `-m pytest tests/test_linear_sweep_transaction.py::test_actual_candidates_sweep_and_outer_rollback -q --basetemp=artifacts/p65-negative-reject-1`: **2 failed, 2 passed** after replacing verifier's status rejection with unconditional acceptance; both real violation and unresolved became incorrectly accepted and tests caught them.

## Final verification

After the depth/weight/huge-motion additions, independently checked the real semi-implicit endpoint values in the transaction tests. Final commands, sequential for actual backends:

```powershell
.venv/Scripts/python.exe -m ruff format --no-cache src/phymeo/geometry/linear_sweep.py tests/test_linear_sweep.py tests/test_linear_sweep_transaction.py
.venv/Scripts/python.exe -m ruff check --no-cache src/phymeo/geometry/linear_sweep.py src/phymeo/__init__.py tests/test_linear_sweep.py tests/test_linear_sweep_transaction.py scripts/smoke_installed.py
.venv/Scripts/python.exe -m pytest tests/test_linear_sweep.py tests/test_linear_sweep_transaction.py tests/test_surface_features.py tests/test_feature_distance_bound.py tests/test_feature_distance_bound_transaction.py tests/test_scene_stepper.py -m 'not tyx' -q --basetemp=artifacts/p65-directed-final-1
.venv/Scripts/python.exe -m pytest tests/test_linear_sweep_transaction.py -m tyx --require-tyx --tyx-mode=interpret -q --basetemp=artifacts/p65-interpret-final-1
.venv/Scripts/python.exe -m pytest tests/test_linear_sweep_transaction.py -m tyx --require-tyx --tyx-mode=cuda -q --basetemp=artifacts/p65-cuda-final-1
git diff --cached --check
```

Results: ruff **All checks passed**; directed NumPy/base regression **241 passed, 4 deselected in 0.71s**; actual interpret **2 passed, 4 deselected in 0.42s**; actual CUDA **2 passed, 4 deselected in 1.09s**. Both actual-mode tests check native `completed_launches > 0`, execution `mode`, backend report and independently expected endpoint positions. Earlier short interpret/CUDA checks also passed 2 each before adding final position assertions; final reruns were justified by those changed assertions. Earlier directed run `p65-directed-1` was 241 passed/4 deselected. Staged whitespace check clean, only normal CRLF-to-LF Git notices.

Self-reviewed proof/traversal and new tests, inspected exact staged five-file scope, and committed. Main final broad wheel/review/performance gate remains parent-owned; no full legacy CUDA/CPU suites or C++ endurance repeated, per task authorization.

Environment: Python **3.12.14** (MSC v.1944, 64 bit AMD64); NumPy **2.5.4**; pytest **9.1.1**; ruff **0.17.2**; `tyx.__version__` **0.1.0a0**, loaded from repository `.venv/Lib/site-packages/tyx/__init__.py`.

Tooling notes: initial sandboxed ruff cache write failed with Windows access denied; `--no-cache` resolved it. Sandboxed Git reported `must be run in a work tree`; authorized escalated Git worked and confirmed the intended branch. A version query using distribution name `tyx` had no metadata, so recorded the actually imported module's version/location instead. None of those environment/helper errors was counted as a numerical RED or concealed as a skip. No unresolved product/test failures observed in the scoped commands.
