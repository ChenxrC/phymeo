# BALL-S compiler composition: source handoff

Status: CODE_READY_TESTS_PENDING / UNEXECUTED, 2026-11-13.
Base: `c550f9c4181b85a0b485fbbe01e3f246a78284df`.
Authority: [frozen implementation contract](p65-ball-scalar-constraints-implementation-20261113.md).
This note records authoring, not source, native, CUDA or installed acceptance.

## Exact author changes

- `_tree_model.py`: `compile_ball_model` first runs complete fixed BALL
  structural admission with explicit constraint permission. It then selects T4
  with the original `needs_constraints` predicate before the no-pair return.
  The BALL base admits only scalar HINGE/SLIDE limited joints and finite
  nonnegative native DOF friction. Explicit excludes remain rejected even when
  the shared scalar-only base would permit them. Primitive shape membership is
  checked explicitly before slot counting and selection of the broader solver.
- `_floating_model.py`: the single FREE-root plus BALL/R/P structural pass
  retains all existing layout, ancestry, actuator, pair, dimension and capacity
  checks. Explicit constraint permission now admits scalar limits and native DOF
  dry rows. Limited FREE/BALL, nonfinite/negative dry and excludes remain rejected.
- `_actuator_model.py`: the stateful FREE+BALL branch uses that full structural
  pass with constraint permission and calls the original T4 predicate before
  choosing normal contact or retaining the no-contact base. Fixed stateful BALL
  continues through `compile_ball_model`.

No public signature, plan field, model kind, device allocation, kernel argument,
constraint row, solver, integrator, publication or checkpoint code was changed.
The original scalar-only branches and all flag-disabled physical arithmetic are
untouched. The source author did not edit old negative tests.

## Route and semantic checks

| Source after complete structure validation | Selected base |
| --- | --- |
| No scalar/dry rows and no possible pairs | Original fixed/floating mechanics |
| No scalar/dry rows and only condim1 pairs | Original normal contact |
| Scalar limits or positive native DOF dry, with or without pairs | Original T4 |
| Any admitted condim3 pair | Original T4 |

The mixed-bound law, native joint/qpos/DOF row addresses, all DOF coupling,
physical mass versus Euler damped mass, KKT and iteration budget remain T4's.
Dry friction is not substituted into passive force. BALL angular limits are not
approximated by three scalar rows. T4 keeps its original pyramidal source-cone
requirement, including sources with joint rows and no contact geometry.

`allow_constraints` defaults to false on both mechanical base compilers. A
BALL source that contains scalar/dry rows therefore still requires explicit
composition permission. The complete structural pass raises before broad T4
selection and before positive dry-row selection. Nonfinite values cannot vanish
through a `frictionloss > 0` filter.

All fixed nv8/nq10, FREE nv35/nq45, bodies128, actors32, activation32, CSR and
geometry128/pairs16/rawslots32 limits remain unchanged. The existing dynamic
contact128/EFC512 caps and overflow behavior remain the solver's responsibility.
F/L/S and conservative E_bound diagnostics do not reserve new buffers or reject
an otherwise valid inactive-row layout; parent runtime owns those diagnostics.

## Shared seams and verification still required

Parent `_model.py` must mirror the stateful FREE route: full floating structural
compile with `allow_normal_contacts=True, allow_constraints=True`, then
`needs_constraints`, then normal/empty-pair handling. Parent runtime owns accurate
new capability labels and joint-row/capacity diagnostics. No fallback on failed
compilation is authorized. Combined activation/transmission records and checkpoint
schemas keep their accepted implementations.

The independent tests lane owns frozen five-family cases, exact old-negative
promotions, same-entry complete native EFC and actuator evidence, separately
continuous native trajectories, sparse-world failure/retry and boundary lifecycle
checks. Parent alone will run new interpret/CUDA and installed qualification.
Only source reading and Git whitespace/diff inspection were performed here; no
import, collection, numerical probe, test, lint, formatter, build or install ran.
