# BALL-S: scalar limits and native DOF dry friction composition

Status: AUTHORING_GRANTED / EXECUTION_PENDING, 2026-11-13. User instruction:
continue the next functional batch. Parent baseline c6a76052; no new deadline.
This contract activates only the first module in
[p65-ball-constraint-next-design.md](p65-ball-constraint-next-design.md).
M-B2 and all earlier accepted runs remain closed and are reused without reruns.

## Deliverable and boundaries

Compose existing scalar HINGE/SLIDE soft limits and native per-DOF frictionloss
with fixed BALL/R/P and FREE+BALL/R/P mechanics and NONE/FILTEREXACT JOINT drives.
Admit no-contact, condim1 and pyramidal condim3 sources through the complete
existing full-J mixed-bound QP. Preserve every force row and its native identity;
no clamp, passive-force substitute, synthetic hinge, extra iteration, state repair
or threshold change. BALL angular limits remain unsupported.

Keep all original fixed nv8/nq10, FREE nv35/nq45, bodies128, actors32,
activation32, CSR, geometry128/pairs16/rawslots32, contact128/EFC512 hard limits.
Keep explicit excludes, multiple FREE roots, FREE targets, unsupported shape/
cone/dimension, BALL springs and other original negatives rejected. Require
finite nonnegative dof_frictionloss before selecting positive rows.
Run complete BALL structure/geometry gates before the broad constraint predicate.
Constraint dispatch precedes the no-contact early return. Joint-only and condim1
with scalar/dry rows must reach T4. Preserve its pyramidal source-cone rule.

Record F positive dry rows, L scalar limited joints, raw S, conservative
E_bound=F+2L+sum_raw_slots(e(condim)), e(1)=1/e(3)=4, requested capacities and
actual required rows. This batch retains actual-overflow rejection; E_bound is
diagnostic and does not silently reserve/raise capacity or reject a source whose
inactive rows fit the original runtime contract.

## Isolated author ownership

Original source author /root/mtyx_t2_audit reuses p65-h1-capacity on a new
codex/p65-balls-source-20261113 branch from this parent contract commit.
Exact product allowlist: src/phymeo/backends/mujoco_tyx/_tree_model.py,
_floating_model.py, _actuator_model.py. Optional NEW report:
docs/development/p65-ball-scalar-constraints-source-notes-20261113.md.
Preserve unrelated untracked review. No shared runtime/dispatch edits.

Original tests author /root/mtyx_t2_tests reuses p65-h1-actuator on a new
codex/p65-balls-tests-20261113 branch from this parent contract commit.
First author NEW docs/development/p65-ball-scalar-constraints-cases-20261113.md,
then NEW tests/test_mujoco_tyx_ball_constraints.py,
tests/test_mujoco_tyx_ball_constraints_transactions.py,
tests/installed/check_ball_constraints.py. If an independent helper is needed,
NEW tests/ball_constraint_cases.py is allowed. Existing tests may be changed only
to promote exact scalar-limit/dry rejection cases into positive route successors;
enumerate those exact files/nodes in the cases document before editing. Preserve
unrelated negatives and all original numerical gates. No product modifications.

Parent /root owns _model.py and ../_mujoco_tyx_runtime.py, contract/reports,
integration, package consumer preparation and the unique project execution slot.
No new agents. No imports, collection, numerical probes, pytest, lint, format,
compile, build, install or profile in either author worktree. Pure text/Git only.
Each author returns small commits, exact file manifest, clean tracked state and
AUTHORING_SLOT_RELEASED. No reset, overwrite, merge-parent, push or cleanup.

## Multi-case verification contract

Freeze exact XML, masses/inertias, q/v/act/control/external inputs, timesteps,
step counts, independent expectations, original gates and test selection in the
cases document BEFORE any project execution. No output-guided fixture tuning.
Implement all five families from the proposal: fixed scalar lower/upper entry/
release plus exact and overlapping margins; FREE+BALL with FILTEREXACT and native
address mismatch; signed BALL/scalar dry slip/interior stick/zero absence;
condim1/3 mixed contact+limit+dry+drives with reversal/unload; W2 sparse ownership,
last-world capacity/iteration/publication failure and actual recovery successors.
Verify full same-entry M/J/bias, complete EFC type/source/side/J/R/aref/bounds/
force and qfrc using independent MuJoCo 3.3.7 decoding; combine original T4 oracle
with R3-A activation checks. Separately advance the continuous CPU trajectory,
never overwrite it from device outputs. Keep original source solver/KKT/mass/
actuation and continuous gates, explicitly cited and copied unchanged.
Negative sources retain limited BALL, FREE actor, shape/cone/dim exclusions,
nonfinite/negative friction, bad range/law/capacity/identity/checkpoint.
No automatic Cartesian expansion, duplicate old green suites or claim of G1/
long support. Both interpret and CUDA plus fresh installed public consumers are
required for this bounded capability. Save all first failures and complete
returned evidence, source snapshots and raw exit receipts.

After both author handoffs and parent text review, parent issues an exact
execution manifest in the new canonical batch evidence directory, runs fresh
entrypoint-aware guards and serial focused validation. Diagnose concrete failures;
only changed/affected tests may rerun before one final full selected set.
No new numerical gate, solver/ABI/row-math change is authorized here: a concrete
need stops this module for a narrower adjudication. Do not claim successful
qualification from authoring, source inspection or a partial test result.

## Acceptance

Compile guards/routes and capability labels must match actual admitted rows;
joint-only sources are not called contact-supported by inference. Preserve
same-entry physical M and damped Euler evidence, single all-world q/v/act/clock/
receipt publication, checkpoint/clone/partial reset and UNKNOWN_RELEASE behavior.
Package source/wheel/installed parity and independent installed consumers follow
the final source result. Update continuation, acceptance matrix and the functional
overview only with observed bounded results; entire P6.5 remains incomplete.
G2b lifecycle allocation budgets stay the next independent capability, unstarted.

