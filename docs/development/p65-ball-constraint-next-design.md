# BALL-tree joint constraints: next module proposal

Status: **PROPOSAL_NOT_GRANTED**, 2026-11-13. This is future-gap planning after
R3-A, not authority to add a feature tonight. It introduces no numerical budget,
tolerance change, capacity extension or completion promise. Only this document
was authored; no product/test code, project import, collection, numerical probe,
test, static tool, build/install or profile was run.

## Source facts and recommended split

The current fixed BALL path supports contact composition, but **not joint limits
or DOF dry friction**: `_tree_model.py:107` rejects both for every BALL tree.
`_floating_model.py:71` independently rejects both for FREE+BALL. Scalar-only
fixed/FREE trees already use the mixed-constraint path. Accordingly the smallest
next generic module is **compose existing scalar soft limits and native DOF dry
friction with BALL-containing mechanics**; it is not a per-robot implementation.
BALL joints remain unlimited in that first module.

A genuine BALL angular limit is a second module with missing row mathematics and
publication identity. Removing the same guard would not implement that row.
Both modules can ultimately use the existing full-J mixed-bound QP and physical
M, rather than a new solver or a three-hinge approximation.

## Fixed native and local evidence

Primary MuJoCo **3.3.7**, Apache-2.0:
[engine_core_constraint.c](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c).
The source actually read is cached at:
`C:/Users/38347/.codex/worktrees/p65-h1-capacity/phymeo/artifacts/p65-mujoco-tyx-t4-research-20261110-01/upstream-3.3.7/engine_core_constraint.c`.
The Phymeo paths below are relative to `src/phymeo/backends/mujoco_tyx/` and refer
to the current parent source inspected during R3-A integration.

| Primitive | Fixed native source | Actual existing Phymeo implementation / gap |
|---|---|---|
| DOF frictionloss | C:641–672 creates one row per positive `dof_frictionloss`, J=e_d, native DOF identity; tendon friction is separate. | `_constraint_model.py:130` retains every native DOF/law; `_constraint_rows.py:77` sets J=e_d and signed force bounds ±loss. This is coordinate friction, not viscous damping or Coulomb contact friction. |
| HINGE/SLIDE limit | C:718–743 processes lower and upper separately, distance to each bound and J=±e_d. | `_constraint_model.py:142` records native joint/qpos/DOF addresses; `_constraint_rows.py:95` and `_constraint_publication.py:107` retain both sides and margin activation. BALL preceding scalar joints already makes qpos and DOF addresses different; preserve the explicit maps. |
| BALL angle limit | C:747–788 normalizes the joint quaternion, calls `mju_quat2Vel`, normalizes the angle-axis vector, uses distance=max(range)-angle, and one nonnegative-force row with J=-axis on three local angular DOFs. C:1102–1103 uses the first joint DOF's `dof_invweight0` for diagApprox. | Static compiler explicitly rejects non-scalar limited joints. Current limit rows and HOST expected-row reconstruction assume two scalar sides and one qpos value. All three require a new kind/width/identity branch; the solver already accepts a complete J vector. |

Do not infer the BALL zero-angle, quaternion sign, or near-pi behavior from a
textbook log map. A later BALL-angle contract must port the exact fixed helper
semantics and test those boundaries. A quaternion component is not an angle,
and three independent hinge stops are not the native BALL limit.

## First module: exact composition seams

Keep existing public MJCF joint `limited/range/margin/solref/solimp/frictionloss`
fields, `MuJoCoRuntime`, controls, stateful JOINT receipts and checkpoint API.
No new public state owner or actuator is necessary. Constrained response remains
a soft native row; it must not become a position clamp or stop repair.

Proposed product ownership for the first module, to be frozen separately:

- Robot author: `_tree_model.py` (`compile_ball_model` and structural gates),
  `_floating_model.py`, `_actuator_model.py`. Permit explicit scalar-limit/DOF-dry
  composition while still rejecting limited BALL joints and all unrelated domains.
- Parent shared: `_model.py` dispatch and `../_mujoco_tyx_runtime.py` capability/
  scalar-address preflight. Source constraints select T4 even with no contact or
  with only condim1 contact; they cannot remain on the normal/direct route.
- Initially reuse unchanged `_constraint_model.py`, `_constraint_rows.py`,
  `_constraint_device.py`, `_constraint_solver.py`, `_constraint_publication.py`,
  all mechanical/activation kernels, and existing typed checkpoint records.
  A demonstrated need to change these files requires a narrower explicit grant.

The routing decision must not use the broad generic `needs_constraints` predicate
without the complete BALL structural pass. In particular it must not accidentally
admit explicit excludes, condim4/6, larger pair/slot layouts, BALL springs, new
shapes, a FREE actuator or multiple free roots. Preserve fixed nv8/nq10 and FREE
nv35/nq45, body/actor/activation/CSR limits and the existing implicit-contact domain.
Original NONE and FILTEREXACT must reach the same full M/J constraint solve; dry
friction is not subtracted as another passive force before that solve.

For the later genuine BALL-angle module, the exact additional source seams are
`_constraint_model.py` (joint kind and complete native addresses),
`_constraint_rows.py` (three-column angular row), and
`_constraint_publication.py` (matching row identity and active-row reconstruction),
plus the same structural dispatch/capability owners. A new scalarized solver or
changes to quaternion time integration are not proposed.

## Capacity accounting and transaction contract

For each admitted source let F be the number of positive native DOF friction
entries, L the number of limited scalar joints, and S the existing raw geometry
slots. A complete conservative static EFC count is
`E_bound = F + 2*L + sum_over_raw_slots(e(condim))`, where e(1)=1 and e(3)=4.
The factor two is required: overlapping margins can activate both scalar sides.
Rows remain in native dry order, native joint/side order, then full contact order.
The current logical count is the actual dry rows plus active scalar sides plus
active contact edges, not a selection of the largest forces.

Retain the original BALL geometry bounds (128 geoms/16 pairs/32 raw slots), the
runtime's original contact/EFC capacities and its absolute 128/512 limits. Record
F/L/S, E_bound, requested capacities and actual required counts. Do not silently
raise defaults, truncate rows or add source iterations. Whether a future contract
requires statically reserving E_bound is a separate explicit decision; existing
runtime overflow must still reject the entire candidate transaction. The same
source iteration/KKT/mass-equation gates apply to the enlarged mixed row set.
Dry interior free variables must use the already implemented feasible-face
initialization/refinement, not a new warmstart or a per-row approximate solve.

Publish q/v/act, generalized and per-row forces, time/tick and complete receipt
once after every world succeeds. Keep both same-entry physical M and damped Euler
integration evidence. No limit-related qpos repair, partial-world publication or
new state bank. Candidate failure retains all accepted states/receipts/clocks;
clear an injected recoverable failure and verify a real successor against a
previously untouched clone. Unknown driver-release errors remain unknown/sticky.

## Five predeclared qualification families for a future grant

| Family | Coverage, with fixed independent inputs before execution |
|---|---|
| 1. Fixed BALL+R/P coordinate limits | Nonidentity BALL quaternion, nonzero scalar velocities/controls, independent lower/upper entry and release, exact margin boundary and overlapping-margin two-row case; no contacts. |
| 2. FREE+BALL+R/P coordinate limits | Nonzero free-root world translation/local rotation, asymmetric inertia and external forces; same native qpos/DOF-address checks and genuine coupled root response. |
| 3. Native DOF dry friction | BALL three local angular coordinates and scalar coordinates, positive/negative slip, interior sticking force, zero friction absence; verify signed bounds and full coupling rather than calling all rows damping. |
| 4. Mixed contact/drive/coordinate rows | Primitive normal and pyramidal3, scalar limit plus dry rows together, mixed NONE/FILTEREXACT and shared/oblique gears; reverse/unload, complete contact COM wrenches and actuator/constraint power. |
| 5. Ownership, capacities and lifecycle | Sparse W2 isolation; actual last-world cap/budget/publication failure; checkpoint/clone/partial reset and real successors; raw accepted clocks/act/control/receipt preserved. |

Each positive family requires same-entry independent MuJoCo M/J/bias, complete
EFC type/source/side/J/R/aref/bounds/force and full qfrc, plus a separately continuous
native trajectory. R3-A act/state gates and the original constrained mechanical
and contact gates must be cited unchanged in the eventual test contract. The
native oracle must not overwrite its continuous state from the device result.

Negative sources retain limited BALL (first module), FREE target, unsupported
shapes/cone/dimensions, nonfinite/negative friction, invalid ranges or law fields,
source/row capacity overflow, malformed identities and incompatible checkpoint
schemas. Exact old limit/dry negatives promoted to support get exact-source
positive successors; unrelated negatives remain. Both backends and a fresh
installed public consumer are needed; source-only results do not confer package
or long-application qualification. No test/node count or new numeric gate is
frozen by this proposal.

## Dependencies and priority against D5

First close R3-A source, compatibility and installed qualification with its final
schema/route bytes. Then one bounded scalar-limit/DOF-dry composition is a lower
risk, reusable increment: existing row assembly/solve/transactions already express
it. Budget/conditioning and correct no-contact-to-T4 routing are the main risks.
Genuine BALL-angle rows should follow only after their native boundary semantics
and three-column publication are specified; do not grow both modules at once.

D5 multibody rigid–deformable support is the larger missing public capability and
has higher overall value than repeatedly subdividing BALL angular-stop cases.
It can proceed independently alongside the bounded coordinate-constraint module.
At this proposal's original source checkpoint, `builders/deformable_scene_lowering.py:129–150` admitted exactly one ordinary
one-body floating participant per node world and root-owned round geometry. Real
multibody support needs complete body ownership, actual COM frames and J-transpose
wrench projection, all original mechanical blocks advanced once, and common
node/rigid publication and failure recovery. Deleting `body_count==1` is not that
implementation. Reuse existing resident scene/transaction machinery where its
interfaces fit, without merging MuJoCo-TYX and resident-D3 state owners by guess.

D4/D6 steady support failures stay visible and do not block this structural D5
capability work; conversely multibody assembly cannot be called a remedy for those
failures. Stop a proposed module if it requires unplanned solver/ABI changes or
repeated unchanged failures; report the concrete gap instead of expanding budgets
or tuning a robot. This document authorizes no implementation or execution.

## Closing source review: concrete composition seams, still not granted

The final text-only review by the original source and test authors found no
source requirement for a new row kernel, solver or device ABI in the first
scalar-limit / native-DOF-dry composition. This is a feasibility finding, not a
runtime result. M-B1 has meanwhile completed its independent-single-body scope;
see the [accepted report](p65-multibody-deformable-report-20261113.md). Non-root
articulated rigid-deformable coupling remains the separate M-B2 gap.

Three compiler seams need a future bounded change:

1. `_tree_model.py` and `_floating_model.py` must retain rejection of limited
   BALL joints while admitting supported scalar limits and DOF dry rows via the
   existing constrained structural checks. Keep explicit excludes rejected for
   BALL trees. Explicitly require finite nonnegative frictionloss: testing only
   `frictionloss > 0` would silently miss NaN after the old blanket guard changes.
2. `compile_ball_model` must finish all BALL structural, shape and capacity
   checks before `needs_constraints`; that dispatch must precede the no-contact
   early return. Joint-only and condim1-with-joint-row models still need T4.
3. FREE activation compilation in `_actuator_model` and the stateless `_model`
   route need the same complete dispatch. Parent runtime capability labels must
   reflect actual limit/dry/contact rows, not unconditionally report BALL limits
   and dry unsupported or label a joint-only wrapper as contact. Preserve the
   existing T4 pyramidal-cone rule even when no contacts are present.

`needs_constraints` recognizes a broader domain than this proposal. It must not
be used to bypass BALL exclusions, capacities, shape checks or the condim4/6
guard. A discovered need to change row mathematics, solver or ABI stops this
narrow composition and requires a separately specified module.

The test work is more than cloning an R3-A XML. Its current checker assumes
contact type6 rows, contact-start zero and the contact working set; it cannot
validate mixed dry type1, scalar-limit type3 and signed bounds. Reuse the
independent full-EFC oracle in `tests/test_mujoco_tyx_constraints.py` (native
row decoding and `assert_receipt`) together with R3-A activation observation and
its independently advanced continuous CPU reference. The former alone omits
activation state; the latter alone misses mixed-row semantics.

The proposed four physical fixtures are fixed BALL with scalar lower/upper
limits and no contact; FREE+BALL scalar limits with FILTEREXACT; signed dry
slip/sticking/zero-absence; and condim3 plus activation, limits and dry rows.
Retain the original five qualification families above, including overlapping
margin rows, sparse-world ownership, late failure and true recovery. Exact XML,
all inputs, unchanged applicable numerical gates and source/installed selection
must be frozen before execution. No new fixture, node count or successful case
is claimed here.

This work was not started before the 06:15 feature freeze: completing and
reviewing the mixed checker plus both backends and installation would risk an
unfinished domain. The result is a concrete next implementation boundary, not
an extra grant or a reason to rerun accepted R3-A tests.
