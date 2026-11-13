# BALL-S fixed cases and unchanged gates

Status **PREDECLARED / UNEXECUTED**. Parent contract: [BALL-S implementation](p65-ball-scalar-constraints-implementation-20261113.md). Base `c550f9c4`; pure authoring, no import/collection/probe/test/static execution. This document freezes inputs before first execution. Native oracle is MuJoCo **3.3.7**; source iteration budget128 and tolerance1e-10 are native input, not a claim that the TYX solver is MuJoCo's iterative algorithm.

## 1. One exact new physical template

All new sources use compiler `angle="radian" autolimits="true" fusestatic="false"`; option timestep`.001`, gravity`0 0 0`, integrator`Euler`, solver`Newton`, iterations`128`, tolerance`1e-10`, jacobian`dense`, cone`pyramidal`. Joint defaults armature`.003`, damping`.04`, solreflimit`.02 1`, solimplimit`.8 .9 .002 .5 2`, solreffriction`.02 1`, solimpfriction`.8 .9 .002 .5 2`. Geom defaults condim1 (or3 in the named pyramid case), friction`.4 .01 .001`, solref`.02 1`, solimp`.8 .9 .002 .5 2`.

A fixed marker body at `2 -1 .4` has sphere`.02` and masks0/0, preserving native body addresses. Main BALL body is at `0 0 .4`; its joint is named `ball`, unlimited BALL, pivot`0 0 0`. Its mass`.7`, COM`.03 -.02 .04`, principal inertia`.035 .045 .055`, inertia WXYZ`.9659258262890683 .2588190451025207 0 0`. Child HINGE named`hinge` has axis`0 1 0`, body offset`.12 0 -.1`, mass`.7`, COM`.03 -.02 .04`, same inertia/frame. Its child SLIDE named`slide` has axis`0 0 1`, offset`0 0 -.201`, mass`.5`, COM`-.01 0 .01`, inertia`.012 .014 .016`. Body/joint frames otherwise identity. Each of the first two bodies has a masked sphere`.035`; tip sphere radius`.1` is masked except in the three mixed cases. Those mixed cases add world plane`size="2 2 .1"`; there is exactly one unfiltered plane–sphere pair.

FREE variants wrap this same BALL/R/P subtree in a massive root at`0 0 .4` and change the BALL body's offset to`0 0 0`. Root joint is `free`; mass2.6, COM`.04 -.03 .02`, principal inertia`.13 .19 .23`, inertia frame`.9659258262890683 0 .2588190451025207 0`, masked sphere`.06`. No new geometry or mechanics formula is introduced.

All sources have four JOINT general actuators: `ball-drive` gear`1 -2 .5` gain`.8`; shared `ball-counter` gear`-.25 .5 -.125` gain`.6`; `hinge-drive` gear`-.5` gain1; `slide-drive` gear`-2` gain1 and affine bias`.01 -.3 -.04`. NONE is used unless the identity says FILTEREXACT: then `ball-drive` has tau`.004`, actearlyfalse, and `slide-drive` tau`.007`, actearlytrue; both ctrl limits`[-.45,.45]` and activation limits`[-.3,.3]`. Counter and hinge remain NONE, so each FILTEREXACT source also checks mixed stateful/stateless actuation. No BALL affine bias.

## 2. Twelve fixed identities, W2 and actual phases

Logical keys `(3,29)`, input mappings inserted in reverse order. Common BALL initial WXYZ is normalizedFP64 `(1,0,0,.125)`; FREE quaternion normalizedFP64 `(1,0,0,-.125)`. Normalization occurs once at source construction; actual TYX entry bytes supply the same-entry native copy, while the continuous native reference keeps its own state. All ordinary scalar coordinates are zero except the variants below. Common BALL rates`(.15,-.2,.1)`, scalar rates`(.125,-.1)`; FREE origin-world linear/local-angular rates`(.07,-.05,.02,.09,-.06,.04)`. World29 multiplies all velocities by`-.75` and offsets root x/y by`(.03125,-.015625)` for FREE variants only; it does not change scalar boundary coordinates. Initial activations are alternating`(.12,-.08)` in native activation-address order. All times start0.

| Identity | Fixed source differences / initial scalar coordinates | Fixed steps / required observations |
| --- | --- | --- |
| `fixed-lower` | H/P limited ranges`[-.25,.25]`, margin`1/32`; H/P q`(-15/64,+15/64)`, v`(+1,-1)` in both worlds |24; initial lower H/upper P rows; each side must release on the same trajectory, exact native event comparison |
| `fixed-upper` | Same limits; q`(+15/64,-15/64)`, v`(-1,+1)` in both worlds |24; upper H/lower P and release |
| `fixed-margin` | Same limits; q`(-7/32,+7/32)`, inward v`(+1/8,-1/8)` in both worlds |2; exact binary equality to margin initially produces no limit row, real interior successor |
| `fixed-overlap` | H/P ranges`[-1/256,+1/256]`, margin`1/128`; q0, v`(+1/32,-1/32)` |2; two sides for each scalar simultaneously, complete four rows, not a clamp |
| `free-filter-limits` | FREE + FILTEREXACT; ordinary ranges/margin as lower case, q`(-15/64,+15/64)`, inward scalar v`(+1,-1)` |24; full root response, nonzero activation and qpos/DOF-address mismatch |
| `dry-positive` | Fixed; no limits; BALL frictionloss`.2` (three native DOFs), H/P`.2`; common q/v |24; complete signed dry rows, nonpositive aggregate dry work, bound slips |
| `dry-negative` | Same as positive, negate all common velocities before the world29 factor |24; opposite signed slip |
| `dry-stick` | Same dry source; q scalar0, all velocity0; all controls`.01`, qforce all`.005`, body wrenches0, act empty |2; nonzero interior signed force strictly within ±.2; full M solve, not hard zero velocity |
| `dry-zero` | Exact dry-positive source except frictionloss0 everywhere; same q/v/control |2; no dry rows, no hidden passive dry force; existing unconstrained route preserved |
| `mixed-normal-none` | Fixed NONE, condim1; H range`[-1/64,.25]`/margin`1/32`; P`[-.25,.25]`/margin`1/32`; q scalar0, dry`.2` on BALL/H/P |24; initial contact+H-limit+dry, command reversal and real contact unload |
| `mixed-normal-filter` | Same mixed source with FREE + FILTEREXACT |24; same full mixed check plus true root/activation response |
| `mixed-pyramid-filter` | Same as previous but condim3 |24; complete four-edge contact with dry/limit rows preceding contact, signed bounds and body reaction |

Common requested controls for actor index a and slot s: `(1 if tick<8 else -1 if tick<16 else 0) * (.08+.01*a+.005*s)`. Generalized applied force: the same phase times`.01*(dof+1)` on all native DOFs; body wrench on the last massive body is phase times`(.03,-.02,.04,.002,-.003,.001)`, other rows zero. Mixed cases additionally use last eight steps' SLIDE generalized force`+64 N`, with all requested actuator controls and other external rows zero. This fixed unload input is separate from motor force and included once in the full mass/power oracle. Short two-step cases use the first phase; `dry-stick` overrides as declared. All controls are observed after original FP32 storage conversion.

No state is reset/repositioned to manufacture entry/release/unload inside any window. The explicit event assertions are part of this predeclared hypothesis; if the continuous native trajectory also fails one, retain the original failure and classify it, do not tune the fixture.

## 3. Independent oracle and original gates

Standalone installed consumer copies independent native decoding locally, not imports from tests/examples. Combine `tests/test_mujoco_tyx_constraints.py::native/assert_receipt` with R3-A analytic activation/transmission observation from `tests/installed/check_free_ball_activation.py`. **Set q, v, act and time before every same-entry `mj_forward`.** Retain a separate continuous MjData per world and advance it by native `mj_step`; never replace it with device outputs.

Native type1 rows map to DOF and bounds±frictionloss; type3 to native scalar joint and actual signed J side; types5/6 to complete geometry-only bijection and native contact offset. Read full J/R/D/aref/impedance/diagApprox/position/margin/force and retain every row, including zero forces. Independently reconstruct full H/b/KKT and physical M residual. For tiny joint-only examples, enumerate all signed box faces (at most six rows), independent of native force and product solver. Contacts retain both body COM wrenches, all row contributions, work and complete EFC address offsets.

Unchanged gates copied from T4/R3-A: geometry position/frame rtol3e-5/atol4e-5; J rtol2e-4/atol5e-5; M3e-4/5e-5; bias/passive/applied4e-4/4e-4; smooth acceleration2e-3/4e-3; constrained qacc/integration3e-3/1e-2; scalar row fields5e-4/4e-4; constraint force3e-3/3e-3; actuator force3e-4/2e-4; act3e-5/3e-6; continuous and same-step q absolute1e-4, v1e-3, time2e-6. Independent normalized KKT and M residual≤5e-5, bound violation≤2e-7; product normalized KKT≤storedFP32 max(source tolerance,2e-5), budget≤sourceiterations. Original quaternion and stored integration gates remain. Dry-zero uses original unconstrained qacc/integration2e-3/4e-3. No new tolerance is introduced.

Record F/L/rawslots, conservative E_bound, requested capacities and actual counts. New domain/capability labels are checked against actual rows, not assumed from presence of contact geometry. Unconstrained zero-dry source keeps its existing route and labels.

## 4. Transactions and actual capacity/budget negatives

Three representative routes: `fixed-lower`, `free-filter-limits`, `mixed-pyramid-filter`. Preserve W2 complete q/v/act/force/clock/control/last receipts, actual checkpoint device-bank bytes and revision across final-world status, completed-readback, publication and verified-restore rejection. Clear the recoverable fault and compare a real successor with an untouched clone. Activation/transmission publication faults apply to the two stateful representatives only. Check selected reset and real restore successors; incompatible source/schema checkpoint is rejected with a valid outer checksum. UNKNOWN opaque-release behavior is reused, not redefined or retried.

Actual capacity source is `mixed-normal-filter`: world3 starts with root z raised1m and H q`1/16` (all scalar limits interior), world29 has the ordinary mixed entry. Record native required counts first and use exact capacities or one fewer actual row; static E_bound must not reject the inactive first world. Overflow rejects both worlds; a fresh correctly sized owner then executes the identical input and is checked fully (capacity is immutable, so do not claim in-place budget mutation/retry).

Budget1 auxiliary source is the exact T4 correlated deep/shallow slider with limits from `test_source_iteration_budget_is_real_and_failure_is_atomic`, plus a disconnected masked unlimited BALL body using the common BALL inertia. Preserve deep sphere z`-.008`, shallow`.0009`, base z`.099`, r`.1`, gravity`0 0 -9.81`, sourceiterations1. This adds zero-coupled BALL DOFs, not a new contact algorithm. World3 shifts the slider up`.1`; world29 starts at original qpos0. Independently require negative unrestricted shallow force and a valid reduced face, hence a bound-hit plus second face; one face must fail late atomically. A fixed recovery command adds vertical generalized force`+256 N` in both worlds, keeping budget1; verify actual zero/nonviolated constraint successor against the untouched clone without changing geometry or thresholds.

## 5. Exact old negative promotions, no blanket exclusions

Only these original parameter identities are removed from old rejection collections. The new suite obtains each original XML from the unchanged source function/constant, compares exact source text and executes two real default-entry steps with zero controls; this does not invent a loaded original fixture. All other negatives remain.

| Original file | Original rejection node | Promoted names |
| --- | --- | --- |
| `test_mujoco_tyx_ball.py` | `test_ball_unsupported_domains_reject_before_device` | `dry`, `scalar-limit` |
| `test_mujoco_tyx_ball_joint_motor.py` | `test_joint_motor_rejects_outside_domain_before_device` | `dry` |
| `test_mujoco_tyx_ball_contacts.py` | `test_ball_contact_domain_rejected_before_device` | `scalar-limit`, `dry` |
| `test_mujoco_tyx_ball_motor_contacts.py` | `test_ball_motor_contact_rejected_before_device` | `scalar-limit`, `dry` |
| `test_mujoco_tyx_ball_scalar_actuation.py` | `test_ball_scalar_unsupported_domains_reject_before_device` | `scalar-limit`, `scalar-dry` |
| `test_mujoco_tyx_ball_activation.py` | `test_ball_activation_rejected_before_device` | `scalar-limit`, `dry` |
| `test_mujoco_tyx_free_ball.py` | `test_free_ball_outside_domain_precedes_device` | `dry` |
| `test_mujoco_tyx_free_ball_normal.py` | `test_free_ball_normal_unsupported_precedes_device` | `scalar-limit`, `dry` |
| `test_mujoco_tyx_free_ball_pyramidal.py` | `test_free_ball_pyramidal_unsupported_precedes_device` | `scalar-limit`, `dry` |
| `test_mujoco_tyx_free_ball_activation.py` | `test_activation_domain_rejected_before_device` | `scalar-limit`, `dry` |

The last three files' direct-compiler `allow_constraints=True` negative parameter `dry` is also promoted to a positive structural compile check. `allow_constraints=False` remains a rejection and is separately checked. No original helper/XML is changed. BALL angular-limit/spring, FREE actor, multiple FREE, shape/cone/dimension/explicit-pair/exclude and hard capacity negatives stay rejected. Invalid friction is tested on the native model after compilation when native XML itself refuses it, with native-invalid versus TYX-admission rejection explicitly distinguished.

## 6. Execution selection and storage

New physical windows: twelve identities, eight24-step identities and four2-step identities, 200steps/world and400worldsteps/backend. Five families include the independent transaction file; its exact parametrized count will be hand-enumerated at code handoff, not inferred from collection. Eighteen original-source promotion identities add two steps each/backend, plus direct-compiler admission checks; these replace changed negatives, not rerun unchanged green suites.

Package consumer runs the twelve new windows plus representative lifecycle using only installed Phymeo, MuJoCo3.3.7, NumPy and stdlib. Complete raw returns precede assertions, JSONL.gz is streamed with input/receipt/native continuous state and failure prefixes, modules are bound to the installed root, and primary/cleanup failures are retained. Parent alone runs static, functional and fresh package qualification. First failure remains evidence; no output-guided input/capacity/tolerance changes.
Pre-execution admission clarification: reversed scalar range is checked as native XML compilation rejection, before any TYX runtime. Negative margin is not newly classified invalid. Direct native-model mutation covers nonfinite scalar law and negative/nonfinite dry friction only. This preserves original admission semantics; tests-first 2e59ce48 remains historical and was never executed.
## 7. Authoring handoff and exact selection (not collected)

Historical authoring status **CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**. Parent subsequently completed the [bounded source and installed acceptance](p65-ball-scalar-constraints-report-20261113.md); the original cases below were frozen before execution. No project imports, collection, numerical probes, tests, lint, formatter, build, installation or CI were executed by this author. Source product/admission changes are outside this lane.

| New source test group | Manual identities per backend | HOST-only identities |
| --- | ---: | ---: |
| Complete new five-family windows | 12 | 0 |
| Exact original-source promotions | 18 | 0 |
| Direct compiler explicit opt-in | 0 | 3 |
| Structural domain rejection | 0 | 10 |
| Native dry parameter corruption | 0 | 6 |
| Independent tiny box oracle | 0 | 1 |
| Nonfinite scalar law / reversed XML range | 0 | 2 |
| Checkpoint/clone/selected reset successors | 3 | 0 |
| Final-world status, readback, constraint publication, verified restore; activation/transmission on stateful routes | 16 | 0 |
| Checksum-valid schema/force record corruption with real successor | 3 | 0 |
| Actual contact/EFC capacity exact and one fewer | 2 | 0 |
| Source budget1 bound-hit failure and fixed +256N recovery | 1 | 0 |
| **Total** | **55 per backend** | **22 once** |

Thus the two new test modules declare 132 combined nodes (110 TYX-mode identities plus22 HOST identities), pending actual collection. The main file is82, transactions50. These are selection counts, not passes or a completeness percentage. Suggested first order: new HOST contracts, twelve interpret windows, exact original-source successors, transactions, then the corresponding55 CUDA identities; do not rerun unrelated old green groups. The source-budget test retains actual first failure scratch before retry, and each late fault captures real completed publication/readback before injection.

Installed CLI (run outside any checkout, with installed dependency profile): `python check_ball_constraints.py --mode interpret|cuda --installed-root ABS_SITE --output FRESH_ABS`. The parent bootstrap may separately consume `--expected-install-root`; this consumer deliberately uses `--installed-root`. Default runs all12 windows (400 worldsteps/backend) plus three representative lifecycle paths (48 worldsteps/backend), total448; `--case` chooses a whole predeclared window and is not a reduced qualification claim. Journal records are streamed, include native same-entry/continuous state and actual returned receipts before assertions, and preserve raw bytes. Every constructed owner closes once; failed close retains a strong reference, and primary/cleanup/evidence errors are aggregated. A storage failure is not called a physics rollback.

The10 modified old test files contain only13 decorator-expression changes:18 ordinary negative identities and the3 duplicate direct-compiler `dry` selections are removed from their respective negative collections. All original `NEGATIVE_XML` elements, source generators, numeric literals and unrelated negative selections are retained. The new tests fetch the exact original XML, execute it without rewriting it, and preserve its hash in evidence. New code uses local copies of the accepted native T4 decoder and R3 activation/transmission oracle; installed execution imports no checkout tests or examples.

Author checkpoints: `3cc7769c` cases, `c2a52476` exact names, `2e59ce48` first main tests, `62719d3d` pre-execution admission clarification, `90f3cc03` transactions, `1288da1c` standalone oracle/consumer, `866720fc` exact old-parameter promotion. All remain unexecuted author snapshots. Any first functional failure remains evidence, including a predeclared event unsupported by the independent native trajectory; no inputs, thresholds, solver budget or row capacity are tuned to obtain a pass.
