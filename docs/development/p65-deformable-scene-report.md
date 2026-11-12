# D5-A public asset / deformable scene assembly

Status: **CODE_READY_TESTS_PENDING / AUTHORING_UNEXECUTED**.
This author has not imported, collected, tested, calculated numerical probes,
linted, formatted, compiled, profiled, built or installed this work. The parent
owns execution and all existing product files. This is new declaration/lowering
functionality, not D4 continuous-load qualification.

Base: 6dcf99c2. Branch: codex/p65-deformable-scene-d-20261113.

## Exact author scope

Four NEW files only:

- src/phymeo/builders/deformable_scene_lowering.py
- tests/test_deformable_scene_compiler.py
- tests/installed/check_deformable_scene.py
- this report.

Tests-first d6b044d2, helper631fb60d, mode/cap preflight e02b6138,
query lifecycle67cac21b, installed consumer c49cc00d.
The parent's shared declaration/factory/source/query/export changes are separate
and required before execution. No existing file was edited by this author.

## Public declaration and static lowering

ParticipantDeformable(worlds, surfaces, *, settings=None, limits=None,
memory_limit_bytes=256MiB, node_memory_limit_bytes=256MiB) owns original
DeformableNodeWorld data and DeformableBoundarySurface installations. Arrays
are immutable bytes-backed copies. Original stable point/facet IDs and oriented
point order remain intact. Every registered D1 surface has one installation.
Original TyxDeformableBatch.requirements(trace=True) validates materials,
initial states, topology and its array reservation before runtime construction.

ParticipantSceneDeclaration selects contact_response="deformable_normal" and
deformable=ParticipantDeformable(...). materials_by_pair is the complete eligible
surface/round NormalLayer or explicit-None table. No default law or omitted-pair
inference is introduced. candidate_capacity bounds that complete eligible domain,
including currently inactive pairs; integer site_capacity<=128 retains D3's
active response capacity. Old host_patch/none declarations retain their route.

Parent _prepare obtains original rigid asset lowering via a temporary none-mode
declaration, then calls prepare_deformable_scene(declaration, rigid_prepared).
The NEW helper checks the D3 domain, uses compile_floating_tree_binding for the
original source/model-token gate, composes DeformableBoundaryModel, and returns
original participant/spec data plus node worlds/settings/limits/budgets.

compile_participant_scene dispatches to CompiledDeformableParticipantScene.
Its direct constructor checks declaration type and exact mode. Exactly one
TyxDeformableContactBatch owns both physical sides; there is no preliminary
independent rigid runtime or sequential commit.

source_map contains participants (original participant→source collider→installed
collider) and surfaces (installed key→world/original surface key/point IDs/facet
IDs/oriented point IDs). Scene identity includes the original assets, controllers,
installations, materials, initial data and capacities. Source mutation is rejected.

## Public lifecycle

- step(inputs,h,expected_revision=...) takes complete world→StepInputs nodal loads,
  then delegates once to D3 shared candidate/publication.
- set_controls(controls,expected_revision=None) validates all original public
  FloatingRobotCommand inputs before the one D3 control commit.
- query(expected_revision=...) and require_query_current(result,expected_revision=...)
  delegate to the parent's explicit accepted-source query API. Control-only
  revision changes make a prior result stale.
- last_step.query is the response entry evidence, not a current accepted query.
- snapshot/checkpoint/restore(checkpoint,expected_revision=...,worlds=None)/close
  delegate to D3 public ownership and transaction semantics.
- reset(worlds=None,expected_revision=...) restores a common checkpoint captured
  after initial controls. Thus declared held force/torque is restored as well
  as both physical states.
- revision/state_epoch/world_keys/participant_keys/identity/source_map/last_step/
  last_attempt/stats expose assembly and original complete physics evidence.

Checkpoints remain same-owner capsules; transaction revision is monotonic.
A constructor failure after allocation closes the owner. A second cleanup error
retains both exception identities and a strong contact_owner reference, with
cleanup_state="unresolved". This does not authorize blind retry of UNKNOWN.

## Finite domain and resources

First slice: 1–8 sparse worlds, each one original D1 distance-compliance model
and one ordinary non-folded one-body floating asset. Sphere/capsule source
provenance, full current device geometry and six-DOF dynamics are retained.
No new static installation, multibody/articulated/affine/folded composition,
friction, CCD, implicit coupling or remeshing claim is made.

All original layer/material/query/dynamics kernels remain unchanged. Each facet
is still an explicit discrete layer spring. Native query+response cap and D1's
separate device+reserved-HOST cap retain their old meanings, not RSS or historical
caller-evidence bounds. Response bytes remain
116*max(1,C)+12*N+28*W+4*(W+1), plus the original query reservation.
New compiler HOST costs are owned declaration/source metadata and its initial
common checkpoint, not a new physical bank.

## Frozen cases and independent reference

Both fixtures retain the original D3 square, five distance constraints,
compliance1/16384, node masses0.5kg (E node1 fixed), IDs101/113/127/139 and
facets503/509. Radius1/8, layer width1/16/stiffness128/damping0.5/mu0,
h1/1024 and 200-sweep/tolerance2e-6 remain unchanged.

- Sphere E/P: sparse worlds7/31; E local collider offset+1/4x and fixed node;
  initial velocity−ez/16. Asset source(0,42,71) installs at(world,202,0).
- Capsule C: vertical yz patch, original sloping endpoints and
  velocity(−1/16,+1/8,0), preserving a nonzero moment and unique finite closest
  feature. This is a programmatic FloatingRobotAsset, not native URDF capsule.

Both predeclare initial external COM force(0,1/32,0)N and torque(0,0,1/64)Nm
to exercise held controls/reset. The independent equations include these loads.
They are not selected from execution output.

Source positive cases each execute16 real steps. Tests alone reuse the original
independent D3 finite-distance/FP64 material/rigid reference helper. A separate
continuous reference starts from its own initial state. Original per-step gates
and continuous2e-4 state/orientation gates remain unchanged.

Hand-enumerated nodes, NOT executed collection: **20 HOST and6 TYX per backend**.
TYX: two positive windows, lifecycle, two real late-failure seams, wrong-owner/
closed-lifecycle. HOST: old modes2, static admission11, immutable/source checks2,
cleanup2, direct constructor3. Parent determines actual collection/results.

Late faults call the actual query or publication preparation before raising a
known Python error. Full public checkpoint/revision/accepted receipt must remain
unchanged, followed by a real retry. This is not UNKNOWN-driver recovery.

## Standalone installed consumer

The NEW consumer imports no repo tests/examples. It directly declares assets
and material data via public package APIs. Its exact finite-distance oracle,
original FP64 ordered recurrence and complete load/moment/work checks are
embedded from the accepted D3 consumer; construction/lifecycle are new.

CLI: --mode interpret|cuda --expected-install-root ABS_SITE --output FRESH_ABS_DIR.

Per backend, planned workload: sphereW2 and capsuleW1 each16 consecutive steps,
then branchA/restore/branchB and selected reset with a real successor.
Total38 batch steps /57 worldsteps /6 explicit accepted queries.
The38 internal entry queries remain in complete step receipts.

Returns are journaled before snapshot/assertions. Same-entry and continuous
reference geometry/load/state are saved before comparison. The gzip contains
all inputs, source identities, receipts/states, controls, checkpoints, reset
events and failure prefix. HOST geometry/material/law producer functions are
guarded after static compilation; expected actual calls0. Reference arrays
never overwrite product states.

Every loaded phymeo module must originate at the supplied installed site.
Script/cwd/import paths must be outside source checkouts. All finalization stages
are attempted, original and cleanup errors are retained together, and provenance
or journal errors prevent PASS.

## Handoff and pending execution

Parent focused static checks, NEW source tests, affected compatibility and the
two installed modes remain pending. No author PASS claim exists. Do not rerun
complete accepted D1/D2/D3 suites merely for this new compiler.

Authored SHA256 before parent formatting:
- helper: ba7fd40d3a135a7a168fea0d4a925c8fcf4e666dd75e498875e178d1deabdbd4
- tests: fce80f92838bb3d0604a789f66e02a7bfcf91d14dc3a17c54b9fcafc45d3c255
- consumer: 4788f62fb032a64db7ea7c5842e928d2121d28ebce8853abce6b28351e47eab2

Hashes identify unexecuted author files. Parent modifications, tested identities
and actual results must be recorded separately.

## Parent source and installed acceptance, 2026-11-13

LIMITED_ACCEPTED. Shared participant declarations/lowering, asset provenance and current-query revision are connected to the existing single D3 common owner. This is the declared sphere/capsule plus D1 zero-friction domain, not arbitrary articulated rigid/deformable coupling.

Source08/09:26 passed per backend (12.16s interpret,10.43s CUDA). A first invalid helper import in run07 was corrected to the existing FiniteStaticShape path before qualification. Runs10/11 passed five affected old controls per backend. After the independently diagnosed D3 witness-frame repair, affected asset windows56/57 passed two per backend; no complete old suite was repeated.

Package113629b3: source/wheel/installed530 product members BYTE_IDENTICAL. Installed81 interpret and74 CUDA each report PASS with38 returned/checked calls,57 accepted worldsteps,6 explicit current queries and57 checked contact sites; cpu_physics_calls=0. Run73 failed before consumer construction because bootstrap argument parsing consumed a duplicate --expected-install-root; raw error retained, then consumer alias --installed-root used. No physical source or gate changed for this CLI repair.

Full commands, raw failures and exact source snapshots live in artifacts/p65-feature-until-0730-20261113-01. This acceptance includes public initial controls, partial reset/restore, common publishing and actual source-external calls. Long steady-support D4 remains separately FAIL, as documented in the continuous-load protocol.

## Compatibility with the optional dissipation product

Product ec0e2997's second wheel includes D6 changes to the shared D1/D3 owner.
Affected public D5 consumers132 interpret and133 CUDA both passed outside
the source checkout:38 returned/checked calls,57 worldsteps,6 queries and
57 checked sites each; cpu_physics_calls remains0. These are ordinary D5
compatibility windows, not newly claimed long support or damped-D5 installed
windows. The dedicated damped public-scene source cases121/122 each passed2.
The complete532-member product parity belongs to run127 and was not rerun.
