# FREE+BALL Batch B author handoff

Status: CODE_READY_TESTS_PENDING / AUTHORING_RELEASE. No import, collection, numerical probe, pytest, lint, formatter, build, installation or profile was run by this author. This report is not a physical acceptance.

Baseline c45a01740d131ed6bebfabe2285169e67f296bd7, author branch codex/p65-free-ball-b-20261112. [Frozen design](p65-free-ball-composition-design.md) records primary MuJoCo 3.3.7/Newton 1.7.0 references, capacities, original numerical gates and parent ownership.

## Delivered behavior and limit

One native FREE root plus BALL/HINGE/SLIDE descendants, no contacts or activation, real NONE direct JOINT motors and full external loads. Complete 1/3-column CSR supports signed/zero gears and shared targets; BALL bias is NONE, original scalar AFFINE remains. Native qpos and DOF addresses are independent. nv35/body128 are unchanged mechanical capacities; nq45 is the explicit representation bound, nu32/nnz96 unchanged.

New model flag native_joint_layout selects native joint rows only for FREE+BALL. Legacy scalar T2 retains its pseudo-row geometry and original scalar/root integration branch; its new equivalent dof_ids table exposes original scalar qadr and root angular/linear columns. The complete physical M/J/bias and physical-M/Euler-step-M factorization are reused. No new dynamics solver, CPU step, coordinate-specific correction or increased precision/iteration budget.

First-slice exclusions remain explicit even if a caller supplies wrapper admission flags: every activation/contact/limit/dry row, multiple FREE roots, BALL/FREE spring, BALL same-body mixed joints, non-JOINT transmission and BALL affine bias. Existing no-BALL routes retain their prior scope. Heterogeneous multi-model batching and newly rejected contact/activation combinations remain future work.

## Commits and exact owner scope

- e98175b1: design and tests-first, before product authoring.
- 7b8c38f24102c1c0bc13d19ddb91871217a176aa: four product files.
- 923a38aa: exact retired-negative positives, explicit bad quaternion/multiple-root/same-body inputs; original case forces/gates unchanged.
- Final consumer/report commit adds no product changes.

MOD:
- src/phymeo/backends/mujoco_tyx/_floating_model.py
- src/phymeo/backends/mujoco_tyx/_floating_kernels.py
- src/phymeo/backends/mujoco_tyx/_floating_device.py
- src/phymeo/backends/mujoco_tyx/_joint_transmission.py

NEW:
- tests/test_mujoco_tyx_free_ball.py
- tests/test_mujoco_tyx_free_ball_transactions.py
- tests/installed/check_free_ball.py
- docs/development/p65-free-ball-composition-design.md
- this report

Parent-owned _model/runtime/public transmission/checkpoint/exports/normal/constraint wrappers were not edited here. Parent must integrate the declared ABI before testing. Original untracked docs/development/mujoco-tyx-t1-review.md remains untouched and unstaged.

## ABI and arithmetic review

- geometry has its original signature. Child BALL uses accepted fixed-tree anchor transport, three final-child-rotation axes, local angular velocity and parent cross terms; native root treatment is unchanged.
- mechanics receives dof_ids instead of pseudo joints, then after act_values receives transmission_enabled/offsets/dofs/moment/fields. Scalar flag-zero arithmetic is retained. Full CSR accumulation and fields are the accepted fixed-tree path.
- solve retains every original parameter through status, then appends native_joint_layout,nq,njnt,joints. Legacy branch copies nv+1 and integrates original scalar/root addresses. Native branch copies nq, reuses root update and right-multiplies each child BALL quaternion at its native address.
- All floating base devices allocate dof_ids and flag-zero CSR aliases; enabled CSR allocates actual complete fields. Only enabled CSR publishes joint_transmission_fields. No new instance-only helper is called on normal/constraint's unbound floating initializer.
- compile_joint_transmission accepts exact builtin tuples (8,10) or (35,45), exact builtin ints, no booleans/lists/unregistered caps. Default fixed domain remains 8/10. Floating structural validation precedes the 35/45 call; complete moment normal-or-zero checks remain.
- Parent new public transmission schema is native-free-ball-joint-motor-transmission-v1; existing class and stage semantics remain, fixed schema retains nv8. Runtime capabilities are free-ball-no-contact-v1 and free-ball-joint-motor-no-contact-v1. New activation schema is not claimed.

Manual source reasoning: root translations contribute world basis columns and local root rotation contributes R_root columns; a BALL contributes three columns of the same child rotation at one anchor. Each descendant inherits all applicable columns. The body-COM J maps those columns before full mass/bias/application assembly; no block diagonal approximation discards root/child or sibling cross terms. For scalar-only legacy layouts, dof_ids[i,1] equals i+1 for i>=6, so the new lookup does not change the old spring expression. No numerical proof or executed comparison is claimed.

## Authored validation

Manual enumeration from final decorators, not collection: 64 nodes across the two new pytest files.

Main 52:
- Eight named structural families x two backends: 24 genuine steps, W2 sparse keys, forward/reverse/unload phases.
- nv35/nq45/nu32 capacity x two backends: two steps, not a scale/performance qualification.
- Fourteen native-valid unsupported sources before device construction.
- Six invalid CSR-capacity types/values.
- Three exact former negative XML sources x two backends, default entry and original zero controls, two steps each.
- Root/child x zero/nonunit quaternion x two backends; before-allocation rejection.

Transactions 12:
- Full checkpoint bytes -> real successor -> restore -> repeated successor, clone and selected original reset.
- Last-world status and late readback fault, complete old state/receipt retained, real retry equals clone.
- Invalid final-world control before any kernel.
- Rechecksummed saved transmission wrong schema or width; rejected restore leaves continuation identical.

The old floating.BALL_XML, fixed-BALL free-and-ball and direct-motor free wrapper have exact new positive declarations. Parent retires only those old unsupported parameters. No old source test was edited by this author.

Dynamics oracle is independent native CPU forward/fullM/jacobian plus original physical and Euler linear equations, and independent complete JOINT force/moment/power. Continuous native data starts independently at the same stored initial state and is never copied from later TYX output. Original tolerances are retained in full in the design and raw evidence. First raw output is appended before comparison. No skip or tolerance enlargement is authorized.

## Source-external consumer

tests/installed/check_free_ball.py is standalone: only stdlib, NumPy, fixed MuJoCo and public Phymeo APIs; no checkout/test/example helper import or sys.path modification. It requires --mode interpret|cuda, --output absolute fresh path, --installed-root absolute actual installation. All loaded phymeo modules must resolve inside that installation; script/cwd and sys.path must not expose checkout sources. Existing dependency site-packages are not rejected merely for being in a venv.

Two explicitly declared structures: FREE+BALL+HINGE+SLIDE and FREE+two BALL branches. Each uses original source-case parameters, 24 steps plus checkpoint/clone/restored/partial-reset successors: 27 main calls and one clone call per structure, 112 accepted worldsteps total per backend if all pass. No measured success is claimed.

Each step keeps full entry, requested/stored controls, same-entry native fields, independent continuous native lane, raw returned acceptance, immediate integration state, full state and typed receipt. The raw return is saved before later observer/oracle checks. Full/integration state matches; actual TYX step and integration observation forbid native mj_step/mj_forward. Quaternion checks cover root and child. Complete raw bytes/dtypes/shape, clocks, forces, CSR, counters, module origins and failure prefixes are retained in a JSONL gzip. Checkpoint files and summary bind their SHA. Close attempts independent owners, preserves primary plus cleanup exceptions, and does not claim recovery after unknown native free failure.

## Parent pending commands and compatibility

No commands below were run by author.

1. Parent static checks on four product and three new Python files, then qualified style-only fixes if needed.
2. Complete tests/test_mujoco_tyx_free_ball.py and tests/test_mujoco_tyx_free_ball_transactions.py in the single serial interpret/CUDA queue, preserving actual collection and every first failure.
3. Narrow old no-BALL floating cases that exercise geometry/passive/integration; original kind3 normal/constraint/activation wrappers affected only by appended ABI. Select already identified source-relevant nodes instead of rerunning all old BALL or robot suites.
4. Build/install once after shared integration, source/wheel/install parity, copy standalone consumer outside source, run each backend with original gates.
5. Acceptance must name actual passed domain and unresolved first failures. These authored counts do not establish physical success, performance, Newton speed superiority, general contact or activation support.

Text checks completed: bounded original-source reads, complete product diff review, exact declared paths and git diff --check; no static/project execution. All numerical evidence and formal acceptance remain pending the parent.

## Parent source execution (2026-11-12)

After shared dispatch, receipt, checkpoint and legacy wrapper integration, the
first full run of both new files (`22-r2-source`) completed **64 passed / 68.32 s,
RAW_EXIT0**, including actual interpret/CUDA parameters and HOST admission
negatives. Original independent native and physical gates remain unchanged.
The author-only history above is retained. Narrow affected old-ABI compatibility
and the two source-external consumers are pending at this checkpoint; no general
FREE+BALL contact/activation, full robot task or scale/performance pass is claimed.

`31-affected-compatibility` subsequently completed **16 passed / 69.82 s,
RAW_EXIT0**: no-BALL mixed7 and offset-root dynamics, kind3 normal/constraint
wrappers, legacy floating activation and stateless routes, fixed BALL activation,
and the two retained FREE+BALL activation/contact admission rejections. Both
device backends are covered by the functional nodes. This is the declared narrow
compatibility selection, not a rerun or claim of the entire legacy suite.

