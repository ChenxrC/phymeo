# B2 reference-displacement representation revision

Status: FROZEN_PARENT_RULING. This is an explicit revision of B2 native representation, not acceptance of a new implementation. Parent adjudication follows the documentation-only proposal at child4b060ed and parent033e438. The user's standing authorization covers architecture decisions and serial validation; no extra generic permission is required.

Read [representation proposal](../../development/p65-b2-native-representation-proposal.md) for the rejected alternative and original failure, and [integration contract](2026-10-30-p65-material-self-gripper-integration-design.md) for all physical scenarios, proof gates and R1-R6. This document takes precedence only on the representation/publication and explicitly listed execution plumbing below. Old failed receipts remain unchanged.

## Decision and limits

Select option A only: immutable per-node trial-start FP64 reference R plus actual native FP32 displacement d, explicitly named reference-displacement-f32-v2. Do not implement two-word FP32, automatic fallback, adaptive recentering, host solver repair, dependency changes or relaxed physics. Existing default world-FP32 v1 remains callable with its original rejection behavior. Full B2 native acceptance now requires explicitly selected v2 under this revised contract; the original v1 B failure is not converted into a pass.

This changes the old claim that published world positions are exact promotions of one FP32 result. Native geometry, prediction, gradient, angle, lambda, projection and reconstruction remain actual FP32 computations. Published world position is the declared correctly rounded FP64 decode of immutable R and actual native d. Effective position precision is nonuniform and is not described as single FP32 world storage or native FP64 dynamics. Velocity remains the exact promotion of native FP32 output.

Only the saved fixed-row storage obstruction is established. Neither general FP32 impossibility nor successful v2 full windows is assumed. Fresh audit rejection may expose another real issue; never tune the physical gate to make this design succeed.

## Exact codec, decoder and supported range

For each fresh original-state trial/substep, R is an owned immutable byte copy of initial.position, d is positive zero with shape(n,3), dtype float32. R never changes within that trial. The next physical substep starts from the published previous state; p-zero-p trials always bind the same original state and fresh d. No hidden carry, warm reference, fixture-specific origin or mode-selected reference is allowed.

Let q denote the exact rational value of a finite binary input. S=q(R)+q(d); canonical world X=RN64(S), nearest/ties-even. Retain exact rho=q(X)-S. Native input encodings use RN32 of the exact binary-input rational expression, nearest/ties-even, with the existing normal-or-zero policy. A nonzero value requiring subnormal/zero publication in a native word rejects; no flush workaround. World binary64 decode must be finite, with its actual exact rounding residual checked; the f32 normal restriction is not silently misapplied to immutable f64 inputs. Canonical computed zeros are positive zero; immutable original byte fingerprints retain original signed zeros.

For an exact rational codec value, numerator and denominator each have a frozen4096-bit bound, checked before unbounded work; exceeded resources reject unknown/unsupported. The binary-format rounding algorithm is an integer/rational input codec, not a geometry/material solver. A float cast alone is insufficient evidence: independently verify output equality, bracketing/tie parity or an equivalent exact rounding identity. Decoder errors, overflow/nonfinite inputs and unsupported native ranges reject honestly. Do not truncate residuals to float for acceptance.

Precompute exactly these immutable directed reference vectors:
- Distance indices(i,j): R_i-R_j.
- Oriented hinge(a,b,c,d): R_b-R_a, R_c-R_a, R_d-R_b.
Their row identities/order and exact RN32 conversion witnesses bind the original material/state/reference. During actual native geometry, each directed vector j-i is computed in the declared order b_hat_ji + fl32(d_j-d_i), with FP32 rounding. Do not recompute world geometry/normal/angle on the host as solver input. Rounded edge tables are not claimed cycle-consistent or exact geometry.

## Actual lifecycle and evidence

Prediction computes original semi-implicit native velocity and d_pred=fl32(h_hat*v_pred), from zero displacement. Preserve all original/effective force and scalar conversion boundaries. Distance then bending use dedicated represented kernels, original element order and constitutive equations, both private lambda vectors, and actual d updates. Do not run the old distance kernel on d alone or rounded world X. Retain the two alpha operations per nonempty law and operation-specific actual launch receipts.

Reconstruct once on device as the declared FP32 evaluation of v_pred+(d_final-d_pred)/h_hat. The independent oracle remains v_pred+(X_final-X_pred)/h in original units; decoder/h/velocity errors are not omitted. Terminal loads are native velocity-only, once, with encoded position fixed. Original pre/post branch guards, numerical-row continuous domain proofs, physical affine segment proof, all dynamic masses and all physical energy/angular/work rules remain.

Private state includes R, d, d_pred, lambdas, codec witnesses, complete raw rows, boundaries and decoder records. A late distance/bending/audit/reconstruction/terminal/diagnostic/close failure publishes none of its partial changes. Outer substep rollback restores all public and hidden state/history/inputs; retry must match fresh. Observation/download is not a new update; re-encoding/recentering is prohibited, not relabeled observation.

Use a distinct owned v2 type family (RepresentedMaterialPath/Sweep/Batch, RepresentationBoundary and NativeCompositeRepresentation) and schema xpbd-composite-material-path-v2. The old MaterialBatch/raw18/raw45 v1 semantics do not change. V2 distance raw18 and bending raw45 contain d before/after in coordinate columns, and unchanged actual scalar semantics elsewhere. Additionally retain actual update-product t per row/node/component and the actual native directed geometry vectors used before/after. Raw native arrays and derived world arrays are separately typed, versioned, byte-owned and fingerprinted. Do not fill a native raw45 array with host-decoded positions.

The new row reader independently decodes X, checks identities/order/status/completed launches and all event bindings, and retains raw arrays, dtype/shape/bytes and failed receipts. Prediction, alpha, each nonempty block, reconstruction and terminal operations have separate real receipts; read-only decode has zero launches. Unknown schema, dropped/duplicated/reordered records, mutated R/edges/d/t/lambda/scalars/status or mismatched final hidden state reject, including mutations that preserve rounded X. Serialization round trips exact f32/f64 bytes and all identity tokens; generic repr/decimal-only dumps are insufficient.

## Frozen noncancelling accounting

For each actual row coordinate, let t be the recorded native update product used by the kernel, d'=native output, and g/delta be observed gradient/increment. The independent world-coordinate ideal still uses original inverse mass w.

a=q(d')-q(d)-q(t)
p=q(t)-q(w)*q(delta)*q(g)
rho=q(X)-q(R)-q(d), rho'=q(X')-q(R)-q(d')

The exact identity q(X')-q(X)-q(w)*q(delta)*q(g)=a+p+rho'-rho must be checked. Charge the conservative numerator |a|+|p|+|rho'|+|rho|, normalized by the original h*Vref, exactly once per row update. Retain the direct absolute world residual as a diagnostic and assert it is bounded by that charge; do not also add that same direct residual as another projection charge. Do not use signed cancellation or remove either decoder residual. This charge may exceed the former direct residual and must still satisfy the unchanged cumulative gate.

Keep original independent gradient, constitutive equation, parameter and lambda contributions and their original units/scales/aggregation. Reference-edge conversion errors are reported and included through independent world-coordinate gradient/equation comparisons; they are not fake point translations and are never a credit. Decoder effects at prediction/reconstruction/publication are explicitly checked at those equations too. Re-reading identical accepted evidence does not append another physical event; every genuine update remains charged. No threshold change follows from a conservative decomposition.

All original tau1e-8(host)/2e-5(native), reference scales1,512 iteration cap, enclosure width tau/64, sqrt256bits/atan64terms/timecells4096/depth32 and per-step/window1e-3J/1e-3Nms budgets remain. Windows sum positive energy and component/domain absolute angular defects without cancellation. NUM005 remains visible. The original R1 external work receipt is never subtracted for internal joint contact.

## Interfaces and exclusive edit scope

Expose optional keyword representation=None on TyxCompositeClothProgram and TyxCompositeClothParticipant. None preserves v1 exactly; explicit NativeCompositeRepresentation(kind="reference-displacement-f32-v2") selects v2 and binds the problem signature. Reject other kinds. A represented workspace may reuse unchanged outer lifecycle and independent bounds/energy functions, but must supply its own typed path/reader and atomic position_sweep; never fake v1 frames. If an extra old shared seam is genuinely necessary, stop that dependent edit and report the exact finding to parent.

The original B2 owner may edit only:
1. NEW src/phymeo/constraints/material_representation.py
2. NEW src/phymeo/backends/_tyx_represented_material_kernels.py
3. NEW src/phymeo/backends/tyx_represented_material.py
4. NEW src/phymeo/validation/material_representation.py
5. EXISTING src/phymeo/backends/tyx_composite_cloth.py: explicit representation option/signature/workspace dispatch only, unchanged v1 branch.
6. NEW tests/test_material_representation.py
7. NEW tests/test_composite_representation_path.py
8. NEW tests/test_tyx_represented_composite_material.py
9. EXISTING tests/test_composite_cloth.py: keyword-only representation forwarding in program_for/settled_workspace only.
10. EXISTING tests/test_composite_material_physics.py: keyword-only representation forwarding in run_window/standalone_replay and metadata only.
11. EXISTING tests/test_tyx_composite_material.py: explicitly select v2 for the same complete native acceptance cases; replace only v1 storage-meaning assertions with v2 raw/displacement/decoder-binding assertions. Keep all physical expectations, case parameters, windows, rebuilds, fault controls and exact test names.
12. NEW docs/development/p65-b2-native-representation-report.md
13. NEW .superpowers/sdd/2026-10-30-p65-material-self-gripper-integration/task-1-representation-report.md

The fixture seam must not send representation into case parameters or change a default host/v1 caller. No old physical assertion, helper formula, budget, window or perturbation may be removed. Preserve current original source/failed evidence and the earlier reports. All old bounds/host solver/energy/trace-v1/distance/runtime/shared contact/pin/workflow files remain unchanged. Parent alone owns later typed shared contact integration; neither v2 nor geometry69 grants that capability.

## Independent controls and serial acceptance

Author the proposal's exact binary-rational controls before execution: x=1,t=2^-25; translations2^10 and length scales1/2,1,2; inverse corrections with noncancelling charges; permutations/disconnected references; ties-even, normal-range failures and finite resource exhaustion. R=2^100 plus actual d=1 must expose nonzero decoder residual and reject under original normalization, not claim exact world publication.

Add independent exact identity/direct-world-bound checks for the above charge, all raw/derived/identity corruption, missing updates, unchanged-X reference mutation, schema-v1 rejection, serialization ownership and late hidden-state rollback. Actual tests must fail if host material solver/gradient updates or host reconstruction replace native execution. Diagnostic bounds remain allowed and never feed solver outputs.

Use the original native test names in test_tyx_composite_material.py for the unchanged complete A/B physical matrix with explicit v2. This is the native representation contract migration, not deletion of the failing test or a substitute smaller suite. Retain original v1 failure artifacts and explicit default-v1 compatibility/rejection controls in the new tests. Existing v1 bending primitive tests remain unchanged.

New host codec/path tests must run under a future sole grant. Original75 host evidence remains historical at72bdbcb: do not rerun it merely because optional test-factory forwarding changed. Verify forwarding-only diff and run focused default-factory parity controls. Any actual shared dependency change requires a parent-selected broader set.

Then complete actual interpret and CUDA, serially: full unchanged physical matrix in test_tyx_composite_material.py plus new represented native controls. First failures stop for evidence-led diagnosis, never gate tuning. Preserve all original exact seven direct consumers from task-1-validation-grant.md: the four host results are reusable if dependencies are unchanged; the three actual consumers are required in both modes. Original five interpret v1 bending primitives are reusable only if their closure stays byte-identical; CUDA v1 primitives remain pending. Record this reuse by exact version, not a new-hash run.

The original pending bounded v1 bending cpu_cpp node remains required. Add one bounded v2 case exercising actual prediction, one complete ordered distance+bending sweep, raw wire/decode and reconstruction with independent equations and appropriate finite/rollback controls. No long C++ lifetime stress or TYX003 closure. All execution (imports/static/host/native/CPP/report audits) is serial and requires the next explicit parent grant; this contract initially authorizes code/tests only.

Full v2 windows/transactions, saved-data audit, final same-snapshot source/evidence receipts and clean/process0 release are necessary before parent acceptance. Full self-contact response/fold/general robot/grasp integration and the single eventual stage review/wheel remain open.
