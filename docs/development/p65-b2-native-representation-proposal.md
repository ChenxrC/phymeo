# B2 native represented-state contract revision proposal

Status: **CODE_DESIGN_PROPOSAL_READY**. This is a documentation-only proposal for
parent adjudication, not a frozen contract, implementation grant, or native
acceptance. Product remains `72bdbcb8c86f65c6cd5be8c13bed2123254fa647`; the validation
report remains `2be6396b11b5f4aaa299e2636e3ad900c769bc57`. No source, test, existing
report, dependency, or physical gate is changed by this document. No import,
numerical/offline probe, test, lint, formatter, compile, build or profile was run
for this proposal. The execution slot was already released and is not reclaimed.

## Recommendation and decision boundary

Recommend option A, **immutable per-node trial reference plus actual native FP32
displacement**, for a separately versioned, explicit composite mode. It changes
where position rounding occurs without requiring a floating-point expansion
library in the pinned backend. Option B, **two-word FP32 position state**, is a
credible alternative if the parent prefers a representation without per-element
reference geometry and accepts a larger arithmetic/trace proof obligation.
Neither option has been implemented or shown to pass a complete B2 window.

Both options **revise the existing promise that native world positions are exact
promotions of single FP32 outputs**. Calling their operations FP32 does not make
the represented world state single precision. Their published world position is
an audited FP64 decoding of immutable input data and actual native outputs. This
is a real precision/publication contract revision, not an FMA optimization,
storage-format-only change, or host audit repair. If single-FP32 world publication
must remain binding, neither option is admissible and the present B2 rejection
stands. No automatic fallback between representations is proposed.

The recommendation is narrower than a universal claim about FP32. Saved evidence
rules out a simple arithmetic/order cleanup for the current dominant fixed rows.
A changed whole solve can have different operands. It remains unknown whether
another single-word algorithm could satisfy all frozen gates.

## Evidence being explained, not rerun

All receipts below are in
`artifacts/p65-b2-validation-20261030-181949/` in this worktree. The immutable
`final-evidence-files-manifest.json` has SHA256
`6f1b75d9944df6f30931a863193072df5d7f41e6614532df32ce78d7430f3e46`.

| Receipt | Preserved observation | Scope |
| --- | --- | --- |
| `12-host-full-after-diagnostic-fix` | RAW_EXIT 0; 75 passed | Product 72bdbcb; six new host files and four exact host consumer nodes, expanded to seven cases |
| `17-audit-saved-host-evidence` | RAW_EXIT 0; 24 complete windows, 640 accepted saved steps; rebuild hashes agree | Existing run data only; no extra physics |
| `13-first-actual-interpret-full` | RAW_EXIT 1; 29 collected, five primitive cases passed, next composite case failed | Original B at h=1/128; first complete private sweep; no publication |
| `14-diagnose-native-budget` | RAW_EXIT 1; original rejection re-raised, exact operands saved | One previously authorized same-case diagnosis; no product modification |
| `15-offline-saved-budget-breakdown` | RAW_EXIT 0 | Saved row operands only |
| `16-offline-nearest-fp32-proof` | RAW_EXIT 0 | Fixed observed ideal_after values only |

At bending row 1, node 3, component y, the normalized noncancelling total first
exceeds tau: projection/storage 2.3420806820402713e-5, gradient
1.890040772992456e-10, equation 4.9449411074473403e-8, conversion and lambda zero;
total 2.3470445235554487e-5 versus the original 2e-5. At the final row the same
coordinate's total is 2.9275224522847993e-5. Its projection-only term is exactly
33694620709861/1152921504606846976 = approximately 2.9225424779765093e-5.

All seven dominant outputs are already the nearest FP32 numbers to their saved
exact ideal_after operands. The two nonnearest outputs elsewhere have avoidable
normalized excesses 4.26204627146376e-11 and 8.520228966801824e-11, and cannot
change that coordinate's bound. This establishes a storage obstruction for this
fixed trace, not impossibility of every FP32 algorithm or every new trajectory.
No accounting mismatch justifying removal of a term was found. The full native
interpret acceptance, CUDA group and bounded actual cpu_cpp smoke remain pending.

## Present seams and what they actually require

References are to product 72bdbcb, not to an assumed future parent integration.
Source paths in this table are relative to `src/phymeo/`.

| Existing seam | Current behavior and consequence |
| --- | --- |
| `backends/tyx_composite_cloth.py`, `_predict`, `_block`, `_reconstruct` | Converts world x to FP32, uses old native distance and new bending kernels, promotes x back to FP64, reconstructs through converted world positions |
| `backends/tyx_bending.py`, `TyxBendingMath.project_bending` | Actual `(m,45)` FP32 trace; `(n,3)` world x; returns promoted native x and lambdas; must retain its v1 meaning |
| `backends/_tyx_bending_kernels.py`, `project_bending_traced` | Ordered lane, actual atan2/analytic gradient/update/post-angle; writes world positions into fields 0:24 |
| `backends/_tyx_distance_trace.py`, `project_node_distances_traced` | Old ordered `(m,18)` world-position trace, shared with distance-only behavior; must not be repurposed |
| `constraints/material_trace.py`, `MaterialBatch`, `CompositeMaterialPath` | v1 has FP64-owned snapshots, fixed widths 18/45, exact schema token and consecutive complete distance-then-bending blocks |
| `validation/composite_material_path.py`, `audit_sweep`, `_require`, `ordered_work` | Requires native rows equal their FP32 conversions; binds before/after world positions; accumulates absolute projection, conversion, equation and lambda errors across all sweeps; checks original-input prediction and reconstruction; computes ordered work from world rows |
| `solvers/composite_cloth.py`, `position_sweep`, `_path`, `_audit` | Publishes a complete audited private sweep, then reconstructs once; final path/energy checks precede returned state |
| `backends/tyx_math.py`, `as_compute`, `TyxMath.call` | Normal-or-zero FP32 input policy; explicit f32/i32 arrays, owned private calls, sync, finite output checks, launch/completion counters |
| `backends/_tyx_numeric.py` | Checked f32 operations; no established two-word/error-free-transformation contract, compiler contraction guarantee, or general subnormal support |
| `backends/tyx_participants.py`, `_NodePort` | Positions read the owned published world array; response/apply use native velocity operations; do not silently pass represented positions through a world-FP32 node update |
| `validation/composite_material_energy.py` | Endpoint physical accounting is independent of representation and consumes actual world state/original inputs; retain it |
| `runtime/load_replay.py`, `SceneLoadReplay.evaluate` | Fresh original-state trial, original requests and selected-pre binding; hidden references/low words must be fresh too |
| `solvers/initial_robot_material_contact.py`, `_make_node_ledgers`, and `_material_contact_balance.py` | Existing old distance-specific dispatch and elasticity are parent-owned; neither new representation authorizes composite acceptance there |

The new path must not be mislabeled `numpy` to evade FP32 checks, nor squeezed
through v1 by placing host-decoded world numbers into columns described as native
FP32 outputs. A v2 reader is necessary.

## Frozen invariants and units

Retain the ordered distance block then bending block, original element ordering,
one prediction, both lambda vectors initialized once per trial/substep, complete
private sweep publication, one reconstruction, final residuals of both laws,
and rollback of all state/history/inputs/hidden state after every late failure.
Retain the backend pin `99f6df7c0fd8fdeef77f97df83914da27b64b7ee` and actual modes.

Retain original tau (host 1e-8/native 2e-5), Lref=Vref=angle_ref=1 in their original
SI units, 512 position-iteration cap, independent enclosure width <= tau/64,
original angle/sqrt/domain proof resources, and all strict supported branch and
degeneracy conditions. No new offset size, residual size or compiler discrepancy
may select a looser tolerance or a different physical fixture.

Retain A: T=1/4, h=1/64,1/128,1/256; B: T=1/8, h=1/128,1/256,1/512; two independent
rebuilds; both signed node-4 perturbations; mass factors 1/2 and 2; proper
rotation/translation; disconnected copies; original replay/terminal controls.
Per-step and whole-window positive energy remain <=1e-3 J, and angular defect
<=1e-3 N*m*s. No per-step cancellation, averaging, or multiplication of the window
ceiling by its number of steps. R1 external receipts, R2 zero extra transfers in
physical windows, R3 positive-part and component/domain absolute sums, and NUM005
remain unchanged. No contact-work subtraction or hidden use of effective inputs
in joint physical balance is introduced.

Use world X, reference R, displacement d, and low/high position words in metres;
velocity in m/s; inverse mass w in 1/kg. Distance C is m, compliance is s^2/kg,
alpha is 1/kg, lambda is kg*m, and its gradient is dimensionless. Bending C is rad
(dimensionless in the algebra), compliance is 1/J, alpha is 1/(kg*m^2), lambda is
kg*m^2, and its gradient is 1/m. In either case w*delta_lambda*gradient is metres.
Changing the position representation must not rescale lambda, compliance or the
existing h*Vref displacement normalization.

## Shared semantics for either new representation

Let q(a) mean the exact rational value of stored binary number a, and RNp mean
round-to-nearest/ties-to-even in the declared binary format. A representation E
has an exact value S(E) and a canonical world view

    X(E) = RN64(S(E)).

The public ParticleState, point geometry, physical ledger and world-coordinate
ideal equations all use this same X, not an unrounded hidden world state. Retain
rho(E)=q(X(E))-S(E) as an exact audit quantity. FP64 addition is permitted only as
this declared decoding (or input encoding/diagnostics); it is never a projection,
force, gradient, angle, or replacement solve. Independently check the mapping
from immutable input bytes and downloaded native bytes. If its rounding cannot
be certified under finite resources, reject.

Every material row's ideal constraint/analytic gradient, original compliance
alpha and original mass are evaluated at the canonical world X_before, using the
existing independent rational/interval mathematics. Native geometry and scalar
outputs remain observed approximations. World X_after is decoded from that row's
actual native output. The original raw row world residual

    |X_after - (X_before + w_original*delta*gradient_observed)|

must remain bounded. A conservative representation decomposition may replace
its arithmetic calculation only with a proved upper bound, never with a smaller
residual. Conversion, representational storage, decoder rounding, gradient,
parameter, constitutive-equation and lambda terms remain visible and accumulate
without sign cancellation per node/component/element over the whole path.

Concretely, retain each original world row's interval equation
`e_C = C(X_before) + alpha_original*lambda_before + D_original*delta`,
its ideal positive denominator interval D, and its lambda update/conversion
error e_lambda. In the position budget, replace only the raw update term with a
proved representation upper bound eps_repr, then still add

    w_original * (abs(delta)*gradient_error
                  + magnitude(e_C)/D.low*magnitude(ideal_gradient)
                  + e_lambda*abs(observed_gradient)).

The element equation sum remains sum(magnitude(e_C)/Cref), the lambda sum remains
sum(D.high*e_lambda/Cref), and position sums remain divided by h*Vref, each
compared against the original tau. Keep prediction's original world/velocity
equations and reconstruction's original world quotient with their existing
stage normalizations; do not move a failed position charge into a separate
uncounted diagnostic or replace a cumulative gate with a per-row maximum.

All conversion events have explicit input/output shapes, physical units, sequence
and immutable identity. Observational download/decoding of an unchanged state is
not a new physical update. Re-encoding, changing a reference, collapsing a low
word, or writing a rounded decoded view back to native state is a distinct event
and must be charged; it may not be described as mere observation.

Only complete successful raw arrays become typed accepted evidence. Preserve
failed-attempt receipts/status/counters separately; a prefix of successful rows
is not an accepted batch. Reference arrays, high/low words, both multiplier
vectors, decoder records and trace checkpoints participate in the same private
transaction. No output may reach the caller before the complete sweep and its
independent audit succeed. After an accepted inner sweep, a later physical,
terminal, diagnostics or close failure still rolls back the outer attempt.

## Option A: immutable per-node reference plus FP32 displacement

### Representation and deterministic reference rule

At each fresh substep/trial, set `R = owned_exact_copy(initial.position)` with
shape `(n,3)`, FP64 metres, and native d=zeros `(n,3)` FP32 metres. Freeze R through
all position sweeps and reconstruction; it is never recentered using a current
iterate or a fixture identifier. Decode S_i=q(R_i)+q(d_i), X_i=RN64(S_i).
At the next physical substep use that substep's immutable published initial state
as its new R, with zero d. Replay p-zero-p always starts with the same original R;
there is no warm start, accumulated displacement carry, or trial-specific origin.

There is no arbitrary global coordinate origin to tune. The rule is generic for
all nodes and translations. The reference fingerprint includes original state,
node identity/order, topology, domain, h, representation version and trial input
binding. An optional user-supplied origin is not proposed in the minimum contract.

Native geometry cannot evaluate a world difference by rounding R+d to world
FP32. Pre-encode immutable reference differences for each declared directed
vector: distance (i-j); bending (b-a), (c-a), (d-b), matching the current kernel's
orientation. These are `(md,1,3)` and `(mb,3,3)` f32 metre arrays, bound to R and
ordered identities. For vector j-i, native geometry evaluates an explicitly
ordered f32 expression equivalent to

    b_hat_ji = RN32(q(R_j)-q(R_i))
    e_hat_ji = fl32(b_hat_ji + fl32(d_j-d_i)).

The encoder must certify nearest-even conversion of the exact binary-input
difference, or record and charge a specified alternative double-rounding error.
It cannot assume that a host FP64 subtraction followed by cast is always the
same exact rounding. The recommended contract is exact-difference conversion,
implemented as an input codec, not a host solver. No per-sweep host recomputation
of geometric edges, normals or residuals is allowed to feed the kernel.

Rounded reference edges may fail cycle closure and may induce geometry error.
They are not declared exact geometry. Every row records its actual native edge
vectors and compares its angle/gradient/equation with bounds at decoded world X.
Reference-edge conversion is a geometry-input error, not a fictitious position
translation; its effect must be included through the independent world-gradient
and equation bounds. It must never be omitted because R itself is exact.

### Prediction, projections, reconstruction and publication

| Boundary | Proposed actual operation | Independent obligation |
| --- | --- | --- |
| Original inputs | Preserve R exactly; convert v0, original/effective force, w, gravity, h, rest and compliance by their existing explicit normal-or-zero FP32 rule | Retain original-input/request receipts; record every conversion, including reference vectors; no synthetic global-position cast |
| Prediction | Actual kernel computes existing semi-implicit v_pred in FP32, then d_pred=fl32(h_hat*v_pred), using zero initial d | X_pred=decode(R,d_pred); compare v_pred and X_pred against original equations and scales; certify initial-to-predicted world segment |
| Constitutive alpha | Retain original compliance, actual converted input and two native scaling operations per nonempty law | Original alpha=compliance/h^2 remains ideal; conversion/equation errors and launch counts remain |
| Distance rows | Actual ordered native geometry uses reference vector+d differences; computes the same C, gradient, delta and lambda laws; updates d_i with actual f32 increments | Audit decoded world row positions, original rest/mass/alpha and all storage errors; old distance-only kernel is not changed |
| Bending rows | Same oriented analytic FP32 geometry/atan2/gradient/lambda equations from reference vectors+d differences; same pre/post native branch guards | Audit independent world angle/gradient and continuous world row segment; actual raw outputs cannot be replaced by bounds |
| End of sweep | Private d plus both lambdas and both blocks retained; decode world candidate, audit complete sweep | Publish all together; no host addition of a correction computed by the audit |
| Reconstruction | Actual kernel computes fl32(v_pred + fl32(d_final-d_pred)/h_hat), retaining the declared operation order | Compare against original world equation v_pred+(X_final-X_pred)/h; decoder error and cancellation/rounding are visible |
| Final/public state | Position is declared RN64(R+d_final); velocity is exact promotion of actual native FP32 output | Finite decode, exact binding, domain, both material residuals and physical budgets before publication |
| Terminal impulse | Existing actual FP32 velocity response/addition only; encoded position and reference unchanged | Once-only terminal work/response and selected-pre fingerprints bind canonical world state |

The native distance kernel must be a representation-specific ordered kernel.
Running the old kernel on decoded world FP32 would restore the observed failure
mechanism; running it on d alone would give wrong geometry. Both are prohibited.

Reconstruction must use the displacement representation directly, not cast
X_final and X_pred to FP32. Yet replacing its output on the host with a more
accurate world-coordinate quotient is also prohibited. The independent audit
may reject a mismatch; it cannot repair it.

### Handwritten storage/accounting argument and limits

For a row coordinate, let t be the actual native FP32 product used in the update,
including converted mass and observed delta/gradient, and let

    d' = fl32(d+t),   a = q(d')-q(d)-q(t).

For normal finite round-to-nearest addition, |a| is at most half a spacing at the
rounded displacement result. More generally, compute |a| from observed bits;
this exact residual is valid without assuming a particular compiler operation
sequence. The decoded world update satisfies the identity

    q(X')-q(X)-w_original*delta*g_observed
      = a + (q(t)-w_original*delta*g_observed) + rho(E')-rho(E).

Thus |a| + |q(t)-w_original*delta*g_observed| + |rho(E')| + |rho(E)|
is a noncancelling upper bound for the original world projection residual.
Use this or a separately proved equally conservative decomposition; do not
subtract the two rho values to hide representation loss. This may be more
conservative than the old single absolute residual. The parent must freeze the
precise charge locations before implementation; no cancellation is proposed.
The existing gradient/equation/lambda terms are then added with original units.
Geometry-edge errors already present in independent gradient/equation bounds
must be reported there; they cannot be silently zeroed or used to reduce a term.

This relocates the per-update spacing scale from |X| to |d|. It does not make
displacement exact or prove acceptance: d can grow large over a substep, many
updates can accumulate, opposite updates cannot cancel their charges, geometry
and lambda errors remain, and world decoding still incurs binary64 rounding at
large |R|. Underflow, overflow, nonfinite native intermediates, failed branch/
area proof, unsupported decoding or exhausted budgets fail closed. No adaptive
rebase is included; adding one later needs explicit conversion events, charging
and rollback proof. No full trajectory error or convergence theorem is claimed.

Effective position precision is nonuniform: an exact stored binary64 reference
plus a binary32 displacement, rounded to the canonical binary64 world view. It
is neither uniform FP32 nor a claim of full FP64 native dynamics. Transcendentals,
gradients, lambda, parameters and velocity operations stay FP32.

## Option B: compensated/two-word FP32 position state

### Representation and actual operations

Use `(H,L)` arrays, each `(n,3)` FP32 metres, with exact S=q(H)+q(L) and the same
canonical world decoding X=RN64(S). Encode the immutable initial position using
H=RN32(x0), L=RN32(q(x0)-q(H)), retaining the exact discarded input residual.
Require a canonical, nonoverlapping pair (normal finite words or exact zero),
including a specified signed-zero rule. Non-overlap does not imply that every
FP64 input or every sum is represented exactly. Original-input conversion loss
is charged even if both words are valid. No host residual is inserted after a
native solve, and no hidden third word persists.

A credible minimal native accumulator uses an explicitly specified TwoSum
primitive. Under separate IEEE round-to-nearest operations, no overflow, and
supported intermediate range, TwoSum(a,b) produces s,e with s+e=a+b exactly.
Its standard general form is

    s=RN32(a+b); bv=RN32(s-a); av=RN32(s-bv)
    br=RN32(b-bv); ar=RN32(a-av); e=RN32(ar+br).

Do not assume those conditions from the existence of `_add`. The pinned backend's
code generation, temporary rounding, contraction/reassociation and subnormal
behavior need independent actual-mode controls. Its pure-function API does not
currently establish a pair-return primitive; use explicit kernel scratch/output
arrays if necessary, without modifying the backend dependency or old helpers.

For one existing FP32 increment t, a concrete two-word update is

    (s,e)=TwoSum(H,t)
    (z,f)=TwoSum(L,e)
    (H',L')=TwoSum(s,z).

If the primitive assumptions hold, q(H')+q(L')=q(H)+q(L)+q(t)-q(f).
The discarded f is a real, noncancelling storage charge, not a compensation
credit. Record it as actual native evidence and independently check the identity
from downloaded words. Even if the primitive identity is not proven for some
backend operation, the exact observed pair-sum residual is still audited; no
small-residual promise may be inferred. Unsupported arithmetic must reject,
not enable unchecked fast math or flush a required low word to zero.

Keep lambda, parameters, gradients, atan2 and velocity single-word FP32 in this
minimal alternative. Improving them too would be another declared revision.
Prediction computes native v_pred and the native displacement product, then
adds it to the initial position pair using the above actual accumulator. Ordered
distance/bending compute relative vectors by native pair subtraction first,
then explicitly collapse those small relative vectors to FP32 for geometric
arithmetic. Collapsing H+L to a world FP32 value before subtraction would destroy
the purpose and is forbidden. Record relative-vector collapse residuals and
retain world-coordinate ideal bounds.

For reconstruction, subtract final and predicted position pairs on device,
collapse the resulting difference to FP32, then perform the declared FP32
quotient/addition with v_pred. Charge all subtraction, collapse, h conversion and
velocity rounding against v_pred+(X_final-X_pred)/h. Publish RN64(H+L) only through
the declared decoder; no host reconstruction. Terminal impulses still change
native FP32 velocity only, with position pairs frozen. Re-encode anew from the
original published state for each fresh replay/substep; no low-word carry hidden
in SceneState history.

### Handwritten accounting and limitations

Let S=q(H)+q(L), S'=q(H')+q(L') and r=S'-S-q(t). Without invoking an error-free
claim, the exact downloaded operands give r directly. The original world update
residual is bounded by

    |r| + |q(t)-w_original*delta*g_observed| + |rho(E')| + |rho(E)|.

When the stated Three-TwoSum construction is valid, r=-q(f), giving a concrete
storage-error witness. For well-scaled nonoverlapping pairs the discarded tail
can have a smaller scale than single-word world storage; no unconditional
relative bound is asserted near cancellation, subnormal intermediates, overflow
or very separated exponents. A fixed O(u32^2) slogan is not an acceptance proof.

Normal FP32 words have finite dynamic range; a second word increases effective
significand information, not the exponent range or branch support. A low word can
be nonzero but too small to survive RN64 decoding of H+L; record the exact rho and
charge it. Do not assume every pair sum is exactly representable in FP64. Small
low-word intermediates are especially likely to encounter the current
normal-or-zero restriction. A future support extension would need parent ruling;
this proposal preserves that restriction and rejects unsupported arithmetic.
Observing a zero does not prove the exact intermediate
was zero on a flushing backend. To enforce this restriction, the pair algorithm
must expose the intermediate operands/results needed for exact normal-range
checks, or have an independently established equivalent checked primitive.
Endpoint pair residuals alone cannot certify every intermediate range condition.

No reference-edge table is needed, and long displacement growth is less tied to
a trial origin. Costs include a larger raw wire, additional state transaction,
canonicalization/input splitting and native pair-subtraction proofs. No speed,
memory-use measurement, backend parity, or passing physical result is claimed.

## Wire, interfaces, serialization and provenance

Propose an owned `NativeCompositeRepresentation` configuration with an explicit
kind `reference-displacement-f32-v2` or `two-word-position-f32-v2`, decoder id,
reference/encoding rule id, arithmetic operation-order id, normal-or-zero policy
and finite proof limits. Bind it to problem signatures and diagnostics. The
existing default world-FP32 behavior and v1 schema remain unchanged. Callers must
explicitly request a ratified v2 representation; no backend/fixture heuristic.

Use a new `xpbd-composite-material-path-v2` type family. A v1 consumer rejects it
by type/schema instead of guessing from array values. Proposed types are
`RepresentedMaterialPath`, `RepresentedMaterialSweep`, `RepresentedMaterialBatch`
and `RepresentationBoundary`. They own immutable copies, including original
inputs, original material identity, representation fingerprint, initial-state
fingerprint, physical domain, h/scales/policy, mode, backend pin, all encode/decode
records, actual raw status, and operation-specific launches/completions. Empty
blocks still exist with zero launches. Sweep/row/event sequence is complete and
consecutive; final represented state and lambdas bind to the last accepted row.

| Payload | A | B |
| --- | --- | --- |
| Owned immutable reference | R `(n,3)` f64, original state identity; directed reference vectors with exact conversion witnesses | Original position `(n,3)` f64 plus split witnesses; no reference geometry |
| Native state at block input/output | d `(n,3)` f32 | H and L, each `(n,3)` f32 |
| Bending base raw array | `(mb,45)` f32; fields 0:12/12:24 are d before/after, **not world x** | `(mb,45)` f32; fields 0:12/12:24 are H before/after, **not world x** |
| Bending low-word sidecar | Absent, not an implicit zero array | `(mb,24)` f32, L before/after; equivalent complete pair wire width 69 |
| Distance base raw array | `(md,18)` f32; fields 0:6/6:12 are d before/after | `(md,18)` f32; fields 0:6/6:12 are H before/after |
| Distance low-word sidecar | Absent | `(md,12)` f32; equivalent complete pair wire width 30 |
| Unchanged scalar semantics | Bending 24:36 gradient, 36:38 theta, 38:40 C, 40:42 lambda, 42 delta, 43 alpha, 44 denominator; distance 12/13 lambda, 14:17 gradient, 17 delta | Same; all are actual native FP32 values, not decoded ideal quantities |
| Additional actual evidence | Per-row t `(m,arity,3)` and geometry vectors `(md,1,3)` / `(mb,3,3)` before and after as used | Same t/geometry evidence, plus discarded tail f and pair subtraction/collapse evidence |
| World view | Owned decoded before/after `(m,arity,3)` f64, independently regenerated from R and raw d | Same shape, independently regenerated from raw H/L |
| Typed integer evidence | Native i32 element identities/status/failure index, wrapper-owned validated identity arrays | Same |

The width-preserving base array in A is **not** the old raw45 contract: its schema
and coordinate semantics change. B cannot omit its sidecar or present H as world
x. Never construct a host 45-field array of decoded positions and label it a
native 45-field download. Native mode/type/bytes and derived view must be distinct
fields. Raw parameter arrays, lambda vectors and prediction/reconstruction
inputs/outputs retain their explicit native dtype/shape too.

`RepresentationBoundary` must name the operation and sequence; bind source and
destination representation ids, shapes/units, exact immutable input bytes,
actual native output bytes, decoded value, rounding/conversion residual witness
and associated native call receipt. A read-only decode has zero launches and a
decoder provenance label, never a fake kernel receipt. Prediction, each nonempty
ordered material block, alpha operations, reconstruction and terminal operations
keep separate real call counts. If an implementation adds a separate encoding or
pair-normalization kernel, its call must be declared; no fabricated count parity.

Serialization must retain f32/f64 bit patterns, shape, dtype, row order, signed
zero convention and version tags, not just JSON decimal arrays or repr output.
Canonical typed binary arrays plus metadata are preferred; a JSON-only export
needs explicit bit/hex encoding and exact reconstruction. Existing generic
`core.scene.fingerprint` and `runtime.program.own_diagnostics` already traverse
dataclasses/arrays, but ownership/read-only and exact-byte round-trip behavior of
the new types still need tests. Unknown versions, missing sidecars, aliased
mutable buffers and incomplete receipts reject. A checkpoint hash includes the
codec and raw evidence; changing a reference or low word is a changed state even
if its rounded world view coincides.

## Domain, physical ledger and downstream consequences

The independent ideal-angle method remains exact binary-input determinant
identities, rational square-root/pi/atan enclosures and analytic gradient bounds.
No Fraction(float atan2) becomes an ideal angle. Bounds evaluate the canonical
world coordinates actually returned to physical consumers. Native edge/angle
rounding remains measured against that independent result.

Native pre/post branch guards remain necessary but not sufficient. Retain B2
continuous certificates for every declared world conversion and row segment,
including distance-induced hinge motion, and for the final affine initial-to-
final world segment. References/low words do not make a segment safe. The exact
meaning of each certified segment is the affine interpolation of its decoded
world endpoints. It is not a certificate of every machine-operation intermediate,
collision CCD, the true ODE trajectory, or an arbitrary curved representation
path. B1 endpoint-only statements remain endpoint-only; no retroactive claim.

`ordered_work` must read independently decoded world rows and expose any
encoding/decoding/storage work discrepancy. Its kinetic changes and material
impulse work are diagnostics, never a cancellation credit for acceptance. Final
energy, original force/gravity work, external held/terminal receipts and angular
defect use the exact same canonical public world positions/velocities. The
NUM005 discrete angular contribution stays visible and is never subtracted.

Point consumers should read canonical world positions; they must not silently
round them back to world FP32 before geometry/contact without a separate declared
and audited conversion. Existing `_NodePort` reads positions directly and its
load response/terminal path is velocity-only, so reuse is plausible without
editing it. This must be verified with represented-state snapshots and exact
request aggregation; it is not inferred from old single-word tests. Unrecorded
interleaved configuration loads remain rejected. A future native contact update
would have to update the represented state and append a typed load event.

Replay p-zero-p, selected-pre candidates and warm restart must compare the full
encoded state and reference fingerprints as well as public X/v. Each trial
rebuilds from original state; no cached low words/reference changes can follow a
rejected trial. Material identity must match through participant/path/physical
ledger. A contact ledger that recognizes only DistanceConstraints or v1 remains
unsupported; parent integration must explicitly dispatch v2 with composite
elastic bounds and original-input joint work. Representation alone grants no
joint material-contact or self-contact acceptance.

## Independent controls and required actual verification

These are proposed future tests, not collected or executed here. Freeze their
exact equations and byte fixtures before granting implementation/verification.
No baseline window is shortened or substituted.

### Hand-derived representation controls

1. **A nonzero correction below one world-FP32 ulp.** Start from world x=1 m,
   apply t=2^-25 m. Single world FP32 rounds back to 1. A uses R=1,d=0 and must
   obtain native d'=2^-25; B begins H=1,L=0 and must retain H'=1,L'=2^-25 under
   its declared algorithm. Canonical X=1+2^-25 is exactly representable in FP64.
   The reference/pair decoded result and all storage residuals are independently
   written binary-rational expectations. This is a codec/update control, not a
   replacement B physical fixture or proof that its gates pass.
2. **Generic translation/scale and cancellation.** Apply the same independently
   written correction after T=2^10 m translation and at length scales s=1/2,1,2:
   x=s*1+T, t=s*2^-25; expected world result is s*(1+2^-25)+T. Reverse the native
   correction and expect return to x, while both absolute error charges remain
   in the journal rather than cancelling. Within a trial the reference/policy
   identity stays fixed; state-word fingerprints change with each update.
   Transformed initial states require new bound reference/split fingerprints.
   Include node permutations and disconnected ownership. These dyadic
   quantities are specified independently of the A/B acceptance scenes.
3. **Finite range/publication limitations.** Encode a nonzero subnormal-required
   word and reject under the unchanged normal-or-zero rule. Exercise overflow,
   exact cancellation, ties-to-even and low-word corruption. For R or H=2^100 m
   with an actual increment of 1 m, RN64 decoding cannot retain that increment;
   require an explicit nonzero decoder residual and budget rejection when it
   exceeds the original normalization. It is not acceptable to claim exact
   publication because the native displacement or low word retained it.

For more general geometric scale tests, use the dimensionally consistent map
X->sX, v->sv, force/gravity->s(force/gravity), rest_distance->s*rest_distance,
distance_compliance unchanged, bending_compliance->bending_compliance/s^2,
masses and h unchanged. Then distance lambda scales by s, bending lambda by s^2,
and kinetic/elastic energy and angular momentum scale by s^2. The derivation is
from the frozen equations, not a measured fit. Keep the original tau/scales and
absolute physical ceilings; similarity does not entitle a scaled trajectory to
pass them. Use codec/row algebra controls for unconditional exact expectations;
any additional accepting physical scale window needs its own prior bound. The
already frozen mass-scale and proper-transform windows remain mandatory.

### Version, provenance and negative controls

- Forge R, reference edge, high word, low word, decoder id, encoding residual,
  gradient, native t, lambda, raw theta, scalar parameter, status or launch count;
  reject even where the final decoded X matches.
- Drop/reorder/duplicate a distance or bending block, row, sidecar or conversion;
  reuse a prior trial/reference, mix mode/pin/version, or relabel a derived view
  as raw native output; reject with no prefix publication.
- Try a raw45 v1 reader on v2 coordinates; reject, not reinterpret. Serialize and
  deserialize typed evidence and require complete byte/fingerprint equality.
- Inject late invalid hinge, unknown continuous proof, exhausted enclosure,
  wrong reconstruction, final hidden mutation and later-substep/terminal/
  diagnostics/close failures. Check represented arrays, lambda vectors, world
  state, history, input and receipts rollback; retry must equal a fresh rebuild.
- Prohibit host B1/distance solver calls, host gradient/angle feeding native state,
  host quotient replacing reconstruction, silent reference recentering and
  low-word carry across trials. Keep legitimate independent audit calls enabled.
- For B, explicitly test compiled rounding/canonicalization and normal-range
  TwoSum premises in each actual mode, using independent rational expectations.
  A passing interpreter is not evidence of CUDA/C++ arithmetic behavior.

### Actual mode obligations and retained consumers

Under a new sole serial execution grant, require the full revised represented
suite in actual interpret and CUDA, including all A/B h grids/windows/two
rebuilds, signed perturbations, mass scales, proper transform, disconnected
copies, explicit external replay, terminal-once, all physical/path limits and
transaction controls. Require the literal A first-sweep independent oracle and
actual operation receipts in both modes. Preserve existing world-v1 primitive
coverage separately; do not relabel its five passing cases as v2 acceptance.

The bounded actual cpu_cpp smoke must exercise the new representation's native
prediction, one complete ordered distance+bending sweep, raw wire/decode and one
reconstruction against an independently bounded case, plus finite/rollback
controls appropriate to that bounded smoke. It is not a C++ full-window claim or
permission for lifetime/stress runs. The originally pending v1 bending smoke
also remains pending until parent explicitly resolves the revised test matrix.

Retain these exact direct-consumer nodes, with original decorators/modes:

- Host: `tests/test_material_ledger.py::test_angular_loss_is_reported_as_material_discrete_contribution`.
- Host: `tests/test_initial_robot_material.py::test_fixed_angular_budget_rejects_coarse_rotating_material`.
- Host: `tests/test_load_replay.py::test_p_zero_p_uses_original_time_state_and_inputs`.
- Host once: `tests/test_load_replay.py::test_actual_xpbd_hidden_multipliers_are_rebuilt_for_each_trial` (no tyx_mode; defaults to NumPy).
- Actual interpret/CUDA: `tests/test_tyx_material_path.py::test_actual_rotating_edge_keeps_original_discrete_loss`.
- Actual interpret/CUDA: `tests/test_tyx_participants.py::test_cloth_workspace_uses_ordered_constraints_and_device_reconstruction`.
- Actual interpret/CUDA: `tests/test_tyx_participants.py::test_node_position_units_reconstruct_only_once_and_reject_bad_state`.

These unchanged nodes establish old-consumer regression evidence. New represented
consumer controls are additionally required; those old nodes do not exercise v2.
Future source/dependency impact determines what is rerun. Host75 at 72bdbcb remains
valid historical evidence for its unchanged dependency closure; it cannot be
reported as a new-head pass or as coverage of a v2 decoder/ledger. In particular,
new represented path/codec host tests are mandatory even if the old host solver
and bounded-angle library are unchanged. Parent-owned dispatch/serializer changes
would broaden the affected test selection before execution.

## Minimum proposed implementation ownership, not permission to edit

Prefer additive v2 types/reader/kernel/adapter so the old B1, distance and current
host paths remain untouched. For option A the minimum scoped set is:

| Action and exact path | Owner and responsibility |
| --- | --- |
| New `src/phymeo/constraints/material_representation.py` | B2 lane: frozen representation policy, v2 typed boundaries/path/batches, strict ownership/schema/shape semantics |
| New `src/phymeo/backends/_tyx_represented_material_kernels.py` | B2 lane: actual FP32 displacement prediction, ordered represented distance/bending and reconstruction; raw evidence/status; B pair helpers only if B is selected |
| New `src/phymeo/backends/tyx_represented_material.py` | B2 lane: deterministic input codec, native adapter and represented workspace, private encoded lifecycle and declared decoder; no host solver |
| New `src/phymeo/validation/material_representation.py` | B2 lane: independent codec/decode checking, versioned world row reader, represented path ledger and ordered world-work diagnostics; reuse unchanged bounds/domain algebra |
| Change `src/phymeo/backends/tyx_composite_cloth.py` | B2 lane only after a grant: explicit representation option/signature and workspace dispatch; v1 default/semantics preserved |
| New `tests/test_material_representation.py` | B2 lane: hand-derived input/decode/update expectations, scale/translation, ownership, byte serialization and finite limits |
| New `tests/test_composite_representation_path.py` | B2 lane: independent world accounting, complete wire/identity/negative/domain/transaction controls |
| New `tests/test_tyx_represented_composite_material.py` | B2 lane: complete actual-mode matrix and independent existing physical equations for v2; no reductions of frozen windows |
| New `docs/development/p65-b2-native-representation-report.md` | B2 lane: new execution status/receipts only when authorized; retain the original failed v1 reports |

Option B uses the same ownership paths, with pair primitives/sidecars and their
additional controls; it does not require editing `_tyx_numeric.py` or the pinned
backend. The two alternatives must not both be implemented speculatively. A
concrete API-factory injection seam for reusing the frozen fixture/equation
helpers needs parent approval before any existing test-helper edit; copying
physical assertions into a weakened substitute suite is not acceptable.

The represented workspace can override native `_predict`, `position_sweep`,
`_reconstruct`, `_path` and ledger construction while inheriting the existing
outer participant/result/energy lifecycle. It must not feed fake v1 frames into
the old native reader. The new ledger may share the unchanged exact angle/domain
primitives; independent row extraction and identical accounting obligations need
parity/negative tests. No change to `constraints/material_trace.py`,
`solvers/composite_cloth.py`, `validation/composite_material_path.py`,
`validation/composite_material_energy.py` or B1 is required by this additive
proposal. If actual implementation exposes a necessary shared seam change,
report it before expanding this minimum allowlist.

Parent exclusively owns any changes to these existing shared seams:

- `src/phymeo/solvers/initial_robot_material_contact.py` and
  `src/phymeo/solvers/_material_contact_balance.py`: explicit composite/v2 identity
  dispatch, original-input joint work and elastic/angular integration.
- `src/phymeo/runtime/load_replay.py`, `src/phymeo/runtime/program.py`,
  `src/phymeo/core/scene.py`, `src/phymeo/core/state.py`: only if new hidden-state
  ownership, signature or diagnostic serialization cannot fit existing protocols.
- `src/phymeo/backends/tyx_participants.py`,
  `src/phymeo/coupling/candidate_audit.py`,
  `src/phymeo/solvers/_finite_contact_replay.py`,
  `src/phymeo/solvers/_thick_contact_replay.py`: only if an actual downstream seam
  would truncate/mutate represented world state or lose selected-pre provenance.
- Shared specs/plans/queue, exports, old tests and precision policy documentation.
  No lane edit to old distance/math kernels, dependency pins or workflows.

## Decisions required before a frozen implementation contract

1. Ratify or reject the public precision revision. Recommend explicit option A
   with immutable per-node initial references and no adaptive rebase. B is the
   alternative, not an implicit fallback. Neither preserves single-FP32 world
   publication; both preserve actual FP32 native operations.
2. Freeze the canonical decoder and exact-difference/split codec, its rounding
   and normal-range rules, signed-zero/serialization conventions and finite
   resource caps. Decide the conservative charge locations for every decoder
   residual; their bound must dominate the original world update residual and
   accumulate without cancellation. No tolerance change follows from this.
3. For A, freeze directed reference-vector ownership/order and native edge
   evaluation order; establish that independent world angle/gradient/equation
   errors cover every converted geometric input. The proposal does not prove
   that those errors fit every frozen window.
4. If B is selected, establish the actual pinned backend's rounding/temporary/
   contraction premises or use observed residual bounds that fail closed without
   an error-free claim. Freeze canonicalization and pair subtraction/collapse;
   do not promise universal ~48-bit accuracy or subnormal support.
5. Ratify schema v2 raw/derived separation, complete event sequence and operation
   receipts, plus the explicit factory seam needed to run the original physical
   acceptance equations on the represented participant. Resolve the bounded C++
   selection without silently removing any original pending obligation.
6. Parent must decide shared material/contact dispatch and physical-state
   precision capability wording. Until that integration is independently
   accepted, standalone represented material validation cannot advertise joint
   native/composite/contact completion.
7. Full-window feasibility remains an empirical and certified-audit obligation,
   not a conclusion from the storage-scale argument. A new failure in geometry,
   lambda, representation, reconstruction or physical budgets requires original
   evidence and adjudication; it does not authorize a fallback, changed h,
   compensated accounting, altered fixtures or budget relaxation.

Documentation authoring ends here. **CODE_DESIGN_PROPOSAL_READY**; the execution
slot stays released and B2 native/composite delivery stays unaccepted. No
representation is implemented or authorized by this proposal alone.
