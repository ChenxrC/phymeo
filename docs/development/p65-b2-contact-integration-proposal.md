# B2 typed internal-contact accounting proposal

Status: **PROPOSAL_NOT_FROZEN**. Documentation only. No imports, numerical probes,
tests, lint, builds, helpers or reviews ran. This proposes an accounting seam and
two independently derived candidate success controls; automatic nonlinear
convergence and complete native delivery/path certification remain explicit
proof obligations, not claimed passes. Parent alone owns future shared edits.

Authority: brief4434638c25b2bcfe807eb8a89bf8d59a6240d03c, read in the absolute parent
integration checkout without merging. R7 reportdc9be828/product40e5090, its accepted
parent merge09867484684643a2440d20f7a9741c234eedd4ee and original failed v1 evidence
are unchanged. The public point adapter's721b5c162f1f7e42dd618c70378bba098af0698e
acceptance is historical, not a new result here. Parent self-response R2 at174e2ab
informs actual-delivery accounting but is not silently generalized to robot contact.
Only this new document is written in the child worktree.

## 1. Alternatives and recommendation

Recommend **typed material/path/elastic dispatch in the existing parent contact
boundary**, using owned binding and certificate types. Retain the actual public
participants, original-state replay, synchronous contact iterations, complete
selected-pre terminal freeze, CCD, history and rollback. No duplicate material
solver, new contact algorithm or case-specific program is necessary.

An additive CompositeThickContactProgram coordinator could keep the legacy public
entry point entirely unchanged and delegate to the current thick workspace.
However it still needs R7-aware replay and terminal publication, typed elastic
energy and two-origin input binding. Copying the workspace duplicates line search,
anchor birth, terminal freeze and rejection behavior. A delegating wrapper with
narrow hooks is credible if parent wants a separate support boundary; it is not
less shared implementation. Prefer explicit typed dispatch and unchanged legacy
defaults. A distinct contact certificate is useful; a second solver is not.

## 2. Actual call sites and blockers

References describe the parent source read for this proposal, not future lines.

| Parent path/range under src/phymeo | Concrete issue or change |
| --- | --- |
| solvers/dynamic_robot_thick_contact.py:84-120,160-174 | Only DistanceConstraints/None are owned today. Explicit composite binding, copied policy/representation and signature identity are required. |
| solvers/_thick_contact_workspace.py:63-80 | Constructs MaterialContactBalance and MaterialPathLedger unconditionally; needs typed audit/elastic selection. |
| solvers/_finite_contact_replay.py:77-149 | Independently requires single-FP32 particle world storage, passes object/Fraction requests, and consumes max_residual. All three differ from R7/composite contracts. |
| solvers/_thick_contact_workspace.py:192-227,299-356,382-406 | Preserve convergence, full selected-pre identity, original-input physical balance and paired phase moments. |
| solvers/_material_contact_balance.py:23-36,38-97,108-129 | Original state/input ownership and joint algebra already exist; distance-only elastic() must dispatch to bounded composite energy. |
| runtime/finite_contact_terminal.py:9-23,96-136 | Must forward an explicit publication policy and complete pre/post participant evidence. |
| validation/finite_contact_terminal.py:31-42,149-185 | A second independent all-domain FP32-world guard rejects R7 even if replay's guard is changed. Preserve velocity/native-response/configuration guards. |
| validation/composite_material_path.py:13-50,223-395 | Typed host path plus complete conservative residual metadata. |
| validation/material_representation.py:211-263,541-826 | Reference identity binds local workspace inputs; global original inputs need an explicit outer binding, not a forged reference hash. |
| solvers/composite_cloth.py:103-139,329-359; backends/tyx_represented_material.py:531-570 | Standalone physical_audit is against effective replay inputs; retain as local evidence, never as the joint verdict. |
| coupling/composite_cloth.py:36-76; backends/tyx_composite_cloth.py:66-108 | Public participant and terminal application already exist. No unrecorded interleaved position load is added. |

A gate-only edit is insufficient:

- Composite readers require FP64 expected-request arrays, whereas current replay
  passes exact object/Fraction mapped requests. Casting cannot erase that ideal.
- Composite readers expose max_equation_error/cumulative_projection_error, not a
  maximum of every checked equation/lambda/angle term. Do not rename one field,
  synthesize max_residual=tau or bypass the tau/8 refinement law.
- R7 reader reference_id includes its constructor input fingerprint. A workspace
  built by replay binds Feff there. A new reader constructed with global F0 will
  disagree for nonzero held replay even with identical R bytes.
- Both replay and terminal single-FP32-world checks must be replaced only for a
  declared versioned R7 domain, never skipped globally.
- PointParticipantProgram (runtime/point_participant_program.py:32-85,225-272)
  supports particles only. It supplies robot-free staging/terminal batching, not
  robot energy or a robot+cloth base. For this seam use existing
  ArticulatedCouplingProgram/TyxArticulatedCouplingProgram with empty old sphere
  contacts and actual robot/composite participants. Do not expand the point
  adapter's accepted role implicitly.

## 3. Proposed public contracts and schema ownership

Names below are proposals, not frozen APIs.

```python
@dataclass(frozen=True)
class CompositeContactMaterialBinding:
    material: ClothMaterial
    audit_settings: CompositeMaterialAuditSettings
    representation: NativeCompositeRepresentation | None = None
    schema: str = "composite-contact-material-binding-v1"
```

DynamicRobotThickContactProgram keeps its existing signature. Values in materials
additionally accept this explicit owned type. Existing None/DistanceConstraints
normalization and signatures remain unchanged. Bare ClothMaterial rejects. The
binding must match the participant's model/material/policy/backend/representation
signature; a trace's shape is not authority to select a reader. Composite-host
requires numpy and representation=None. New native composite contact requires
explicit R7. Standalone native v1 defaults and original failure remain callable;
no automatic fallback or migration occurs.

New validation/contact_material.py supplies an audit object, not a solver:

```python
make_contact_material_audit(binding, *, domain, model, initial,
    original_inputs, h, settings, context, participant_signature)
# methods on the returned original-state-bound object:
require_pre(candidate, effective_inputs, exact_held_impulses,
    participant_diagnostics, *, sweeps) -> ContactMaterialCertificate
elastic_interval(position) -> Interval
publication_binding(pre_state, pre_diagnostics) -> MaterialPublicationBinding
```

The immutable new contact certificate binds global original and local solver
inputs separately; domain/model/material/policy/representation; original state/h;
complete typed path and affine-domain certificate; exact/requested/effective/native
transfer boundaries; and conservative normalized residuals. It is neither a new
raw18/raw45 meaning nor CompositeExternalImpulses.

Additive terminal keywords, with None preserving existing behavior/signatures:

```python
FiniteContactTerminalProgram(base_program, requests, *, registry,
    reference_angular_speed=1.0, material_publications=None)
FiniteContactTerminalLedger(scene, before, requests, settings, context=BackendContext(), *,
    reference_angular_speed=1.0, registry=None, history_fields=(),
    material_publications=None, pre_diagnostics=None)
ledger.require(after, *, post_diagnostics=None)
```

material_publications is an owned exact domain map selected at program binding,
not inferred from a submitted trace. For R7 validate the pre decode and require
unchanged exact R/d/d_pred/lambda/path/reference identities and bitwise world
configuration after terminal. Actual FP32 velocities, original request mapping,
once-only application and all non-requested domains remain checked. Missing or
extra policy/evidence rejects. Default require(after) remains valid for old users.

Recommend an additive `certified_residuals` return mapping in both composite
readers: accumulate every exact _error result and all accumulated projection,
constitutive-equation/lambda and angle bounds under original scales; round the
maximum outward. Tracking is invocation-local and transactional. Existing keys,
paths, equations and thresholds remain. The contact certificate combines these
with transfer errors. Neither an incomplete field nor fabricated tau is an
acceptable measured maximum. These reader changes are parent-owned and trigger
fresh changed-dependency validation.

| Binding | Reader/schema | Publication | Elastic energy |
| --- | --- | --- | --- |
| None | old MaterialPathLedger no-material branch | old host/native meaning | zero |
| DistanceConstraints | old distance-v1 | old host FP64/native promoted FP32 | old elastic() |
| explicit composite numpy | CompositeMaterialPathLedger/composite-v1 | host FP64 | independent elastic_bounds |
| explicit native R7 | RepresentedMaterialPathLedger/path-v2 | RN64(q(R)+q(d)); actual FP32 v | same world elastic_bounds |

An empty composite remains a composite identity. Material identity includes both
ordered laws, units and parameters. No v1 relabeling, fake NumPy evidence, host
solver/gradient repair or world-FP32 conversion is allowed. R7 normal-or-zero and
4096-bit codec bounds, directed vectors, actual t/geometry, raw displacement
columns, alpha/lambda receipts and exact serialization remain unchanged.

## 4. Original versus effective inputs and delivered transfer

Let F0 be immutable global external input; P the exact mapped nodal held impulse
from actual stored mapping blocks and world proposal; P_req the submitted FP64
request; Feff the actual replay input. Bind all four. Independently check
w*|h*(q(Feff)-q(F0))-P|/Vref under unchanged contact tolerance. Retain separate
noncancelling |P_req-P| and |h*(q(Feff)-q(F0))-P_req| terms. Native force conversion
already charged by the local path is not charged again under a renamed event.
Independent ideal-map error remains separately visible. Fraction(float(P)) is
not the exact mapped ideal.

Preserve existing R7 identities by constructing the **local solver-path reader**
from Feff and calling require(pre,Feff,zeros_fp64,diagnostics,sweeps=...). Label
this origin local_solver_inputs in the outer certificate. The outer independently
binds F0/original state/P and performs the exact transfer checks above. Joint
energy always uses F0. This is explicit composition of two audits, not treating
Feff as global original or authorizing work credit. A dual-input reader API is a
possible alternative contract revision; do not combine both conventions silently.

Every held/rebuild/zero/retry starts from original state/history, fresh d=0 and
lambdas=0. Equal R bytes can legitimately have different reference identities
when Feff differs. Terminal-only replay must match the complete selected pre
state, typed path/raw wire, diagnostics, stage metrics/counts, history and effective
inputs, as require_selected_pre already requires. No rebase or hidden carry.

Parent S2 R2 adds an important distinction: requested owner pressure is not an
exact measured native pressure. For particle held delivery use exact
ell_H=h*(q(Feff)-q(F0)), with observed predictor uncertainty
 e_H=|q(vpred)-q(v0)-h*(g+q(w)*q(Feff))|/q(w).
For terminal use ell_T=(q(vpost)-q(v_selected_pre))/q(w), not the nominal request.
Retain all original request/native input data and full selected-pre binding.
These quantities certify delivery only; they do not supply external work credit.

Do not transplant S2's full-rank inverse blindly. Its A=B M^-1 B.T and
L=A^-1 B M^-1 require positive rank/LDL and strict kappa_inf<1e12. Existing thick
contact may have redundant owners: six 3D anchors on two adjacent triangles over
four nodes plus one prismatic DOF yield18 rows but at most13 generalized DOFs,
so A is singular. Even normal-only duplicate vertex rows can be dependent.
Dropping owners, regularization, fitted pressure or asserting unique native
pressure is forbidden. For robot+particle delivery, a generalized mass/map
witness must include M_R and original force/effort/bias uncertainty, and any
nonunique owner feasible-set/capacity treatment needs a separate parent ruling.

Where a rank-admitted decomposition is valid, R2's p_eff=L ell, remainder
r=ell-B.T p_eff and |L|e / |I-B.T L|e bounds remain certification quantities,
never changes to state or load. Use safe lower normal capacity and conservative
upper tangent/slip bounds over the full uncertainty interval; never clip an
unresolved negative lower pressure to zero or spend capacity twice. Charge the
remainder at every actual/ideal extreme and in resultant/couple/work gates; its
orthogonality does not make it harmless. Our mu=0 controls avoid friction-disk
spending but do not prove a general normal-delivery or nonunique-owner theorem.
Until parent freezes that rule, the typed seam is an accounting proposal, not
native friction/contact acceptance. Legacy contact behavior is not rewritten by
this document.

## 5. Original-input joint energy and internal work

For each particle domain d, evaluate actual decoded world x and actual stored v:

E_d = sum_i m_i |v_i|^2/2 - sum_i m_i g_d dot x_i + U_D(x) + U_B(x),
U_D = sum_{alpha_e>0} (|x_i-x_j|-ell_e)^2/(2 alpha_e),
U_B = sum_k (theta_k-rest_k)^2/(2 alpha_b,k).

Distance compliance has units m/N=s^2/kg; bending compliance1/J. Bending C is
dimensionless radians, gradient1/m, lambda kg*m^2, delta-x=w*lambda*gradient.
Hard distance retains its old zero stored-elastic contribution and constraint
meaning; bending compliance stays strictly positive.

Use independent bounded theta/gradient from bending_bounds/elastic_bounds, not
Fraction of float atan2 or producer C. Preserve sqrt256/atan64, supported smooth
branch, each numerical-row segment certificate and full physical affine segment
certificate. Exact endpoints alone do not prove continuous safety. Unknown cut,
degeneracy or enclosure/resource exhaustion rejects. No oracle feeds the solver.

Robot contribution is K_R^0(v)=v^T M0 v/2 and original frozen free generalized
force f_R^0=free_integral/h, exactly the existing FrozenRobotLedger convention.
With bounded energies, the joint defect interval is

D_E = [K_R^0(v1)+sum_d E_d(x1,v1)]
    - [K_R^0(v0)+sum_d E_d(x0,v0)]
    - f_R^0 dot(q1-q0) - sum_d F0_d dot(x1_d-x0_d).

Require max(0,D_E.upper)<=beta_E. Never subtract internal held/terminal work,
R2 remainder work, or Feff contact work as an external credit. The selected
all-prismatic domain and one-axis controls have constant mass; no inference to
true M(q1) energy, arbitrary revolute robots or unmodeled guide work is made.

Preserve existing material enclosure requirements. Additionally propose contact
final joint-defect interval width<=min(existing applicable energy ceiling,
beta_E/4096) per substep,
including every initial/final energy enclosure in that exact width. This
stricter allocation needs parent adjudication,
including the dimensional1 J reference if retaining the old numeric tau/64
expression. It is not a larger physical budget or automatic resource escalation.
A32-step window consumes at most beta_E/128 in these widths. Unachievable
certification rejects unknown; rounded display floats never decide acceptance.

Held work remains a diagnostic of actual represented input delivery:

W_H_delivered = sum_d (ell_H,d/h) dot(x1_d-x0_d)
              + (ell_H,R/h) dot(q1-q0).

Also retain W_H_mapped formed from exact stored-map P and its difference from
W_H_delivered and independently ideal-map work. The map is frozen initially,
so endpoint generalized work need not equal final material-point displacement
work for deforming normal-offset geometry. Nothing authorizes pretending that
difference cancels. The physical ledger uses actual states and original external
inputs regardless of the diagnostic decomposition.

At selected pre configuration, terminal generalized/nodal impulse L satisfies

Delta K_T = sum_d [L_d dot v_pre,d + (1/2) sum_i w_i |L_di|^2]
          + L_R dot v_pre,R + (1/2) L_R^T M1^-1 L_R + epsilon_native.

Aggregate on physical nodes and robot DOFs before squaring; preserve every
shared-node/robot cross term. Distinguish mapped/requested/FP32-input/actually
delivered impulses. Existing actual response and terminal energy checks bound
epsilon_native, nonpositive contact energy and input conversion. The measured
terminal momentum identity is diagnostic, not permission to overwrite requests
or apply a repaired load. For frozen-M0 energy comparison retain supported
prismatic M0/M1 consistency; do not assume it for general robots. Terminal changes
no configuration/elastic energy and its transfer is internal to the joint D_E.

CompositeExternalImpulses remains external-purpose only and retains its
internal_boundary rejection. Never issue one for these paired loads. Existing
workspace physical_audit against Feff may remain under an explicit
local-effective-input label and retain its original gate. Its external-work
subtraction and standalone1e-3 limits never decide the stricter joint gate.
If that local diagnostic rejects a candidate, preserve the failure; bypassing
or changing it requires a separate parent decision.

## 6. Discrete angular accounting and thick offset couple

Preserve the particle-domain NUM005 defect:

A_d = sum_i [x1_i cross(m_i*v1_i)-x0_i cross(m_i*v0_i)]
    - sum_i x0_i cross[h*(F0_i+m_i*g)+P_i]
    - sum_i x1_i cross L_i.

Here P/L remain the bound exact mapped requests for the original discrete
moment contract; report actual delivered differences separately and conservatively
charge them, not a redefinition that hides missing delivery. Require the sum
of componentwise absolute defects across domains<=beta_L. Rotational invariance
of ideal bending force does not erase discrete integration/projection angular
loss. No cancellation across domains, components or time is allowed.

For y=sum b_i*x_i+s*n(x), J_i=b_i I+s*D_i n. Independently verify
sum_i J_i^T p=p;
sum_i x_i cross(J_i^T p)=y cross p;
sum_i v_i dot(J_i^T p)=p dot(sum_i J_i v_i).
Robot reaction is -J_R^T p. Keep the paired held/terminal moment discrepancy
sum_anchor |(y_cloth-y_robot) cross p| under its existing physical budget.
It does not disappear merely because linear impulses are internal.

Explicit independent hostile oracle: a=(0,0,0), b=(1,0,0), c=(0,1,0), s=-t.
For tangent p=(px,py,0), the normal-derivative term contributes nodal z impulses
(-t*(px+py), t*px, t*py). Its resultant is zero, but its moment is
(t*py,-t*px,0)=s*ez cross p. Barycentric-only mapping can therefore pass resultant
while failing moment and virtual power. Use nonzero tangent impulse in this
control; normal-only success scenarios cannot expose the omission by themselves.

Prismatic guides can supply unmeasured world torque. Particle discrete defects
and paired-couple checks are not a total free-general-robot angular theorem.
That stronger closure remains an explicit separate obligation.

For a window sum max(0,D_E.upper) at every step and sum each component/domain
absolute angular bound, with the same beta_E/beta_L ceilings for the full window.
Retain all per-step/contact/terminal/mapping/CCD gates. Do not use net endpoint
energy or opposite angular errors as a substitute. Internal residual work is
never added to external work to forgive creation.

## 7. Two independent candidate success controls

These are new candidate acceptance data, not replacements for any old test.
They start the actual contact solve from zero; no impulse is handed to the
solver or copied from its output. Require nonzero bending C/multiplier and
nonzero actual coupled transfer in saved evidence. The following physical
bounds are independent and rigorous conditional on explicit test predicates.
They do not prove automatic fixed-point convergence or native path success.

### Shared setup and conservative envelope

Coordinates [m]: a=(0,0,0), b=(1,0,0), c=(0,1,0), d=(0,-1,0).
Each mass m=2^-48 kg, inverse mass2^48. Distance edges(a,b),(a,c),(a,d), rest1 m,
compliance2^48 m/N; hinge(a,b,c,d), compliance2^48/J. Declare both prism surfaces
(a,b,c),(b,a,d), half-thickness t=1/64 m. Robot is one prismatic +z link, mass4m,
zero COM, identity orientation and positive inertia diagonal4m kg*m^2. Its patch
covers [-2,2]x[-2,2] in the plane z=-t at q=0. Robot gravity, original effort,
force and torque are zero. Friction coefficient is0.0. Particle force arrays
are zero. Public actual robot/composite participants share one articulated base,
with empty old sphere contacts; no duplicate physical domain or auxiliary solver.

Window T=1/8 s, h=(1/64,1/128,1/256) s; two independent rebuilds at every h.
References are1 m,1 m/s,1 rad. Keep512 iterations, contact tau1e-8 host/2e-5 native,
contact energy/angular ceilings1e-8 host/1e-5 native in J/N*m*s, and all original
B2 proof/representation limits. Standalone1e-3 is not a contact tolerance.

Mass/compliance scaling is transparent analytic homogeneity, not tuning after a
probe: masses/forces/inertias scale by m, compliances by1/m; geometric/velocity
solve equations are invariant, impulses/lambdas/physical energy/angular values
scale by m. Powers of two avoid parameter-conversion loss. These inputs and
main expected loads are well inside native normal range, even with alpha/h^2.
This gives tight-budget accounting controls, not unit-mass coupled acceptance;
finite native arithmetic still requires the full unchanged path certification.

Freeze extra independent predicates for every stored state: |x|_infinity<=2 m,
|v|_infinity<=1 m/s, |q|<=1 m, |v_R|<=1 m/s, |edge_length-1|<=1 m,
|theta-rest|<=1 rad, |g|<=1/8 m/s^2, and each nodal held/terminal impulse component
<=8m N*s. Also require each active world impulse component<=8m N*s and paired
point separation<=4 m. Exact values/certified intervals check these predicates;
a violation rejects the case rather than enlarging the predicate. They are test
oracles, not implementation clamps or generic support heuristics. Preserve all
independent area/angle/segment/CCD proofs; coordinate caps do not replace them.

Then K_total<=8m J, U_total<=2m J, |G|<=m J. Since original forces/robot work are
zero, positive energy defect per step<=12m J;32 steps give<2^-39 J plus the
separately allocated enclosure width. Each angular component/node is bounded
by8m for endpoint difference,36m for original/held moment and32m for terminal
moment:76m N*m*s. Four nodes/three components yield912m per step and a32-step
sum<2^-33 N*m*s. A crude six-anchor, two-phase paired-moment window bound is
<2^-31 N*m*s under the stated predicates. All remain below1e-8 even with mass
factors1/2 and2. If geometry discovers more than six owners, use the independently
computed aggregate moment bound rather than assuming the six-anchor estimate.
No signed cancellation or external internal-work credit enters these bounds.

The tiny mass makes these accounting gates easy to certify; strong normalized
contact/path/mapping/independent-oracle checks remain necessary to exclude an
incorrect solve. These controls cannot stand in for unit-mass physical fidelity.

### SC1: closed support with a prestrained hinge

Initial shared positions, all velocities zero; particle gravity(0,0,-1/8).
Rest angle r=2^-10 rad gives theta0=0 and C0=-r!=0. Patch touches both lower
faces. Nonzero bending force competes with a compressive body load on a dynamic
support. This exercises active material even if final contact suppresses folding.

Independent zero-stiffness reference: four particles supported by robot mass4m
have common acceleration g/2. Per physical node held P=-m*h*g/2 along +z;
robot reaction is the negative sum, terminal impulse zero for uniform closed
motion. Shared-vertex anchor splitting is not uniquely prescribed; assert only
physical-node aggregate quantities for this reference.

At the flat oriented hinge the independently derived gradient is
(2ez,0,-ez,-ez), so force is m*r*(2ez,0,-ez,-ez). Maximum bending acceleration2r
is strictly below1/8, leaving an initial compression margin. This proves active
opposed material force initially. It does not turn the zero-stiffness reference
into the exact nonlinear XPBD answer. SC1 requires nonzero saved C/lambda,
positive held transfer, the full unchanged gate set and the explicit envelope
through the complete window. Automatic convergence across all h is unproved.

### SC2: open impact with an active bending-rate perturbation

Translate all four nodes upward by g0=3/1024 m; patch stays at z=-t. Particle
gravity zero, rest angle0; va=vb=vc=(0,0,-1/8) m/s,
vd=(0,0,-1/8+1/64) m/s, robot at rest. Theta0=0 but
Cdot=sum g_i dot v_i=-1/64 rad/s, independently activating the hinge before or
at impact. This uses bending-rate excitation rather than SC1's static prestrain.

Independent unperturbed reference (vd=-1/8): for a remaining impact-step gap g
with0<=g<=h*u, u=1/8, per physical node
P=m*(u-g/h)/2, L=m*g/(2h), P+L=m*u/2.
Final common velocity is -u/2. Initial joint kinetic2m*u^2 drops to m*u^2.
These follow from momentum plus endpoint gap/terminal relative-velocity equations.
At exact endpoint arrival g=h*u, P=0 and L=m*u/2 is a terminal-only impact;
do not eliminate that branch to simplify acceptance.

The specified d-velocity perturbation/material changes individual values. Keep
the formulas only as a zero-perturbation comparison, never an alleged exact
nonlinear oracle. SC2 requires positive selected terminal delivery, nonzero
bending rows, complete original gates and the explicit envelope. A complete
independent interval solution or contraction proof for its nonlinear contact
fixed point and its actual native cumulative bound is still missing. This is
a real pre-freeze decision, not permission to probe for easier parameters.

For both: retain both rebuilds and mass factors1/2,2. Independently perturb SC1
rest to r plus/minus2^-12 rad and SC2 d-relative speed to1/64 plus/minus2^-12 m/s.
Use consistent node/triangle permutations, the proper cyclic axis permutation
Q(x,y,z)=(z,x,y), and translation(1/4,-1/4,1/4) m. Transform robot placement/axis,
patch geometry, gravity, velocities and forces consistently; neither a reference
frame nor a contact impulse is tuned to an observed result. Keep original legacy
h/mesh/lifecycle controls unchanged. A refined-mesh material-transfer rule and
a unit-mass tight-
budget coupled success case remain separate unsolved extensions. The redundant
owner/rank issue from section4 also applies; it cannot be hidden by mu=0.

## 8. Hostile and backward-compatible controls

1. Full None/distance/composite-host/R7 dispatch matrix. Reject wrong law order,
   material/policy/backend/representation/domain/h/original/effective input,
   stale reader, unknown schema and extra/missing bindings. Preserve legacy
   distance types and the original native v1 default/rejection evidence.
2. Mutate R/d while rounded world is unchanged; remove native t/geometry/operation
   or an entire sweep; alter d_pred/lambda; relabel raw v1 as v2; replace required
   affine proof by endpoint guards. All reject without repairing world or trace.
3. Forge an external receipt that would cover positive internal joint energy.
   Pure ledger negative: four masses2^-48, zero original force/gravity, initial
   rest, then synthetic x unchanged and each v=(2^16,0,0). K=4*m*(2^16)^2/2
   =2^-15 J>1e-5. A receipt claiming that work must not make internal contact
   pass. This is a hostile ledger input, not a physical trajectory/handoff.
4. Opposite defects across steps/domains cannot cancel. A single excessive
   positive energy or component/domain angular defect rejects even with net zero.
5. Tangential offset oracle from section6: omit s*Dn, one endpoint, robot reaction
   or shared-node cross terms; substitute centroid/zero offset or cancel only the
   resultant. Independent map/moment/power/response checks must fail, regardless
   of the tiny success-control energy/angular values. Preserve original
   large-impulse and zero-impulse coefficient guards too.
6. Preserve nominal pressure but alter actual held effective input or terminal
   delivered velocity. R2-style uncertainty/capacity and original physical
   ledgers must notice. Include on-range delivery, within-map-ceiling inverse-
   response amplification, off-range residual and open-birth one-phase mutation.
   No unique measured pressure or harmless remainder is inferred.
7. Change complete selected-pre diagnostics/path/iteration count/history/inputs
   while leaving state identical. Require full identity, not state-only equality.
   Duplicate/canceling terminal rows aggregate before squares; second application,
   changed R/d/configuration, wrong M1 or hidden state mutation rejects.
8. Fail on second/third outer substep after an earlier accepted one, at native
   row, path, mapping, joint energy, terminal, CCD, diagnostics or close. Discard
   public/history/input and private encoded state; retry equals fresh; close each
   owned resource once. Unknown proof must not be hidden by an earlier best trial.
9. Existing finite-corner failure/unknown, original lifecycle, h and mesh controls
   remain unchanged. Do not replace them with easier small-mass controls or count
   expected rejection as accepted physical support.

## 9. Proposed exact file ownership and validation selection

All future implementation paths below are **parent-owned**. Naming them is not
permission to edit them in this documentation task. Minimum recommended source
allowlist if the proposed signatures and support domain are frozen:

1. NEW src/phymeo/constraints/contact_material.py: owned binding, certificate and
   versioned publication types.
2. NEW src/phymeo/validation/contact_material.py: typed dispatch, exact original/
   effective/delivery envelope, bounded elastic callback and certificate assembly.
3. src/phymeo/solvers/dynamic_robot_thick_contact.py: explicit material binding and
   signature validation at the section2 ranges.
4. src/phymeo/solvers/_thick_contact_workspace.py: typed audit construction and
   consumption only; keep solve, freeze, contact/CCD/history behavior.
5. src/phymeo/solvers/_finite_contact_replay.py: typed publication, input binding
   and path reader use only; legacy branch remains explicit.
6. src/phymeo/solvers/_material_contact_balance.py: elastic dispatch, interval
   widths and composite report scope, preserving original-input algebra.
7. src/phymeo/runtime/finite_contact_terminal.py: explicit publication and complete
   pre/post evidence forwarding only.
8. src/phymeo/validation/finite_contact_terminal.py: explicit R7 publication rule;
   preserve original request/velocity/robot/configuration checks.
9. src/phymeo/validation/composite_material_path.py: complete conservative residual
   metadata only, with transactional collection.
10. src/phymeo/validation/material_representation.py: same residual metadata only;
    no change to R7 representation/equations/codec/raw rows.

New proposed tests:
- tests/test_contact_material_binding.py: exact dispatch, two-origin load binding,
  residual metadata, energy/angular/credit/publication negatives.
- tests/test_composite_thick_contact.py: independent SC1/SC2 data, original
  equations, full windows/rebuilds/covariance, selected-pre and rollback.
- tests/test_tyx_composite_thick_contact.py: complete actual modes and a bounded
  active-bending/contact C++ node.

Parent additionally owns the frozen design/plan/report. Do not edit R7 kernels,
codec/raw schema, standalone energy ledger, participant solver, point adapter,
original mapping/geometry/friction laws, dependency pin or old expectations for
this minimal dispatch proposal. If actual-delivery/capacity certification needs
more than these paths, freeze that additional parent seam explicitly first.
Do not disguise it as a material-type extension.

Future focused validation, only after a separate sole serial grant:

- Complete new host binding/physical files. Existing
  tests/test_composite_material_path.py and test_composite_representation_path.py
  complete, because both readers change; include original independent bounded
  energy/power and external/internal receipt controls.
- tests/test_thick_contact_mapping.py complete;
  tests/test_thick_robot_transaction.py complete; original thick contact nodes
  test_two_jaws_static_and_low_friction_single_step,
  test_open_impact_full_three_corner_oracle,
  test_distance_material_pre_ledger_and_terminal_post,
  test_reordering_and_shared_ancestor_are_equivalent,
  test_two_jaw_window_halving_and_rebuild and lifecycle/unknown cases under their
  original selection. Do not alter the legacy distance-material fixture's
  explicitly wider historical budget; new composite cases use strict contact
  limits, and neither result is relabeled as the other.
- Complete new interpret then CUDA with --require-tyx and no v2 skip/fallback.
  Existing actual nodes test_actual_three_corner_impact_and_material,
  test_actual_two_jaw_support_and_slip, test_actual_failure_and_all_trials_count
  and selected-pre/finite-terminal consumers directly exercise touched seams.
  Preserve existing lifecycle/h/mesh gates; parent freezes exact decorated node
  selections before execution rather than replacing a host-only test by flags.
- Residual metadata touches R7 dependency closure. Do not attach old acceptance
  to new hashes. Prefer one final same-snapshot complete R7 interpret/CUDA group
  after shared edits, with new host/path checks and appropriate original host
  consumers, not repeated baseline runs during authoring. Parent determines
  changed-dependency breadth; old receipts stay immutable.
- The point adapter is reusable only with unchanged dependency closure. If later
  robot-free composite self-contact is included, add focused public adapter+
  actual composite+terminal tests; do not infer acceptance from its distance-only
  bounded native control. No robot energy is added to that adapter.
- Bounded cpu_cpp: one automatic active-bending/contact step covering prediction,
  ordered material sweeps, held replay, selected terminal, raw wire and original
  audits. No long lifecycle stress/TYX003 closure. Do not promise counts or
  compilation success before collection/execution.

Freeze commands, current source/test/pin/import manifests, fresh basetemps and
first outcomes. Save full same-run states/history, global original and local
Feff inputs, exact/requested/native/delivered impulses, paths/raw bytes, all
joint energy/angular intervals, mapping/CCD/affine proofs and failed receipts.
No extra physics solely for reports. Audit saved evidence, then final source/
evidence hashes, clean tree and process0 release. Parent owns the single later
integrated-stage review; none is dispatched now.

## 10. Decisions and honest remaining proofs

1. Freeze typed dispatch versus an additive wrapper and the public type names.
2. Freeze the explicit local-Feff path plus global-original delivery certificate,
   or deliberately revise reader input identity; never mix the conventions.
3. Require complete conservative residual aggregation. No synthetic tau maximum,
   incomplete renamed field, selected-pre weakening or storage bypass is approved.
4. Freeze explicit versioned publication in both replay and terminal checks,
   including unrequested represented domains and exact hidden-state binding.
5. Adjudicate energy-enclosure width allocation and its dimensional reference.
6. Generalized actual-delivery witness and redundant owner/rank/capacity semantics
   need a parent proof. S2 R2's valid full-rank certification decomposition is
   informative but does not solve singular robot-contact owner sets. No fitted
   inverse, regularization, silently dropped owners or external-credit residual.
7. SC1/SC2 have fixed data and independent physical envelopes. Complete nonlinear
   interval solutions/contraction, automatic convergence, strict native cumulative
   path bounds and actual supported terminal deliveries are not established.
   They are candidate success controls, not ready-made pass promises. If fully
   proved success data are required before implementation, this is real remaining
   mathematical work. No probing/tolerance tuning is authorized by this proposal.
8. Small-mass accounting controls do not certify unit-mass coupled fidelity,
   refined-mesh material transfer, lawful incident folds, general rotating robot
   energy/angular closure or full folded grasp. Those mandatory gates stay open.

Document authoring ends at PROPOSAL_NOT_FROZEN. No implementation or executable
validation result is claimed, and no executable slot was taken.

## 11. Follow-up after parent adjudication (documentation only)

Authority: parent adjudication9ea19b75492b372effd770d4e79a67663eaa2123, read from
the absolute parent tree; no merge, executable work or source/test edit. Sections
1-10 above preserve the original585-line proposal/history and its candidate data.
This follow-up supersedes their undecided architecture status only as follows:
parent selected typed dispatch, both explicit publication guards, local-Feff /
global-F0 binding, and complete conservative residual aggregation. Public names,
additional witness files, physical support and acceptance remain NOT_FROZEN.
S2's existing full-rank R2 and active execution are unchanged.

SC1/SC2 at tiny mass are SUPPLEMENTAL ACCOUNTING ENVELOPES. They are not the two
primary coupled-fidelity success controls. Section16 below states exactly what
is still missing at ordinary mass; no success is inferred from mass scaling.

### 11.1 Exact quotient identities, independently derived

Use m owner-coordinate rows and n physical generalized coordinates. Keep all
three world rows for every owned contact in the actual B, including duplicates.
Restrict this candidate to constant-mass translational coordinates: q has units
m, v has m/s, M has kg, and both generalized and owner impulses have N*s. Let
W=M^-1 be the exact inverse of the admitted symmetric positive frozen mass and
A=B W B.T. B is dimensionless. For any owner vector z,

    z.T A z = (B.T z).T W (B.T z).

Positive definiteness of W proves ker(A)=ker(B.T), hence range(A)=range(B).
For an exact complete basis N of this nullspace, N.T N is positive definite.
Pi_N=N (N.T N)^-1 N.T is symmetric, idempotent and basis-independent; Pi_N B=0.
Pi_R=I-Pi_N projects onto range(B). A Pi_N=Pi_N A=0, and A restricted to Pi_R
is positive definite. These claims require exact rank/span receipts, not merely
B.T N=0, which would also hold for an incomplete or empty asserted basis.

For nonzero rank, s=||A||_inf>0 has units1/kg. Set H=A/s+Pi_N. Split z into its
orthogonal range and null components. Its quadratic form is the sum of a strictly
positive range term and the squared null component, so H is positive definite.
H Pi_N=Pi_N and H Pi_R=A/s. Therefore

    H^-1 Pi_N=Pi_N,
    H^-1(A/s)=(A/s)H^-1=Pi_R,
    L=(1/s)H^-1 B W,
    Pi_N L=0,                 L B.T=Pi_R.

Thus p_eff=L ell is zero-nullspace and Q=B.T L satisfies

    Q^2=Q,       range(Q)=range(B.T),       Q.T W=W Q,
    r=(I-Q)ell,                         B W r=0.

The W-self-adjoint identity follows directly from symmetry of W and H^-1.
It proves that Q is a mass-weighted orthogonal projector in impulse space,
not an owner-selection or physical-update algorithm. In particular,

    ell.T W ell = (B.T p_eff).T W (B.T p_eff) + r.T W r.

The residual term is nonnegative; orthogonality does not make it zero. Terminal
work also contains r dot v_pre. It must remain in physical accounting.

An equivalent independent characterization minimizes
(ell-B.T p).T W (ell-B.T p). Its exact stationary equation is A p=B W ell.
All minimizers differ by a vector in ker(B.T); p_eff is the one with minimum
Euclidean owner norm. On-range ell=B.T p gives p_eff=Pi_R p and r=0. This is
uniqueness of a chosen certificate representative, not of measured owner pressure.

If rank(A)=m, Pi_N=0, H=A/s and L=A^-1 B W exactly, recovering S2 R2. Under
uniform physical mass scaling M->gamma M, gamma>0, W,A,s scale by1/gamma and
H,L,Q stay unchanged. This is an algebraic identity, not physical-budget relief.
No mixed angular/linear coordinate metric follows from it. Orthogonal coordinate
changes preserve the algebra but need not preserve the infinity-norm condition
number; the explicitly frozen sufficient conditioning gate still applies.

### 11.2 Exact illustrative identities and limitations

These scalar restrictions illustrate the algebra; an implementation retains the
full owned 3D rows and all physical columns.

- Duplicate rows B=(1,1).T and one mass M=m0 give
  A=(1/m0)*[[1,1],[1,1]], s=2/m0,
  Pi_N=(1/2)*[[1,-1],[-1,1]], H=I and L=(1/2)*(1,1).T.
  Hence p_eff=(ell/2,ell/2), r=0, and kappa_inf(H)=1. No owner was dropped.
- Opposing normal rows B=(1,-1).T give H=I and
  p_eff=(ell/2,-ell/2). For ell>0 this representative is not nonnegative on both
  scalar normal axes, although p=(ell+t,t), t>=0, has the same delivery and can
  be nonnegative. The certificate rejects the chosen witness; it does not deny
  existence of another physical representation. Adding its self-stress (t,t)
  to p_eff to create capacity would violate the proposed rule.
- With B=(1,0) and M=I, ell=(a,b).T has p_eff=a and r=(0,b).T. Stored relative
  response sees no b, but an independent extremum row C=(0,1) sees W r=b and
  the residual quadratic energy is b^2/2. This is an explicit reason to keep
  ideal-extreme and energy checks, even when B W r=0 exactly.

The same nonnegative pressure distribution is not guaranteed after projection
for different owner rays, phases, friction coefficients or geometry. Repeated
geometric rows also do not license new owners or repeated capacity. Identity,
phase and once-only semantics remain independent admission requirements.

### 11.3 Conditioning, resources and failure obligations

Verify stored exact B/M, owner order, positive mass, complete rank/nullspace,
Pi_N symmetry/idempotence/annihilation, H positivity and all inverse identities
independently of any producer claims. A deterministic rational elimination with
fixed owner/coordinate ordering can produce rank and nullspace witnesses; exact
pivot and span receipts must be checked. No small singular-value threshold,
regularization, fitted inverse, diagonal shift or dropped owner is permitted.

Require positive exact LDL for H and strict
||H||_inf*||H^-1||_inf<10^12, using exact rational row sums. Equality rejects.
This bounds H, not the full physical amplification |L| or every ideal-extreme
map. Independently propagate |L| and reject if the resulting uncertainty exhausts
original phase/map/physical ceilings even when H passes conditioning.

Rank zero is not covered by division by s. A separate empty/unloaded identity
case would need zero delivered/requested loads and all original motion/geometry
checks; otherwise reject. Do not infer positive contact capacity from it.

The parent must freeze finite matrix dimension, pivot/operation and rational-bit
limits for this NEW witness. A4096-bit numerator/denominator ceiling consistent
with the existing exact codec is a possible conservative choice, not permission
to extend that codec or assert that all matrices fit. Limits apply to intermediate
as well as final certificates, with deterministic operation counting and no
unbounded retry. Exhaustion is unknown/reject and publishes nothing. Reusing the
number4096 alone does not supply an algorithmic resource contract. No matrix
witness or condition result has been computed for a physical case here.
## 12. Generalized observed delivery and original-input uncertainty

Use the original state's constant block mass M over particle velocities and all
admitted robot translational velocities. M need not be diagonal on the robot.
Do not replace it by independent per-joint masses. Actual stored values are
interpreted as rationals q(.) in the following equations, not recomputed states.

For each robot generalized coordinate, the original free generalized force is

    f_R0 = effort0 + J_linear(q0).T force0
                    + J_angular(q0).T torque0 - bias(q0,v0).

This is the existing FrozenRobotLedger free_integral/h definition, including its
canonical frozen mass and original bias. For admitted all-translational fixed
orientations the angular Jacobian vanishes; retain the original term/binding
rather than silently replacing the original force model. Particle free force is
m_i*g_i+F0_i. Record every term, canonical mass and exact original-input identity.
The reference here is that declared frozen law, not a proof of arbitrary robot
energy. Any ideal-versus-stored mass/force enclosure remains a separate receipt.

Let ell_H=h*(f_eff-f0). In the existing robot contact replay only effort changes;
force/torque/gravity/bias and the original evaluation configuration must remain
identical, so ell_H,R=h*(effort_eff-effort0). Do not use an undocumented change
of forces/torques as contact delivery. Bind exact requested B0.T p_H, the FP64
load arrays and actual Feff separately, as in section4.

The actual pre-material free predictor gives the observed momentum residual

    delta_H = M*(q(v_pred)-q(v0)) - h*f0 - ell_H,
    e_H = abs(delta_H)                         (componentwise, N*s).

The exact nominal predictor is only an audit equation,
v0+W*(h*f0+ell_H); it is never published or substituted for actual v_pred.
For a particle block this reduces exactly to
abs(v_pred-v0-h*(g+w*Feff))/w, the R2 expression. For the robot it includes native
mass solve, original effort/bias, force conversion and predictor rounding through
the ACTUAL output. No diagonal approximation or particle formula is used there.
Equivalently compute the exact velocity residual and multiply by M; a bound
|M|*abs(velocity_residual) is conservative but may be looser than abs(delta_H).

The signed delta_H is retained. The nominal ell_H plus its symmetric e_H box is
conservative delivery evidence; using the observation does not authorize force
repair or replacement of nominal requests. Material projection/reconstruction
must not contaminate this measurement: use the actual pre-material predictor,
not final cloth velocity containing bending impulses. For a robot-only free
participant with no other constraints, final free-trial v may serve only when
its phase/provenance proves it is that predictor. If current diagnostics do not
expose or prove this boundary, that is an additional parent-owned interface gap.
Do not infer v_pred from positions, ideal equations or an accepted final state.

Independently retain robot motion error
q(q_pred)-q(q0)-h*q(v_pred), with the original h*Vref normalization, and every R7
particle prediction/decode/storage receipt. It cannot be charged as pressure
and then silently removed from configuration or point-motion checks.

At terminal the complete selected-pre identity is fixed and no material step
intervenes. Actual observed delivery is

    ell_T = M*(q(v_post)-q(v_selected_pre)).

Use the full constant robot mass, matching the admitted selected-configuration
M1, and the particle masses. Compare with original B1.T p_T and native input
receipts without substituting either for ell_T. In the declared constant-mass
domain M1=M0 must be justified by the model, not inferred from close floats.
For variable mass, mixed revolute coordinates or changed configuration, this
follow-up provides no terminal momentum/energy extension.

If an independent ideal frozen model has errors |M_star-M|<=E_M and
|f0_star-f0|<=e_f, the same observed velocity implies the additional conservative
momentum error E_M*abs(v_pred-v0)+h*e_f, plus any separately bound ideal delivery
error. At terminal use E_M*abs(v_post-v_pre). Do not assert these errors are zero
merely because stored M passed LDL. Existing independent mass/map requirements
are retained; the proposed ordinary single-axis model has an analytic constant
mass, but no physical run is claimed. Bit/interval exhaustion rejects.

For each phase propagate its complete admitted impulse uncertainty e through

    p_center=L*ell,     e_p=abs(L)*e,
    r_center=(I-Q)*ell, e_r=abs(I-Q)*e.

Save the joint linear dependence on the SAME delta-ell as well as the boxes.
Do not spend one discrepancy independently as two favorable pressure allowances.
Every uncertainty event has one source ID and a list of uses (delivery, cone,
map, energy, angular). Multiple conservative uses do not create multiple budgets.
No native conversion/storage charge from R7 is removed or counted as new credit.

## 13. A sufficient cone and KKT enclosure rule

The following is a conservative candidate audit of the quotient witness. It is
not a new friction solver, proof of owner uniqueness or change to S2. Retain each
owner's exact normal ray n, its phase and actual final relative velocity. Let

    lambda = n dot p / sqrt(n dot n),
    T = p - n*(n dot p)/(n dot n),
    U = u - n*(n dot u)/(n dot n).

Use independently outward bounded square roots and all pressure/velocity/map
uncertainty to obtain lambda in [lambda_lo,lambda_hi],
||T||<=t_hi, ||U|| in [u_lo,u_hi], and T dot U in [w_lo,w_hi]. Never normalize by
an unproved nonzero n. Keep exact normal-ray/phase identity, not a rounded guessed
unit normal. Interval dependency loss is conservative and may reject a valid
contact; tightening cannot change a physical threshold or use solver answers.

A directly sufficient strict sign rule is lambda_lo>=0 for each phase. A negative
lower bound is uncertainty/rejection, not max(0,lambda_lo) capacity. This can
exclude native near-zero-pressure cases; that limitation must be accepted or
explicitly resolved by parent under original gates before implementation.
Unresolved upper pressure with zero safe capacity still requires the full audit.

Initially closed owners use C_H=mu*[lambda_H_lo,lambda_H_hi] and
C_T=mu*[lambda_T_lo,lambda_T_hi]. Initially open owners retain ZERO held tangent,
and only a proved birth may use
C_T=mu*([lambda_H_lo,lambda_H_hi]+[lambda_T_lo,lambda_T_hi]) once. Project each
phase under its own normal before summing. Owner generations, duplicate rows,
old history and proposal nullspace components supply no extra capacity. An
open-held tangent enclosure containing unresolved nonzero values cannot simply
be declared zero; it must satisfy the original zero-held rule or reject.

For each admitted capacity C=[c_lo,c_hi], require the original speed-scaled
ceilings, with the existing positive response scale kappa and Vref:

    R_cone = kappa*max(t_hi-c_lo,0)/Vref,
    R_diss = kappa*sup |w+c*z| / Vref^2,
    R_work = kappa*max(w_hi,0)/Vref^2,

where w in [w_lo,w_hi], c in [c_lo,c_hi], z in [u_lo,u_hi]. All endpoints give a
safe bound because c,z>=0; use exact/outward arithmetic, not a midpoint. These
are the conservative enclosure versions of _dynamic_friction_law.audit: cone,
maximum-dissipation and positive-work checks, not an unconstrained SOC minimizer.
For zero tolerance the admitted stick case has U=0 and ||T||<=C; sliding has
T=-C*U/||U||. The interval test covers both without selecting a favorable capacity
or slip from uncertainty. Actual coupled normal/tangent velocities remain used.

Held Signorini additionally encloses every combination of
abs(min(k_n*lambda,gap/h))/Vref, with existing penetration/gap gates. Terminal
requires max(0,-u_n,abs(min(k_n*lambda,u_n)))/Vref. Here k_n is the original normal
response scale and u_n is the actual normal relative speed. Check the complete
lambda/gap/speed intervals, not the requested pressure at one chosen endpoint.
Retain all ideal-extreme directions, including zero-pressure extrema, and every
contact response/map ceiling. The stated formulas correspond to current
_thick_contact_workspace.py:191-212,363-405 and _dynamic_friction_law.py:24-51;
they add a sufficient delivery-witness audit, not removal of existing guards.

These inequalities can reject a quotient representative although another owner
pressure could satisfy Coulomb. That narrower admitted set is intentional in the
candidate, but parent still must freeze it. It does not establish existence of
two useful ordinary-mass physical controls or cover all redundant contacts.
## 14. Ideal extrema, self-stress and physical residual limits

At each phase retain B, its independent ideal-map enclosure B_star, full actual
requested owner p_requested, p_eff, p_N=Pi_N*p_requested, and
p_requested-p_eff. If abs(B_star-B)<=E_B componentwise, the ideal nodal mapping
error for the actually requested impulse is bounded by

    abs((B_star-B).T*p_requested) <= E_B.T*abs(p_requested).

The RIGHT side uses the FULL request, including p_N, not only p_eff. Although
B.T p_N=0, generally B_star.T p_N=(B_star-B).T p_N need not vanish. Thus a large
proposal self-stress cannot manufacture capacity and also cannot disappear from
ideal-map/resultant/couple/work uncertainty. Keep the original raw/native input
construction errors and exact cancellations distinct from this enclosure.

For any independently certified extremum relative-point map C_star, the frozen
configuration response to remainder r has bound

    abs(C_star*W*r) <= abs(C_star*W)*(abs(r_center)+e_r).

Use outward interval matrix products when C_star is bounded. Add the separately
bound actual-versus-ideal evaluation error (C_star-C)*v_actual and effects of
independent ideal-map/mass uncertainty without omitting correlations or allowing
cancellation. Check every required clip extremum and velocity direction, not
only loaded rows or the centroid. Positive B W r=0 evidence does not replace any
of these checks; even at stored rows, map changes between phases matter.

This linear response identity is valid for a frozen terminal velocity update
and for the free predictor. It is NOT a bound on final held position/velocity
after nonlinear distance/bending sweeps. Multiplying it by h is not a proof of
nonlinear endpoint gap. That would require a separately bounded material-map
sensitivity. Keep direct actual final-state gap/relative-velocity audits, full
material path and physical-affine proof, and CCD; no sensitivity is assumed here.

For particle blocks, exact resultant and moment operators are S*r_particle and
C_x*r_particle, where S sums physical nodal impulses and C_x uses each actual
world position cross-product matrix. Conservative magnitudes use abs(S)*r_bar
and abs(C_x)*r_bar, not a signed sum of errors across domains. Robot generalized
impulses need the declared original Jacobian dual and existing paired world-point
mapping receipts. A constrained prismatic guide does not furnish an arbitrary
invertible world-force lift. Do not infer a total free-robot world momentum or
torque theorem from translational coordinate units alone. Retain the original
particle NUM005 and paired moment gates and all robot generalized response checks.

Held residual endpoint-work magnitude can be conservatively accounted as
r_bar dot abs(Delta y)/h, where y comprises world particle positions and admitted
robot translational coordinates. At a frozen terminal configuration, with
a=B.T p_eff and the exact orthogonal decomposition ell=a+r,

    Delta K(ell)-Delta K(a)
      = r dot v_pre + a.T W r + (1/2)*r.T W r
      = r dot v_pre + (1/2)*r.T W r.

This is an accounting identity only, never a recomputed or repaired terminal
state. A general enclosure retaining all uncertain/ideal cross terms is
r_bar dot abs(v_pre) + a_bar.T abs(W) r_bar
+ (1/2)*r_bar.T abs(W) r_bar. Exact stored orthogonality may remove its stored
cross term but not an ideal-map or ideal-mass cross term without proof.

These contributions stay INSIDE the original ceilings. The full actual-state
joint D_E and componentwise angular/window gates remain authoritative. If a
split proof bounds a physical defect by base plus residual/uncertainty, charge
each event once in that upper bound; never add a new permitted allowance or
subtract residual/internal work as external credit. If the full actual defect
already includes that event, its explanatory decomposition cannot forgive it.
There is no averaging across phases, owners, time, domains or components.

## 15. Follow-up witness controls and exact ownership implications

These are unexecuted future requirements, in addition to section8, not new files
created by this task:

1. Independently derive the exact duplicate/opposed/off-range examples in11.2;
   retain every owner. Check complete row permutations and invertible nullspace
   basis changes N->N*S yield the same projector and witness. An incomplete or
   falsely full nullspace, invalid rank, nonsymmetric mass or wrong inverse fails.
2. Add p_requested->p_requested+N*t with unchanged exact stored delivery. Certified
   p_eff/capacity is unchanged; the FULL proposal remains in independent ideal-map
   error and binding. For opposing rays, reject negative quotient normal even
   when a positive self-stressed alternative exists. Never fit that alternative.
3. For B=I and M=diag(1,10^12), H=diag(1,10^-12); the exact condition product is
   10^12 and must reject equality. Nearby exact below/above choices, positive LDL
   failure, zero rank, incomplete receipts and resource exhaustion are separate
   controls. They are algebraic negatives, not physical support claims.
4. Hold nominal request fixed while mutating actual robot predictor or terminal
   v. The full observed M delta-v, not per-joint masses or nominal force, must
   change uncertainty/delivery and capacity. Mutating original effort/force/
   torque/bias or selected-pre identity is independently rejected.
5. Distinguish a truly unchanged original robot force from an unauthorized changed
   force whose effect is absent from effort-only ell_H. Bind the unchanged bias
   and original evaluation configuration. Do not hide native mass-solve error by
   calling its ideal predicted velocity the observed predictor.
6. Change one phase at open birth. Recompute its own B/Pi_N/L/normal/uncertainty
   under its saved phase identity and combine capacity only once. A held projector
   cannot be reused at terminal solely because row counts match. Retain original
   full selected-pre/path/history/effective-input and once-only terminal controls.
7. Off-range r that is invisible to stored B but visible at another ideal extremum
   must exhaust the relevant original velocity/couple/work bound when oversized.
   Include full requested null self-stress magnified by E_B, zero-pressure ideal
   extrema, omitted normal derivative and missing cross terms.
8. Preserve the late second/third-substep failures, exact source/raw evidence and
   rollback/retry contract from section8 for every new certificate failure.

The ten paths in section9 remain an architectural candidate, not a new allowance.
A quotient/predictor witness should be an explicitly parent-owned additional
validation module if selected; it must not silently change S2's full-rank reader
or existing robot law. The precise path, public immutable certificate, finite
resource limits and robot predictor evidence seam require a parent freeze before
source authoring. Every guard remains outside the solver, and none supplies a
replacement state, pressure, map, normal or material response.

## 16. Ordinary-mass coupled fidelity: exact remaining gap

No pair of primary ordinary-mass success controls has been proved in this
continuation. Sections7's tiny-mass controls remain supplemental only. For a
precise reference to the missing work, take their SAME geometry/rest/rate/load/
h/window/rebuild data at particle masses1 kg, robot mass4 kg, distance compliance
1 m/N and bending compliance1/J (the ordinary-mass homogeneity counterpart).
These values define UNPROVED analysis targets, not accepted new tests or a change
to the original candidate data. No mass, rest angle, h, perturbation, friction,
iteration limit or budget was adjusted after observing execution.

Some local facts do survive independently: SC1 has initial C=-2^-10 and gradient
(2ez,0,-ez,-ez), nonzero bending force, and maximum initial material acceleration
2^-9<1/8 m/s^2. SC2 has initial Cdot=-1/64 rad/s. The unstrained reference impulse
formulas remain valid for their explicitly unperturbed limit only. Neither fact
bounds the full strained finite-step path or its owner capacity. In particular,
SC1's initial compression margin is not a nonlinear window convergence proof;
SC2's perturbed normal-offset geometry can change birth time and its terminal
response, so the reference gap/u arrival time is not a perturbed-case theorem.

The original coarse envelope becomes up to384 J of positive-defect bound over
32 steps at unit mass, and its angular bound is likewise far above the physical
ceiling. It is therefore USELESS as a coupled-fidelity oracle at this scale.
That failure of an upper bound does not prove the physical cases fail, nor does
it imply a universal FP32 obstruction. It means there is no independent tight
bound here yet. Small-mass scaling cannot cure this missing fidelity evidence.

What is required for EACH of the two primary controls is concrete:

- An independently chosen non-vacuous trajectory box and impulse box covering the
  complete h grid, windows and rebuilds, with a strictly positive lower bound on
  actual bending activity and on the relevant held/terminal transfer. Boxes must
  discriminate a wrong trajectory/load, not merely bound all coordinates by2.
- Independent area/edge/angle/offset-map and derivative bounds on that box,
  including supported angle branch, every material row and full physical affine
  segments. A bound on the Hessian of the bending energy alone does not certify
  the ordered XPBD row map or the contact/material composition.
- A nonlinear residual enclosure for the complete declared finite material
  sweeps, lambda history and endpoint gap/terminal equations, including shared
  nodes and robot response. A force equilibrium or continuous ODE solution is not
  the same equation as this discrete algorithm. Rank-reduced proof coordinates
  may analyze the quotient but may not delete owner obligations in the result.
- A certified contraction/interval-Newton witness or another independent residual
  error bound giving trajectory/impulse inclusion, plus sign/phase/slip margins
  sufficient for section13. Such a witness needs explicit derivative constants,
  a domain mapping into itself, error amplification and finite iteration effects;
  none has been derived here. Existence of a continuum contact solution does not
  imply this replay's512-iteration success or two reproducible rebuilds.
- For SC2 specifically, a certified perturbed birth-time bracket and positive
  terminal-delivery lower bound, covering endpoint-only birth if it occurs. For
  SC1, a full-window compressive capacity/geometry margin under the active hinge.
- Tight ORIGINAL-input bounds on sum_j max(0,D_E,j) and the original sum of
  absolute angular component/domain defects, with all mapping/paired moments.
  Energy dissipation at impact cannot cancel a later positive step. Ideal torque
  invariance cannot erase discrete NUM005. No internal transfer credit is used.
- Actual R7 decode/raw/native evidence and complete FP32 predictor/projection/
  reconstruction/publication uncertainty within unchanged normalized path gates.
  Analytic geometry bounds do not establish these native inequalities in advance.
  They remain distinct from the ordinary-mass physical enclosure and from a
  universal impossibility claim.

These are specific missing mathematical certificates, not permission to replace
the two primary controls by handoff fixtures, tiny masses, zero material activity,
solver-derived expected trajectories or tuned easier parameters. A proposed
implementation could legitimately reject the current targets; no pass is promised.
Parent must decide and freeze stronger independently justified controls/support
before claiming this physical integration accepted.

## 17. Width allocation, decisions and release of this continuation

For a true per-step defect d_j and independent interval [a_j,b_j] enclosing it,
max(0,b_j)<=max(0,d_j)+(b_j-a_j). Thus a noncancelling window needs a bound on the
physical sum PLUS the sum of every initial/final energy enclosure width. All terms
have units J; no normalized tau is itself an energy allowance. A fixed allocation
eta*beta_E for widths is defensible only after a physical bound leaves at least
that margin. Angular intervals need the corresponding absolute-component bound
and N*m*s units. Preserve finite sqrt/atan/codec limits; failure to certify the
allocated width rejects rather than increasing proof resources or physical limits.

Consequently the earlier beta_E/4096 example is NOT frozen here. Its final
allocation depends on the primary controls' physical margins and finite step
counts. No independent bending-width result or physical trajectory was evaluated
for this follow-up, and original R7/host/native receipts remain unchanged.

Parent decisions still needed:

1. Admit this conservative quotient representative and its narrowed cone domain,
   or decline it; algebra alone gives neither friction validity nor full coverage.
2. Freeze complete finite witness resources and exact parent-owned module/API,
   including observed robot free-predictor provenance if not already exposed.
3. Freeze sufficient phase/cone/ideal-extreme/error allocation using original
   ceilings and once-only capacity semantics, with no nullspace self-stress credit.
4. Supply the two primary ordinary-mass nonlinear fidelity enclosures and their
   physical/angle/native margins; section16 is the precise unresolved gap.
5. Then freeze dimensional enclosure allocation, full source ownership, focused
   host/interpret/CUDA/bounded-C++ groups and the later integrated review scope.

Status remains PROPOSAL_NOT_FROZEN. This continuation is documentation only:
no import, test, numerical/offline probe, lint/formatter, compilation, build,
installation, helper/reviewer, merge or remote action. S2 owns execution; no
executable slot was taken. Documentation authoring is released at its checkpoint.