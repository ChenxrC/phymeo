# S2-F conforming material and lawful-fold proposal: documentation only

Status: PROPOSAL_REQUESTED_NOT_FROZEN. This is the next distinct S2-F material
contract after accepted S2-L response. It is not another review or rewrite of
the old nonincident proposal. Continuous P6.5 authorization includes parent
architecture adjudication; no generic stage permission question is needed.

## Owner, starting point and only writable file

Use the original p65_self_contact_module in its existing isolated worktree
C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo, starting from
clean f504f53e77407426c24cae769cc80b7c08c395dd. Parent independently checked that
clean checkpoint and that the following file is absent.

Create ONLY docs/development/p65-s2-conforming-shell-proposal.md. Read-only source,
mathematical derivation, document file IO and Git are allowed. No imports,
collection, tests, numerical/offline probes, lint, formatter, compile, build,
install, profile, source/test/export edits, parent merge, push or helper/reviewer.
Do not edit old proposals/reports, accepted S1/LG/response or the parent adapter.
Preserve your original release and evidence. Parent alone owns old shared seams.

Read these from the absolute current parent tree, without merging it:
- docs/development/p65-s2-self-contact-proposal.md, especially sections5 andD;
- docs/superpowers/specs/2026-10-30-p65-material-self-gripper-integration-design.md,
  section10 and R4;
- docs/superpowers/specs/2026-10-30-p65-self-response-design.md and R1/R2;
- docs/development/p65-s2-response-parent-math-audit.md;
- docs/development/p65-b2-contact-adjudication.md and the completed quotient spec;
- actual constraints/triangle_prism_surface.py, constraints/cloth_self_response.py,
  geometry/cloth_self_anchors.py, validation/cloth_self_mapping.py and current
  runtime/point_participant_program.py / public replay and terminal contracts.

The accepted nonincident response used None/legacy Distance, unique exposed ray
and constant-normal held continuation. R7 composite dispatch and a complete
physical cone were not added. Quotient acceptance is unconnected exact algebra,
not a pressure law or permission to spend nullspace self-stress as capacity.

## Required concrete design result

The working architectural direction already chosen by the parent is a NEW
conforming simplicial material body. Preserve legacy TrianglePrismSurface as-is.
Do not repeat the old alternatives at length: briefly compare a fully specified
derived-director model with an independent-director model, then recommend the
smallest coherent finite domain that admits actual non-flat lawful folds.
A rigorous alternative legacy gluing model is only relevant if a complete
material-address relation and its verifier can actually be supplied.

1. Define the reference oriented manifold triangle complex, boundary and vertex
   links, global top/bottom labels, explicit per-vertex half thickness in metres,
   and deterministic conforming prism-product tetrahedralization using a global
   vertex-ID order. Specify orientation/parity so adjacent faces really share
   identical side triangulations. Internal tetrahedra/facets are material, not
   separate friction surfaces. Explain why this is a new piecewise-affine body,
   not a warped bilinear wedge, six-point convex hull or legacy normal prism.

2. Give an exact director law and owned data. For a candidate
   a_i=rho_i*S_i/||S_i||, define S_i and all weights, winding and boundary behavior
   precisely; distinguish fixed reference weights from current geometry weights.
   Prove translation/proper-rotation covariance and state when S_i or the law is
   unresolved. Projected thickness and orientation restrictions need explicit
   admissibility, not example-dependent angle/epsilon rules. Independent directors
   require real inertia, velocities, energy and equations; a kinematic copy is
   not an alternative with free dynamics.

3. Derive the COMPLETE sparse map for each tetrahedral material point
   y=sum_a lambda_a*(x_i(a)+sigma_a*a_i(a)), with fixed owned barycentric material
   coordinates. For normalized S, include
   D a_i = rho_i/||S_i|| * (I-n_i*n_i.T) * D S_i
   and every incident-star dependency; include any derivative of nonconstant
   weights if you choose them. Show sum_j J_j=I and
   sum_j J_j*(omega cross x_j)=omega cross y from the actual law.
   Derive virtual work, resultant and torque of J.T*f and full shared-base
   response including star overlap. Explain the declared mass model: geometry
   slaved to middle particles does not automatically supply volumetric inertia.
   Identify required material-energy terms/derivatives and limits of existing B2.
   No frozen-normal, three-node-only, finite-difference or missing-torque shortcut.

4. Supply an independently verifiable reference-material incidence rule for ALL
   tetra pairs, including same cell, face/edge/vertex neighbors and distant cells.
   Injective individual tetra maps and positive determinants are necessary but
   insufficient. A candidate finite PL embedding certificate is that each
   geometric intersection is exactly the image of the shared reference simplex
   (or empty for disjoint reference simplexes), plus valid reference topology and
   boundary orientation. Establish its sufficiency and a finite implementable
   verification strategy; reject unknown instead of claiming theorem by naming it.
   Shared IDs alone, a k-ring/collar/angle exemption, convex hull filling, common
   connected intersection component, or all-fold rejection is forbidden.

5. Define the actual continuous path, not just endpoint geometry. Existing
   affine middle-node paths lead to nonlinear normalized directors and several
   dependent radicals. Identify exact identities to factor at permanent shared
   features, complete time/material coverage, interval/radical degeneracy logic,
   finite resource caps and independent checks. Ordinary strictly positive
   separation cannot certify a permanently shared vertex. Do not relabel the
   legacy two-root solver or multiply a frozen velocity bound by h to claim a
   nonlinear path bound. State a proved, nonempty domain for successful folded
   paths; arbitrary-depth unknown rejection alone is not successful fold support.

6. Derive external shell/robot contact from the actual oriented material boundary,
   including edge/vertex features and all supported maximal ties, full material
   coordinates, both actual stored and ideal maps, and complete CCD. Internal
   subdivision faces must not create robot contacts or capacity owners. Specify
   how full-star loads reach one particle base exactly once via the existing
   public replay/terminal seam and how original external work, internal work,
   energy and angular defects remain distinct. State how loaded changing
   directors/curved paths affect held replay and terminal reconstruction.
   Geometry certification alone does not accept response or a native path.

## Independent successful controls and negative controls

Provide at least two independent non-handoff success controls with explicit
ordinary positive masses, units, reference topology, thickness, initial states,
material/director law, time interval and three h values, two independent rebuilds,
and hand-derived exact equations or nonvacuous certified bounds. Choose values
from analytical design, never numerical trial fitting. Include:
- an ordinary non-flat fold across a shared material edge through a nonzero
  fold interval; prove surrounding cell and star validity and no fictitious
  internal contact load;
- a different reference-valid connected multi-face vertex star with a nonzero
  director derivative, including an external point/finite-patch load that
  exercises full-star virtual work/resultant/torque and reaction on the other
  participant. Distinguish a static or instantaneous map oracle from complete
  physical trajectory acceptance; provide the latter's equation/bound or label
  its exact unresolved obligation. No two-face fixture presented as global-star
  coverage, no unstrained reference solution presented as strained dynamics.

Retain covariance under proper rotations/translations, cell/node order with
stable material identities, and identical reconstruct-from-published-state.
Require negatives for remote incident overlap in a valid connected mesh,
interior-time crossing with legal endpoints, positive-tetra but overlapping
star, inverted/degenerate tetra, vanishing director sum, nonconforming diagonal,
forged incidence/boundary label, incomplete star Jacobian and omitted torque.
Each rejected negative is separate from the two successful physical controls.
No pin/weld/object-coordinate-time branches, tiny-mass escape, broad incidence
exclusion or relaxed old contact budgets.

## Interfaces, costs and handoff

List exact proposed types/functions, immutable schemas/fingerprints, coordinate
and precision declaration, resource policies, unknown reasons, proof producer
and independent verifier separation, and exact NEW file allowlist by dependency.
Identify every parent-owned old seam separately. Explain host/actual interpret/
CUDA/bounded C++ implications and which existing consumers truly change; no
unchanged old suite re-run just for reporting. R7 FP64 reference plus actual FP32
displacement is explicit and is not native FP64 or single-FP32 world state.

Separate proved commitments, proposed algorithms and still missing proofs in
the document. Do not fabricate a theorem, successful native outcome or complete
physical freeze. If one part is unresolved, still complete the independent
topology/law/Jacobian/interface derivations and give the smallest precise missing
mathematical obligation. Existing local scopes remain accepted on their versions.

End with PROPOSAL_NOT_FROZEN, exact proposal hash, clean small documentation commit
and authoring release. Parent will read and adjudicate before a source grant.
There is NO executable slot in this task and no authorization for old-source
integration. Full B2 contact, legal folds and whole P6.5 remain mandatory.
