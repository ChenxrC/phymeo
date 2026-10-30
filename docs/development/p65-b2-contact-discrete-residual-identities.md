# B2 ordered projection: exact endpoint residual identities

Status: HAND_DERIVED_CONDITIONAL_IDENTITIES_ONLY. No executable check, new pass, physical implementation grant or proved SC1/SC2 nonlinear enclosure. This note complements [the pressure-direction finding](p65-b2-contact-pressure-direction-audit.md) and preserves all original path, NUM005, R7, energy and contact gates.

## Scope and observed values

Use fixed positive mass M and W=M^-1 in a coordinate space consisting only of translations in metres. The ordinary target has diagonal particle masses and a constant prismatic robot mass. Mixed rotational coordinates, configuration-dependent mass and general articulated robot bias/energy need separate identities. All q(stored) values below denote exact rational binary values; nonlinear ideal constraint values/gradients require independent outward enclosures on their original branch.

Take one complete selected material replay, with immutable initial state/input and a fresh lambda history. x_hat is its observed predicted configuration, x is its final selected pre-terminal configuration, and D=x-x_hat. This is a computational trace, not a sequence of additional physical substeps.

Enumerate EVERY ordered configuration transition s, including explicit conversion/publication transitions. A material row j(s) has its recorded delta multiplier delta_s and recorded compute gradient g_s; use original W and define the observed residual

    epsilon_s = x_after_s-x_before_s-W*g_s*delta_s.

For a conversion/publication-only transition, take the row contribution zero and epsilon_s equal to the entire configuration change. No synthetic row impulse is applied. Complete linked transitions must telescope to D. Untouched coordinates, duplicate physical node IDs and all shared-node additions are retained.

For R7, reconstruct each trace configuration only through the declared immutable R/raw-d/rounding schema. A decoded internal-row world configuration is a defined observation, not a claim of native FP64 world storage or another physical time state. Preserve the existing separate noncancelling a/p/rho, gradient, parameter, equation, lambda and reconstruction charges. This identity cannot cancel those charges or bypass the original path ledger.

Let G_j(x) and C_j(x) be the IDEAL final constraint gradient/value, independently bounded at the actual selected final configuration, not a producer-supplied final gradient. For every original material row define

    zeta_j = lambda_final_j-lambda_initial_j-sum_{s:j(s)=j} delta_s.

This retains actual lambda conversion/arithmetic discrepancies. Fresh zero initial multipliers must be bound to the actual trial; do not assume an absent/carrying history is zero.

## Exact ordered-gradient drift

Simple telescoping gives

    D = W*sum_j G_j(x)*lambda_final_j + eta,

where

    eta = sum_s epsilon_s
          +W*sum_rows delta_s*(g_s-G_{j(s)}(x))
          -W*sum_j G_j(x)*(zeta_j+lambda_initial_j).

This is an identity, not an assumption that all gradients were evaluated at the final state. The middle term includes both changing-configuration gradient differences and observed compute-gradient error. It must not be labeled pure geometric curvature. A trace with zero final row residual can still have nonzero eta.

It is precisely this drift that is lost by replacing an ordered finite XPBD trace with a final-force-equilibrium equation. Any inclusion proof for the actual replay must retain and bound it, or supply another equivalent full-history error term. Interval evaluation uses a complete linked trace and independent final ideal-gradient enclosures; a floating diagnostic sum is not the certificate.

For a componentwise conservative upper bound, take absolute values BEFORE summing each epsilon, delta-gradient and lambda-history contribution, then use |W|. Keep the direct signed identity only as a diagnostic/equality check. Do not use observed cancellation to buy new normalized, phase or physical budgets.

## Positive-compliance constitutive residual

For rows with physical compliance alpha_j>0, define the true original-parameter residual

    r_j = C_j(x)+(alpha_j/h^2)*lambda_final_j.

All units stay those of the row: distance and angle are not equated. For U(x)=sum_j C_j(x)^2/(2*alpha_j),

    (M/h^2)*D + grad U(x)
       = sum_j G_j(x)*r_j/alpha_j + (M/h^2)*eta.

Each summand has generalized force units. This identity uses the original compliance and h, not merely their converted compute alpha. The existing equation/parameter/conversion tests remain independently mandatory. A zero-compliance row cannot be divided by alpha; it needs its own multiplier/KKT formulation and is outside this elimination.

Let f_eff be the original declared gravity force plus actual effective force used by this replay, and define its observed prediction discrepancy

    epsilon_pred = x_hat-x0-h*v0-h^2*W*f_eff.

Then the selected endpoint satisfies the exact backward-Euler force residual identity

    R = (M/h^2)*(x-x0-h*v0)-f_eff+grad U(x)
      = sum_j G_j(x)*r_j/alpha_j
        +(M/h^2)*(eta+epsilon_pred).

This statement is conditional on validated observations and independently enclosed ideal C/G. It is NOT proof that this residual is small, that the endpoint lies in a selected box, that the finite iteration count converges, or that the backward-Euler solution is the same as exact continuum motion.

## Internal contact and original inputs

In a future joint constant-mass replay, write the exact original input f0 separately from represented held transfer. If a declared attachment map B_H sends the requested held impulse P to the generalized coordinates, then the intended force increment is B_H.T*P/h. Actual effective-minus-original inputs and their mismatch must remain separate exact observations.

An independent contact/root analysis at the FINAL configuration would instead use the derivative of its fully specified gap/support law. Even if written B_gap(x), it cannot silently be identified with B_H. The difference

    (B_H.T-B_gap(x).T)*P/h

and actual representation/delivery discrepancy enter the force residual. Fixed normal directions, moving normal/attachment derivatives, finite support ties and gap branch membership all require proof. A frozen-map response bound times h does not establish this endpoint relation. The [pressure-direction note](p65-b2-contact-pressure-direction-audit.md) further shows why the physically declared pressure space must precede a quotient certificate.

The local material reader can be bound to f_eff. Joint physical accounting remains bound to the global original f0. Neither eta, internal contact work, map drift nor the force residual is a new external-energy credit.

## Actual reconstruction and energy

Define u=(x-x0)/h and the observed reconstruction discrepancy r_v=v_selected_pre-u. With the same original constant M,

    K(v_selected_pre)-K(u)
      =u.T*M*r_v+(1/2)*r_v.T*M*r_v.

For an actual terminal impulse ell_T=M*(v_post-v_selected_pre) at unchanged configuration,

    K(v_post)-K(v_selected_pre)
      =v_selected_pre.T*ell_T+(1/2)*ell_T.T*W*ell_T.

These are exact algebraic identities of stored states. Requests, represented input batches and actual ell_T are separately bound. Paired terminal transfer is internal to the joint system; its work is retained in the actual total balance, not subtracted as externally supplied power. No identity proves its sign without the physical law and all actual/ideal uncertainty checks.

Elastic and original gravitational potential intervals still enter the DIRECT joint endpoint energy defect. Per-step positive parts and their whole-window sum remain subject to the original1e-8 J host/1e-5 J native contact ceilings, not B2's standalone1e-3. If these identities are used to construct an independent upper bound, do not also add the same event as an extra energy defect. Direct defect and explanatory decomposition are distinct reports; original noncancelling representation charges stay separate.

## Particle angular bookkeeping and NUM005

For a particle with scalar mass m_i, x_i=x0_i+h*u_i implies x_i cross(m_i*u_i)=x0_i cross(m_i*u_i). Subtract the initial-position moment of original plus represented held predictor impulse h*f_eff. Using the complete trace,

    D_L_pre =
      sum_rows,i x_before_s,i cross(g_s,i*delta_s/h)
      +sum_rows,i (x0_i-x_before_s,i) cross(g_s,i*delta_s/h)
      +sum_i x0_i cross(m_i*(sum_s epsilon_s,i+epsilon_pred_i)/h)
      +sum_i x_i cross(m_i*r_v,i).

The first term is the observed local row moment; an ideal rotationally invariant gradient would have zero summed moment at that row configuration, but a stored observed gradient need not. The second term is the discrete configuration-to-velocity mechanism behind NUM005. The final terms retain conversion, prediction and reconstruction effects. Dropping the second term because a row's local ideal torque is zero is invalid.

Terminal delivery adds sum_i x_i cross ell_T,i at the unchanged endpoint. Comparing against requested terminal moments needs the separately observed actual-minus-requested impulse and complete phase binding. These particle identities do not define angular momentum for a one-dimensional robot coordinate or prove a general articulated joint's reaction torque. Robot/world paired moments and missing generalized physics remain separate obligations.

The accepted composite_material_path diagnostics already expose related local/discrete row contributions. Those float displays and ideal moment invariance do not establish the tight ORIGINAL contact per-step and noncancelling whole-window angular ceilings. No previous passing scope or NUM005 status is changed here.

## What remains before ordinary-mass physical freeze

A residual identity is useful evidence, but it is not the required small residual or an error-to-solution bound. For BOTH unchanged ordinary targets still derive:
- a non-vacuous trajectory/impulse neighborhood, complete area/branch/support/birth margins and positive material/contact activity through every h/window;
- independently bounded ordered-map/final-gap derivatives and all drift terms above, plus a residual-to-state/impulse inclusion or contraction/interval-Newton proof on the entire neighborhood;
- exact ownership/direction rules for redundant pressures, including the newly identified frictionless world-gauge obstruction;
- original-input energy/angular upper bounds and their enclosure-width allocations without temporal cancellation or internal credit;
- actual R7/native evidence and original finite resource/iteration limits, which remain separate from the analytic inclusion.

No value, mass, rest angle, perturbation, h, window, fixture or tolerance was fitted or changed in this hand derivation. No source/test/implementation task or executable probe is authorized by this note.
