# SC2 first contact under the prescribed hinge-rate perturbation

This hand derivation bounds first contact in the zero-pressure material replay used to detect impact. It covers the three original time grids and all three prescribed velocity perturbations. A second result bounds the terminal impulse in the special case where that free replay reaches contact exactly at a grid endpoint. The bounds use the ordinary target with particle mass 1, robot mass 4, unit compliances and zero rest angle. They also hold with fixed unit compliances for uniform particle mass m in [1/2,2] and robot mass 4m. [Control and acceptance obligations](p65-b2-contact-integration-proposal.md).

Under the declared homogeneous scaling from unit mass, multiplying every mass by s and dividing every compliance by s divides both terms in each row denominator by s. Starting at zero multipliers, induction multiplies each multiplier and its increment by s and gives identical position increments. The terminal impulses below then scale by s as well. A common translation and a proper rotation applied to the entire setup preserve the estimates in patch-normal coordinates.

## Free material motion through the candidate impact step

Write u=1/8, g0=3/1024 and eta in {63,64,65}/4096 for the upward velocity of node d relative to the three other nodes. Use the original length, time and mass units. The robot is at rest in the free replay. Its square patch occupies the plane z=-rho, with rho=1/64. The initial four nodes share the translation g0 e_z. Define the ballistic comparison by translating every node through (g0-u t)e_z and adding eta t e_z to node d only. At a flat state the angle gradient contracted with the initial velocity is -eta.

Set T=1/32 for the free comparison, including the continuation beyond its first touch. Each of h=1/64, 1/128 and 1/256 divides T. In step k the predictor is y_(k-1)+h v_(k-1). Let d_k denote the complete finite material displacement relative to that predictor. Reconstructing velocity from the actual ideal displacement gives

    v_k - v_ballistic = sum_(l=1..k) d_l/h,
    y_k - y_ballistic(kh) = sum_(l=1..k) (k-l+1)d_l.

Use the four-node Euclidean product norm. For each step the multipliers start at zero. Any complete sweep count between 1 and 512 is allowed. Fix P=1/1800. If the predictor is within P of the translated flat reference, its initial constitutive residual is at most 3P. All four constraints vanish at that reference and their gradients have norm at most 3 throughout the intervening segment. Since P<1/128, the finite ordered-row bound then gives ||d_k||<=26h^2(3P). [Local geometry and sweep bounds, sections 1–6](p65-b2-local-row-derivative-bounds.md).

To prove that premise at every step, induct from the original velocities. The predictor error relative to its ballistic value contains only terms with l<k in the position sum. Bound these using preceding steps. Including the new displacement after its sweep gives the same conservative bound for the endpoint. For kh<=T the total error bound and the translated-reference check are

    E = 78P T(T+h_max)/2 = (117/2048)P < 1/30720,
    eta T + E < P,       h_max=1/64.

The second inequality follows from 65*1800 + 117*64 < 131072. This also starts the induction at k=1, when the predictor error is zero. The affine path between endpoints retains both position bounds because its ballistic comparison and reference translation are affine in time. Each physical segment velocity differs from its ballistic velocity by at most V=78P T=13/9600.

## The first-touch interval on all three grids

For clarity let b_i be the height of a middle node above z=0. Its lower prism vertex lies at height g_fi=b_i+c_f relative to the patch plane, with c_f=rho(1-n_f,z). On the smaller domain just proved the raw oriented area differs from e_z by at most 4P+4P^2<1/400. Both face normals are upward with ||n_xy||/n_z<1/399. Hence

    0<=c_f<1/(128*399^2),       E+1/(128*399^2)<1/30000.

Let gamma be the minimum across both faces and all six lower addresses. Each node has ballistic height at least g0-u t. Each face contains a node descending with the full speed u. The position and tilt bounds give

    |gamma(t)-(g0-u t)| < 1/30000     for 0<=t<=T.

This minimum equals the exact signed full-prism/patch gap minimized over the two pairs. Indeed each face has lower heights of absolute value less than 1/256, since |g0-u t|<=3/1024 and its middle-node deformation and tilt correction together are bounded by P+1/(128*399^2). Upper heights exceed 3/256 by n_z>1/2. All projected vertices lie strictly inside the square with coordinate magnitudes less than 33/32. The support-function calculation for the complete square now applies with |ell|<1/256, H=3/256 and M=31/32. It gives the unique maximizing direction +e_z for each prism. All tied lower addresses on that face belong to its original lower vertices. [Finite support calculation, section 12](p65-b2-sc1-active-material-reference.md).

There is a strict closing-speed margin along every open segment. Subtracting the common downward velocity leaves product norm at most eta+V<1/56. A face normal is invariant under common translation and its derivative norm is less than 4 in the outer domain. Thus each lower-vertex vertical speed, including the director derivative, is at most

    -u+(1+4rho)(eta+V) < -1/8+17/896 < -1/10.

Continuity and the finite piecewise-affine middle path extend strict descent across step boundaries. Therefore the minimum crosses zero at a unique first-touch time. Simultaneous owners may share that time. For L=47/2048 and U=49/2048 the common ballistic heights are +1/16384 and -1/16384 respectively. Their magnitudes exceed the gap uncertainty. Even the earlier gap exceeds the largest permitted h*tau because 1/16384-1/30000>2e-5/64. Both first entry into the contact budget and first zero gap are strictly inside (L,U).

The coarse grid therefore brackets free impact inside its second step. The finer grids have 3/128 as a grid boundary inside the bracket. They retain both adjacent step possibilities and exact endpoint arrival. This establishes a first-touch bracket for the perturbed, finite-sweep free replay independently of which support vertices attain the minimum.

The bend is active there. On the ballistic path its signed angle is -atan(eta t) with the original winding. For 0<=eta t<=65/131072 the derivative of atan is greater than 3/4. The angle error caused by the material displacement relative to ballistic is at most 3E, using the full gradient bound along the segment between the two configurations. For any t in [L,U],

    -theta(t) >= (3/4)(63/4096)(47/2048) - 3/30720 > 1/8192.

With unit compliance, at a free grid endpoint in this bracket whose exact final constitutive gate |theta+h^(-2)lambda_B|<=tau holds, the final bending multiplier exceeds h^2/10000. The statement about angle applies throughout the time interval, while the multiplier statement uses only such a grid endpoint.

## Terminal impulse for the endpoint-only branch

Suppose first zero gap occurs at a grid endpoint reached by the free replay with zero held transfer. Let I contain every touching owner. Use the current full triangle-offset Jacobian and the prismatic robot coordinate. With normal +e_z the relative normal velocity at owner i is B_i v. Here B_i includes the current director derivative on the face and robot coefficient -1. At this endpoint the reconstructed free velocity is the incoming segment velocity, so B_i v_minus < -1/10 for every i in I.

For the ideal perfectly inelastic, frictionless terminal law, project v_minus in the full kinetic metric M onto the closed cone B_i v>=0 for i in I. This cone contains zero. A common upward cloth velocity with stationary robot makes every inequality strict. Positive definite M yields a unique velocity minimizer. The linear constraints give nonnegative multipliers L_i supported on I, with

    v_plus = v_minus + M^(-1) B_I^T L,
    L_i (B_i v_plus)=0 for every i in I.

Keep separate material owners even when rows coincide. The summed impulse J=sum_i L_i admits bounds independent of its split. The cloth part of each row has norm at most 1+4rho=17/16, so every entry of B_I M^(-1) B_I^T has absolute value at most (289/256+1/4)/m=353/(256m)<3/(2m). Choosing any touching row gives J>m/15 from its required increase in speed. Even the weaker condition B_i v_plus>=-tau gives J>(2m/3)(1/10-tau)>m/16 for this pure-normal response.

Complementarity gives v_plus^T M(v_plus-v_minus)=0, so the projection decreases kinetic energy by ||v_plus-v_minus||_M^2/2. Before impact the cloth product velocity norm is bounded by 2u+eta+V<9/32 and the robot is stationary. Afterwards the robot speed has magnitude J/(4m). Consequently J<9m/16. These are impulse bounds for the ideal endpoint-only branch. They are independent of any selected solver output.

A step whose free endpoint penetrates before the terminal phase also includes a held replay from the original step entry. Its endpoint, active set and incoming terminal velocities need separate inclusion. Stored R7 motion, actual phase delivery, selected-pre binding, later contact iterations, the complete physical window and the original energy, angular and transaction checks remain open.
