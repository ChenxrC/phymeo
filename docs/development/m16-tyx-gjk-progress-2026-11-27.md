# Finite-hull simplex progress at the saved moving M16 state

The original 98-by-9-vertex pair exhausted GJK's 64 iterations after the
first accepted 1 ms M16 step. The saved GPU trace identifies a numerical
selection error: an interior triangle improves the squared closest distance
over its edge by 6.6637483e-12, but the old rounded projections reverse that
ordering by 1.1641532e-10. The reducer therefore repeatedly adds and discards
the same support point. The independent comparison uses the exact FP32
simplex vertices retained by the kernel, not a different model.

The shared finite-support reducer now keeps high/low FP32 lanes through the
interior edge/plane projection and squared-norm comparison. It reuses the
existing copy scratch before simplex compaction. Closest coordinates and
barycentric weights retain their existing public storage types. This is
compensated FP32 arithmetic, not hardware FP64. Curved full-surface projection
and tie handling retain their prior path. No support vertices, iteration
budgets, convergence tolerances, or UNKNOWN handling change.

The exact case now resolves in 10 iterations. Its original distance is
approximately 1.0202469333 mm. Five CUDA regression variants cover the saved
case, reversed pair order, reversed vertex order, a rotated frame, and a
translated frame. These and 214 affected geometry regressions pass (219 total);
the final five-case formatting rerun overlaps that total. Ruff and diff checks
also pass. Re-querying the original eight-world moving state resolves all
51,650 candidates. This establishes the concrete geometry repair, not general
long-trajectory accuracy or performance qualification.

## Stopped moving-state validation

Two separately registered component probes preserve the original integrator,
model, physical criteria, and same saved starting state. Each has an independent
64-world-step cap, charges its first 16-step reservation, and closes with zero
known GPU/native integration, 48 unused steps, and no unsettled reservation.
No unused budget is permission to retry.

The first probe rejects contact capacity: one world requires 800 contacts,
also present in the native reference, while the old buffer holds 640. Its
overflow status prevents consumption. A separately registered probe uses
1,024 bounded slots throughout contact and solver workspaces. No contact is
silently dropped or admitted with an overflow status.

The corrected-capacity probe passes geometry and the coupled solver's own
checks, then stops before integration on independent native force balance:

| World | Normalized balance residual | Original criterion | Contact counts GPU/native |
| --- | ---: | ---: | ---: |
| 1 | 5.00969320996e-6 | 2e-6 | 470 / 470 |
| 4 | 1.21939758387e-3 | 2e-6 | 453 / 452 |

World 4's extra pair is geom 350/1736 (body 1/2): GPU gap
-8.61027915633e-11 m, native distance query zero. The same gap and normal exist
in the parent query, before this GJK fix. This is a near-zero activation
disagreement, not evidence that either contact decision is physically correct.
It is not an identified source of all force errors: world 1 has equal contact
sets and still fails the balance criterion.

Across matched contacts, point error is at most 3.72e-10 m and normal error
about 1.07e-7; mass matrix error is at most 3.80e-14 and smooth-force error
1.63e-8. World 4 also fails the original componentwise acceleration criterion
`0.002 + 5e-4 * abs(native)`, with maximum absolute difference 0.00785655
(generalized coordinates have mixed units). These observations narrow the
next investigation to near-zero activation and moving constraint response;
they do not establish a unique root cause or justify relaxing tolerances.

Evidence is preserved under
`D:/cxr/alphanist-online-20261026/phymeo_tyx_gjk_progress_20261127`, including
both closed ledgers, raw state/contact/solver arrays, independent comparison,
the failed old trace, and repaired trace. Follow-up should use these saved
states to separate geometry certificates, activation decisions, Jacobian /
reference-acceleration / constitutive response, and solver residual. Further
movement requires a concrete correction and a new bounded plan.

This change is a geometry component fix. The four-step original-model gate
remains failed; there are no new policy samples, PPO updates, training release,
32/256-world qualification, or claims that the old native crash is repaired.
