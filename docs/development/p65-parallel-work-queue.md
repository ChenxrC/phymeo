# P6.5 parallel code and serial execution queue

User decision 2026-10-30: module code may be developed concurrently; tests run serially because of resources. Later user decision explicitly defers remote CI to a new machine; remote CI is no longer a local execution prerequisite. CI and Documentation are manual-only in delivery50a5eeb; complete job definitions retained.

## Worktree ownership

| Lane | Owner | Worktree / branch | State |
| --- | --- | --- | --- |
| Existing delivery Task3 | thick_friction_implementer | C:/Users/38347/Documents/ChatGPT/phymeo / codex/p6-5-handoff | Local Task3/4 complete and released; e211138 remote3CPU pass/5timeout retained; remote CI deferred by user, policy50a5eeb pushed |
| Integration and G1 | parent | C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo / codex/p65-parallel-modules | G1 code and tests authored, unexecuted |
| B1 bending | p65_bending_module | C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo / codex/p65-bending-module | B1 verification queue grant now assigned to original agent from ee667918; no previous tests, results pending |
| S1 self-contact | p65_self_contact_module | C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo / codex/p65-self-contact-module | CODE_READY_TESTS_PENDING, commit8d6a403; six files clean, zero execution |

All new lanes may read/edit/Git concurrently. No pytest, numerical script, solver import, compile, lint/format, packaging, dependency installation or performance run without a parent queue grant. The original Task3 command is not interrupted or restarted.

## Current execution order

1. Original Task3/4 local tests/audits/build/wheel are complete and will not be repeated. Remote CI41007037491 ended with3CPU success/5timeout, Documentation success. User deferred remoteCI; no rerun/poll or further optimization solely for this hosted limit. Policy50a5eeb preserves every job and disables automatic push/PR triggers.
2. **B1 owns the sole local executable-validation slot.** Original p65_bending_module continues from ee667918 in its worktree. Merge the policy/delivery checkpoint if clean and it changes no underlying physical source; use existing Python environment with explicit child-src import path. Static checks and tests/test_bending.py plus tests/test_bending_trace.py, fresh basetemp/rawexit/source manifest. Preserve initial post-code results honestly; fixes stay in B1 allowlist. Parent and S1/G1 do not execute checks concurrently.
3. On B1 natural completion/clean/process0/report release, grant S1 its candidate/operator tests, then G1 builder tests, one group at a time. No baseline whole-suite or old long-window replay.
4. Parent audits versioned results, integrates additive modules and freezes B2/S2/G2 shared work. Parallel isolated coding remains allowed, shared seams have one owner, all physical/static/build validation uses the same serial queue. Local correctness and installation gates remain; remote CI coverage is explicitly deferred until user provides a new machine.

Code checkpoints are not physics acceptance. Full local H2/P6.5 acceptance still requires native/material/contact/hold/slip/fold/transaction evidence. Do not relabel cancelled remote jobs as passing or let their deferred state stall unrelated local tests.

## Parent clarifications

B1 only checks each discrete projection's pre/post smooth branch and rejects abs(theta_after-theta_before)>=pi. It does not certify the entire affine correction segment. Both endpoints must satisfy the original angle/rest branch and geometric admissibility. Later accepted-step continuous geometry still requires certification. The B1 agent was informed before implementation completion.

S1 requests are in the supplied base point-row order, matching public TriangleOffsetPointPort indices. Its factory uses a single physical base and existing same-domain operator. Conservative AABB candidates are not collision or CCD acceptance.

G1 model/input tests are written but have not been imported, collected, linted, compiled or executed. No grasp simulation or force-balance success is claimed.
