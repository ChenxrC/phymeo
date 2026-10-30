# P6.5 parallel code and serial execution queue

User decision 2026-10-30: module code may be developed concurrently; tests run serially because of resources. No module here has executed tests yet.

## Worktree ownership

| Lane | Owner | Worktree / branch | State |
| --- | --- | --- | --- |
| Existing delivery Task3 | thick_friction_implementer | C:/Users/38347/Documents/ChatGPT/phymeo / codex/p6-5-handoff | Original source frozen, CUDA89050 currently owns execution; parent waits for explicit release |
| Integration and G1 | parent | C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo / codex/p65-parallel-modules | G1 code and tests authored, unexecuted |
| B1 bending | p65_bending_module | C:/Users/38347/.codex/worktrees/p65-bending-module/phymeo / codex/p65-bending-module | Coding from frozen contract1ccab186 |
| S1 self-contact | p65_self_contact_module | C:/Users/38347/.codex/worktrees/p65-self-contact-module/phymeo / codex/p65-self-contact-module | Coding from frozen contract1ccab186 |

All new lanes may read/edit/Git concurrently. No pytest, numerical script, solver import, compile, lint/format, packaging, dependency installation or performance run without a parent queue grant. The original Task3 command is not interrupted or restarted.

## Next execution order

1. Existing Task3 CUDA completion, same-run audits, final immutable report/resource/index release.
2. Existing delivery Task4: parent evidence check, fresh build/outside-wheel smoke, new exact-head CI under unchanged30min matrix. CI and local execution grants must not create competing physical runs.
3. One frozen B1/S1/G1 verification group at a time (dependency-ready order); all authored tests currently UNEXECUTED. Formatting/syntax checks precede that group's numerical work in the same grant.
4. B2/S2/G2 and shared integration: explicit contracts and one owner for shared changes, parallel independent coding, serial native/integration verification.

Code checkpoints do not satisfy any physics gate. Current accurate CI remains prerequisite for merging new module code into delivery; coding is allowed now in isolated worktrees by the newer user instruction. Do not reinterpret this as a request to stop overall P6.5 or shorten tests.

## Parent clarifications

B1 only checks each discrete projection's pre/post smooth branch and rejects abs(theta_after-theta_before)>=pi. It does not certify the entire affine correction segment. Both endpoints must satisfy the original angle/rest branch and geometric admissibility. Later accepted-step continuous geometry still requires certification. The B1 agent was informed before implementation completion.

S1 requests are in the supplied base point-row order, matching public TriangleOffsetPointPort indices. Its factory uses a single physical base and existing same-domain operator. Conservative AABB candidates are not collision or CCD acceptance.

G1 model/input tests are written but have not been imported, collected, linted, compiled or executed. No grasp simulation or force-balance success is claimed.
