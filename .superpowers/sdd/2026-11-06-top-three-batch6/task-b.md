CLOSED — accepted within B2 declared scope. HOST18, interpret1, CUDA1, affected compatibility4 andsource-outside wheel passed. Report: docs/benchmarks/p65-dual-turn-mesh-round.md. No further authoring or execution grant. Instructions below are historical.

# B2 author brief

Status: TEST_AUTHORING_ONLY. Parent only serial project execution.
Plan: docs/superpowers/plans/2026-11-06-top-three-batch6.md.
Reuse original rr author/tree; clean check, ordinary fast-forward supplied parent commit; no reset/WIP overwrite.

Only NEW:
- src/phymeo/geometry/_dual_turn_mesh_round.py
- tests/collision/dual_turn_mesh_round_cases.py
- tests/collision/test_dual_turn_mesh_round.py
- tests/collision/test_tyx_dual_turn_mesh_round.py
- docs/benchmarks/p65-dual-turn-mesh-round.md

First freeze all cases/oracles and tests, no implementation yet. Public main test imports existing API and fails due actual both-rotate unsupported status; missing module/import errors are invalid RED. Parent owns collision_pipeline.py and all existing shared files.

Proposed DualTurnMeshRoundQuery(prepared_mesh,round_shape,mesh_path,round_path,mesh_first) inherits MeshRoundQuery and original five-field result.
In initial mesh coordinates finite points x_i(t)=p_m+R_m(t)^T[d+t*v+R_r(t)*a_i]. Both actual nonzero axes/moving pivots retained. Conservative speed S=|v|+|theta_r|max|a_perp_r|+|theta_m|[max(|d|,|d+v|)+max|a|], exact rational upper norms. Rotation enclosure errors and storage/world reconstruction all accumulated. No unsound finite sampling as path exclusion. Reuse closed mesh/finitesegment-distance/bounded search. Map both material sides, normals, offsets and all tied addresses at actual witness t; unknown stays unknown under old caps.

Eight predeclared inputs:
1 capsule, 2 sphere: box[-2,2]^2 z[7/8,9/8]; capsule0->Y or sphereY,r1/8; mesh X angle-2A,round Z angleA,A=storedpi/2. root pi_real/(6A), endpointclear/interiorhit; independent pi interval, not exact1/3.
3 mesh pivot travel(1/8,-1/16,3/32),round additional(0,0,1/64). root sin(2At)cos(At)+(t/64)cos(2At)=3/4 bracket[5/16,1/3]. Independent Taylor interval oracle.
4 box x±1/2,y±2,z[1,5/4] Zangle1; capsule(-3,1,0)->(3,1,0) Xangle3,r1/8. first asin(7/8)/3, end balls all clear; independent finite axis span.
5 main box bottom1 genuineclear margin7/8-4/(3sqrt3)>0.
6 actual concave U, axis±(1/4,0,0),r1/8,meshX.9/roundZ1.3,clear>=5/8. Independently verify U placement convention before fixture; no output-tuned constants.
7 containment box±2,axis±(.5,0,0),angles.9/1.3 first0.
8 puregrazing mainboxz[5/8,7/8],angles-A/+A,maxheight1/2: hit or unknown, neverclear.

Mirror/global rigid transform/swap/worldidentity, full witness reconstruction, capacity/invalid mesh/device failure recovery included. Real interpret/CUDA broad phase comparisons; no new GPU CCD claim. Tests on fixed independent controls; new producer cannot serve oracle. No project execution/import/numerics/static/build/agents. Tests-only small commit, exactREDnode/anticipatedoldguard, then wait parent.
