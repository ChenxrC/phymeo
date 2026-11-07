# Collision capability registry

The callable registry is [collision_capabilities.py](../../src/phymeo/geometry/collision_capabilities.py). It is scoped by unordered shape pair, operation, owner mode and backend. It records facts; runtime dispatch still owns the actual algorithm choice.

The accepted source baseline is 8128440, with original acceptance reports linked from each verified row. No old suite was rerun. Implemented-but-unverified routes are deliberately distinct from verified cases. A point witness does not imply a pressure manifold, geometry does not imply robot response, and a hybrid TYX broad phase with HOST narrow phase does not imply device-only collision. Unknown combinations return an explicit missing record.

| Family | Reused route | Current expansion |
|---|---|---|
| sphere / finite capsule | rational finite-axis geometry and closest-set witnesses | common weighted patch/load adapter |
| box / general hull | full hull support patches; box remains only a special hull | robot compiler must retain arbitrary hull materials |
| mixed round / hull | support-map point geometry and dedicated finite OBB-capsule method | complete contact patch where existing route is only a point |
| mesh / round | finite closed mesh and existing path queries | rotation-sample TYX face candidates in B |
| triangle / hull or triangle | full finite material features and adjacency | keep separate deformable owner contract |
| tetrahedral cells | positive-volume cells and material owners | reuse existing accepted geometry |
| cylinder / cone / ellipsoid | scoped support-map geometry | distributed pressure/owner coverage remains separate |

The new independent library has all ten S/C/B/H unordered pairs, each at +0.02, 0 and -0.01 m parallel-support gaps. H is a ten-vertex pentagonal prism, not an OBB proxy. Inputs preserve unique owner keys and original vertex IDs. These are declared cases, not claims that all geometry and dynamics already passed.

2026-11-07: the first registry run failed because the new module did not exist (raw exit 2). After implementation, the focused registry/material-input suite passed 37 tests (raw exit 0). This verifies capability lookup, evidence requirements, scope isolation, immutability and case construction only. Real patch, load, dynamics and device tests remain in the [active plan](../superpowers/plans/2026-11-07-generic-collision-until-0800.md).

Receipts: artifacts/p65-generic-collision-20261107-01/01-a-k0-red and 02-a-k0-host. No whole-engine completion percentage is inferred.
## 2026-11-07 A/K3 专项接受

25c39d8 接通十种 S/C/B/H 组合的固定/浮根真实所有者及机器人–自由体一般凸包。最终98项专项/兼容性通过；公开CC/CB/HH三组各两步长连续0.5s，共1152步，末态位移差约1.43736e-6m、总冲量相对差约1.54171e-5。旧pair保留原采样，混合一般H不改变旧摩擦/pad；完整资产名称歧义先校验。详[当前三线结果](../benchmarks/p65-generic-collision-20261107.md)。

注册表新增 patch_adapter/manifold 和 robot_robot/response 的限定接受域；HOST浅接触及唯一法向，不升格CUDA、一般深穿入、CCD或任意摩擦历史。robot_rigid只含机器人S/C/B/H与自由体S/B/H（胶囊只能在机器人侧）。N4 Newton静态留出、组合wheel仍待本轮完成。
