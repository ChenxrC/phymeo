# mujoco-tyx T1：固定根旋转/平移树源码交接

**当前父验收：T1_LOCAL_VALIDATED。** `71ea1dc0` 的解释21、CUDA17、相邻8及源码外wheel两路径均通过；详[实际验收](../benchmarks/p65-mujoco-tyx-rotational-tree.md)。下文保留作者执行前交接历史，不将通过结果倒灌给原作者检查点。

**CODE_READY_TESTS_PENDING / UNEXECUTED，2026-11-10。** 产品作者分支 `codex/mujoco-tyx-t1-source` 基于父已验 `af3a78545d00cfbfa18d4efeb5da324352923a81`，保留旧 `codex/h1-foot-support` 分支。本文对应源码检查点 `f358489d`；作者没有 import、collection、数值 probe、测试、lint、formatter、编译、安装或新执行槽。父唯一串行执行，独立案例由原 actuator 作者维护；本作者没有修改测试文件。

## 新增的实际路径

- [_tree_model.py](../../src/phymeo/backends/mujoco_tyx/_tree_model.py)：只打包原 source-bound native 模型的静态字段。保全部 body ID/parent、每个 body 的完整 joint range、joint pos/axis/qpos0、qpos_spring、COM、完整非主轴 body-frame COM inertia、armature/damping、actuator 控制与力范围。**不读单个姿态的 M/J/bias 来冻结一般旋转树。**
- [_tree_kernels.py](../../src/phymeo/backends/mujoco_tyx/_tree_kernels.py)：每个实际物理步的 geometry、mechanics、solve 三个 TYX pass。
- [_tree_device.py](../../src/phymeo/backends/mujoco_tyx/_tree_device.py)：自己的静态设备字段与动态 scratch，复用原 TYX 资源方法和双状态 bank 发布协议。
- 原 [_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py) 与 [runtime](../../src/phymeo/backends/_mujoco_tyx_runtime.py) 只增加 T1 路由、设备类型、能力标签和 readonly dynamics receipt。未改 T0 核、旧接触律、公共 dispatch 或 exports。

T1 路由对含 hinge 的模型启用；原纯 prismatic T0 路径保持原行为。T1 支持固定根 1–8 个 native hinge/slide DOF，含同一个 body 内多个 scalar joint 的原顺序、分支以及零 joint 的 welded body。body 显式容量为 128，超过就拒绝，不能丢静态成员。原 T0 “所有 hinge 均 unsupported”的负控不再代表当前功能；父需把它更新为仍未支持的 ball/free-tree 或其它真实缺项。

## 每步算法与坐标

1. **原 body/joint 顺序 FK。** 从 parent 和原 body local pose 开始，依次施加同 body 的每个 joint。几何位移使用 `q-qpos0`；hinge 围绕实际 joint anchor 旋转并补 body origin 位移，slide 沿当前实际轴平移。保 authored body frame，不把它换成 principal-inertia frame。
2. **world 运动递推。** 同一入步 q/v 上计算每个 body origin 的 v、ω 以及 qddot=0 时的 a、α，含 hinge 轴运输、Coriolis 和 centripetal 项；再计算 world COM 和 COM acceleration。welded body 同样递推并参与后续质量和力，不能因为没有 DOF 就删除。
3. **完整 J/M/bias。** COM Jacobian 用世界 linear/angular 两组行。`M = Σ(m JvᵀJv + Jwᵀ Iworld Jw) + diag(armature)`；`bias = Σ[Jvᵀ m(aCOM-g) + Jwᵀ(Iworld α + ω×Iworld ω)]`。这是世界 COM 形式的 RNEA/虚功装配，所有量在本次设备 pass 重算；不是按 joint 的独立 mass，也不是 qpos0 的 M。
4. **原输入与被动力。** `applied` 包含 actuator、原 qfrc_applied 和全部 body COM world force/torque 的完整 Jᵀ 投影；`passive = -D v - K(q-qpos_spring)`。stateless affine actuator length 使用 raw native q×gear，不减 ref。单 actuator 力夹紧发生在 gear 投影/聚合之前。
5. **物理 M 与步矩阵分开。** 两次有界 dense Cholesky 分别解 `qacc=M⁻¹(applied+passive-bias)` 和 `a_integrate=(M+hD)⁻¹(applied+passive-bias)`；随后 Euler 更新 v、q。D 在 passive 中只出现一次；步矩阵中的 hD 是原 Euler 隐式阻尼，不再额外加一份阻尼力。非正/非有限 pivot 明确失败，不做 epsilon 或 eigenvalue clip。

驱动源语义另保一处审阅发现：按 CPU MuJoCo 3.3.7，disabled group 先令 actuator force=0，但之后仍参与统一 forcerange clamp。因此力范围不包含 0 时，不能把整个 disabled actuator 分支跳过。T1 新核按 `0 → 可选 gain/bias → 统一 clamp → gear 聚合` 执行；原 T0 的对应修复由父独占处理。

## `last_step.dynamics` 冻结接口

所有字段来自**同一次实际入步 kernel/readback**，不是随后 CPU forward 的补算。数组拥有不可变字节；q/v 是原 native scalar joint 坐标/速率。

| 字段 | shape / 定义 |
|---|---|
| `body_position` | `(nbody,3)`，原 body origin 的 world position，含 world 与所有 welded members |
| `body_rotation` | `(nbody,3,3)`，authored body→world |
| `body_com` | `(nbody,3)`，真实 COM world position |
| `jacobian_com` | `(nbody,6,nv)`，world linear 前三行、world angular 后三行 |
| `mass_matrix` | `(nv,nv)`，本次姿态完整 physical M，含原 armature |
| `bias` | `(nv,)`，native `qfrc_bias` 的 C+gravity 语义 |
| `passive` | `(nv,)`，原 damping/spring，不含 applied 或 gravity |
| `applied` | `(nv,)`，actuator+qfrc+实际 COM world wrench 的 Jᵀ 投影，不含 passive/bias |
| `smooth_acceleration` | `(nv,)`，上述 physical M 解；无接触时也就是本步 `qacc` |
| `input_q/input_v/time` | 本次原状态 bank 的实际设备读取；`phase='tyx-euler-input-dynamics'` |

原 receipt 的 `integration_acceleration` 单独保 `(M+hD)` 解；它不能冒充 `smooth_acceleration`。snapshot 仍是原显式 detached CPU 端点观察，phase 保持 `endpoint-cpu-forward-recomputed`；此观察不会进入 T1 step 求解。step 不调用 CPU `mj_step/mj_forward/mj_fullM/mj_rne/mj_jac`、HOST dynamics、NumPy linalg 或 Warp。

## 准入与事务边界

T1 要求**没有可能的原生接触对**，按原 weld/parent/OR mask 过滤静态判断；不能只查初态 ncon=0。free root、ball、equality、tendon、flex、plugin、mocap、userdata、激活驱动、复杂传动、joint-wide summed force clamp、非 Euler、非默认 enable/disable flags、fluid、gravcomp、dry friction 仍明确拒绝。原 source timestep、solver metadata 与 native 控制身份继续保留。

有限关节限位仍只支持 strict interior：实际入步和候选都必须严格在 range/margin 内域；进入 stop 域拒绝发布，不 clip 到限位。NaN、非正定质量表示、FP32 overflow、晚 readback 或 receipt 构造失败都不交换状态 bank，返回原完整 all-False publication mask。已有 reset/clone、held control、独立 world 和 tick 协议继续使用；不会把失败候选写回物理状态。

首批新增能力仅是**无接触 fixed-root tree**。它不表示一般 free-root tree、足地支撑、摩擦或 G1 已迁移；这些仍依赖 T2–T5。

## 提交与独立验证

产品小提交为 `d9bf39cd`（静态字段）→`25cf3758`（每步 kernel）→`f358489d`（设备/receipt/最小路由）。本报告和 NOTICE 为随后文档提交，不改变这些待执行源码字节。

独立测试作者已从原 MuJoCo 预登记偏心单摆、非零 v 双摆、同 body R/P、多分支、七/八 DOF、非主轴 COM/I、massive welded tool 真实 world wrench、真实限位和 late-failure/clone 等案例。由父分别执行解释和实际 CUDA，并保首次失败、实际 source/commands/readings、同次入步 M/J/b/a 与短窗原 native oracle。不在这里预报通过数量，不重跑已绿 T0 同源组来替代 T1 验收。

## 来源与归属

算法字段与分层参照固定 [mujoco-warp v0.0.2 smooth.py](https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/mujoco_warp/_src/smooth.py) 的 kinematics/com_vel/crb/rne、[passive.py](https://github.com/google-deepmind/mujoco_warp/blob/v0.0.2/mujoco_warp/_src/passive.py)、原 scalar transmission/Euler；T1 改写为 TYX 数组和世界 COM 递推/完整 Jacobian 装配，未包装 Warp 执行。沿用包内 Apache LICENSE/AUTHORS，NOTICE 追加具体修改和这批文件对应关系。原 native MuJoCo 是独立 oracle，不借产品结果生成答案。
