# BALL-C 之后的机器人组合路线

状态：**RESEARCH_ONLY / NOT_IMPLEMENTATION_GRANTED**。2026-11-12 06:10–06:35 的一次有界源码研究；本文件没有产品实现、项目执行或新验收结果。父分支已告知 BALL-C 新组 52 项、兼容 6 项和安装后两后端各 64 个 world-step 通过；本文复用该结论，不重新核证据。源码观察基线为父 `2f97aecfaa88afe0eba728604d049084ad88d9f1`，作者树只新增本文件。

## 1. 推荐顺序与目前真实边界

推荐下一机器人实施片为 **BALL-M1：通用 native JOINT actuator moment 表 + 固定 BALL 树的直接 NONE 扭矩电机**。这是传动层的可复用纵向片，不再为每个球轴写单独力学分支。先限定无接触/无 activation 的新目标域，验收后沿同一映射补 FILTEREXACT 与已接受 BALL-C 接触的组合。整个机器人组合矩阵并没有因此完成。

| 顺序 | 路线 | 已有可复用能力 | 真正缺口 | 改动与风险 |
| --- | --- | --- | --- | --- |
| 1 | 直接 BALL JOINT motor | BALL 的原生 qpos/DOF 地址、完整 M/J/b、四元数积分、公共 runtime/事务 | 三列 actuator moment、标量力到完整广义力的统一 scatter、源/收据绑定和准入 | 中等；不改积分器。不能只删除 target 拒绝门，现代码只写一个 DOF |
| 2，可独立并行 | SceneDeclaration → ROB1 多 participant | ROB1 四 family、共同入步载荷、每 mechanical block 一次推进、world 原子发布 | 一 world 多 participant 的声明、源 collider/owner/control 身份 lowering、静态/动态工具归属、组合资格 | 中等；主要是装配/API。不是新增接触求解器，也不是把 MuJoCo BALL 自动带入 resident tree |
| 3 | FREE+BALL | 固定 BALL 三列几何、T2 单 FREE + scalar 的根运动学，各自已验 | T2 native joint→DOF/qpos 多宽度表、祖先三列、convective bias、子球积分与可变 nq | 较高；不仅 dispatch。目前 T2 的 `nq=nv+1`、每个非根 joint 一 DOF 假设都不成立 |

公开的两条路径必须继续分别标域：

- MuJoCo-TYX：`compile_mujoco_xml(...)` → `MuJoCoRuntime(..., implementation="tyx", mode="interpret"|"cuda")` → control/step/snapshot/integration_snapshot/checkpoint。BALL-A/C 指这个 native-model 路径。
- resident：`ResidentParticipantKey`、`ResidentParticipant`、`ResidentParticipantGeometry`、`ResidentSceneBatch` 与 `ResidentParticipantContactPlan`。ROB1 是共同入步接触载荷、各机械块一次求解、world 一次发布；不是跨树未知接触力的隐式 KKT。
- `SceneDeclaration`/`compile_scene` 当前是另一装配入口，每 world 一个 articulation。多机器人 HOST 场景已有自己的 participant 实现；它不等于已经接通 resident SceneDeclaration，也不等于 MuJoCo native 多 FREE 根。

## 2. 源码证据：哪些是接线，哪些是新数学/ABI

以下本仓库路径相对 repository root；行号固定于上述观察基线，后续实现后按符号定位。

| 观察 | 实际位置 | 推论 |
| --- | --- | --- |
| BALL-C 先完整 structural tree 检查，再按 possible pairs 分 normal/constraint | `src/phymeo/backends/mujoco_tyx/_tree_model.py:38–54` | 接触组合已接通；不能绕 structural pass 放开其它球能力 |
| BALL actuator target 仍只允许 HINGE/SLIDE | 同文件 `74–86`；全部 actuator 再检查 `148–167` | 直接球目标是真缺口，当前拒绝正确 |
| actuator 编译仅保存 gear[0]、一个 qadr/DOF | 同文件 `236–260` | 不是单一能力标签/dispatch 的问题 |
| mechanics 以 scalar length/velocity 算一个力，再写一个 DOF | `_tree_kernels.py:349–364` | 球 qpos 的首个数不能当角度；gear 后两分量也不能丢 |
| 通用 scalar activation 的 ids/parameters 和 receipt 是显式 scalar-v1 | `_actuator_model.py:54–115`、`mujoco_actuation.py:38–64` | 后续复用 activation 积分，但不能悄悄重新解释旧 schema；na=0 也不能创建假 activation |
| T2 只准 leading FREE + scalar，祖先后续只置一个 DOF | `_floating_model.py:33–56,147–189` | FREE+BALL 需 native 地址表升级；固定球树不能通过 wrapper 冒充浮根 |
| T2 对 i>=6 直接积分 q[i+1]，根 quaternion 单独积分 | `_floating_kernels.py:493–530` | 子球需真正 quaternion 积分；不能模拟三个 hinge |
| SceneDeclaration 检查 world ID 唯一，每个 world 只有一个 ArticulationWorld | `builders/scene_compiler.py:1–12,218–236` | 应加显式多 participant 声明，而不是删重复 world 检查 |
| 现 scene lowering 已识别四 family，collision key 只按 world 分配 | 同文件 `337–418` | 可复用 asset lowering，但 owner/name/control 都需加入 participant 维度 |
| ResidentSceneBatch 已用统一事务 service | `backends/resident_scene.py:31–52`；`_resident_participant_layout.py:37–54,270–334` | scene adapter 应调用公共 resident API，不调用 bank、不复制积分器 |

固定一手来源：

1. [MuJoCo 3.3.7 transmission](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L934-L971)：JOINT scalar moment 为 gear[0]；BALL moment 是 native 球 DOF 的三维 gear。JOINTINPARENT 会另转轴，首片不支持。BALL length 使用 quaternion rotation vector，不能等同 qpos 第一项。
2. [MuJoCo 3.3.7 actuation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L418-L444) 与 [最终力投影](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L516-L521)：先 actor scalar force 和 clamp，再 moment 转置乘 force。此次还只读了原固定源码缓存，未调用 native。
3. [Newton 1.7.0 SolverMuJoCo](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/mujoco/solver_mujoco.py#L7282-L7362)：BALL 保留 4 个 qpos/3 个 DOF；其球执行器按 native BALL 的各 gear 轴建 actuator，并有额外 target-coordinate 映射。它不是三个 hinge。此处只借用传动结构，不宣称已复制 Newton 球姿态控制模式。
4. 固定安装源码复核：父 `.venv-newton-1.7.0/Lib/site-packages/newton/_src/solvers/mujoco/solver_mujoco.py:7282–7362`，其 dist-info 明确版本 1.7.0。已存 `artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src/smooth.py:1814–1827` 的 Warp 0.0.2 同样区分 scalar 与 BALL moment。CPU 3.3.7 是行为 oracle，不能因 Warp 类似就继承资格。两上游均 Apache-2.0；移植时保留归属与许可证。

## 3. BALL-M1 的有限通用合同建议

### 3.1 输入、映射与力

首片拟能力名 `fixed-ball-joint-motor-no-contact-v1`。沿现有固定 BALL + R/P 模型，nv<=8、nq<=10、nbody<=128、nu<=32；新 BALL 目标要求 JOINT/NONE/FIXED/BIAS_NONE、na=0。可以与已支持的 scalar NONE/affine-bias actor 共存。新球 gear[:3] 是任意有限向量，包括斜向、负数和零向量；gear[3:] 必须为零。

编译一个 immutable **native JOINT transmission table**，而不是专用“单球三轴”对象：

- `row_offsets[nu+1]`、`dof_indices[nnz]`、`moment[nnz]`，绑定 native actuator ID、joint ID/type、qpos address、DOF address 和原 gear。
- scalar 每行 1 个 native DOF；BALL 每行完整 3 个 native DOF，含系数为零的列。nnz<=3*32=96；构造先验证地址、完整行覆盖、dtype/finite 和容量，再分配设备内存。
- actor 仍只产生一个 scalar force。统一按 native actuator 顺序将各行 `moment * force` 累加进完整 qfrc_actuator；共享目标允许叠加，不覆盖前一个 actor。零 gear 不删 actor 或漏掉其 clamp/身份。
- public 输入依旧为 MuJoCoControl。NONE 控制先按 ctrllimited clamp；FIXED gain 产生力；disabled 保留原 native 零 raw-force 后 force clamp 的顺序（正的 forcerange 下不能擅自保证最终力为零）；最后才传动。
- 保留原 physical M 与 Euler step M+hD 两个用途。不得由 actuator 子片修改 M、J、b、积分器、根/球 normalization 或原失败门。
- BALL BIAS_NONE 不依赖 length。首片不得把四元数首项当 length，也不发布一个伪造的 actuator_length。未来 BALL affine bias/position target 必须补真实 rotation-vector length、velocity 和相应 native 边界测试。

建议让新的 public transmission evidence 记录原表、actual device scalar forces、完整 generalized force 和同一次 input/control binding；作为新 optional typed extension，不重新解释现有 `native-scalar-filterexact-actuation-v1`。严格要求实际 device 中间量，不能由 HOST 期望公式填充“actual”收据。

本表未来也给 FILTEREXACT stage 使用：先沿现有更新算 scalar force，随后调用同一 moment mapping；activation 仍按 native actadr/actnum 归属。此次不提前更改已验 scalar activation ABI。

### 3.2 拒绝域与公共生命周期

对**有新 BALL-target actor**的模型，在任何 device allocation 前拒绝：FILTEREXACT/其它 dyn、BALL affine bias、JOINTINPARENT/SITE/TENDON/BODY transmission、非零 gear[3:]、FREE、可能接触、joint limits/dry、ball spring、equality/closed loop、同 body 多 joint、joint-wide actuator force clamps和超容量。旧无球目标的 BALL-A/C、ROB2b 域必须原样保留，不能把这个新首片的限制全局施加给旧域。

全部 q/v/ctrl/time、完整 diagnostics 和新 transmission receipt 一次候选共同发布；晚失败保留原 accepted bank/控制/clock/receipt。沿用 checkpoint/clone/partial reset/restore。没有新 actuator state，na=0 保持真实空 act。不得引入 CPU mj_step 回退、隐式模型重编译或 step 时 CPU dynamics。

## 4. 预声明十组案例与独立 oracle

这是待实施固定矩阵，不是本轮通过数。作者在 tests-first 提交完整 XML/输入后，父再授权唯一串行执行。两 mode 都运行；W={3,29} 为稀疏键，两 world 自己持续推进，不能每步把 CPU 状态覆盖成 TYX 状态。共同连续窗口 h=0.001、24 步；命令分 0–7/8–15/16–23 为 +0.8/-0.75/0，world29 的非零命令另加 0.015。每例保实际非零初速度；范围内会产生真实运动，不要求长期稳定。

| 组 | 固定几何/传动及主要检查 |
| --- | --- |
| M01 | 单固定 BALL、偏心 COM 和非对角惯量，分别 unit X/Y/Z gear；验证三列 native moment、全轴响应与完整 M/J/b |
| M02 | 同模型非单位斜 gear=(1,-2,0.5)，非 identity ball quaternion；负/混合轴不能误作世界 torque 或只留第一列 |
| M03 | 同球多个 actor，含共享/反向 gear 和不同 gain；独立按 actor 顺序累加，验证无覆盖、无重复加力 |
| M04 | 两个 BALL 分支 + 一个 scalar（nv=7），native joint/qpos/DOF 地址不同；检查 ancestor/全 M 交叉、只作用目标的局部 moment |
| M05 | BALL + HINGE + SLIDE，球 NONE 与 scalar NONE/affine bias 共存，qfrc_applied 和非零 COM xfrc 同时存在；验证所有力共同进入同一次运动 |
| M06 | ctrlrange ±0.6、forcerange ±0.7、gain=2；依次 reversal/unload，证明先 scalar clamp 再映射，而不是各 torque 分量单独 clamp |
| M07 | disabled group + 正 forcerange [0.1,0.25]；另 zero-gear actor。保 native 最终 force 行为、零广义力及完整身份，拒绝“disabled 全零”错误 oracle |
| M08 | 同一 unit quaternion 的 q/-q 两个独立入口，非零局部 angular velocity；只比较等价姿态/真实 native 后继，保原 quaternion 字和源地址，不任意规范化 raw 证据 |
| M09 | W2 checkpoint→真实下一步→restore→同控制下一步、clone、partial reset；末 world publication 故障与失败后 retry。旧 accepted state/controls/clock/receipt 一起保留 |
| M10 | 预分配拒绝域：新球目标 FILTEREXACT、affine bias、JOINTINPARENT、FREE、可能接触、limits/dry、尾 gear、nu33；非法 control/四元数也必须保持原 bank。每个子项独立断言原因，不能仅 catch 任意异常 |

每步有两条不同 CPU 参考：fresh native data 在同一实际 TYX 入步状态做 mj_forward/mj_step，用于 full M/J/b/力与单步；另一份 native data 从原初态独立跑完整 24 步，用于累计 q/v。native oracle 允许单独执行，TYX step 区间禁止 CPU mj_step/mj_forward。另以原 gear、native地址和 authored gain/clamp 独立重建 scalar force、moment 转置投影及功率关系；不调用产品 compiler/producer 当 oracle，不用 native 输出反写产品状态。

沿已验 BALL 原门：body position/rotation/COM rtol2e-5 atol3e-5；J 2e-4/5e-5；M 2e-4/3e-5；b/passive/applied 3e-4/2e-4；smooth/integration acceleration 5e-4/8e-4；连续 q atol5e-5、v atol3e-4、time atol2e-6。scalar/generalized actuator force 沿 BALL-A rtol3e-4 atol2e-4。不以此说明任意高 gain、惯量或长窗都合格。坏条件模型应拒绝/报失败，不用该片调门。

## 5. 文件所有权与实施次序

以下是建议 allowlist，**尚未授权产品实施**。先冻结真实 table/receipt ABI，再开编码；不得在原 schema 中插入未声明列。

| owner | 文件范围 | 责任 |
| --- | --- | --- |
| 原机器人作者 | NEW `src/phymeo/backends/mujoco_tyx/_joint_transmission.py`；MOD `_tree_model.py/_tree_kernels.py/_tree_device.py` | 静态 native moment 表、实际标量力/统一 scatter、表与设备资源；复用原 mechanics/solve |
| 父共享接缝 owner | `_model.py`、`_mujoco_tyx_runtime.py`，必要 NEW `mujoco_joint_transmission.py` public record | 明示能力/路由、immutable receipt 和共同发布；不碰旧 checkpoint 字节或 act schema |
| 父共享接缝 owner | 若 mechanics ABI 改动，仅 `_normal_device.py/_constraint_device.py` 的调用参数 | 保留旧 wrapper 的完整表/字段与 scalar 调用；不准入未测新球电机接触 |
| 原测试作者或机器人作者独占 | NEW `tests/test_mujoco_tyx_ball_joint_motor.py`、`..._transactions.py`、`tests/installed/check_ball_joint_motor.py` | 十组固定矩阵、独立 oracle、源码外有限消费者；不能导入私有 bank |
| 另一个独立 author lane | 仅先 SceneDeclaration/ROB1 装配合同与 tests-first | 不碰上述 native files；父审后另授前端文件，不能与本片混合提交 |

本片不需要修改 `_floating_*`、`_actuator_kernels.py`、contact solver、training/PPO 或 G1。如果实现要求这些改动，应停止当前片并报具体依赖，而不是顺手扩域。旧 scalar 的 arithmetic/order、3-launch stateless路径和资源释放必须有 focused compatibility；只补实际受影响资格，不重复已绿全套。

实施顺序：table/拒绝域与 tests-first → 通用 stateless scatter 接线 → 两后端十组及事务 → 一次新 wheel 的公共有限消费者。每次只修实际 source/API 问题；原失败原样保存。验收后将 BALL-M1 列为矩阵中一格，随后同表接 FILTEREXACT，再资格 direct-ball motor + BALL-C 接触，不能用分片数量当“通用框架完成”。

## 6. 两条后继线的明确入口

**SceneDeclaration → ROB1**：建议新增显式 participant assembly 版本，不破坏现 SceneDeclaration.worlds 的唯一 world 含义；一个 logical world 下按稳定 participant ID 组织 asset/state/control 和 collider owner，静态环境独立表。调用已有 ResidentSceneBatch 和 ContactPlan，不访问 storage。首批只复用已验 R01 双自由体、R02 臂推盒、R04 四 family 混合及 W2 隔离/晚失败/partial reset；R03 两个已知数值失败不得借 adapter 声称解决。HOST patch、33 physical bodies/world、原 site/candidate caps 不变；公共装配不意味着 native BALL/MPM/柔体可导入。该线独立于 direct motor，可并行开发前端但执行仍串行。

**FREE+BALL**：先写一张 native joint→qpos/DOF/ancestor 表和完整 root/child angular transport ABI，再选单 FREE + 单 BALL + 至多两个 scalar 的首域。nq 应按每 FREE/BALL 的宽度计算，不能沿用 nq=nv+1；root world-linear/local-angular 与子球 local-angular 需分别验证。独立案例至少 free rigid covariance、球链 COM 偏置、full M 交叉、bias、两个 quaternion 更新、受力/无载/世界隔离。没有这些实例就不能删 FREE 拒绝门；更不能同时引入多 FREE 根、球 limit、motor/contact全部组合。

## 7. 成功与停止规则

成功必须同时有：native moment 身份/轴正确、实际 scalar force 和完整广义力一致、同入步动力学与独立连续轨迹通过原门、故障原子保留/恢复、两后端、源码外公共消费者。只输出非零姿态或只有 motor 辅助类不算连通。

第一轮失败先区分固定输入/参考/单位/API 与 actual 算法；不通过改 gear、质量、控制窗口、初态或容差追绿。连续两轮不能将同一失败缩到一个可证伪层时，记录最小输入、实际差异和下一问题，暂停该格，继续不依赖它的 SceneDeclaration 装配线。不得用无止境精度证明阻塞整个机器人/训练路线。

本研究到提交即结束；没有自动长 G1 重跑、性能超越声明或新执行授权。
