# 驻留接触规模：分页 bundle 与原型实例化研究

**RESEARCH_NOT_IMPLEMENTED，2026-11-10。** 仅阅读父 `7e378f7cec775ca7e4c7f7c8cd92c4048fb8d0b1` 源码、原 K1 未提交计划及 Newton 1.7.0 官方资料；本轮没有项目 import、数值、测试、编译、安装或实现。下文新类型/方法均为讨论边界，不是已存在接口，也未冻结实施合同。目标是 W256 各自的动态物体与有限静态地面，在保留每页 256 shape 等原容量的前提下，经过**一次物理发布**。

父任务已报告 W256 混合普通推进通过，512 shape 的单模型注册被原容量拒绝。这说明动力学批量与接触注册仍是不同层；不能用普通 world-step 数外推受载能力。本文不重复那些执行或验收。

## 当前源码究竟提供什么

| 现有入口/限制 | 已读到的事实 | 对扩展的影响 |
|---|---|---|
| [CollisionModel](../../src/phymeo/constraints/collision_shapes.py) | 单模型 `MAX_SHAPES=256`、总顶点 8192、单凸体 64 顶点、网格 128 三角；`CollisionKey.world` 为非负 i32。 | 512 shape 不能塞进一个原模型；不能用 world=-1，也不能把各世界地面别名成 world=0。 |
| [TyxWorldBatch._step](../../src/phymeo/backends/tyx_world_batch.py) | 目前只接一个 `ResidentContactPlan`。上传持有控制；begin attempt；每个 tree storage 推进一次；finish/check；最后交换 active bank、epoch 和接触收据。 | 连续调用两次 `step_contacts` 就是两个物理步；传 list/bundle 当前会被类型检查拒绝。 |
| [ResidentContactPlan](../../src/phymeo/backends/_resident_contact_plan.py) | `_begin_attempt` 对其 storage 做 entry mechanics，读实际状态并 HOST 采样有限接触，上传每个 storage 的全尺寸 wrench/reject；`_stage_tree` 再投影、solve、candidate geometry 和 finalize。 | 同一 storage 可含多个页的世界，但每个旧 plan 都会遍历整个 `physical.count`，不是只推进本页行。 |
| [ResidentCollisionPlan](../../src/phymeo/backends/_resident_collision_plan.py) | 每 shape 明确绑定原 world；同世界 dynamic/static 配对。可选动态对仅限同一完整树的非相邻不同组。静态世界必须有实际动态绑定。 | 可以研究把不同完整 world 装到不同小模型；不能靠改 key 消除世界隔离，亦不自动增加“每 world 多个独立动态物体”。 |
| [接触桥](../../src/phymeo/backends/_resident_contact_bridge.py) | 仅有限 sphere/capsule/unrounded hull，`RobotContactMaterial.friction==0`，discrete entry，实际材料点与双侧力矩/功。`sample_tree_contacts` 的 group_count 目前仍限 1..17，尽管 frame group_index 可到 32。 | 29 关节设备动力学不等于 30-body G1 驻留受载已支持；这里还有 group 容量与摩擦域两个独立边界，本研究不改它们。 |
| [ResidentContactStepper](../../src/phymeo/training/resident_stepper.py) | 借用原 batch/plan，严格检查 `ResidentContactPlan` 与 owner。 | bundle 若获准实现，训练入口的类型边界也需显式补充；当前不能直接借 duck typing 接进去。 |

尤其不能把几个旧 plan 的 `_stage_tree` 放进循环：每个都会重写同一个 mechanics/vector/candidate bank，可能擦掉另一页载荷并再次 solve。即使各页 world 不重叠，这也不成立。

K1 的[原未提交计划](../superpowers/plans/2026-11-08-device-convex-patch.md)只研究设备完整接触集合及权重；它明确不接载荷、Jᵀ、solve 或发布，K2 才消费。这次不能修改旧 WIP 或把其解释器局部记录当完成。当前受载链仍调用 HOST `contact_patch_sample`，分页本身不应声称设备窄相或零往返。

## 固定 Newton 1.7.0 的参照与边界

官方 worlds 文档把 body/shape/joint 等实体按 world 分组，提供各类 world-start 索引；不同局部 world 隔离，全局 -1 可用于公共静态物体。文档建议同构世界使用 `add_world`/`replicate`，并提醒异构求解器支持有范围。[1.7.0 worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html) · [world collision filtering](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html#world-indices)

固定 tag 的 `replicate` 批量合并实体，`_merge_builder_copies` 为各 world 展开实体数组并偏移引用；部分几何对象引用可共享。它不是“全部 shape/state 只存一份”的零复制原型证明。借鉴点是静态几何来源、实例身份和 world 状态分离，而不是照抄 -1 语义或宣称已有同等性能。[v1.7.0 builder.py：replicate/merge](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/sim/builder.py#L3004-L3234)

## 两条路线比较

|  | A：有限页 + 单次发布 bundle，建议先做 | B：每 world 原型实例化，后续研究 |
|---|---|---|
| 注册模型 | 每页原 `CollisionModel`，所有原 cap 保持；首片按完整 world 分区。 | 原型几何与局部配对表有 cap；独立实例表绑定真实 world/body/group/material 身份，需新注册/结果类型。 |
| W256×2 shape | 两页，每页 128 个完整 world、256 shape；batch 仍是同一个 W256 owner。 | 两形状模板可复用几何，但 256 份动态状态、控制、接触和反力身份仍分开。 |
| 力学改动 | 共用一次 entry mechanics，合并后每 storage 一次投影/solve/finish。可复用当前 HOST law。 | 需要 instance-aware bounds、world/local-row 分派、地址重建和缓存版本；不能直接向旧结果塞 512 个 key。 |
| 主要收益/风险 | 先解除全模型 shape 注册门；额外页查询/设备与 HOST 复制开销尚未知。 | 有机会复用几何、局部配对和实例布局；异构参数、选择复位、计费、失败域都需独立证明。 |

A 的第一片要求每个 contact-bound world 恰在一页；可保证不漏跨页物理对，因为不同 world 本来隔离。单个 world 若已超页容量，先明确拒绝，不能把它随便拆两页而漏掉跨页形状对。未来若改成 pair-domain pages，必须登记每个 canonical pair 恰归一页、重复 shape 只是 scratch 引用；所有站点及载荷按原 world 合并。那是另一次合同。

B 需要将几何原型与 `(world, original body/shape/material IDs)` 实例身份分离；共享只读地面顶点不等于共享地面 key 或动态状态。当前没有 global-static world 接口。每个静态实例仍须有真实 world key，静态反力分别保留，汇总仅用于观测。原型 revision、实例拓扑、原树 graph token/绑定、实例参数和每 world 控制都需要版本边界；不能把模板行号当持续材料站点身份。

## A 的最小方法边界与一次推进

可讨论 `prepare_contact_bundle(page_specs, ...) -> ResidentContactBundle`，再让 `step_contacts(bundle,h)` 或一个新专用入口调用原 `_step`。这两个名字**目前不存在**；不建议通过继承绕开原 plan 构造/owner 检查。bundle 拥有新编译的页资源，借用一个原 batch；关闭它不关闭 batch。每页原模型继续检查形状、顶点、mask、revision、完整材料域和绑定。

一次 attempt 应拆成以下阶段，共用原实现而非复制第二套力学：

1. 在同一个 batch lock/device context 下冻结 owner、topology_generation、state_epoch、src/dst、持有控制及每页 registration/shape revision。所有 page 的 query 都读同一个 accepted src；不保留借用 view 到锁外。
2. 先完成全部页候选/容量检查。每种实际 storage 的 entry mechanics、完整输入/速度/J/M 和 entry packet 只准备一次；页只消费所需 world 的几何、配对与 HOST law，不重建力学。
3. 收集每个 world 的全部原站点、FP64 COM wrench 和分别绑定的静态反力。按 canonical pair/site 顺序汇合，先完整检查预算，**总 FP64 wrench 只量化一次到 FP32**；不先各页量化再相加，也不重复加原 external/drive。
4. 每个 storage 一次投影、solve、candidate geometry、finalize。实际候选与原 entry 做一次完整 motion/work、方程、驱动和反作用审计；失败 world 保留原完整状态。
5. 预构造 aggregate world receipt/status、所有页 result/pending 与资源统计。最后原 batch 只交换一次 bank、推进一次 attempt epoch、发布一个完整 bundle 收据引用。不能先发布页 0，再执行可能失败的页 1 finish/IO。

现有 `_check_attempt` 检查 owner/topology/query epoch、active bank、batch held 数组和各 tree held 控制副本；bundle 至少保留这些检查，并在最终发布前重验所有页，拒绝混 owner、已关闭页或 reset 后的旧 attempt。当前没有单独控制 epoch，不应假称已有。reset/restore 会推进全 batch state_epoch；即使某个 world 未变，也不能复用旧查询。

原 `_step` 在全 False 时仍交换保留状态的 bank 并推进一次全局 state_epoch；物理 tick 不推进的 world 仍旧值。新路径应沿用这一差别，不能把 epoch 当物理步数。page result 的 epoch 表示原入步来源，不得冒充末步 query。`completed_steps` 每次原调用加一次，不得按页计数。

## 容量、部分接受与晚故障

- 每页 shape/vertex/几何上限不变。当前 candidate_capacity 默认 4096、合法上界 256×255/2=32640；这是页 query 的 cap。即便只含隔离 worlds，原 pair scratch 仍枚举页内形状组合，不能假设它已经是实例化线性存储。新增 page 数、总设备字节和 bundle candidate 总量应显式有界，默认数值待合同/实测裁定，本文不猜。
- 页的候选必须来自原允许域。bundle 先查 pair 归属唯一，再统计实际候选；不能删掉失败/未决候选、截断或在页间重置计数。world-complete 页可直接保留原 per-world site_capacity（0..4096，默认128）；以后一个 world 跨页时，该预算必须针对所有页总和，只计一次、不得每页享用一份。
- 原站点先全部计数再算力；不能按非零力筛预算。当前只有 `ContactSiteCapacityExceeded` 转为该 world 拒步；candidate overflow、非法几何等是 plan/attempt 异常。第一片应保留这些分类：任何页 candidate overflow 在发布前中止整 attempt，不能自行发明部分成功；站点超额的 world 清除所有页贡献并保留状态，其余 world 可接受。
- late query/finish/旧控制检查失败：所有 q/v/tick、已发布页布局与收据保持原版本，不能留下半页新证据。已持有控制输入按原语义保留，不伪称回滚了用户命令。无法同步的设备故障沿原 unavailable/checkpoint 恢复路径处理；不能在无法确认发布时给已知全 False。
- 正常返回仍是覆盖原 batch 全部 world 的 bool map。部分接受由既有环境层 poison/full-reset 语义处理，拒步不生成 reward/episode transition。训练 `ResidentContactStepper` 对新 bundle 的所有权/关闭检查需要明确扩展与独立验证。

## 独立多案例验收与下一次决策

| 新目标组 | 独立观察与应通过的门 |
|---|---|
| 单页等价、分页边界 | 新 bundle 单页与原小规模有效路径对照；W127/W128/W129/W256 以每 world 两 shape 覆盖页边界，真实入步速度/不同载荷，最终 world 集合不变。旧已绿全套只复用，不为此重跑。 |
| 至少四种力学域 | 原允许范围内的小型 ordinary tree、affine tree、floating tree、floating-affine tree；包含转动胶囊、有限盒面、非共线外载，检查完整 q/v、原 full/reduced Jᵀ、双侧合力/力矩、材料功及实际驱动力。不能只验复制同一个无载球。 |
| 隔离与排列 | 各 world 放在相同物理坐标仍不互碰；world key/注册顺序/页顺序重排与页面重新装箱，材料 ID 与反作用不串世界；一 world 多个静态支撑可在同一页真实合载。 |
| 部分拒绝与晚故障 | 页边界 world 的 site cap不足只拒该 world；最后一页 candidate overflow、关闭/错 owner、过期 epoch、finish 及同步故障保留原正确失败域与完整 masks；主动核对无第二次 solve、无半页发布。 |
| 训练与规模证据 | 选择 reset 后只应有目标 world 的物理初态改变，全部旧 attempt 失效；有载 observation/reward/done 来自真实接受态。解释器/CUDA 各做最小新增覆盖，另留原输入、完整收据、页成员与 bytes/launch/readback 统计。 |

推荐先由父裁定 A 的 world-complete 页成员、bundle 入口和全局资源边界，再冻结小范围开发/失败注入合同。功能门是 W256 真实受载一次发布，且每 storage 只 solve 一次；不是吞吐宣称。通过后才测分页查询与 HOST narrow phase 的真实成本，并决定是否值得投入 B 或 K1→K2 设备消费。页数、拷贝量、候选和站点分布尚无新测量，不能承诺快于 Newton，也不能借本研究开启 G1 GPU 接触、摩擦或大规模异构动力学扩域。