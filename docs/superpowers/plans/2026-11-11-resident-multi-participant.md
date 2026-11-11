# ROB1：驻留设备上的同世界多参与者接触响应

日期：2026-11-11。状态：**ROB1c_PARTIALLY_QUALIFIED / R03_PRECISION_FAILURE_RETAINED**。最新四family服务、机械臂/自由盒、W1/W2/W4及混合world事务在两后端各20个节点通过；两个R03精度失败保留。原R01/事务因共享接缝变更做了两后端各11项回归，公共CUDA推盒40步也通过。详[ROB1c实际结果](../../development/p65-resident-multi-participant-robots-report.md)。每world<=8参与者/<=33 physical bodies、HOST候选/patch边界保持，不能称所有大场景或隐式跨树求解已支持。下一步ROB1d已合格范围包装和独立的R03保存入步PD算术诊断。原研究基线 `95212cff12cd6437566fdce80b93ddc7c4b64fe7` 及下方设计保留；历史未实现项以本段与实际报告为准。主入口：[持续计划](2026-11-11-newton-parity-continuous.md)、[当前全景](../../development/phymeo-newton-panorama-2026-11-11.md)。

## 1. 要补的能力与已有事实

一个逻辑 world 中，两台独立机器人、自由刚体和静态环境在同一入步状态上生成接触；双方真实运动及关节/根反力随之更新；整个 world 只有一次接受和可见发布。多案例通过才接受，不能由 G1、Panda 或一个对称静止例代替。

| 研究时基线代码（不是最新实现状态） | 当时已有能力 | 本合同要补的缺口 |
|---|---|---|
| [multi_robot_scene.py](../../../src/phymeo/runtime/multi_robot_scene.py) 的 `build_multi_robot_scene/step` | HOST 多固定/浮根树及自由刚体，共同采样、逐参与者计算、全场景一次发布 | 构造明确拒绝非 HOST；不是驻留设备多参与者 |
| [scene_compiler.py](../../../src/phymeo/builders/scene_compiler.py) 的 `SceneDeclaration/CompiledScene` | 普通、affine、浮根、浮根 affine 导入及编译；每 world 一个 articulation | declaration 要求 world 唯一，不能登记同 world 两个独立模型 |
| [tyx_world_batch.py](../../../src/phymeo/backends/tyx_world_batch.py) | 四类完整树、双 bank、原控制与恢复；每 storage 一次推进 | world 直接映射一个机械条目；尚无独立 participant 维度 |
| [resident contact mechanics](../../../src/phymeo/backends/_resident_contact_plan.py)、[bridge](../../../src/phymeo/backends/_resident_contact_bridge.py) | 真正完整 J/M/b、双侧 COM wrench；同树动动与动静接触；候选/站点/晚失败分层 | contact frame 的 group 是单树局部 index；一 world 的力只送到一条树 |
| [collision plan](../../../src/phymeo/backends/_resident_collision_plan.py)、[bundle](../../../src/phymeo/backends/_resident_contact_bundle.py) | 设备状态宽相、已验 device patch、多页 world-complete 接触 | 绑定仍假设 world 对应一个 storage row，分页不是多独立树的共同响应 |

复用 [C8](2026-11-08-resident-tree-dynamic-contact-feedback.md) 和 [B3](2026-11-11-paged-device-contact.md) 的已验底层，不重做原完整矩阵。旧 W256 是原隔离单树世界资格，不倒灌为本新 API 的规模结果。

**首片的“共同响应”精确定义：** 同一入步状态一次跨 participant 接触装配，完整各树 J 投影，各独立机械块一次求解，world 共同接受/发布。当前 [sample_patch_loads](../../../src/phymeo/solvers/contact_patch_loads.py) 使用既有 compliant normal / velocity-regularized friction law，接触力在入步状态确定；它不是跨树未知冲量的隐式 KKT、静摩擦历史或 MuJoCo 约束求解。父总计划中更宽的“共同求解”应按此解释。

## 2. 首域及硬边界

- 新 API 只支持 `mode="interpret"|"cuda"`；HOST 共同场景继续走原 API，不伪造新 HOST 设备路径。
- 每 world 1–8 个独立 participant；类型是原 `FixedRobotTree/AffineRobotTree/FloatingRobotTree/FloatingAffineRobotTree`。纯自由体用 `FloatingRobotTree` 的真实 root、零 child；不使用固定替身、虚拟根或焊接。
- 保持各模型已有关节、质量、惯量、armature、阻尼、PD、delay 和 affine 图资格。每树原 32 scalar-joint 上限不变；free root 不计为伪关节。
- 首片 **单 CollisionModel / 单查询页**，资格 W1/W2/W4，允许稀疏 world ID。世界不能拆页；旧 bundle 多页适配另行资格，不加入本片。
- 原 model 的总 shape/vertex、候选、device patch 和 site 限额原样；site 默认 128、原最大 4096，按完整 world 全部 pair 合计，不能乘 participant 数。candidate capacity 按本单页全部 world 合计。每树 mechanical body rows 单独计；世界 row offsets 是显式登记元数据，不扩大几何/站点预算。
- 原有限 sphere/capsule/unrounded convex hull 与相应 static bindings；box 使用现有有限 hull 表示。只调用当前已支持 patch 后端，`host_patch` 或 `device_patch` 在 prepare 时显式固定，不自动回退。
- 本片保留 `sampling="discrete_entry"` 和原材料参数域。无 CCD、persistent friction、跨树硬连接、闭环、柔体或新形状；这些不是多参与者注册的附带功能。
- 共址跨 world 必须隔离。同 participant 的直接父子过滤继续按原实际 parent 表；跨 participant 的局部 index 相同、同名 link 或各自 root 都不能触发父子/同体过滤。
- 注册全部预检先于设备分配；不完整 law、重复 owner、错误来源、容量越界当场拒绝。准入成功后也保模型/形状 revision 与图绑定检查。

## 3. Additive 公共接口

新类型从 `phymeo` 显式导出；原 `TyxWorldBatch`、`SceneDeclaration`、原 binding 构造与 checkpoint 契约保持源兼容。本片不扩大旧 declaration，也不让调用方写私有 port/storage。

```python
ResidentParticipantKey(world: int, participant: int)

ResidentParticipant(
    key: ResidentParticipantKey,
    system: FixedRobotTree | AffineRobotTree
            | FloatingRobotTree | FloatingAffineRobotTree,
    body_ids: tuple[int, ...],
)

ResidentParticipantBinding(
    participant: ResidentParticipantKey,
    binding: ResidentTreeBinding | ResidentAffineTreeBinding
             | ResidentFloatingTreeBinding | ResidentFloatingAffineTreeBinding,
)

ResidentSceneBatch(participants, *, mode, cache_root=None)
scene.participant_keys
scene.world_keys
scene.layout
scene.set_controls({participant_key: original_typed_control})
scene.prepare_contacts(
    model, bindings, materials_by_pair, *,
    candidate_capacity=4096, site_capacity=128,
    sampling="discrete_entry", intra_participant_pairs=False,
    narrow_phase="host_patch", device_patch_site_capacity=None,
)
scene.step(h)                         # 无接触的同样事务
scene.step_contacts(h, plan)          # {logical_world: accepted_bool}
scene.snapshot(worlds=None)          # {world: ResidentSceneCheckpoint}
scene.restore({world: checkpoint}, *, expected_epoch)
scene.reset(worlds=None)              # 原完整 world 初态/控制/历史
scene.stats()
scene.close()
```

- key 是非负整数、拒 bool。participant ID 只在 world 内唯一；登记顺序不决定身份。
- `body_ids` 长度恰为 system 的全部实际可动 physical bodies，含浮根；affine 用 full physical N 身份，不用独立 K 替代。无 collider 的实际 body 也必须占登记项。
- 每 world 的 body ID 与 static body ID 全局唯一。一个 physical body 可有多个 shape；它们必须保持同一 body ID、participant 与 local body。不同物理体不能共用 body ID，一个物理体也不能冒两个 body ID。
- wrapper 里的原 binding 保持 authored `key.world == participant.world`、原 local points、原 full topology/model/graph/source proof。不把 collision world 偷换成 participant storage slot，也不由 body 整数猜 group。
- static binding 不带 participant；fixed root 的静态 collider 明确登记 static owner，不制造 active group。
- controls 仍用各 system 原 typed input；一次 mapping 全部校验后才更新所选 held controls。缺省 key 保持原 held control；未知 key、错误形状/模型必须无部分更新。
- 新 plan/result/checkpoint 由本 owner 管理，不接受旧 batch 的 plan、他场景的同形状 packet 或 caller 自造 storage ID。关闭 plan 不关闭 scene；scene 关闭其全部已拥有资源。

## 4. 唯一登记表与数组行地址

`scene.layout` 是不可变、只读的公开说明，包含稳定 key 顺序、每 participant 模型/图 fingerprint、配置/速度/physical-body 数、各三类 offset 和内部 storage family/row 的只读诊断。内部 slot 只说明地址，不能用于提交 action 或 reset。

- canonical 顺序为 `(world, participant)`；配置 offset 按实际 nq，速度 offset 按实际 nv，body offset 按 full physical body 数分别累加，禁止 q/v 共用 stride。
- 四类 storage 各自保存其批行；`(world, participant) -> (family, row)` 与反向表都精确一一对应。逻辑 world 的 participant rows 可跨 family，仍共享 world 接受状态。
- world 范围内 `(participant, local_body) <-> CollisionKey.body` 是显式双射。shape key、material、source、revision 全保留；输出给用户始终是逻辑身份。
- contact 的世界 body 槽仅用来收集 wrench；随后按登记 body offsets scatter 到每 participant 原完整 body rows。局部 group 不能跨树误合并。
- 原 ContactLoads 的单树 33-group 检查不能直接当整世界上限，也不能静默删除。父抽出共用采样循环；旧入口仍 33，新入口以登记布局验证全部 body rows，site/candidate 上限不变。最多 8×33 的机械行范围是显式新布局范围，不是新增 site 配额。
- 每个 stage 的 source owner、topology generation、state epoch、bank、world tick 与 participant key 必须绑定。所有参与者 entry tick 一致才可求该 world；错 tick 不能靠复制邻树状态修复。

## 5. 一次共同入步和发布事务

1. 在同一个 scene 锁及 runtime 借用中验证完整 held controls、registry/revisions；固定 accepted bank/epoch。每个使用到的 storage **只 stage 一次 entry mechanics**，其 M/J/b 由原 TYX 核生成，不用 HOST FK/M 替代。
2. 一次世界几何查询取得全体 participant/static frames；所有 pair 的相对速度都来自同一 entry。按原 patch 权重和材料律每 site 只算一次力；另一侧严格取相反力，两个 COM 力矩分别使用真实 `(loadpoint-COM)×force`。
3. 对 participant p，完整广义接触载荷为 `g_p = Σ(Jvᵀ F + Jωᵀ torque)`。点 Jacobian 是 `Jpoint = Jv - [r]×Jω`；跨双方的相对速度行是在世界速度 offsets 上拼接的 `[-Jpoint_A, +Jpoint_B]`。静态方列为零，但真实 static reaction 不丢。
4. 独立 participant 的质量矩阵物理上是 `diag(M_p)`；所有 participant 内部 cross terms 保留。原 PD/隐式控制方程仍各块完整求解。已知 entry contact loads 下，各块一次原解等价于此直和系统；不制造非物理跨树 M 项，也不声称求了未知 contact force 的耦合 QP。
5. affine 先保 full physical wrench/J，再用原 P/c/C 图；即使投影广义力为零，完整物理载荷和图反力仍在 receipt。每个 storage **只 solve 一次**，不得每 pair/participant 调公开 `step` 后拼结果。
6. 所有候选状态、几何和原诊断形成后，按 world reduce 全部 participant 的 status。任一 participant 拒绝，该 world 所有 participant 的候选 q/v/history/tick 必须恢复为 entry bank；这一步在任何可见 swap 前完成。其它完整 world 可按原局部拒绝语义前进。
7. 原 candidate overflow、未知几何、设备/上传/读回异常及 receipt 晚失败继续全 batch 不 swap；site overflow 继续按 world 拒绝。不能把两种失败合并成静默丢 pair。
8. 全部 per-participant/per-world receipt、结果 mapping、几何发布对象先构造/校验，再一次提交 active bank/epoch 和完整公开结果。snapshot/geometry/receipt 读者不能见到一半 participant 新、一半旧。
9. 成功 set_controls 后发生 physics 拒绝/异常，已提交的 held controls 保留；物理状态/历史/最后 accepted receipt 按上述拒绝规则保留。失败不能宣称回滚先前独立成功的 action 提交。

checkpoint 是 world 完整包：registry fingerprint、world tick、全部 participant 原完整 payload、held control/PD history、浮根 local origin 与控制延迟状态。只支持完整 world reset/restore；拒绝缺一 participant 或越 world 的包。restore 先验证全部源和 expected_epoch，再 stage 全部世界候选并一次 commit；不能重建初态冒恢复。reset/restore 使旧 contact/source packet 过期，后继 query 必须重新绑定 epoch。

## 6. 多案例与独立 oracle

先提交 tests/fixture 固定数据，再实现；源码研究阶段不以运行调参数。h、时窗、材料、初态、阈值在首 tests commit 记录，不因结果逐例更改。本片沿原 C8 的工程门：单步物理 atol 5e-6 / rtol 5e-5，40 步 atol 2e-4，原 M 2e-5、PD 5e-5；材料来源 1e-10 和 stored R 5e-5 是不同的输入一致性门。不新增精确零穿透或严格有限步能量守恒要求。

| 固定族 | 输入及不可省略的变化 | 必须验证 |
|---|---|---|
| R01 双自由体 | 两个独立零-child root，mass 2/1 kg、I=diag(1/8)，sphere r=1/8，中心 x=±7/64，g=0，k=64/d=mu=0，入速 x=±1/64；h=1/1024、40 步。加共同刚体变换、登记反序 | 首步每侧力 2 N、a_x=(-1,+2) 的独立闭式；非零真实入速、双方 q/v、净动量/角动量及一次 tick。独立逐步径向弹簧 oracle，不调用产品 bridge/M/J |
| R02 机械臂推自由盒 | 独立普通 2-joint R/P 臂 + 正质量偏心自由 box，非零 q/v/control，接触点偏离两侧 COM；40 步固定窗口，补非零 friction 的切向滑动子例 | full off-diagonal M/b、full J 和两侧力矩；臂与盒都真响应，根 rotation 不被清零；独立 HOST 动力学/几何参考与原式残差，不能仅对比另一个产品入口 |
| R03 双臂与工具 | 两个独立 2-joint 臂 + 自由工具，分别对称与质量/接触点非对称，局部 group/body 名可相同；固定拉开→返回命令阶段，多次同一连续轨迹 | 对称例相消仍保存两侧非零力；非对称例工具平移及转动；分离时载荷关闭、重入后来自新几何；不能用单一静止平衡作为通过 |
| R04 模型/图变体 | R02 的普通/affine、浮根/浮根 affine 变体；一个 physical-force 非零而 K 投影零的图例 | 每类型独立验收；原 full N 反力、独立 K 运动/历史与浮根 q≠v 保真；不得把一个类型通过赋给其它类型 |
| R05 世界/登记 | 同物理各自 W1；W2/W4 同坐标、稀疏 world/participant IDs，反序登记、不同类型混装 | world 间无力/状态串扰；与独立同输入轨迹一致，完整 layout 双射、shape/body/material 不串；计数按实际 storage 批处理，不要求不同批型计数相等 |
| R06 中途恢复 | 非零速度/控制/延迟历史的 tick19 snapshot；继续21步，再完整 world restore 后继续同21步；只 reset W2 中一个 world | 全 participant 后继状态/力/历史相同；邻 world 不变；错源/错 epoch/缺 participant/半世界恢复拒绝 |

R02/R03/R04 的完整构造数据须写入 tests-first fixture 表，并附不依赖设备输出的首步方向/力矩或图约束手算；父只审可运行性和独立 oracle，不要求重开数学研究。持续接触事件用预声明窗口/控制序列验收；若手推的事件前提不成立，应在执行前纠正 fixture 并记录，不能运行后按输出挑事件或调整物理门。

负控与事务最少包含：重复 participant、重复 body、一个 body 冒两 owner、错误 local group/model/graph、静态 body 冲突、缺 law/与过滤冲突、跨 world pair、错 source/epoch/bank、坏 control 最后一个条目、8→9 participant 越界。错误必须在对应边界拒绝，不能靠后面无关错误掩盖。

故障固定落点：第一个 world 的最后 participant 发生 site/solve 拒绝；最后 world 的最后 participant candidate finalize；全体计算后的最后 receipt 构造；最后 restore upload/check。逐一保存旧 snapshot/receipt 和失败信息，确认无部分 world 发布，解除故障后真实重试成功。比较全部 q/v/tick/history/held control 与 source 身份；不只比较计数。未触发故障的已有 world 按异常种类验证继续或全 batch 保留。

## 7. 独占文件表与共享接缝

| Owner | 精确 allowlist | 内容 |
|---|---|---|
| ROB 作者，新文件 | `src/phymeo/backends/resident_scene.py` | 上述 additive 公共 owner、immutable participant/checkpoint/result schema |
| ROB 作者，新文件 | `src/phymeo/backends/_resident_participant_layout.py` | 完整预检、稳定身份/三 offset、family/row 与 body 双射 |
| ROB 作者，新文件 | `src/phymeo/backends/_resident_participant_contacts.py` | 世界采样/完整载荷 scatter、entry/candidate receipt 与 world 资格 |
| ROB 作者，新文件 | `src/phymeo/backends/_tyx_participant_transaction.py` | 仅 world status reduce/所有 participant candidate retain；不重写 dynamics |
| ROB 作者，新文件 | `tests/scale/test_resident_multi_participant.py`、`tests/scale/test_resident_multi_participant_transactions.py`、`tests/test_resident_participant_layout.py` | R01–R06、准入/事务及原公共 API 测试 |
| ROB 作者，新文件 | `docs/development/p65-resident-multi-participant-report.md` | 文件 manifest、能力/限制、实际结果留给父验证后填写 |
| 父，共享旧文件 | `tyx_world_batch.py`、`_resident_collision_plan.py`、`_resident_contact_plan.py`、`_resident_contact_bridge.py`、`_resident_floating_contact.py`（均在 backends） | 最小 storage/事务服务复用、logical owner vs mechanical row adapter、通用采样循环；旧单树行为保持 |
| 父，必要新内部文件 | `src/phymeo/backends/_resident_batch_transaction.py` | 若需抽取共同 stage/check/commit 服务，旧 batch 与新 scene 共同使用；禁止复制一套 integrator |
| 父，共享出口/后验收 | `src/phymeo/__init__.py`、本计划、总计划/全景与验收文档 | 公开导出、接缝合入、唯一验证与源码外消费 |

父内部服务必须显式提供：注册表到原 storage 的受锁借用、controls 预备/提交、entry mechanics once、candidate solve/finalize once、world reduce/retain、prepare-publication/commit/abort 与计数。新 owner 不直接篡改旧 batch 的 private state，也不通过私有 port 规避原来源/事务门。服务可薄适配既有 storage；不用为抽象重排无关实现。

**不在 allowlist：** 四类原动力学/PD/affine 数值核及方程、任何窄相几何/patch 算法、`_device_convex_query.py`、旧 bundle、公用 `SceneDeclaration`、训练/learner、MuJoCo-TYX、G1 模型/策略/观测。确有不可避免的共享修复，先交最小源码事实，由父单点处理，不自行扩文件表。COL 作者不修改 ROB 表；训练作者不修改本场景数据布局。

## 8. 分片、验证与停止条件

1. **ROB1a：预检与公共身份（中）。** NEW layout/schema + tests-first，覆盖重复/错源、q/v/body offset 和 source 兼容。父完成共享服务契约，原单树路径保持；本片不需要先运行 Newton。
2. **ROB1b：两自由体共同响应（高）。** 最小 R01 在两 backend 真正双方受力/运动、完整 world 一次发布；同时完成最后 participant 失败。首节点未过不扩抓取 UI。
3. **ROB1c：臂、工具、图与多 world（高）。** R02–R06，full J/图反力、分离重入、中途恢复与全部故障。每类型/模式独立记录，W1/W2/W4 是本片规模边界。
4. **ROB1d：集成交付（中）。** 父一次针对性旧共享接缝兼容，完整新保存证据和一次源码外 wheel 公共调用；当前同源已绿原套不重跑。一个含非零控制的臂推盒例须可由用户调用，不能仅内部单测。

作者只纯编码/文本/Git；项目 import、collection、pytest、静态、离线检查、build、wheel 消费全部由父唯一串行槽。父按实际实现冻结准确 nodes 后执行，不预写通过数。每个新实际窗口保存原 inputs、layout、controls、全部 participant q/v/history、M/b/J、full body wrench/广义载荷、sites/来源、候选/站点计数、status、tick/epoch、raw transfers/launches/owned bytes。HOST 采样/装配与诊断成本明确计入，不称全流程无读回。

功能成功是公共 API 下三类场景和负控成立，两运行后端各有证据，任何失败不造成半世界可见；有限精度偏差按预声明工程门验收。测试数、文档数或单一“看起来没动”不代表完成。连续两轮针对同一障碍未缩小原因，保存最小失败输入/字段，标出精确缺口并交父裁定；其它独立分支继续，不能靠无限理论或不断扩矩阵拖延。

MuJoCo-TYX 多自由根是独立后继：另冻 nq/nv/joint/body/native source 映射、完整跨模型约束、同输入原生对照和事务，不能继承 resident 通过。G1 的部署来源和零 margin 激活边界仍是旁路诊断，不阻塞 ROB1/GPU 训练，不作为本片成功或失败的唯一标准。
