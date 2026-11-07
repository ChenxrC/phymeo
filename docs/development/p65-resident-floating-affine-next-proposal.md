# 常驻浮根 affine/mimic：后继最小增量提案

状态：**PROPOSAL_NOT_FROZEN / AUTHORING_ONLY / UNEXECUTED**。2026-11-08。

本文只整理源码调研，不授权产品、测试或执行。读取依据为父 `4927ef9414754cb13bf3a2edab97f334025e49e8` 及已冻结 C10 合同；C9 与 C10 正式限域接受是本片实施前置。本提案不改变正在进行的 C9 验收，也不提前解除现有浮根 contact 拒绝。原 C9/C10 输入、门和已保存证据保持。

## 1. 已有能力与真正缺口

- `src/phymeo/builders/floating_mimic_urdf.py:9–28` 已有公开 `load_floating_mimic_urdf`，一次 `_read_tree`、共同 mimic XML 提取、普通浮根装配及 fixed 工具折叠。无须另造 loader 或临时 URDF。
- `src/phymeo/dynamics/_mimic_coordinates.py:48–151` 已编译同类型、无环、非空 affine 图、原行顺序和联合位置/速度边界。`floating_mimic.py:95–125` 已复用它并建立 `S=diag(I6,P)`；139–173 行严格 full/reduced 转换，267–282 行完整 M/b 投影、一次约束求解和原浮根漂移。不得复制图机制。
- `floating_mimic.py:183–238` 已保存原始每条约束的乘子、完整反力、根六维零约束余额、逐行及总功。`runtime/_floating_mimic_participant.py:136–188,191–240` 已支持独立 PD、full COM 载荷、真实几何 owner 和一次 HOST 场景推进。此片缺的是设备接入，不是 HOST mimic 功能。
- `backends/_world_step_layout.py:229–258` 当前只分发 fixed、fixed-affine、ordinary-floating。`_resident_affine_tree.py:28–40,141–162` 只接受固定根 `MimicRobotAsset`；`_resident_affine_storage.py:102` 直接组成固定根 `TreeDeviceStorage`。不能把浮根包装传进去或给根伪造六个标量关节。
- `_resident_floating_tree.py:139–182,270–317` 的公开 `FloatingRobotTree` 只接受完整普通浮根模型与 full state；其真实 physical FK/J/M/b、双 bank、root COM 速度和指数漂移可以复用。`_resident_floating_storage.py:341–428` 已把 mechanics、solve、candidate geometry、finalize 分阶段；无需再分叉一套运行时。
- `_resident_floating_binding.py:56–61` 的 `from_asset` 只接受普通浮根资产；固定 affine binding 在 `_resident_affine_binding.py:22–43` 只接受固定根资产。新图身份需要接入，但完整材料来源证明和浮根 bounds kernel 不需要重写。

## 2. 拟公开声明与四种数量

拟新增 `FloatingAffineRobotTree(asset, state, *, kp, kd, effort_limit, delay_steps=0, control_max_iterations=512)`；`asset` 为已有 `FloatingMimicRobotAsset`，初态为其独立 `FloatingRobotState`。名称和字段仍待父冻结。

| 数量 | 浮根 affine 含义 | 存储/使用位置 |
|---|---|---|
| n | 完整物理标量关节数 | 原图、完整 FK、full joint state |
| k / N | 独立轴数 / 真实电机数 | drive、PD、cap、延迟队列；根无电机 |
| B=n+1 | 完整刚体数 | root0、child i+1，COM wrench/FK/材料 owner |
| Dfull=6+n | 完整广义速度数 | 原 J/M/b、full 外载、加速度及约束反力 |
| Dreduced=6+k | 独立广义速度数 | reduced M/b/载荷、受约束根与独立轴速度 |
| Qfull=7+n / Qreduced=7+k | 完整/独立配置数 | 同一根 origin/四元数加两种关节数组 |

首域建议沿已存在的 `1<=k<n<=16`：B≤17、Dfull≤22；非空 mimic 不包含 n=0。root-only 继续走普通 C9，作为混合批兼容正控。k 与 n 不允许按 body 数猜测，不能把从动轴原 effort 字段当独立电机 cap 或约束反力上限。

图 identity 同时绑定原 FP64 物理模型、原关节名称/limits、原关系行及 P/c；不能只以舍入后的 P 判相同。独立初态、展开 full 初态、原行、联合位置/速度域均预检。以 C9 最终接受版本的有限输入/FP32 算术和资源域为准，不在此提案新增放宽门。

控制沿既有设备 triplet：`inputs(drive[K,3], external=None)`。external 保留 full `FloatingRobotInputs` 的 B 行 COM force/torque，full joint effort 必须为零；电机仅从独立 drive 入口来。root COM 力矩不丢失、不塞入 PD 前馈。当前命令与延迟队列分开：只 drive 延迟，COM 外载仍为当前输入；失败物理步不消费队列，不改变原 `set_controls` 与 `step` 各自的事务边界。

## 3. C9 fused kernel 的最小父 OLD 拆分

当前 `_tyx_resident_floating_kernels.py:440–803` 的 `floating_solve` 同时使用 n 作为电机/关节数量，执行：完整 D Cholesky及自由响应（483–528）、PD/box迭代与真实候选 q 核查（529–685）、加速度/外载（686–700）、关节漂移/队列（701–712）、根指数旋转与 COM→origin 漂移（713–792）、失败保留（795–803）。直接复制并把 n 改 k 会把物理 FK、body 和控制维数混为一谈。

建议父先做一个受影响范围明确的提取，再由 NEW 坐标层组合：

1. **保留原完整 mechanics**：原 physical floating storage 对入口 full state 只算一次完整 Jv/Jw/M/b/外载。不创建 reduced 几何模型，不跑一次无约束 `stage_advance`。
2. **提取原代数求解段为唯一共享实现**：接显式 generalized count、motor count、独立 q/v、gains/caps、当前/延迟 drive，以及矩阵和向量。普通路使用 D=6+n/N=n；新路使用 Dreduced=6+k/N=k。根六维不设 PD。原 Cholesky、全交叉 mobility、迭代顺序、真实 `q+h*v` 控制核查和方程门保留。gains 的现有 `properties[body=motor+1]` 索引须改成明确 motor 参数来源，不能把 k 行假作真实 body。
3. **NEW 只做坐标投影/展开与原行账**：形成 `Mr=SᵀMS`、`br=Sᵀb`、`gr=Sᵀgfull`；共享代数求解只产生一个 reduced candidate。由它得到 `vfull+=S*vreduced+`，独立关节 q+ 只积一次，再用 `P*u+ + c` 建完整关节 q+。原始 full follower 不能先独立漂移再被覆写。
4. **提取根漂移为唯一共享实现**：原 root exp-map、同半球选择、真实根 COM 速度与本体 COM 偏移补偿均保持；普通与 affine 在已求出的真实六维候选速度上各调用一次。根漂移不依赖 reduced k-body 几何，不从 world Q 反减 local。普通关节漂移保留其原路径，affine 使用上述一次 canonical 展开。
5. **保留原候选 FK/energy/finalize**：full candidate 完成后调用 C9 的真实完整 FK/twist/energy。原 full finalize 产生唯一 world 决定，新增 reduced retain 使用同一决定保留/接受，原 batch 在所有状态及 receipt 审核后才 swap。不得先发布 full 再补 reduced。

提取可为同文件私有 TYX 函数或同一 factory 的共享阶段；最终形式由父依据 TYX 可编译接口冻结。提案不要求复制一个长期分叉的完整 integrator。若原 fused 形式不能安全提取，先交精确编译/接口反例并止步于 step-only 接入，不用“暂时复制整个求解器”绕过。

完整反力从同入口 full M/b/外载与真实候选加速度恢复：`r=M*a+b-gfull-E*tau`，E 只向原 independent joint 行注入电机。它须等于 `[0_6,Cᵀlambda]`；保留 root residual、Sᵀr、原图每行反力与入步功。不能以 S*tau 给 follower 另加电机，也不能把链图 flatten 后只存聚合反力。

## 4. C10 接触、材料、检查点与共同发布

待 C9/C10 都正式接受后，先交本片 step/restore，再接触。沿 C10 的有限 S/C/未圆角 H、mu=0、`discrete_entry`：accepted full FK/current bounds → 原 HOST patch及法向law → 每真实 body COM wrench → **full Jᵀ一次** → **Sᵀ一次** → 原 reduced solve → full candidate → 实际材料/application运输与功 → 同次诊断 → 唯一发布。保持原力律、共同作用点和完整双侧材料地址；这是 HOST narrow + TYX 动力学反馈，不是全设备窄相。

新 binding 可仿 fixed-affine 的“普通来源证明+graph token”，委托现有普通 floating `from_asset/compile` 和 bounds：full root0/child i+1、原 collider IDs/local vertices/fixed-tool frame 均不变。same-tree 双动态仍按真实物理 parents/filter 决定，不能按 mimic 图边或独立坐标判邻接；跨独立 participant 仍拒绝。当前类型/域 guard 直到冻结并验证后才由父改。

检查点完整保存：同一 root local origin/quat/v、reduced u/v、full q/v、full generalized applied、真实 K motor effort、K 延迟历史、四块 held control、tick、原模型+graph token。C9 `_resident_floating_tree.py:319–390` 的 `local_origin_m` 与 anchor64 展开检查必须保留；大 anchor 的 world Q 是观测，不能单靠它逆恢复 root。full/reduced 不一致不能静默投影修复。原 selected/actual drive 是 K，contact/input external 是 Dfull，body wrench 是 B×6，诊断同时给 Mr/br/gr 和原 full mechanics。

每世界正常拒绝保留全部 full/reduced/根/history/FK/energy/tick，允许别世界按原规则继续；upload/readback/receipt 等晚异常则沿原无 swap 规则。复用批的 lock、epoch、stale attempt 检查和资源栈；半构造、close、恢复重建均不留额外设备资源。接触完整保存与未绑定 affine-floating world 的正常独立前进都要覆盖。

## 5. 五组独立首步锚与有限窗口

所有下列数值为已有固定输入或手推候选，不是已执行结果；后续 tests-first 冻结时仍须独立法向前提成立才触新 API。

| ID | 声明与独立首步锚 | 区分能力 |
|---|---|---|
| A01 | 原 `floating_mimic_jaws.urdf`：root/两子质量2/1/2 kg，平行+X，`q2=-q1+.004`；零速/重力，独立1 N。平移块 `Mr=[[5,-1],[-1,3]]`，根 a=1/14、独立 a=5/14、从动 a=-5/14，原行 lambda=-4/7 N，root constraint=0。 | 负比例、非零偏置、完整根耦合和真实内部驱动。原 HOST 锚在 `tests/test_floating_mimic.py:76–115`。 |
| A02 | 同 jaws 原右指有限+X面与原 HOST 首载球位姿，声明对手为有限静态体；先核原2048 N/m法向合计2 N。full load 的 root-X/right-joint 都为-2 N，reduced=(-2,+2)，根/独立 a=(-2/7,4/7)，lambda=2/7 N；静态反力+2 N。 | 从动体真实接触通过 full Jᵀ再Sᵀ影响自由根；新的静态窗口不冒称原双机器人窗口已验。原首载锚 `tests/test_floating_mimic_scene.py:214–242`。 |
| A03 | 原 jaws 增0.5 kg gauge，所有COM/轴共X。fan 与 chain 的 P=(1,-1,1/2)，c分别(0,.004,.002)/(0,.004,0)。原物理4×4块：M00=5.5、M0j=mj=(1,2,.5)、关节块diag(mj)；原两C行、外力(1/8,1/4,-3/8,1/2) N、独立motor3/8 N、入速.02照 `test_floating_mimic.py:194–228`。直接用这些literal构造full鞍点oracle。 | 相同P不代表相同原图身份/逐行反力；联合限位/速度与非零入速。 |
| A04 | 原 `floating_mimic_elbow.urdf`：非平行Z/Y转轴、ratio1.5/offset-1/8、偏心fixed工具；保原casebook三姿态/非零根及轴速度、部件质量/惯量、外载、h1/4096。独立从四 authored 部件求几何/J/惯性和，再用C9二阶time-jet路线构造bias和full KKT。 | 偏心COM、非主轴姿态、旋转和工具材料来源；不读产品M/J建立expected。原输入 `docs/development/p65-floating-mimic-casebook.md:44–67`。 |
| A05 | C10 F03原root2/左2/右1 kg、球r1/8、offset±7/64、原mu0/k64首载±2 N；新增关系qR=-qL，初q/v=0。`Mr=[[5,1],[1,3]]`、reduced force=(0,-4)：根 a=2/7、独立 a=-10/7、从动 a=10/7，lambda=-2/7 N，总线动量增量0。 | 同一完整树两个真实动态body同时响应，不拿静态对手冒双侧；此新增约束系统不沿用普通C10的加速度expected。 |

短控制留出复用原 HOST jaws 的根外力/PD Schur解析式、真实非零cap/超载饱和与delay0/1；device triplet和显式前馈沿既有设备合同，不偷搬不同入口的自动前馈。大 n 使用零PD与独立 full 参考，小 k≤4非零PD可复用独立 box-KKT。follower effort不是被动约束反力上限。

再做：初态/注册顺序与刚体协变、连续受载→分离→重入、不按时间改force；W1/W4混合fixed/fixed-affine/ordinary-floating/新类型/scalar及未绑定新world；每代表档40步、tick19恢复后21步逐字段重放。资源边界包含n16/k1、n16/k15、body16及非法body17，root-only普通world继续可用。W32仅在短档成本和内存可行时做有界代表，避免把全矩阵复制32份后长时间只等执行。

## 6. 作者边界、依赖和最小交付顺序

拟 NEW 作者产品（当前均不存在、尚未授权）：`src/phymeo/backends/_resident_floating_affine_tree.py`、`_resident_floating_affine_storage.py`、`_tyx_resident_floating_affine.py`、`_resident_floating_affine_binding.py`。分别只负责公开声明/图编译适配、组合full/reduced存储、P/S投影展开与原行账、来源+图绑定。独立 NEW scale功能/事务tests、reference和外部consumer按父后续冻结分工，不能与产品作者同时修改同文件。

父 OLD：`_world_step_layout.py`、`tyx_world_batch.py`、`_resident_floating_storage.py`、`_tyx_resident_floating_kernels.py` 的上述共享提取、`_resident_collision_plan.py`、C10最终 `_resident_contact_plan.py`、顶层exports。已有 `_mimic_coordinates.py`/loader/HOST模型应直接调用，不计划改动。body/velocity/motor数量接缝沿C10抽象，不能为新kind复制整条contact plan或把“非ordinary”全判affine。

1. C9/C10正式接受并冻结实际共享源后，先 A01：旧 HOST literal M/1 N/root reaction 前提通过，再在函数内查询拟公共 spec/调用入口，形成有效RED。禁止 missing-import 或拼错fixture先失败。
2. 父共享提取保普通C9一组实际受影响节点；新增 step/PD/graph/full-reduced/checkpoint/根local/事务两backend逐组验收，不重跑C5–C10整套。先首步短档，再原输入有限窗。
3. 接 A02/A05 full wrench闭环、A04工具/旋转、容量/latefault/stale/close与混合；保持真实 entry/candidate 时态，原动力学门与完整材料记录，不以较宽 new tol 换通过。
4. 同次保存固定membership的完整输入、full/reduced states、M/b/J、逐侧载荷/材料/原行反力功、history/tick/status和失败retained。stdlib后审只核完整字节/字段/重放/身份，不再重跑producer或解算。
5. 独立Newton原生范围见下；最后新wheel复制到源码外，只public API，核实际wheel SHA、所有加载模块成员字节、混合40+21重放及拒步恢复。仓库test/reference不能充当wheel产品。结论按实际范围写，不预填通过或速度。

## 7. 本机 Newton 1.7 参考边界与止损

本机 `.venv-newton-1.7.0/Lib/site-packages/newton/_src/solvers/featherstone/kernels.py:67–180` 已有 inertia/force reduction和acceleration expansion；`solver_featherstone.py:1005–1061` 用它们包围原dense solve。可作为独立 FREE+R/P、同articulation mimic 的原生动力学对照，不是复制其kernel到本产品。

该solver源码99–103行明确 effort/velocity limit等并未支持，126–128行要求mimic同articulation；119–124行也说明FREE root的BODY_PARENT_F是递归余额诊断，不能当原mimic约束乘子。因此首native优先A01 motor及A03规定外载/无PD，原约束行反力以独立full KKT为oracle；接触另取同声明首载/真实native force，公开各自law和时态。Newton没有对应cap不构成删除本项目cap、PD、联合速度/位置门的理由；不同积分/控制/接触输出只并列，不硬算性能倍率。

有限成功域为普通浮根上的常量同类型无环affine图、已验证finite模型/材料、有限静态或同完整树非相邻双动态、mu0与离散入步。跨独立participants、非线性约束/闭环、持续摩擦、CCD、全GPU窄相和训练梯度均不在本片。

停止线：full/reduced或图身份不一致、root约束余额/原行账不闭合、实际PD或方程门失败、晚错误产生半发布、资源恢复不完整时停止扩域并保存原请求；不改原输入/门或用HOST逐world推进冒设备批。若共享提取需要长期复制整套integrator，先缩回可审的step-only提取。长档成本不合适则保留短档功能和原失败，明确未跑配置；不让该后继阻断C9/C10接受，不以反复证明替代公开可调用能力。

本文件仅文本/源码研究与手推锚，未导入、执行、数值试算、测试、格式化、构建或修改产品。后续实施须父另行冻结范围。
