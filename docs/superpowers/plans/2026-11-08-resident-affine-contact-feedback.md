# C7 固定根 affine-mimic 常驻接触反馈

状态：**PRODUCT_IMPLEMENTED / FIRST_NODE_VERIFIED / FULL_MATRIX_PENDING**，2026-11-08。持续任务，无08:00截止。C6产品48eeb89的HOST50、解释74分组、CUDA74、相邻15和同次112完整保存记录已通过，源码外wheel两backend已完成，C6正式限域接受；C7产品改动与执行继续由父唯一持有。本文冻结[提案](../../development/p65-resident-affine-contact-proposal.md)第1–6节的有限功能与案例，不将其手推当实测。

## 目标与复用

扩展同一个 `prepare_contacts/step_contacts`，支持已接受的 fixed-root affine-mimic 树对有限 static sphere/capsule/unrounded hull。沿原mu=0、discrete_entry、C6实际stored几何与HOST法向律、TYX原full/K mechanics/PD/积分及唯一batch发布。不增加第二求解器、Scene、runtime或法向模型。C6普通树与未绑定ordinary/affine/scalar行为保留。moving/moving、floating、切向历史、设备窄相和CCD另排。

必须保持 `q=P s+c,v=P u`、full N惯量/COM/J及全部交叉项。每storage一次入步mechanics；临时full COM wrench经实际J散回加入physical.vector的external，再执行原Pᵀ投影、K bounded solve、full展开/原图反力、candidate FK、full finalize与K retain。不能调用会重做mechanics的旧完整advance覆盖接触，也不能末尾kick或把接触写入held。drive延迟与current full loads分别保留。原质量2e-5、PD5e-5、512默认/4096上限不变。

ordinary容量状态保7；affine原7约束残差/8全方程残差保留，新增 **9=SITE_CAPACITY_EXCEEDED**。reject核显式接受对应码。按world完整站点帽拒绝N/K/FK/energy/history/held/tick和旧receipt，邻world可推进；未知几何/IO/来源/晚故障全批不发布。所有receipt和复核在原一次swap前完成。

## 接口与数据边界

沿原 ResidentAffineTreeBinding 和 C6 bridge，不放宽其来源/有限几何门。plan按ordinary physical与affine.physical两个实际storage管理scratch/row/packet，不假设row==world slot。不为无接触storage伪造包。

复用原 ResidentContactReceipt：wrench始终N×6、input_external/contact_generalized始终full N；affine selected_drive/actual_drive为K×3/K，原entry/accepted checkpoint及AffineTreeKinematics保完整N/K字段。ordinary字段语义不变。diagnostics为affine补coordinate_kind、N/K、graph_token、P/c/C、实际full motor/acceleration与reduced M/b/total external/acceleration、原reaction/row multipliers/powers；读取原C5实际诊断，不重新HOST求解。Pᵀg_contact若在HOST复算，只标重构值，不冒设备独立输出。actual reduced total external必须来自同次设备projection。

保全部原site/material IDs、surface/common point、双侧力、static reference reaction、stored entry/candidate frames、实际两类位移/功。entry虚功、有限步work、KE变化分列，不加严格能量守恒门。原5e-5 stored-R域与1e-10记录构造一致性门不变，真实物理仍按原5e-6/5e-5与40步2e-4。

## 固定多案例与验收

输入、精确首步M/载荷/a/原图λ、几何和独立公式均以提案第4节为准，不按执行输出调参数：

- C7-01：A01F follower真实3N替代原外加3N，Mr6、a=1、λ=-1；球、有限胶囊和盒的总3N不能因站点数翻倍。首先证明旧asset/FK/binding/patch成立，再在当前ordinary-only公开入口取得有效RED。
- C7-02：负比例fanout，原motor7+第三group接触3N；a=11/14、完整full反力，负gain不能取绝对值。
- C7-03：原chain与运动等价flatten，a=4/7；状态可相同而原行λ不同，身份不可互换。原Newton链拒绝证据复用。
- C7-04：R/P分支交叉惯量，a=(-512,935)/741、λ=53/247；原重力留出a=(-1360,469)/741、λ=25/247；首步、40步、重排与整体轴循环/平移。
- C7-05：负gain转动与非零角offset、入步u=.5，a=27/20、λ=27/40；保持真实转弧材料与作用点有限步功。
- C7-06：固定u0=.25/ff=-6的卸载重入，h=1/1024与1/2048、总上限7/8s；逐步独立标量公式，首真实卸载后正载可停止，无reset/延长窗口/运行后挑事件。
- PD：原K bounded PD，delay0/1、两步命令、低cap饱和；current接触不跟drive延迟。
- W1/W4：非连续world keys、逆序注册、重复local IDs；ordinary、两种affine和scalar真实40步；合法restore后的同源重放与无恢复参照，保全部N/K/source/owned记录。
- 容量/故障：两bank方向、某affine站点超帽/零力几何计帽/原图码7与8不混淆、整批candidate帽；实际第二scratch分配/上传/candidate/receipt晚故障、同tick恢复失效、关闭/重建；不是重复原C5/C6整套。
- 空域、仅affine无ordinary的W1必须真实工作；错误graph/source/group/world/law/dynamic pair先拒绝。不支持moving/moving/浮根不能称分离。

新增代码的基础验证顺序：有效RED→单3N/PD/容量→各首步与40步→连续卸载重入/混合事务→interpret/CUDA→少量真正受改动影响的旧检查→同次完整保存审计→新wheel源码外消费。已绿同source不重复；失败保存首日志，修复后只补受影响或尚未执行组。保真实launch/读回/上传/内存，不先许诺性能。

## 所有权与当前启动范围

原RD现在只 **TEST_AND_CONSUMER_AUTHORING_ONLY**，在原隔离树新增：
1. tests/scale/test_resident_affine_contacts.py
2. tests/scale/test_resident_affine_contact_transactions.py
3. tests/wheel_consumers/resident_affine_contacts.py
4. docs/benchmarks/p65-resident-affine-contacts.md

允许复用已接受C5独立reference case/full-component oracle或在NEW文件写纯独立公式；不能以产品M/J/接触输出生成预期。先提交C7-01有效RED节点，再完整作者；所有项目import/collection/数值/pytest/static/build均禁，作者报告UNEXECUTED/clean/AUTHORING_RELEASED。不改OLD测试/父接缝、merge父、新建助手或重跑旧Newton。

父独占OLD _resident_contact_plan.py、_resident_affine_storage.py、_tyx_resident_contact_kernels.py、tyx_world_batch.py及必要公共记录文档；C6 bridge/原求解核原则上原样复用。**C6正式接受前不修改这些产品文件，也不授C7执行槽。** 原robot继续完成C6新wheel独立使用示例，不重复派审阅。发现必须新增不同本构/第二求解器/放宽预算时给最小具体矛盾，不能先实施。

### C6完成与C7诊断字段补足（2026-11-08，C7执行前）

C6已正式限域接受，见[报告](../../benchmarks/p65-resident-tree-contacts.md#2026-11-08-父执行与正式限域接受)；C7仍仅NEW作者阶段，原RD首test checkpoint043c200已提交但未执行。精确affine receipt.diagnostics字段固定为 `coordinate_kind='affine'`、`full_dof_count`、`independent_dof_count`、`graph_token`、`projection_matrix`、`offset_vector`、`constraint_rows`；完整实际full/reduced mechanics放 `affine_diagnostic`，复用原owned AffineTreeDiagnostic类型及字段，同次candidate真实数据且swap前构造。原C6通用诊断字段语义保留。公共tree_diagnostics只作同次实际输出的一致性核，独立case/full-component oracle仍是物理真值来源，不据产品输出制造预期。

### C7 首节点接入（2026-11-08，仍待完整矩阵）

父产品 `37897e3` 已把真实 full N 接触载荷加入原 affine 投影之前，薄提取原 project/K solve/expand/finalize/retain；普通树仍走原 full solve。两个 storage 各自保存 row/scratch，所有回执和原 AffineTreeDiagnostic 在唯一 swap 前构造；affine 容量码9与原约束7/全方程8分开。

- 359 首有效 RED：原 asset/FK/binding/真实3N成立，只在 ordinary-only 公开入口失败。
- 368 原首节点 interpret：1 passed/1.32s/RAW_EXIT0。
- 369 同节点实际 CUDA：1 passed/3.97s/RAW_EXIT0。独立 a=1、full a=(1,2)、原行 lambda=-1 与接触力共同通过。
- 363 首有效 exact Ruff0；364 exact format1；365仅两个产品文件格式化；366/367四产品文件最终 Ruff/format0。NEW test 的最终格式化留待完整作者交接，不改原子任务 WIP。
- 360 在项目执行前的进程 guard 遇 CIM 内存错误，未启动 Ruff、无 RAW_EXIT；361/362 的嵌套作业数组遗漏文件选择，分别为只读全仓 format-check1/Ruff0，标记 INVALID_SCOPE_READONLY，不冒充有效专项静态检查。原命令/快照/失败/修正都在 artifacts/p65-generic-collision-20261107-01；source before/after 无改写。
- 原 RD 继续仅四 NEW 完整矩阵/consumer/report；原 robot 新增只读 C7 四接缝审阅，不执行/改源码。父仍唯一项目执行 owner。

本次只完成第一个端到端节点。负gain/fanout/chain/RP交叉、40步、PD延迟、卸载重入、混合/故障/完整保存审计、新wheel源码外仍待验收，不授予整个 C7 或 P6.5 完成。C6原“affine入口明确拒绝”是旧能力边界历史，已由C7正控取代；不重跑该旧拒绝断言或C6整套。

### C7 多世界审阅修复（2026-11-08）

原robot对固定37897e3只读审阅发现一项blocking：动态shape局部group索引遮蔽外层storage组，第二个同storage接触world会TypeError。370用既有C6单个W1/W4对照在新source准确复现（1 failed/8.48s），父仅rename为group_index；371同节点1 passed/14.96s，372/373该文件Ruff/format0。该对照包含真实W1/W4完整结果，非重复旧同源绿集；原C7两affine混合矩阵仍需执行。审阅其余载荷顺序、full/K统一发布、诊断与资源边界无第二项阻断，READ_ONLY_REVIEW_RELEASED；所有首失败保留，不重新派同diff复审。