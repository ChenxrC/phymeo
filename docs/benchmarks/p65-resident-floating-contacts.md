# C10 普通浮根常驻接触：本机验收（URDF来源限定）

状态：**LOCAL_VERIFIED_SOURCE_QUALIFIED**，2026-11-08。父产品9d7ee5f；后续f09d8e/cf46193/f48dadc只纠正新测试与保存观察口径，物理输入和产品工程门不变。新wheel构建HEAD2966aa9；审计逻辑88a9292。全部执行结束，C11可接续。完整P6.5仍ACTIVE，本片不代表整个Newton功能对齐。

[冻结开发计划](../superpowers/plans/2026-11-08-resident-floating-contacts.md) · [多案例及原失败](p65-resident-floating-contact-tests.md) · [保存审计说明](p65-resident-floating-contacts-saved-audit-plan.md) · [实现边界](p65-resident-floating-contact-implementation.md)

## 交付能力

公共TyxWorldBatch.prepare_contacts / step_contacts沿同一接受态、碰撞记录和事务入口，支持ordinary FloatingRobotTree的0–16个真实关节、真实root COM与完整6+n力学。root/child有限表面载荷及同一完整树内非邻接双动态组的反作用，使用完整body wrench、J转置和根–关节交叉响应；不把浮根当固定根，不建立假关节，不二次施力或发布。

已验球、有限胶囊和凸多面体的原HOST有限几何/patch，mu=0、离散入步法向律；TYX执行完整响应、PD与共同发布。包含偏心转动、非对角惯量、fixed工具折叠、N9/N16真实末body、持续载荷→分离→重入、世界混合、40步与tick19恢复21步、两bank容量/晚故障/过期状态拒绝及资源关闭。每种能力以多案例和独立原输入验证，不能只用递布或单个接触点代替验收。

## 实际执行

全部在父唯一串行槽、fresh入口进程guard下执行，无远程CI、安装依赖或pin变更。原目录为 artifacts/p65-generic-collision-20261107-01。

| 项目 | 原任务与结果 |
|---|---|
| 有效RED与最小正控 | 496真实公开入口RED RAW1；504 interpret、505实际CUDA各1通过 |
| interpret完整唯一覆盖 | 506成功16前缀 + 510两项 + 511成功6前缀 + 519成功20前缀 + 528一项 + 529三十项，加504，共76；不同源码组分别保存 |
| CUDA完整唯一覆盖 | 505一项 + 507剩余75项/365.80s/RAW0，共76 |
| 最小相邻兼容 | 512 interpret、513CUDA各3项：旧零归约载荷反力、原chain/flatten关系行、浮根CCD明确拒域 |
| Newton 1.7原生 | 514CPU、515CUDA各F01/F03两例×40步=80步/端，完整原生contact/RNEA/COM/原参数保存，均RAW0；不是matched外力 |
| 新wheel源码外消费 | 516构建/520离线安装，521interpret、522CUDA均RAW0。每端W4的40+21、1个真实bank对齐步、两次容量拒绝及足额重试，另F03四十步；合计302接受/2拒绝world-step |
| 保存审计 | 536原失败保留；541仅补审原5组，复用原9完成组，RAW0且明确来源限定。14原始jobs、两端各76唯一节点、133完整gzip（含原失败前缀）、5176文件 |
| 构建来源 | 新wheel431产品成员逐一等于516原source snapshot/before/after与当前产品；安装成员和loaded origins复用521/522核 |

新wheel SHA256：0721c7ccfb2e6b32038a1357b2077427b437c59c99fbbb7e129a8056c0edd170。541报告SHA256：623724c9ebbe9346ef87f8ed261e48bb7153ad038fc042865109c293f435cda5。原解压逻辑bytes唯一484015828，保守审计累计收费514589710，保原2GiB上限；未重生成物理、未落盘全解压或删字段。

## 原失败与来源限制

506的F07 CC把声明FP64小数当成实际FP32 offset，误差3.576nm触发原1e-10材料重建门。父保FP64声明锚，另从已声明的上传转换独立构造stored几何锚。511把非零初态的delay首target误写零；519混合旧affine复用helper误用立即motor7，实际delay1首held motor0。三者只修新观察器，原输入、门、产品和首失败收据全部保留；不重复已绿物理节点。

536另发现原506/507/529快照漏列tests/fixtures/urdf/floating_fixed_composite.urdf。原run HEAD和当前clean文件均指向Git blob473cdaeecf77d7ec21797d28d8810f30fbb06e75，历史未改；同次完整加载模型/材料及独立F06锚继续检查。但这些**不能证明当时working-file bytes或临时WIP**。F06及source_revision节点明确带retrospective_git_identity_not_same_run_bytes、same_run_bytes_verified=False；后者无gzip，只有原pytest断言。不得称本片所有输入均有原当次raw before/after快照。

wheel reader的closed字段错误则按真实public schema修复：plan.closed保留，batch依资源/live plans/readbacks归零与原关闭后RuntimeError。原536失败不是物理失败，也未被覆盖。541状态严格为saved_records_consistent_with_source_qualification；原9组结果及完整原errors以报告SHA复用，只有5个未完成组补审。原RD独立只读复核无阻断。来源补件与431成员构建核分别见c10-fixture-retrospective-01/receipt.json和c10-final-wheel-source-parity-01.json。

## 后续

C11接入浮根affine/mimic：完整物理n/B/D和独立K保持分离，同一次求解/根漂移，保原图关系行反力、root余额、full/reduced检查点和共同发布。原子agent的产品/测试/native/wheel已文本交接，saved auditor作者进行中；尚未执行或宣称通过。

跨独立participant设备联合求解、设备窄相、持续摩擦、CCD及更广软体/传感/训练工作流仍开放。K0凸体距离见证仅为候选清单，不能将单见证当完整受载patch。上述Newton窗口不代表全求解器等价或性能领先。
