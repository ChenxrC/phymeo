# Phymeo 当前续作入口

更新：2026-11-13；**M-B1多独立刚体刚柔、8参与者容量及mixed布局、G2-R0恢复清理已完成源码与第四包安装资格；R3-A原资格保留**。
D5-A、R3-N/P、G2a、D6-A 已获各自有限源码/安装资格；D4 原稳态功能仍 FAIL。
D6-A 的 FUNCTIONAL_PASS 与旧弹性 STRICT_FAIL_RECORDED 同时保留，不能合并成严格通过。
整个 P6.5 / Newton 功能对齐未完成；到07:30计划内执行与安装已提前收口，见[本批交付概览](p65-feature-batch-summary-20261113.md)，不自动启动后继大域。

## 从这里继续

1. [本轮功能计划与当前队列](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)：冻结范围、唯一执行者、停止条件。
2. [功能验收矩阵](p65-feature-acceptance-matrix-20261113.md)：源码、后端、安装、应用和性能分列；不估整体完成百分比。
3. [当前连接架构](p65-feature-architecture-draft-20261113.md)、[统一编译架构](../architecture/unified-scene-compilation.md)：声明、设备来源、查询、载荷与共同发布。
   [离线架构总览图](p65-functional-architecture-20261113.html)可直接打开；R3-A与M-B1现均完成有限源码/安装资格；图中未连接的域仍明确分开。
4. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：下一通用模块与未完成范围；本轮详细事实以验收矩阵及原收据为准。
5. [D6用法](p65-distance-dissipation-usage.md)、[D6设计](../superpowers/specs/2026-11-13-distance-dissipation-design.md)、[D6-B应用协议](p65-damped-normal-load-application-20261113.md)。
6. [Batch C](p65-three-track-batch-c-20261112.md)、[Batch B](p65-three-track-batch-b-20261112.md)、[旧历史入口](p65-continuation-history-through-20261112-0035.md)仅按需引用，不恢复旧周期任务或重做旧审计。

本轮证据总入口：`artifacts/p65-feature-until-0730-20261113-01`。
当前事实依据包含父文档提交 `e25df478`、矩阵 `b1e95c70`、架构 `3e8d0df7`；D6-B以新结束的143收据覆盖旧pending状态。
本页是续作导航，不替代原完整输入、输出、原始失败、source snapshots 与保存收据。

## 已验公共功能与边界

| 功能 | 当前资格 | 仍然不能声称 |
| --- | --- | --- |
| D5-A 公共刚柔声明 | `ParticipantSceneDeclaration(contact_response="deformable_normal")`＋`ParticipantDeformable` 自动装配唯一 D3 owner，资产球/程序胶囊短窗及生命周期源码、安装通过 | 任意关节机器人或持续稳态支撑；多独立单体见M-B1 |
| R3-N 浮根球树法向接触 | FREE＋BALL/R/P、NONE JOINT motor、primitive condim1；完整同入步与连续参考及安装通过 | 人形长舞、limits/dry 等全部组合；activation见R3-A |
| R3-P 浮根球树摩擦接触 | 同一约束/CSR路径的 primitive condim3 正控、安装通过；mixed1/3仅编译路由资格 | condim4/6、elliptic、FREE＋BALL limits/dry 已放行 |
| R3-A 浮根球树有状态执行器 | FREE＋BALL/R/P的NONE/FILTEREXACT JOINT，no-contact/normal/pyramidal；源码117＋兼容6及安装两后端各352worldsteps通过 | limits/dry、多FREE根、任意机器人/刚柔组合 |
| G2a 原容量内异构规模 | W1、W8/B3、W16/B3、W16/B8 源码格及晚失败/retry通过；W16两个格各有两后端安装资格 | W1/W8额外包外资格、单异构kernel、跨桶碰撞或W32以上 |
| D6-A 可选轴向耗散 | 保存 elastic reconstructed velocity 后一次 ordered 速度pass；D1/D3/D5有限源码通过，D1/D3安装 FUNCTIONAL_PASS | 精确守恒、旧elastic短门通过、全网络BE、全局drag、所有膜振型衰减 |

原 S1–S3 resident 多参与者查询、M1/M1-A 资产来源、D1设备节点、D2有限球/胶囊–triangle 与 D3共同发布资格继续复用。
D3/D5经M-B1扩展为每world一个节点模型＋多个ordinary单body动态球/胶囊；总participants≤8、worlds≤8；入步采样、逐facet NormalLayer、零摩擦，存在HOST载荷/诊断往返。
原几何 CONDITIONING 不得丢弃后继续施力；查询不自动授予响应、CCD、实体内外或面积压力。
Resident 刚柔与 MuJoCo-TYX 机器人是分别准入的路径，不能把支持表相乘成任意联合求解。

详细范围：[D5](p65-deformable-scene-report.md)、[R3-N](p65-free-ball-normal-report.md)、[R3-P](p65-free-ball-pyramidal-report.md)、[G2a](p65-heterogeneous-scale-protocol.md)、[D6-A](p65-distance-dissipation-report.md)。

## 安装产品与保留失败

- 首包 `113629b3`：530成员同源；R3-N 75/76、G2a W16/B3 77/78 与 W16/B8 79/80已验，按原证据复用。
- 第二包产品 `ec0e2997`：run127核532成员 source/wheel/installed 同字节；不是全部历史套件重跑。
- 第二包 D6 128/129各44 worldsteps，均 FUNCTIONAL_PASS，同时每后端20项旧弹性精度记录、`strict_elastic_precision=FAIL_RECORDED`。
- 第二包 R3-P 130/131各112 worldsteps，D5 132/133各57 worldsteps，均有限安装通过。
- D6专用安装消费D1/D3；带耗散D5声明有源码资格，不能借普通D5安装消费者声称另一个带耗散D5包外矩阵。

**原D4无阻尼N1：** run50 interpret、run69 CUDA各完整3072步/3秒，入层、释放、重入等事件发生，但两预定稳态窗支撑力均值与速度RMS不合格，均保留 FAIL / DIAGNOSTIC_ONLY。
这不是仅精度问题；另有run46第301步旧短速度门失败，原其余三格未执行。生成载荷subnormal窄修前后的两轨迹不冒称同一产品快照。
**D6-A：** run104原总动量严格失败、run112旧elastic重构速度严格失败及安装strict记录保留；完整算术余项解释不修改状态、请求冲量或原阈值。
固定反力包含外/接触、elastic及damping冲量；requested与实际速度冲量、理想功与有符号舍入差均分别保存，不使用能量补贴。

## D6-B 实际结果与下一步

新案例统一四径向边 `c=4096 N·s/m`，来自事先静力/线性化量纲选择；原四格几何、质量、初态、3秒时序、h、物理门全部保持。
这是独立的新材料应用，不覆盖原D4 FAIL；接触阻尼单位仍s/m，不能与材料系数混淆。
新短合同 run140 interpret14通过，run141 CUDA单返回故障1通过；**这些不是长窗资格**。
包外首launch142因辅助目录不在search path而0step失败；原记录保留，未发生物理推进。
run143仅补明确的源码外helper路径，已完成3072步/3秒，returned=checked=3072、control3、close1，453.315秒；状态 **FAIL / DIAGNOSTIC_ONLY**。
两窗支撑均值21.46127079/19.36648514 N、刚体RMS .30439415/.21362712、中心RMS .16520125/.11754240，均未达到原稳态门。

1. 143入层/卸载/再接触、逐步phase/力/算术/连续长参考完成；peak约51.95 N、gap约−.03037 m、原侧约+.09463 m通过原门。7项旧elastic短精度失败仍记录。
2. 按预登记停止这一系数的应用扩展：不启动该格CUDA或其余固定格，不扫描c，不修改质量/h/时间窗/验收门。
3. 新D6-B失败与原D4失败分别保存；完整运行和可解释耗散不是稳态成功，不能把未运行格或新模型外推成已验。
4. R3-A已完成117源码＋6兼容、第三包532成员parity和164/165两后端各352worldsteps；详[正式报告](p65-free-ball-activation-report.md)。M-B1已完成多独立单body与同片的地址/反作用/共同发布；新增28源码节点及S8/M3容量4节点通过，两后端安装各68worldsteps/132participantsteps，见[正式报告](p65-multibody-deformable-report-20261113.md)与[用法](p65-multibody-deformable-usage-20261113.md)。
5. 异构扩容先补完整生命周期内存预算；live/peak无真实observer则UNKNOWN，Newton公平性能比较另立阶段。

## 执行与保护边界

父集成：`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`，分支 `codex/p65-parallel-modules`。
父/root独占旧共享接缝、合入及唯一串行项目执行；原三作者仅在既有隔离worktree按allowlist并行作者工作。
继续前先核实时agents/Git/WIP和当前grant；已释放任务不重派，同源已绿不重跑，同hash证据不重审。
所有import/collection/数值/probe/pytest/lint/formatter/compile/build/install/profile必须由父fresh entrypoint-aware guard后串行执行。
06:15冻结新增范围，07:30按真实已验/失败/未执行汇总；时间边界不授权伪造通过或自动扩展任务。
不建周期任务，不poll/rerun/dispatch远程CI，不push/mainmerge/发包，不改pin或盲跑寿命stress。
三份无关未跟踪稿、证据Junction及其canonical归档、用户Torch和当前venv/pins全部保留，不跟随Junction删除或全解压补副本。
观察器在物理返回后失败不能称物理回滚；UNKNOWN_RELEASE保owner/lease，不能重复free或伪报零live。

旧G1长舞46.29s XY门/131.48s完整动作、J/R学习、R03两项净加速度、B-point touch-SE仍开放。
通用摩擦/换面历史、CCD、更多刚柔机器人组合、传感、可微物理、密集布/体软体、杆缆与MPM/流体仍是后继；本轮不因安装绿而关闭。

第三包R3-A产品08d7a1b6、全部164/165返回和压缩trace已保留canonical installed-closed-archive-01/wheel-03；不重跑同源已绿矩阵。M-B2非根关节树与BALL limits/dry已有独立研究提案，未授实施。

## 本轮后段接受与下一交付

- 第四包产品23757ae6：534成员同字节，191/192 M-B1各44worldsteps/88participantsteps；201/202 S8/M3各24/44。S8=4world/8participants，不外推8world满配。
- G2-R0：178预期RED，185源码12GREEN、193包外12GREEN；实际恢复晚失败保原发布、原异常与cleanup，确切候选强owner及真实关闭。UNKNOWN未被重试，资源完整预算仍未实现。
- 新canonical `artifacts/p65-feature-until-0730-20261113-01/installed-closed-archive-04/wheel-04`：33文件，原件保留，禁止重复归档/全解压。没有新项目执行owner；已绿同源不重跑。
- 下一三条独立功能线：M-B2非根关节刚柔；BALL树标量limits/DOF dry与完整mixed checker；G2b真实allocation manifest/生命周期预算。先冻结每片多案例与接口，原作者隔离编码、父共享接缝及唯一串行验证。
- D4/D6-B只按[可否证的短试样后继](p65-sustained-support-next-experiments-20261113.md)另冻实验；不选第二个c盲跑。本轮沒有新增该实验或BALL limits产品执行。
