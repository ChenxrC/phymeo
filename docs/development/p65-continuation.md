# Phymeo 当前续作入口

> 2026-11-14 第三批：M2-B1 公共 GPU 端点 body 观察已完成源码资格（`f1c3d9a0`）。`prepare_device_workspace(include_bodies=True)` 从 accepted q/v 导出 body origin 世界位置/线角速度及 body→world 矩阵，与 native 状态共同发布。新增解释6/CUDA8及受影响兼容58项通过，见[本批报告](p65-body-observation-report-20261114.md)和[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。下一批按[计划第11节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)实现 B2 accepted-entry 接触持久双 bank、同相位状态/载荷与 reset/checkpoint，再接 B3 Env 生命周期。接触 token、真实操作可控性、语言编码及共享策略训练仍未完成；无新截止、当前执行槽已释放。

> 2026-11-14 第二批历史：M2-A公共异构GPU状态/动作/reset接口已完成源码资格（2de488b9）；八份真实操作场景静态24项通过，并在同GPU各32步/256环境步与独立CUDA W1逐字节一致、全部出现物体支撑接触。见[本批报告](p65-heterogeneous-manipulation-io-report-20261114.md)及[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。合计143个不同验证节点通过。下一批M2-B是端点对象/接触特征与Env目标生命周期；M0可达性、实际抓放/PushT、语义编码及共享策略训练仍未完成。该批执行已结束。

> 2026-11-14 新应用主线：单GPU上8个异构机器人环境、抓取/放置/PushT共用一个自然语言条件策略。参见[研究与实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。首批M1共享多树动力学已完成源码资格：产品0c217310，新增28项多树测试和95项受影响兼容节点通过，见[M1报告](p65-mujoco-tyx-forest-report-20261114.md)。固定臂与独立完整FREE物体已可共同进行CUDA接触/运动；下一批是公共异构GPU训练接口及8任务场景。预训练语义编码、共享策略训练、完整抓放/PushT成功率与新安装交付仍待完成。下文原验收保留。


更新：2026-11-13；**G2b-A共享CUDA数组预算与32构型展示已完成限定源码/安装验收**。产品e1401fbf，源码64项通过、新包536成员同字节；包外interpret12步、CUDA256步（8,192worldsteps）均STRICT_PASS。257帧实际CUDA轨迹的视频已生成，见[报告与证据](p65-g2ba-resource-gallery-report-20261113.md)和[公共用法](p65-g2ba-resource-gallery-usage-20261113.md)。原两作者及父执行槽均释放。BALL-S、M-B2/M-B1、G2-R0与R3-A原有限验收复用。
D5-A、R3-N/P、G2a、D6-A 已获各自有限源码/安装资格；D4 原稳态功能仍 FAIL。
D6-A 的 FUNCTIONAL_PASS 与旧弹性 STRICT_FAIL_RECORDED 同时保留，不能合并成严格通过。
整个 P6.5 / Newton 功能对齐未完成。旧07:30计划、M-B2、BALL-S与G2b-A均已收口。下一资源模块是G2b-B完整HOST/native/context/JIT等观察与外部owner计量，再另冻更大规模及公平性能格；不继承旧截止或session，不重跑已绿组。

## 从这里继续

0. [G2b-A验收报告](p65-g2ba-resource-gallery-report-20261113.md)、[公共用法](p65-g2ba-resource-gallery-usage-20261113.md)：六solver-bank数组manifest/预算及32无接触异构构型；不是总显存预算、融合kernel或跨world接触。[BALL-S](p65-ball-scalar-constraints-report-20261113.md)原混合约束资格复用，BALL角限位仍未实现。
1. [M-B2验收报告](p65-articulated-deformable-report-20261113.md)及[公共用法](p65-articulated-deformable-usage-20261113.md)：非根连杆、完整J/M投影与共同发布已验；[实施合同](p65-articulated-deformable-implementation-20261113.md)已关闭。[上一批计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)仅作历史。
2. [功能验收矩阵](p65-feature-acceptance-matrix-20261113.md)：源码、后端、安装、应用和性能分列；不估整体完成百分比。
3. [当前连接架构](p65-feature-architecture-draft-20261113.md)、[统一编译架构](../architecture/unified-scene-compilation.md)：声明、设备来源、查询、载荷与共同发布。
   [离线架构总览图](p65-functional-architecture-20261113.html)保留G2a时点的8/16容量标记，当前32及预算以本页/新版文字架构为准；图中未连接的域仍明确分开。
4. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)与[新版离线全景图](phymeo-newton-panorama-2026-11-13.html)：下一通用模块与未完成范围；本轮详细事实以验收矩阵及原收据为准。
5. [D6用法](p65-distance-dissipation-usage.md)、[D6设计](../superpowers/specs/2026-11-13-distance-dissipation-design.md)、[D6-B应用协议](p65-damped-normal-load-application-20261113.md)。
6. [Batch C](p65-three-track-batch-c-20261112.md)、[Batch B](p65-three-track-batch-b-20261112.md)、[旧历史入口](p65-continuation-history-through-20261112-0035.md)仅按需引用，不恢复旧周期任务或重做旧审计。

当前G2b-A证据：`artifacts/p65-g2ba-resource-gallery-20261113-01`，产品e1401fbf、536成员同源，22正式CUDA及25/29视频已经结束；完整压缩轨迹/状态/视频保留，不重跑或解压补副本。原BALL-S证据 `artifacts/p65-ball-scalar-constraints-20261113-01`、d7105a9d、534成员资格复用，不重启原已结束组。
M-B2原证据 `artifacts/p65-articulated-deformable-20261113-01`、快照 `6b4a639c` 与验收 `cdccbe44` 复用；旧D6-B仍以143结束收据覆盖历史pending状态。
本页是续作导航，不替代原完整输入、输出、原始失败、source snapshots 与保存收据。

## 已验公共功能与边界

| 功能 | 当前资格 | 仍然不能声称 |
| --- | --- | --- |
| G2b-A共享数组预算与异构32 | 六solver-bank实际manifest、跨owner/边界峰值、UNKNOWN计费；64源码项；包外W32/B32解释12步与CUDA256步STRICT_PASS，真实视频 | 完整进程资源峰值、融合kernel、跨world接触、更大W、全部约束组合或Newton性能胜出 |
| BALL-S 球树混合约束 | fixed/FREE BALL/R/P中scalar软限位、DOF dry与NONE/FILTEREXACT、无接触/原condim1/3共同求解；源码132＋恢复兼容8、包外各448worldsteps通过 | BALL角限位、G1长舞、BALL刚柔、更大异构规模；下方R3各行限制是其原首片历史范围 |
| D5-A 公共刚柔声明 | `ParticipantSceneDeclaration(contact_response="deformable_normal")`＋`ParticipantDeformable` 自动装配唯一 D3 owner，资产球/程序胶囊短窗及生命周期源码、安装通过 | 任意关节机器人或持续稳态支撑；多独立单体见M-B1 |
| R3-N 浮根球树法向接触 | FREE＋BALL/R/P、NONE JOINT motor、primitive condim1；完整同入步与连续参考及安装通过 | 人形长舞、limits/dry 等全部组合；activation见R3-A |
| R3-P 浮根球树摩擦接触 | 同一约束/CSR路径的 primitive condim3 正控、安装通过；mixed1/3仅编译路由资格 | condim4/6、elliptic、FREE＋BALL limits/dry 已放行 |
| R3-A 浮根球树有状态执行器 | FREE＋BALL/R/P的NONE/FILTEREXACT JOINT，no-contact/normal/pyramidal；源码117＋兼容6及安装两后端各352worldsteps通过 | limits/dry、多FREE根、任意机器人/刚柔组合 |
| G2a 原容量内异构规模 | W1、W8/B3、W16/B3、W16/B8 源码格及晚失败/retry通过；W16两个格各有两后端安装资格 | W1/W8额外包外资格、单异构kernel、跨桶碰撞或W32以上 |
| D6-A 可选轴向耗散 | 保存 elastic reconstructed velocity 后一次 ordered 速度pass；D1/D3/D5有限源码通过，D1/D3安装 FUNCTIONAL_PASS | 精确守恒、旧elastic短门通过、全网络BE、全局drag、所有膜振型衰减 |

原 S1–S3 resident 多参与者查询、M1/M1-A 资产来源、D1设备节点、D2有限球/胶囊–triangle 与 D3共同发布资格继续复用。
D3/D5经M-B1扩展为每world一个节点模型＋多个ordinary单body动态球/胶囊；M-B2另支持每world一个ordinary R/P浮根关节树，子连杆球/胶囊真实受载、完整B×6与J/M共同发布。两类world可混batch。总participants≤8、worlds≤8、每world bodies≤33；入步采样、逐facet NormalLayer、零摩擦，存在HOST载荷/诊断往返。
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
旧06:15冻结/07:30收口已经结束。新一批按用户继续指令与M-B2合同执行，没有新的时间截止；仍不授权伪造通过或自动扩大模型域。
不建周期任务，不poll/rerun/dispatch远程CI，不push/mainmerge/发包，不改pin或盲跑寿命stress。
三份无关未跟踪稿、证据Junction及其canonical归档、用户Torch和当前venv/pins全部保留，不跟随Junction删除或全解压补副本。
观察器在物理返回后失败不能称物理回滚；UNKNOWN_RELEASE保owner/lease，不能重复free或伪报零live。

旧G1长舞46.29s XY门/131.48s完整动作、J/R学习、R03两项净加速度、B-point touch-SE仍开放。
通用摩擦/换面历史、CCD、更多刚柔机器人组合、传感、可微物理、密集布/体软体、杆缆与MPM/流体仍是后继；本轮不因安装绿而关闭。

第三包R3-A产品08d7a1b6、全部164/165返回和压缩trace已保留canonical installed-closed-archive-01/wheel-03；不重跑同源已绿矩阵。M-B2现有独立实施合同，BALL limits/dry与G2b仍为后继提案。

## 本轮后段接受与下一交付

- 第四包产品23757ae6：534成员同字节，191/192 M-B1各44worldsteps/88participantsteps；201/202 S8/M3各24/44。S8=4world/8participants，不外推8world满配。
- G2-R0：178预期RED，185源码12GREEN、193包外12GREEN；实际恢复晚失败保原发布、原异常与cleanup，确切候选强owner及真实关闭。UNKNOWN未被重试，资源完整预算仍未实现。
- 新canonical `artifacts/p65-feature-until-0730-20261113-01/installed-closed-archive-04/wheel-04`：33文件，原件保留，禁止重复归档/全解压。没有新项目执行owner；已绿同源不重跑。
- M-B2已验后，下一两条独立功能线：BALL树标量limits/DOF dry与完整mixed checker；G2b真实allocation manifest/生命周期预算。两者仍为提案，先冻结各自多案例与接口，再由原作者隔离编码、父共享接缝及唯一串行验证。
- D4/D6-B只按[可否证的短试样后继](p65-sustained-support-next-experiments-20261113.md)另冻实验；不选第二个c盲跑。本轮沒有新增该实验或BALL limits产品执行。

## M-B2 本次交付

- 产品/测试快照6b4a639c：新解释15＋CUDA10，受影响旧路径兼容39项通过；没有功能测试失败或数值修复。
- 新包534产品成员与源码/安装同字节，包外解释/CUDA各40worldsteps、147完整事件，三个owner均一次关闭成功。完整原始结果、格式首错与source snapshots保留，新安装归档9文件/3,959,181 bytes；原件保留，不重归档或重跑。
- 实际覆盖R–P串联与非对称R/P分支、两world、真实非零入速/控制变化、完整body来源与力学、晚失败/retry、restore/partial reset。多关节机器人同world、MuJoCo BALL刚柔、摩擦/CCD/持续支撑及新关节D6组合仍未获本批资格。
- 现无项目执行owner；下一批从上述两条提案选独立节点，不恢复已关闭M-B2任务。
