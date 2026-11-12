# 功能补齐执行计划：2026-11-13 至 07:30

状态：**IMPLEMENTING，用户已授权按计划执行**。北京时间2026-11-13约01:01开始，07:30截止不顺延；06:15冻结新功能。计划基线 `6dcf99c2`，上一批已验产品 `3534f742`。原三位作者已从该基线在各自原worktree切换新隔离分支，父/root是全部项目执行的唯一串行owner；当前实现不等于验证通过。执行收据入口为 `artifacts/p65-feature-until-0730-20261113-01`。

## 1. 本轮原则与交付优先级

以“用户可调用的新功能或新支持组合”作为主交付。连续测试、容量测试和安装调用是验收手段；不以测试数量、重复审计、只写计划、扩大数字上限冒充功能完成。不再追求精确零穿入或无限精度；新长窗先登记工程误差、物理输入与通过门。

依赖分两条：

- 现有D3共同响应 → **D5-A统一场景声明接线**；现有D3 → **D4-N持续法向载荷**。两者可独立开发和验证；前者不必等待全部摩擦D4，但不因此宣布完整D5关闭。
- 现有FREE+BALL动力学＋原normal求解器 → **R3-N浮根球关节接触**。与变形线路独立，可并行编码。
- 现有G1异构分桶 → **G2a原容量内功能规模**。作为配套，排在上述功能及其安装闭环之后，不抢占物理组合开发。

| 优先级 / 节点 | 难度 | 用户获得的能力 | 完成的可观察标准 |
| --- | --- | --- | --- |
| P0 / D5-A | 高 | 用公共场景声明创建资产球/胶囊＋可变形片，编译后直接step、读状态和力、reset/restore；不手工拼接内部owner | 两类不同公共场景的完整身份、双方载荷和共同状态一致；解释/CUDA及源码外调用通过 |
| P0 / D4-N | 中高 | 当前法向模型可在同一真实轨迹上承载、受控卸载、离层和再接触，给出可复用连续观测 | 有支承的可变形片、多形状/材料/h；全窗状态与力一致、有限压入、事件与载荷门通过；无reset伪造事件 |
| P1 / R3-N | 中高 | 带浮动底座及球关节的机器人使用现有primitive法向接触，与环境或自身分支产生真实动力学响应 | 双足支撑与分支互碰两族；底座/关节完整力和运动、独立native参考、故障恢复及安装通过 |
| P2 / G2a | 中 | 在明确的原容量内运行更多不同模型world，公开完整质量和资源记录 | W1/8/16及8桶边界；每world完整输出/参考、恢复/超容量拒绝；不以零控制或抽样world替代 |
| 必须收尾 | 中 | 一个包含本轮已接受增量的安装包、更新后的功能图与接手入口 | 源码外解释/CUDA调用、产品字节同源、原失败保留、功能/限制表明确 |

目标是优先关闭P0，再争取R3-N和G2a；每片独立验收。不会因为时间到而将未过的组合标成支持。

## 2. 已完成的针对性源码核对

本计划复用[Batch C结果](../../development/p65-three-track-batch-c-20261112.md)和[三主线路线](2026-11-12-scene-deformable-scale-roadmap.md)，没有重新泛化调查Newton或重跑已绿组。

- 场景编译器当前只有 `host_patch/none`，D3响应另有共同owner；真正缺少声明、身份、资产来源与公共生命周期接线。独立 `compile_scene(SceneDeclaration)` 仍是旧入口，不能误称所有编译器已统一。
- `_deformable_response_source.py` 当前明确拒绝from_asset。资产binding的topology可为None，必须复用原source-shape/model-token校验；仅删除拒绝会丢失来源约束。
- D3每真实步重新查询并执行无记忆normal law，已有完整反力与事务。持续法向资格暂无证据需要新solver/history状态；原全free片和单固定点案例不能直接证明固定环境承载。
- FREE+BALL在 `_model.py` 被送往无接触编译器，`_floating_model.py` 又拒绝全部潜在接触。原normal wrapper已有完整DOF/CSR/M/J及FREE/child BALL积分参数，可以做有限组合；runtime能力标签必须同时区分free与fixed BALL。
- G1为8模型/16world功能上限；累计allocation/upload计数不是live/peak内存。旧同构S256不是异构资格；扩到32/128/256需要完整生命周期资源预算，今晚不先改上限。

## 3. 开发与案例节点

### D5-A：公共场景到共同响应

增加显式 `deformable_normal` 模式和不可变变形体声明，经现有 `compile_participant_scene` 静态分派、原资产/控制/安装身份校验，直接构造唯一D3共同owner。不能先造独立刚体场景再套D3，不能顺序step两侧再称共同提交。

公共接口委托原D3的step、snapshot、checkpoint、restore、close。初始控制提交后捕获共同初态，partial reset恢复同一初态及held controls。查询必须区分“本次响应入步query”和“当前accepted状态query”；后者只有实现明确revision/currentness接口才公开。

首域保留每world一个ordinary单body浮体、球/胶囊及一个D1模型、零摩擦；不同时扩多刚体或关节刚柔组合。

- 资产球压片：稀疏W2、非零入步速度、偏心COM、原材料/面身份、完整层表；partial reset与restore后真实后继。
- 程序化资产胶囊推竖直片：W1、不同朝向/接触端点、真实平移和转矩。胶囊不冒称URDF原生语法已支持。
- 来源篡改、缺材料、world不匹配、末次共同发布前故障、构造清理错误；原direct路径受影响兼容。

### D4-N：连续承载作为功能资格

选定一个统一的新案例族：四角固定、中心自由的五点膜；四扇形facet各保独立离散弹簧语义，材料内力/固定反力来自真实ordered history。四面初始共法向可作量纲参考，变形后不能把4k或某个高度当精确平衡答案。

拟登记四格：球、法向胶囊、球的材料柔度变化、球的h减半；由同一实际轨迹完成入层→承载→主动向外卸载→离层→再接触。初始速度非零；不混入另一个三固定点候选的输入/门。

执行前由测试作者提交唯一cases协议：原始坐标/拓扑/质量、层参数、h、载荷时序、总时长、事件窗、工程误差与参考算法。当前3s候选作为起点审阅；只做一次执行前物理可达性核对，随后冻结。主动上提明确称“受控卸载”，不冒称无外功自然反弹。暂态与稳定支撑门分开：未达到预登记稳态窗不能移动窗口或称稳态；若模型仍有有界振荡，按原暂态门和实际动力学解释，不先归咎浮点。

每步检查完整facet/round身份、原层力、双方COM力矩、固定反力、真实材料变形和独立运动参考；多接触参考按(surface,round,facet)计，不能按facet去重。工程长窗漂移另登记，原短窗门不改。原法则若已足够，只新增公共runner/观测和验收；出现实际产品缺陷再窄修。

本轮两位原作者已一致选定下列主输入与门；完整稳定ID、惯量、初态数组和序列化绑定在cases提交中一次补齐，首运行前检查，不执行探针选参数：

| 项 | 本轮D4-N登记值 |
| --- | --- |
| 支承/材料 | 四角(±1,±1,0)固定、中心(0,0,0)自由；四个扇形facet与四条径向距离约束；中心质量0.5kg、刚体2kg；原柔度1/16384，N3为其两倍 |
| 接触/初态 | 原逐facet法则：r=1/8m、width=1/16m、k=128N/m、d=0.5s/m（原乘法速率阻尼）；初始gap=width+1/128m、法向速度−1/16m/s；胶囊轴长1/4m，以最低轴端定义gap，COM比同gap球高1/8m |
| 时序 | T=3s；h=1/1024s，N1另h=1/2048s；零gravity、COM外载[0,1)为−16N、[1,9/8)为+16N、[9/8,3]为−16N，均沿法向 |
| 支撑窗 | [0.75,1)与[2.75,3)；总接触力的法向分量均值为L的0.8–1.2；刚体COM与自由中心法向速度各自RMS≤0.1m/s，不以相对速度小替代 |
| 连续/工程门 | 入层、支撑、层外零力、再接触依序出现；事件至少持续1/128s（粗8步、细16步）；峰值≤8L、gap≥−r/2，并独立保持膜原侧。无符号距离不能把翻面认作再接触；不宣称CCD |
| 长窗参考 | 新position≤2e−3m、velocity≤1e−2m/s、orientation≤2e−3rad；旧逐步几何/法则/材料/M/J门原样保留。连续参考从自己的初态推进 |

四窗共15,360物理步/后端，两个后端合30,720步，运行时间尚未测量。先跑完整粗步N1，观察实际成本及首失败，再排剩余固定格；若来不及，整格标未执行并转独立功能，不删除后半轨迹或收窄统计窗。证据逐步流式写入，避免历史receipt嵌套和内存累积。

摩擦、一般换面测度/历史、CCD和任意柔性接触不随D4-N通过关闭。

### R3-N：FREE+BALL法向接触组合

先做完整floating结构准入→潜在pair全枚举→原primitive/condim1/cap检查→原normal编译；无失败后fallback。NONE JOINT电机保原完整CSR；不放开activation、limits、dry、tendon、FREE target或任意shape。

主节点复用动力学、几何、normal solver和积分器；改组合准入、显式路由及能力标签。保持nv35/nq45、actor32、geom128/pair16/slot32及原数值门/求解预算。

- 非对称自由底座双足：一支BALL、一支SLIDE，偏心惯性、非零root/child速度、斜gear及非零控制，核接触支撑和底座反应。
- 自由底座下两条BALL运动分支互碰：不等质量、偏心COM、多电机/共享target，核双方接触与共同祖先广义力。
- 各两稀疏world、原24步正向/反向/卸载；完整same-entry和独立连续native参考。
- 晚失败、retry、checkpoint/clone/partial reset真实后继；未支持组合与非法四元数保持拒绝。

原normal数值门保留；不是只解开guard或让模型构造成功。pyramidal condim3在本片完全关闭且时间有余时另登记；FILTEREXACT需要独立路由/schema工作，今晚不顺带放开。

### G2a：为功能服务的规模资格

原三模型保持h=.001、原24步非零控制及全部原门：

- W1三格分别运行；W8按3/3/2；W16按6/5/5。
- W16/B8边界：8个不同源身份、每桶2world，原三类物理参数不变。
- 每个world保完整力/运动/native参考；末桶故障/真实retry、checkpoint/partial reset后继；W17/model9前置拒绝。
- 分别记录构造、step、完整观察、参考、保存和边界操作成本；峰值未完整观测则标UNKNOWN。不能用累计上传量作峰值、不能称快于Newton。

不为做大数字删输出、减接触密度或改数值门。若P0/P1仍未闭环，G2a顺延。

## 4. 并行作者与共享文件

仅沿用原三位agent、现有隔离worktree；父唯一项目执行者。下列纯作者grant已派发；作者不运行任何项目命令。

| owner | 独占工作 | 关键文件边界 |
| --- | --- | --- |
| mtyx_t2_source | D5-A静态lowering、公共compiled包装、短窗用例与报告 | NEW builders/deformable_scene_lowering.py及专用测试/安装consumer；不直接改旧共享文件 |
| mtyx_t2_tests | D4-N唯一cases、独立参考、连续runner与事务负控 | NEW examples/deformable_normal_load_cases.py、run_deformable_normal_loads.py、专用测试/协议；与D5用例按接口协作，不双写 |
| mtyx_t2_audit | R3-N组合准入与独立机器人矩阵 | 隔离树内_floating_model.py的限定准入、新专用测试/安装consumer；不改父runtime或公共分流 |
| 父/root | 共享声明/分流/来源校验/导出、G2a工具、合入和全部执行 | participant_scene_compiler.py、_deformable_response_source.py、必要D3公共query、_model.py、_mujoco_tyx_runtime.py；精确allowlist在代码前登记 |

D5初始control与subset reset使用原公开checkpoint/restore共同事务；D3新增公开query/currentness由父统一补，禁止作者私改checkpoint或访问私有bank绕过。子agent只做编码/文本，所有import、collection、数值、pytest、lint/format、build/install/profile均父串行且每条fresh guard。父共享allowlist为 `participant_scene_compiler.py`、`_deformable_response_source.py`、`tyx_deformable_contact.py`、`phymeo/__init__.py`、MuJoCo-TYX `_model.py`及`_mujoco_tyx_runtime.py`；后续实际缺陷需先窄定位再记录增量。

## 5. 到07:30的时间安排

时间块为资源分配，实际测试时长未知；不是承诺每一项必过。作者可以并行，执行队列绝不并行。测试就绪可提前填空档，但按功能优先调度。

| 北京时间 | 作者工作 | 父串行队列与检查点 |
| --- | --- | --- |
| 00:30–00:50 | 一次冻结每片公共接口、case输入/门、allowlist和资源边界 | 核新基线/WIP/owner；不重新调查已清楚的旧模块 |
| 00:50–02:15 | D5-A、D4-N、R3-N三路并行作者；先交测试与小提交 | 共享接缝窄改；已交增量静态/最小首格。出现实质反例立即归类 |
| 02:15–04:15 | 根据真实失败修实现；完成D5和R3剩余负控 | 优先D5短窗/事务＋D4连续首格，再完整两后端；原同源绿组不重跑 |
| 04:15–05:30 | 完成独立安装consumer；P0未完则回到P0 | R3-N完整矩阵和必要旧兼容；父小提交功能结果 |
| 05:30–06:15 | 只补已接受功能的收口；有空档再G2a | G2a功能规模或处理前序实质问题；不启动新的大域/新物理模型 |
| 06:15–07:10 | 冻结本轮候选产品，只修阻断安装的明确问题 | 统一wheel及受影响源码外interpret/CUDA消费者，保存核对和提交 |
| 07:10–07:30 | 不再开新功能 | 更新差异全景/架构连接图/续作入口，列实际通过、失败、未执行与下一步；07:30交付概览 |

06:15是范围冻结点；未成熟增量留隔离分支，不混作已验产品。临近截止不启动预计无法收尾的长命令；已在运行的命令在可安全结束的边界收束，任何未完成项按真实状态报告，不能为截止伪造PASS。较晚启动时先削去G2a和pyramid，再削减R3-N扩展，不取消P0与安装收尾。

## 6. 成功、失败与防止原地循环

- **功能完成**：公共API在不同案例中复用；力与运动共同接受；有限误差/资源和错误语义明确；两后端和源码外调用实际通过。支持表新增一格，才算功能增量。
- **验收补足**：D4-N/G2a可能无物理核改动；报告为既有功能获得连续/规模资格，不虚构“新solver”。
- **有限失败仍有产出**：保原输入、完整首次失败前缀、定位到准入/来源/参考/物理/事务哪一层；不因某一片失败阻塞独立片。
- **停止当前假设**：同一原因两次有针对性的修复/诊断仍未缩小，或单片连续诊断超过约60分钟没有新证据，就固定最小反例并转独立节点；不改fixture/h/质量/时间窗/预算追绿。
- **避免精度死循环**：只处理影响可运行性、错误力/运动、失稳、漏接触或超过预登记工程门的问题；不把允许的有限穿入变成精确数学接触要求。
- **避免测试死循环**：同源已绿不重跑；只有源码实质影响才跑相应兼容；保存核对只针对本次新结果，不重复旧压缩档案/hash审计。
- **范围控制**：本轮不做无假设G1长舞/重训练，不改pin或运行寿命stress，不追Newton速度冠军；不扩成全摩擦/CCD/全软体/MPM项目。

## 7. 07:30应交付的概览

1. 新功能表：公共场景刚柔接线、持续法向资格、浮根球树法向接触、异构原容量资格，各列已实现/已验/未过/未执行。
2. 当前架构图：场景声明→编译→设备状态/查询→同入步载荷→共同发布；MuJoCo-TYX机器人分支独立画出，不暗示两求解器已经任意耦合。
3. 每片commits、代表案例、工程限制、真实首失败和安装产品版本。
4. 下一步按剩余依赖排：摩擦/换面→更广D5；pyramid→activation；完整内存预算→扩容→公平性能。

沿用[续作入口](../../development/p65-continuation.md)和[差异全景](../../development/phymeo-newton-panorama-2026-11-12.md)。禁止周期任务、remote CI、push/mainmerge或发包；不改用户环境，不清理旧证据Junction或三份无关草稿。实际结果另记，不能用本文件的安排宣称验证通过。

### Functional diagnosis after the retained strict precision failure

Run46 reached 301 actual returned steps, then failed the unchanged short
same-entry velocity gate: 2.27667386e-5 m/s absolute discrepancy. This strict
qualification stays FAIL. Following the user's functional-first direction,
the runner now offers explicit --diagnose-short-state: it streams each original
short position/velocity/multiplier assertion failure, with unchanged thresholds,
actual and reference arrays, while continuing the same fixed trajectory.
It can only return DIAGNOSTIC_COMPLETED, never PASS. Every force/geometry,
stored-arithmetic, long-window drift, oriented-side, penetration, peak and
support/unload/recontact requirement remains active. No input, damping, h,
statistics window or state is altered; this diagnoses real functional behavior
without pretending the original precision qualification passed.
### 04:00 functional checkpoint and remaining queue

D5-A, R3-N, R3-P and G2a have their bounded source/installed qualifications.
The original undamped D4 N1 full3-second windows both completed but failed
the predeclared steady-support means/RMS. They remain FAIL; other original
D4 cells were not launched merely to accumulate steps.

D6-A adds an explicitly declared axial velocity-dissipation material, with
source and second-wheel functional qualification and retained strict elastic
precision limits. Its API/operator does not imply D4 steady support.
The new [D6-B application](../../development/p65-damped-normal-load-application-20261113.md)
predeclared c=4096 N s/m from a separate dimensional estimate, unchanged
original four-case geometry/load/h/windows, and separate case identities.
New short contracts140/141 passed14 interpret and1 CUDA node respectively.
First package launch142 failed before construction (0 steps) because the
outside helper directory was not in PYTHONPATH. Launch143 adds only that
explicit outside helper directory and runs the full coarse N1 interpret
window under predeclared diagnostic-short-state mode.

Next: inspect that complete fixed window; if functional support is achieved,
continue the remaining fixed cells/backend qualification. If the actual
functional gate fails, retain the result and diagnose the model/observer
once before assigning another independent feature; do not scan c or alter
time windows, masses or acceptance gates. All project work remains root's
single guarded execution queue. Architecture and qualification documents
are being updated independently in the original author worktrees.

### 04:30 queue: retain D6-B failure; qualify R3-A composition

Run143 finished 3072 returned/checked steps with both steady windows FAIL.
The c4096 application expansion is stopped as predeclared; run148 only renders
the saved old/new observations, with no new physical stepping. See the
[closed result](../../development/p65-damped-normal-load-result-20261113.md).

The independent R3-A source investigation and narrow seven-product-file grant
are frozen in [the design](../../development/p65-free-ball-activation-next-design.md).
Original three agents author compiler routing, six fixed-source/native consumer
and transactions separately; root owns five shared seams and every guarded
project command. Four exact old unsupported activation XMLs become explicit
two-step positives; unrelated negatives and numerical gates stay. Next queue:
focused static checks, complete new interpret/CUDA source and transaction
matrix, affected old routes, then fresh wheel and installed consumers.
No kernel/device ABI or capacity change is authorized by this slice. Two
unsuccessful diagnoses of the same issue freeze that defect; the original
06:15 feature cutoff and 07:30 result overview remain unchanged.

## 05:00实际进度与下一批

R3-A已完成完整117新源码节点＋6兼容节点、第三包532成员parity、164/165安装每后端352worldsteps，见正式报告。D6-B稳态FAIL保持，不继续扫描系数。
下一批M-B1按[冻结合同](../../development/p65-multibody-deformable-implementation-20261113.md)仅补同一节点片多个独立浮体的稳定地址、逐body载荷与共同发布。三原作者纯编码，父共享接缝/唯一serial验证；M-B2关节树与BALL limits/dry仅提案。06:15冻结、07:30收口保持。

## 05:30 已实现功能的容量资格收口

M-B1 新源码28唯一节点及旧兼容各8已通过；G2-R0真实恢复晚失败清理缺口经178 RED、185完整12 GREEN完成窄修。统一wheel04产品固定23757ae6，源码/包/安装534成员相同；安装消费者串行进行。

最后新增的验证范围只有M-B1已实现容量，不增加产品能力：S8为四独立world(7,31,43,97)各原A两body，共8participants，4连续步；43/97仅重标原7/31的world和安装key，几何/anchor/质量/速度/外载/门保持。M3为world7原A首body、world31原A双body，验证混合单/多body布局，4步。完整原独立oracle/逐body来源/力矩/连续运动/一次发布；预期每后端24worldsteps/44participantsteps。不是性能、W32或稳态资格。两NEW tests/consumer由原测试作者纯编码，父唯一串行验证；执行后不调fixture或阈值追绿。06:15之后不增功能。

## 实际收口：范围内执行、安装与文档提前完成

M-B1、S8/M3与R0的源码和第四包验证均已结束，包外新增容量各后端24worldsteps/44participantsteps与预声明一致，R0安装12项通过。最新产品23757ae6、534成员parity及canonical归档已完成；不重跑已绿矩阵。原D4与D6-B稳态FAIL、D6严格精度失败分别保留，其他未执行应用格不填PASS。

本计划的功能节点和其已登记增量已经收口，不为填满至07:30的剩余时间启动新的BALL约束大域。原作者完成了compiler与mixed-EFC checker的具体接缝调查，但该新域仍需单独冻结后执行。M-B2、G2b和持续支撑短试样同样作为后继，不混入本次已验产品。

[本批交付概览](../../development/p65-feature-batch-summary-20261113.md)、[当前全景](../../development/phymeo-newton-panorama-2026-11-12.md)和[离线架构图](../../development/p65-functional-architecture-20261113.html)取代本文件中各时段的pending描述。源码、产品包与应用结果按实际版本分列；整个P6.5未完成。所有执行槽释放，无新周期任务、远程CI、push或环境改动。
