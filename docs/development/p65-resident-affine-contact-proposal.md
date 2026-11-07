# C7：设备常驻 affine-mimic 树的有限静态接触反馈提案

**PROPOSAL_UNEXECUTED / NOT_FROZEN。** 2026-11-08，作者只读源码、已有报告和本机 Newton 1.7.0 文本，未运行 import、数值计算、测试、静态检查或设备命令。本文件是后继方案，不能把 C6 目前已报 HOST 50 项及 R01 interpret 1 项扩称 C7 已通过。读取父工作树 HEAD `042a79d60afa9a8c54068fed3f4bdf8713d6bfe8` 及其当时 C6 接缝；C6 后续实测/改动仍以父正式交付为准。

## 1. 最小可交付能力与现状

沿原 `batch.prepare_contacts(model, bindings, materials_by_pair, ...)` / `batch.step_contacts(plan,h)`，允许已接受的 `ResidentAffineTreeBinding` 参与同 world 的有限 static sphere/capsule/unrounded hull、原 `RobotContactMaterial(friction=0)`。保持 `discrete_entry`、原有限几何/法向律、实际 stored frame 与原一次 batch swap。固定根、恒定无环 affine 图，`1<=K<N<=16`；不加 floating、moving/moving、持续摩擦或 CCD。普通 tree 与没有接触绑定的 affine/scalar world 继续原路径。

已有可复用事实：

- `backends/_resident_affine_tree.py:137–200, 267–361` 已将 K 独立 state/drive、N 物理 COM loads、原 FP64 图 token、P/c/原约束行、full/reduced checkpoint 分开；不是新建 K-body 假模型。
- `_resident_affine_storage.py:101,237–314` 组合原 `TreeDeviceStorage`，依次 full mechanics → P 投影 → 原 bounded PD/Cholesky K solve → full 展开/原图反力 → full geometry/energy/finalize → K retain。`491–550` 已返回 full 与 reduced M/b/J、外载、原行乘子/功。
- `_resident_affine_binding.py:15–48` 现有 `from_asset` 保原 source/model/graph proof，复用 full 物理 group 的材料编译；`_resident_collision_plan.py:354–382,438–443,490–498,589–592` 已可从实际 affine full FK 做设备 bounds，仍拒未排除的树动态/动态对。无需另一套几何绑定或 bounds kernel。
- C6 `_resident_contact_plan.py:97–99` 明确拒 affine 类型，`171–181,217–220,337–387,430–482` 将 scratch、packet、mechanics、solve、receipt 固定到 `batch._trees`。这是实际接入缺口，不能只删类型 guard。
- `_resident_contact_bridge.py` 的 full N×6 COM wrench、原站点/静态反力和真实末态锚点运输不依赖独立坐标维数，可直接复用；不再做一份 C7 HOST law/Scene。

首个有效 RED 应先证明 C5 原 affine asset/真实 full FK/材料绑定有效，旧有限 patch/law 的指定接触力成立，再调用现公开 prepare_contacts，准确停在 ordinary-only 边界；不能用缺模块或不合法 fixture 作 RED。

## 2. 一次受力与唯一提交

令 `q=P s+c, v=P u`，E 将 K 个真实电机放入 full independent_indices，`PᵀE=I`；C 保原图每行 `q_f−a q_d=b`。P 恒定，无 Pdot 项。所有计算以同一 accepted bank/epoch/tick 的 full FK、COM、速度为入步。

1. 同锁 stage 原 scalar/ordinary/affine held controls；current full external 不延迟，只有 K 维 drive triplet 按现 delay0/1 队列选取。每个相关 physical storage **只做一次**原 full mechanics，保 `M_N,b_N,Jv_N,Jw_N` 和不含接触的 `e0_N`。
2. 由实际 full N 状态生成 C6 `ContactShapeFrame`：原 group-local 材料点经 stored R/p；COM 与 v/ω来自真实设备。HOST 旧 patch/law 在共同作用点计算入步力，每站只乘一次归一权重。每个 world 保存完整 `w_N=(F_COM,T_COM)` FP64 与上传 FP32，两边材料见证、static 等反力均保留。
3. 用 C6 原 `contact_project` 对 **physical** J 散回 `g_c,N=Σ(JvᵀF+JwᵀT)`，仅加到本次 `physical.vector[...,1]`。不写 held loads、不把接触当下一步外载、不再注入 constraint reaction。随后复用 C5 原投影：`Mr=Pᵀ M_N P, br=Pᵀ b_N, er=Pᵀ(e0_N+g_c,N), Jr=J_N P`，保全部 cross blocks。
4. K 维原 bounded implicit PD 解 `Mr a=τ+er−br`；保持 512 默认/4096 上限、PD 原 5e-5 与质量方程原 2e-5 门。一次 `u+=u+h a; s+=s+h u+`，原 expand 得 `q+=P s++c, v+=P u+`。不调用 ordinary N 维 solve 后改 follower，也不能完整 `affine.stage_advance` 再加力——后者会重做 mechanics 覆盖本次接触。
5. 原 full balance 使用已经包含接触的 full external：`r_N=M_N(Pa)+b_N−Eτ−e0_N−g_c,N=Cᵀλ`。原 reverse-row 处理 chain/fanout；同时保 `Pᵀr_N`、各原行 `λ_j(C_j v)`、总 `r_N·v`。这些是传动约束反力，不等于 static contact reaction；fixed base 可传力，不一般声称全树线动量守恒。
6. 构造 full candidate FK/energy；容量/域/数值拒绝在 full finalize 前生效，full finalize 负责 N/FK/energy/tick 回留，原 affine_retain 依据同一 accepted 标志回留 K/history。所有 world 的实际 candidate 读取、双侧锚点运输、owned receipt、source/epoch 复核完成后，原 batch 才一次 swap/epoch，再发布对应记录。

功与能量不混同：`g_c,N·v = (Pᵀg_c,N)·u` 是同一入步虚功；FP32 实算按现误差门比较，不能要求两种加法次序位相等。`entry_power*h`、真实 application work、真实 material work、末态 KE 变化分别保存，不发明严格接触能量守恒门。COM 仍是实际设备值，不从独立舍入的 p/R 重建；C6 的 5e-5 stored-R 域和 1e-10 点构造一致性语义不改。

## 3. 必须先冻结的内部接缝

### 3.1 复用 affine 原阶段，不能复制 solve

建议父从 `_resident_affine_storage.py:237–314` 原样抽取私有“从已备 full mechanics 开始”的 project/solve/expand 阶段，以及原 full finalize/K retain 阶段；普通 `stage_advance` 继续按原顺序调用。具体签名父冻结，例如：

```text
stage_solve_from_mechanics(h,src,dst,tick_src)  # 仅原 project/K solve/expand
stage_candidate_geometry(dst)               # 委托 self.physical
stage_finalize_and_retain(src,dst,tick_src,tick_dst,accepted)
```

contact plan 在 solve 前用现 kernel 加 full 接触，candidate geometry 后、finalize 前注入容量拒绝。无接触普通 step 不改变核公式、次数和顺序。已有 C5 `check_advance/finish_advance` 继续负责原 pending 诊断。

### 3.2 full/K、row 与状态码

- plan 按两个真实 storage 管理 bounded scratch：ordinary storage 与 affine storage.physical，各自 rows/nmax；不为每 world 创建 runtime。`_handles(storage)` 和 `_stage_tree(...,storage,...)` 需显式知道是哪种 storage，不能靠相同局部 row 恰巧对齐。batch 全局 slot、storage row、full group、independent coordinate 四种索引必须分开。
- affine packet 同 bank 读取 N/K 两状态与同 tick，调用原 `spec.checkpoint(world,reduced,full,tick,_held_drive,_held_loads)`；geometry 用 `AffineTreeKinematics`，保 N 物理 position/velocity 和 K independent 字段。不能拿 public snapshot 的 active 值冒 candidate。
- **已发现的真实冲突**：C6 `contact_reject` 写 7，ordinary `STATUS_NAMES[7]` 为容量；affine `AFFINE_STATUS_NAMES` 以 7 表约束残差、8 表全方程残差（`_resident_affine_storage.py:18–22`）。文本检索现 resident 状态/赋码未发现 9 使用。建议 affine 新增 **9=SITE_CAPACITY_EXCEEDED**，reject kernel 显式接收或选择对应容量码；ordinary 保7、affine 保7/8原义。最终由父冻结，不提前放宽现 C6 affine 拒绝。
- 站点容量仍每 world 0..4096、包含零力几何站点。接触 scratch 每次清零；局部超帽不得推进该 world 的任一 N/K/control/history/energy/tick 或旧 receipt，邻 world 可接受。任意设备/IO/来源/完整 receipt 晚故障则全批不发布。

### 3.3 原 receipt 上追加明确坐标说明

复用 `ResidentContactReceipt`，不新建平行 records 系统。`law_wrench_fp64/uploaded_wrench_fp32` 永远 full N×6；`input_external/contact_generalized` 保 full N，`selected_drive/actual_drive` 在 affine 明确 K×3/K。diagnostics 增 `coordinate_kind,N,K,graph_token,P,c,C`、原 full motor `Eτ`、actual full/reduced acceleration 与 reduced mass/bias/**实际 reduced total external**、原 reaction/row multipliers/powers。

full `contact_generalized` 必须是原设备散回值。若由公开 P 与其在 HOST 复算 `Pᵀg_c,N`，只能标重构诊断，不能冒一个未运行的设备输出；不以 `Pᵀ(e0+g)−Pᵀe0` 的非结合浮点差偷当精确接触量。保原 C5 reduced total external 实际读数即可核真实接入。保持正常 ordinary receipt 现有字段含义。全部 accepted motion/work 来自同一个 candidate，不重采第二次 law。

## 4. 独立案例库（候选输入，待父冻结）

统一原 C5 物理 model/惯量/图/限位/命令源，`h=1/1024`、短窗40步；不改旧 C5 cases。新增碰撞几何是明确的安装声明，质量惯量不随形状重新推算。初始验证以下均取 C5 原初态；后继时刻仍用独立 full-component 导数/KKT oracle，加独立有限平面内球法向公式，不能拿产品 M/J 或采样力反造预期。

除说明外 `k=64 N/m,c_damp=0,mu=0`。每个 sphere 半径1/8，中心置相应 physical group 的实际 COM；static 有限箱右面为 `x_face=x_center0−1/8+δ0`，x厚1/4，y在初始球心±1/4，z±1/4，右面外法向+x。箱固定在此初态声明位置，绝不跟随当前球。正控 `F=(64δ0,0,0)`，COM矩0，static等反力−F。只声明该接触 pair，非目标 moving pairs 显式排除。

| 案例 | 冻结机制、载荷与精确首步预期 |
|---|---|
| C7-01：follower 真实载荷替代 | 复用 A01F：两平行P、m=(2,1)、P=(1,2)ᵀ、c=(0,1/8)、s=u=0、motor0、g0；将原外加 follower F_x3 **移除，仅以新接触产生同3N**。球心(1/8,1/2,0)，δ0=3/64。M=diag(2,1)、Mr=6、g_c=(0,3)、er=6，a_s=1、a_full=(1,2)，原行 λ=−1，r=(2,−1)。首步u=h、s=h²，follower q=1/8+2h²。另同一物理安装换成z向有限capsule轴±1/16或box半宽(1/8,1/16,1/16)，总3N不随2/4站点翻倍。 |
| C7-02：负比例 fanout | 复用 A02：三个并列+X P，m=(2,1,4)，P=(1,2,−1/2)ᵀ、c=(0,1/8,−1/16)，原 motor7。仅第三组接触3N，初球心(−1/16,−1/2,0)，δ0=3/64。Mr=7、g_c=(0,0,3)、Pᵀg_c=−3/2；a_s=11/14、full a=(11/14,11/7,−11/28)。原两行 λ=(11/7,−32/7)，r=(−38/7,11/7,−32/7)。正x力在负gain下给负独立增量，不能把gain取abs。 |
| C7-03：chain 与等运动 flatten | 复用 A03，m=(2,1,1)，原行q1=2q0+1/8、q2=−q1/2+3/16，P=(1,2,−1)ᵀ、c=(0,1/8,1/8)，motor7。第三组球心(1/8,−1/2,0)，δ0=3/64，真实3N。Mr7、er_contact−3、a_s4/7；r=(−41/7,8/7,−25/7)。原chain λ=(41/14,−25/7)；明确flatten为q2=−q0+1/8时λ=(8/7,−25/7)。状态可等值，原行反力不能换标签，graph token/checkpoint/binding不可互换。 |
| C7-04：R/P full cross coupling | 复用 A05：root Rz，两个P分支在(1,0,0)/轴+y、(0,1,0)/轴+x，m=(2,1,1)，原惯量不变；q2=2q1+1/8，P=[[1,0],[0,1],[0,2]]，s=u=0，原full effort=(0,1,0)，motor0。第三组COM(1/8,1,0)接触3N，δ0=3/64。full g_c=(−3,0,3)，reduced=(-3,6)，Mr=[[161/64,−1],[−1,5]]，a_s=(−512/741,935/741)，full a=(−512,935,1870)/741；原λ=53/247、r=(0,−106/247,53/247)。无驱动R也响应，不能对角化。重力留出A05G保持g=(0,−2,0)，只把原COM F_x3替换为接触3N，直接复用已验 a_s=(−1360/741,469/741)、λ=25/247。 |
| C7-05：非零角offset与转动功 | 复用 A04 两Rz：长度1、COM各1/2、m=(1,1)、Izz=1/12，q1=−2q0+π/2，s0=0/u0=1/2、motor1、g0。第二组球心(1,1/2,0)，δ0=1/32，真实2N；full g_c=(−1,−1)，reduced1。原 M=[[5/3,1/3],[1/3,1/3]]、b=(0,1/8)，Mr5/3、br−1/4；a_s27/20、full a=(27/20,−27/10)，λ27/40、r=(27/20,27/40)。入步接触功率1/2W，原constraint功0。连续40步使用实际圆弧FK和原surface/common运输，不能用hJv冒末态位移。 |

上述首步的 N、COM矩、full/reduced generalized load、a、原图λ均需逐项核，不只核某个q；λ正号沿C行的 follower+1。static reaction不是λ或固定根完整支撑。几何换序/全局axis-cycle和平移沿 C5 已有安装变换：同一物理预期转坐标，接触材料ID和graph名映射不变。保持 C5 原单步 `atol5e-6,rtol5e-5`、40步 `atol2e-4,rtol5e-5` 和原反力门；不为新片扩大旧预算。

### 独立卸载/重入与 PD 两步

- **C7-06 连续卸载/重入**：C7-01 同原m/P/c/有限箱与3N初压力，仅作为明确新物理留出设置 `u0=1/4`、恒定 independent ff=−6N、kp=kd=0、cap16。不改控制或重置状态。有限面内独立标量律为 `a=[−6+2 max(0,64(3/64−2s))]/6`；每步严格原半隐式顺序。初a0=0，入层连续参考ω²=128/3，首次离开阈s=3/128，理想连续t_exit=asin(sqrt(3/8))/sqrt(128/3)，u_exit=sqrt(5/128)；层外a=−1，首次重入在t_exit+2sqrt(5/128)，落在固定总7/8s窗内。此连续公式只帮助核事件，不冒离散首触时刻。h/h2固定1/1024与1/2048，最多896/1792实际步，首真实卸载后的再正载接受步可提前停；逐步独立标量更新、原力、history/graph约束与无残留接触核对。
- **C7-PD 两步短控**：C7-01 起点，kp0/kd6/cap1/4，delay1，初队列(target0,rate0,ff0)，已持新命令(target0,rate1/4,ff1/8)，原3N入步。第一步仍选旧队列，`τ=−6h/(1+h), a=1/(1+h)`；第二步选新队列并饱和为+1/4，当前接触由已变化s重采。delay0首步直接+1/4。另低cap=1/1024旧零rate正控为τ=−1/1024，a=1−1/6144。完整held/queue/full N loads分别存，不能为证明PD把接触改成held外载。

## 5. 世界隔离、事务与验收顺序

W1先C7-01；W4建议非连续键(503,91,1207,6)对应 ordinary C6-R01、affine C7-03、affine C7-04、原FreeSphere，按反序注册并跨world重用局部body/shape IDs。全部40步，原FreeSphere用已验z2/g−8/m2/h条件，`v_n=−8nh,z_n=2−4h²n(n+1)`；ordinary与affine分别依独立原oracle，不调批量版输入。tick19保存、同tick restore（epoch必变）、21步重放与不断点参照一致。

必须另有：

1. affine某world cap0时完整几何站点超帽，该world N/K/queue/held/FK/energy/tick/last receipt保持；ordinary、另一affine、scalar继续。按原同source重建足帽plan、同请求重试，与独立暂停该world的参照等值。code9具名可见，不能显示AFFINE_CONSTRAINT_RESIDUAL。
2. 双buffer均经历新路径；真实 contact `_buffer` 第二分配失败、上传失败、candidate读回、finish owned receipt失败、poison/rebuild、close异常照 C6 对应边界测试，**不是重跑原整个故障套**。late全局故障前后全部公开状态/held/history/epoch/旧receipt等值；重建从原checkpoint，不手写内存救活。
3. graph/source错配：同full几何但不同chain/flatten图、不同P/c、错physical group、换source形状、跨world、无材料、非法μ、未排除dynamic/dynamic均在分配/step前拒绝。readonly数组与外部修改不污染旧snapshot。
4. h/h2、full graph约束/原行反力、实际正常力与两类功是功能验收；完整wall成本/launch/上传/读回只按实记录，不设猜测速度目标，不将HOST逐world几何称全GPU。

阶段顺序：有效入口RED → C7-01首真实响应/PD与容量 → C7-02/03/04/05初态和40步/重排变换 → mixed/late失败/双buffer/重放 → C7-06两窗 → 同source interpret/CUDA → 新源码外public consumer。短窗按40步明确成本，长窗按上述上限；墙钟待首真实小组报告后排队，不做额外预性能probe。只运行新增和确受phase拆分影响的少量原C5/C6相邻检查，不重跑整旧矩阵。

## 6. 文件所有权和避免第二求解器

父独占 OLD：`_resident_contact_plan.py` 的两storage packet/scratch/receipt、`_resident_affine_storage.py` 原阶段提取/码9、`_tyx_resident_contact_kernels.py` rejection码参数、`tyx_world_batch.py` 的storage显式分派与统一提交。按需要仅文档化现 public receipt；原 `AffineRobotTree`/binding/model/graph/solver核/旧桥和 law 不必重写。

拟后续作者 NEW：`tests/scale/test_resident_affine_contacts.py`、`tests/scale/test_resident_affine_contact_transactions.py`、`tests/wheel_consumers/resident_affine_contacts.py`、`docs/benchmarks/p65-resident-affine-contacts.md`。本提案不预设额外产品模块或 kernel；若父阶段薄接缝不足，先给精确缺字段/顺序再冻结，不能复制 C5 大段动力学或创建另一 Scene/runtime。首片依赖 C6 正常完成验收；默认 `step()` 与现 C6 ordinary 域仍原样。

## 7. pinned Newton 1.7.0 对应与不能混称之处

只读本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA:3` 为1.7.0；以下均该安装树，不以main最新行为代替版本事实：

- `_src/solvers/featherstone/kernels.py:67–130,133–180` 实际将 full惯量和广义力按两侧/一侧multiplier累加到独立reference，再展开加速度；与本片“保full M、PᵀMP/Pᵀ力”是可对照的层次，不意味着两端控制律/接触采样相同。
- `featherstone/solver_featherstone.py:707–744,953–1022,1051–1126` 在实际FK COM twist下读取contact，累加body_f，再做mimic reduction/solve/expand/integrate；原body force和contact不能重复注入。该文件`94–103,126–128` 明示joint effort/velocity limit等不支持、mimic须同articulation；不能把native内置PD称Phy bounded PD的等同验收。
- `_src/sim/builder.py:6619–6622` 仍拒reference自身已是mimic或follower又被别人引用。C7-03原chain不能拿flatten通过冒原图支持。C5已有400步CPU/CUDA参考与链拒绝记录直接复用，不重跑；未来若补native接触，只针对新例并记录实际Contacts和solver状态。
- `semi_implicit/kernels_contact.py:465–558` 从两侧contact point/COM读取速度，法向弹簧和只在接近时的阻尼后写双侧body wrench；它不是Phy归一patch站点的同一law。即使本片μ=0且首库c=0，box/line点数与总刚度口径仍须据原生实际Contacts核对，不强拼速度比或力等价。`body_parent_f`（Featherstone文件`106–124`）是原递推COM joint wrench，不是本片原图每行λ；从native全方程派生λ应另标派生量。

停止线：若需要放宽既有C5/C6门、丢full/J交叉项、重新生成材料而不保source、把错误图flatten后标通过、增加第二次完整step猜接触，立即停在最小反例。实际FP32或非线性运动误差先保原收据并按已有工程门分类，不进入无限精度证明循环；本片不依赖持续摩擦持稳例是否通过。
