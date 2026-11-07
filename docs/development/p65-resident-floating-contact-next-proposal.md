# C10 普通浮根常驻接触反馈：下一片提案

状态：**NOT_FROZEN / PROPOSAL_UNEXECUTED**。本文件只有文本调研和候选案例，不授实施，也不提前接受 C9。作者树起点 `1bddde9`；父源码读取至 `1ba1c8058aa2d08b8d0334096d17bbdf09dfcd40`，C9 完整验证仍在进行。没有项目导入、数值计算、测试或性能执行。以下数值是手推首步锚点，均不是实测结果。

## 1. 首片能力与前置条件

在 C9 正式接受后，复用同一 `TyxWorldBatch` 的普通浮根完整树，增加：

- 有限 sphere / capsule / convex hull 对有限静态 S/C/H 的 `mu=0`、`discrete_entry` 接触反馈。
- 显式 `tree_dynamic_pairs=True` 时，同一个完整浮根物理树中不同、非直接相邻 body 的双侧接触；两侧真实运动、双方 COM 力矩均进入同一 full mass solve。
- `n=0..16`，`B=n+1`，`D=n+6`，`Q=n+7`；root 为 body 0，joint i 的 child 为 body i+1。root 六维不是六个伪电机，motor 数仍为 n。

这是 **HOST 有限窄相/原法向律 + TYX 完整动力学与单次发布**。当前浮根 contact binding 的拒绝必须保留，直到父冻结并接好完整适配。当前姿态候选不代表 CCD；不覆盖 floating affine、跨独立 participant 的共同求解、持续摩擦、未知深包含/非唯一接触斑块。已有 geometry 拒绝不改成零载荷。

## 2. 已存在什么，真正缺什么

以下行号属于上述父源码快照；后续排版可移动。

| 源码事实 | C10 最小处理 |
|---|---|
| `backends/_resident_contact_plan.py:97` 明确只接 fixed / affine / static；`:193` 将非 ordinary storage 当成有 `.physical` 的 affine。 | 增加明确 floating 分支，不能只放开 binding 类型。浮根 storage 没有 `.physical`。 |
| 同文件 `:195–204`、`:409–431`、`:527–595` 用同一个 n 作为 body、广义速度和 drive 数；`_tyx_resident_contact_kernels.py:60` 的投影也是 n×n。 | 分别使用 B、D、n。临时 wrench 为 `[T,Bmax,6]`，projected 为 `[T,Dmax]`；root 与全部关节列均保留。 |
| `_resident_contact_bridge.py:113/288/351` 把 group index 限到 15、body count 限到 16。 | 父有界提升为 index≤16 / count≤17，并仍以实际 spec 校验。固定/affine 物理上限不变；不能让普通 n=16 接受不存在的 body16。 |
| `_resident_floating_storage.py:119–121,216–220` 已有 roots、joints、FK、全 B COM/world twist、energy 双 bank；`:341–420` 已有 full mechanics / solve / candidate / retain。 | 复用这些阶段；不另调 HOST participant.prepare/integrate，不造第二 runtime 或解算器。 |
| `_resident_contact_plan.py:252` 的速度核只支持 rootless joint tree。 | 浮根直接读取指定 bank 的真实 `twist[B,6]`，不以固定根速度核猜 root twist，也不重新 HOST FK。 |
| `_resident_contact_bridge.py:384–416` 已按共同作用点取双方速度、调用原 patch law、逐侧合 COM wrench；`:439–499` 已运输入口材料点和共同作用点并分别算功。 | 几何、权重一次、两类功和 owned 记录原样复用；不另造“浮根接触记录”。 |
| `_resident_collision_plan.py:465–518` 已有 floating current-bound、同树/父图/过滤域；`tyx_world_batch.py:527–548` 已有所有检查先于一次 bank/epoch 发布。 | 复用 domain/source proof 和共同事务，保 same-body/mask/exclusion 先过滤、未滤 direct parent-child 拒绝。 |

HOST 对照的真实接口是 `FloatingRobotPointPort`，不是 `FloatingWorldPointPort`。`coupling/floating_port.py:82–92` 对全部点合 `Jᵀ impulse` 后只解一次 full M；`dynamics/floating_robot.py:145` 的 root/world 六列与 joint 列可作坐标语义核对。`runtime/_floating_robot_participant.py:340,411,458` 已显示完整 M/b/J、COM wrench 和一次浮根推进，但不直接搬进设备执行。其 PD 采样时序不自动等同 resident 的接触后 full solve；首物理锚点先用 kp=kd=0，非零 PD 使用 C9 已有独立 box-KKT reference。

## 3. 复用公开入口与一次发布时序

公开入口建议不增第二套 plan：

```python
plan = batch.prepare_contacts(
    collision_model, bindings, materials_by_pair,
    sampling="discrete_entry", tree_dynamic_pairs=False,
    candidate_capacity=4096, site_capacity=128,
)
# bindings 可含已经存在的 ResidentFloatingTreeBinding / .from_asset。
# same-tree 双动态显式 tree_dynamic_pairs=True。
batch.step_contacts(plan, h)
```

共享接缝只使用三个独立数量：**B=physical body count、D=full generalized velocity count、N=actuator count**。现 ordinary fixed 为 `(n,n,n)`，fixed affine 为 `(n_full,n_full,k)`，ordinary floating 为 `(n+1,n+6,n)`。每个 world 的物理输入统一为 `wrench[B,6]` 与 `Jv/Jw[B,3,D]`，输出为 `g_contact[D]`；控制输入另为 `[N,3]`。批分配分别使用 Bmax/Dmax/Nmax，不能共用 nmax。affine 原 Pᵀ 归约和浮根原 motor 选择仍由各自现有 solve 完成。

这是原 contact plan 的有界数量/packet 接缝，不新造机器人类别框架，也不为每类复制采样、法则、事务或 receipt。浮根专用适配仅负责已有 bank 布局及原阶段调用。

`ResidentContactReceipt` 沿原字段，浮根尺寸明列：`input_external/contact_generalized` 为 D；`selected_drive` 为 n×3；`actual_drive` 为 n；两个 wrench 为 B×6；状态/几何为原 `FloatingTreeCheckpoint/FloatingTreeKinematics`。diagnostics 建议 `coordinate_kind='floating'`、`joint_count=n`、`body_count=B`、`velocity_count=D`、`position_count=Q`，并保存同次候选的原 `FloatingTreeDiagnostic`；generic full equation residual 必须与该读数一致。不得用 swap 后另一份 active 状态补候选诊断。

1. 原锁内固定 `(owner, topology, epoch, src bank, controls)`；query 直接取该 accepted bank 的当前保守包围。材料覆盖及 pair 过滤沿原规则；不额外做一个完整 step 猜接触。
2. 原 `stage_mechanics(src)` 只做一次，得到 D×D M、D bias、B×3×D J、原外载及 drive 延迟选项。读同源 entry FK、COM、twist，保 authored origin、真实 COM、principal inertia frame 的不同语义。
3. HOST 原 bridge 用 `v_COM + omega × (p_load−COM)` 采样。两个正反力都写各自 B×6 行；同树对不得只留一侧，静态反力继续按原静态 reference point 记录。
4. 上传 B×6 FP32 wrench，设备只新增 `g_c = Σ_body(JvᵀF + Jwᵀτ_COM)` 的 D 列投影；加到现有 external 向量一次。保持外部 joint effort、COM force/torque、重力、实际 motor 各一次。用原 full solve/PD/drift；不把 root 与 joints 分开积分。投影算术沿 C9 最终接受的 floating-only 策略，不重新引入旧中间乘积下溢拒绝，也不改旧 fixed/affine 算术。
5. 原 candidate FK/twist/energy 与 finalize 保留机制；读 dst，`finish_contact_loads` 运输入口锚点、构造所有 public owned records，再作状态/epoch/命令复核。全部可能分配、读回、构造失败都须在唯一 swap 前；最后只做原 assignment。材料功、作用点功、入步功率×h、动能变化分别报告，不强造能量相等或新增严格守恒门。

浮根 `SITE_CAPACITY_EXCEEDED` 可沿原 floating/ordinary 状态码 7，须经过原 root/joint/FK/twist/energy/tick/queue 全保留；affine 的 7/8/9 不改。容量数完整几何 sites，含零力项。未绑定 floating worlds 仍原 `stage_advance`；混合 fixed / affine / scalar 接触与非接触 world 不可误访问 `.physical`。

## 4. 候选案例库：先固定输入，再写有效 RED

公共基准沿 C9 `h=1/1024`、state same-entry `atol=5e-6, rtol=5e-5`、40 步累计 `atol=2e-4`、full equation `2e-5` 和原 PD 门。法则默认 `(k,d,mu)=(64,0,0)`，所有长度 m、质量 kg、惯量 kg·m²。未另述时零重力/外载/速度、单位姿态、kp=kd=0；原旧案例不修改。独立 reference 复用 `tests/reference_newton/robot/resident_floating_reference.py:57–145` 的 time jets / full inertia sum，不从产品 M/J/solve 生成期望。

| 例 | 固定输入与首步独立预期；后续范围 |
|---|---|
| F01 root-only 静态 | n0/B1/D6；root m2、I=(1/2,1/2,1/2)、COM0，球 r1/8、中心 z3/32；有限箱 `x,y∈[-1,1],z∈[-1/4,0]`。压缩1/32 → Fz=2、a_z=1、τ=0、静态反力−2z。40 步用独立球–有限顶面 gap 与半隐式递推。n0 drive 必须空，root 不得被 padding 当 child。 |
| F02 单 P 完整耦合 | 原 P03 root2+child1，COM/关节原点重合、Px、惯量原值，只有 child 球 r1/8；静态壁 `x∈[-1/2,-3/32],y,z∈[-1,1]`，Fchild=+2x。x/q 子矩阵 `[[3,1],[1,1]]`：无 motor 时 `(a_root,qdd)=(0,2)`；原 +3 N motor 留出为 `(-3/2,13/2)`。另将接触绑定 root：`g=(2,0)` → `(1,-1)`、child 世界加速度0。每种力不能重复计入外载。 |
| F03 同树双 P | root m2；两个 Px child m2/m1、COM0、I 各1/4、offset x=−7/64,+7/64，球 r1/8。两 child 为非相邻兄弟，真实 F=(−2,+2)x，`g_root6=0,g_joint=(-2,+2)`；x/q full block `[[5,2,1],[2,2,0],[1,0,1]]`，`a=(0,-1,2)`。40 步双方均动，完整线动量/同一共同点合矩核对；反序注册、轴循环、共同平移和共同速度留出。不得将对方静止。 |
| F04 旋转工具与入步速度 | 原 P04 root2/I1/2、child1/I1/4、Rz、两 COM 原点重合；child 球局部 `(0,1/4,0)`，F02 静态壁；root ωz=1/16、joint v=−1/32、motor=1，d=4。入步点 vx=−1/128 → N=65/32；根 COM a_x=65/96、根 αz=−2、joint qdd=127/32。40 步独立实际圆弧/有限面距离与 full time-jet 参照，记录材料/作用点功差，不用末态速度重采力。 |
| F05 COM 与主轴两例 | (a) 原 P02_origin 的 COM=(1/8,0,0)、I=(1/4,3/8,1/2)，原点球受有限壁 +2y：τ_COM,z=−1/4，a_y=1、αz=−1/2。(b) 原 P02_frame 的 I=(1/4,1/2,1/2)、principal frame 绕z45°；同 root 两球局部 z=−1/4,+1/4，各有限壁施 +2y/−2y，净F0、净τx=1 → α=(3,1,0)。显式排除未声明形状对；不能把多接触平均成零载荷。 |
| F06 authored fixed 工具 | 原 `floating_fixed_composite.urdf` 真 loader+from_asset：n0、M3、COMx1/12、I=(3/10,53/120,13/24)，tool_box 中心x1/4、边长1/8。有限壁 `x∈[1/8,3/8],y∈[-1/4,-1/64],z∈[-1/8,1/8]`，工具面压缩3/64，总F=+3y、τ_COM,z=1/2 → a_y=1、αz=12/13。无另加外力；质量/工具坐标只折叠一次。当前有限 H 斑块及材料点全保存。 |
| F07 有限 S/C/H 留出 | 新增 root m2，两个 child 的质量2/1及轴、offset、local shape 全部取 C8 D06 同变体（hull/cap_side/cap_end/cap_miss/CC/非盒 HH），保其原相对几何。首 normal 依次为 `[2,2,2,0,16/25,16/25] N`，正载完整 patch 权重和为1；保真端点、线/面积归一及非盒质心。先各2步几何/响应门，不将这组扩称任意旋转 H 全轨迹独立真值。 |
| F08 分支与末 body | C9 P05 原 n=3/9/16 非轴 R/P、偏心 COM、principal frame、非零 q/v/外载全部保留；末 body 绑定球 r1/8。静态有限壁以独立声明变换得到的初始球心 c 为中心，法向+x、面x=c_x−3/32，半宽1；首载2N，不从产品输出反推外载。n16 必验 body16/B17/D22 完整反应，并加 body15 留出。n3 另用 C8 D02 原 Rz→Px/Py 兄弟几何：两±2x，root6及共享R广义接触载荷为0、Px为−2、Py为0，真实 root/各分支响应由独立 full reference 判定。 |
| F09 卸载重入 | F01 改初始 z=1/8−1/4096、vz=1/8、g_z=−8，原 k64、h 不变；初始 N=1/64，a_z=−1023/128。以 `N=64 max(0,r−z)`、`v+=h(N/2−8)`、`z+=h v_new` 独立递推40步，必须出现受载→分离零力→重入，不用 reset/时间表切载荷，也不预填切换 tick。 |
| F10 多世界与事务 | W1/W4/W32，非连续 world keys，n=0/1/9/16 混合，每档完整40步与 tick19→21步恢复重放。选 F01/F02/F03/F08 有载输入；另混合旧 ordinary/affine contact、scalar、未绑定 floating。全 state/geometry/输入队列/sites/反力/诊断按 world 比较，复位只影响指定 world。容量0与足额重试、投影上传后/候选读回/receipt 构造晚故障、stale plan、close/rebuild均单独保存。 |

F08 的完整坐标由既有固定声明链/独立解析变换生成，是确定输入构造，不用碰撞输出或期望载荷调整位姿。复杂分支先核独立 same-entry M/b/J 与真实 body wrench 投影，球–有限面窗口用独立距离继续；需要未知几何分支时按原 unsupported 留证，不新增自动容差/步长搜索。

必须相邻负控：非法 body17；group16 对 B16 不合法；旧 fixed/affine body15/count16 有载仍成立而不存在的 body16 仍拒绝；直接父子未排除、跨完整模型、错 topology/owner/world、伪 source、未知 law/非零 mu、布尔伪值及非有限载荷。只跑确受维数接缝影响的少量旧节点，不重跑 C6–C9 全矩阵。

## 5. 原子性、资源与验收边界

- capacity 为 world-local 数值拒步，原 finalize 将该 world 候选 bank 的 root/joint/applied/FK/twist/energy/tick/drive history 从 src 完整保留，其他世界按原合同完成；不要求未发布 scratch 字节不变。设备/读回/构造异常则本次全局不 swap，不发布新 query/receipt；同一输入可重试且完整重写所需候选，恢复失败时原规则要求从完整 checkpoint 重建。
- 源绑定含模型原 FP64 token、实际 B/D、root inertia frame、shape/local points、group、world、topology/epoch；entry/candidate 同来源。大 anchor 的 `local_origin_m` 原样保存，不能以舍入 world Q 反算；最初几何构造的一致性检查不替代实际 COM。有限几何还须在 FP64 world points 中可表示；C9 的大 anchor 可逆检查点不自动证明该处微小表面可用于接触。
- plan 分配中途失败关闭已分配资源；batch.close/plan.close 均幂等。真实 owned bytes 归零、分配次数保累计；报告真实 projection launch/上传/读回/同步，HOST public record 构造仍在 HOST。W32 能运行不等于获得吞吐收益，本片不预设速度比或另建性能门。
- 原全 M SPD、方程、PD、domain、finite、候选审计不变；接受 C9 的浮根算术后才组合。共同点双侧 wrench 的瞬时合力/合矩与全状态的有限步动量变化分别检查；材料功、作用点功与离散能量残差各记，不把功率或静态相消冒充精确能量守恒。

## 6. 文件 owner 与实施顺序建议

父仍唯一 OLD owner：`_resident_contact_plan.py`（显式种类/维数分派、同次 receipt）、`_resident_contact_bridge.py`（17-body 有界校验）、`_resident_floating_storage.py`（纯候选 packet/diagnostic 读取复用）。`tyx_world_batch.py` 原提交逻辑原则上不改；`_resident_collision_plan.py` 既有浮根 current-bound/父图只接线，若需修改必须先列具体缺口。根导出无需新求解 API。

候选 NEW 产品仅 `backends/_resident_floating_contact.py`（指定 bank 的原存储适配）与 `backends/_tyx_resident_floating_contact.py`（B×D wrench 投影，不含 solve/integrator）。NEW `tests/scale/test_resident_floating_contacts.py`、`test_resident_floating_contact_transactions.py` 和一份报告；独立 C9 reference 直接复用。具体 allowlist/API 由父下一 grant 冻结。

1. **T0 有效 RED / 静态首片**：先在原 HOST patch 验 F01 的2N和完整浮根质量前提，再由公开 prepare_contacts 触当前拒绝；随后实现根0/child/末body16、原完整求解与 packet。F01/F02/F04/F05/F06/F08 的代表短步通过后才开窗口。
2. **T1 同树双侧与生命周期**：F03/F07、分支双端、卸载重入及真实双方速度；默认 False 与父图过滤负控。完整反力/反矩进同一 full M，不增加跨 participant 联立系统。
3. **T2 实际 interpret/CUDA、混合事务与安装**：冻结短矩阵和40步保存集合，W1/W4/W32重放、晚故障、少量相邻旧回归；源码外 public consumer 可复用同输入。只在具体失败时诊断那一步，不重复旧长前缀或同源已绿套。

若 finite patch 本身未实现，保失败归几何后继；若 C9 full state/恢复尚未接受，C10只保持 proposal；若需要改摩擦律/跨独立树联立/CCD才可通过，则拆后继，不调原输入和门。floating affine 依赖 full-to-reduced 接触投影与原 mimic 反力；跨独立 participants 依赖新共同系统/事务域；持续切向与 CCD 分别依赖其材料历史和连续路径，均不塞入本片。

## 7. 本机 Newton 1.7 对应与不可偷换部分

依据本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA:3` 的版本，读取同安装目录源码，没有运行 Newton，也未用 main 分支推定本版行为：

- `newton/_src/solvers/featherstone/solver_featherstone.py:61–99` 使用 reduced-coordinate CRBA，浮根须显式 FREE joint；`:592–605` 将外载及 FREE wrench 放入 body force；`:717–746,767` 将 contact body wrench 送入同 articulation 的动力学。可作完整浮根而非逐刚体推进的架构参照。
- 同文件 `:108–123` 说明 FREE root 的 `body_parent_f` 是递推力平衡诊断，不是真正固定根约束反力。C10同样不能把 root 接触响应当静止支座反力。
- `newton/_src/solvers/semi_implicit/kernels_contact.py:461–558` 以两侧 body-local contact points、真实 COM 速度/力矩计算 penalty force；本版 damping 只取 closing 部分。Phymeo 的 `solvers/sampled_box_contact.py:33–45` 为原双向速度 spring/dashpot 后 clamp，`contact_patch_loads.py:33–40` 为 whole-patch 权重一次。因此多点数量、阻尼、作用点口径不自动等价。
- `newton/_src/sim/contacts.py:273–306,443–453` 有原 contact points/normal 与可选 force 通道，但是否填充依 solver；上述 Featherstone 调用的直接输出是 body_f。后续原生参照应锁定真实读取来源/时态，不能把未填 buffer 当零力或从位移反演成原内部力。

最小原生参照建议只选 F01 单球/有限面与 F03 同 FREE articulation 双 P 的 mu=0、d=0独立输入；先核真实接触数/合力/质量与自由坐标约定，再比状态。多点 H 或 damping 不匹配时并列实际物理量，不调整材料拟合、不报跨框架速度或模型等价。
