# 接触模型架构：检测、接触律与共同动力学

日期：2026-11-17。源码核验基准：`codex/exp` 的 `17a1cf4e9a5b1040bddfa3f09c137b8cbda360ee`。

本文与[离线架构图](assets/contact-model-architecture-20261117.html)描述**目标架构和分步接入路线**，不是现有功能全景。本文仅新增说明，不修改求解器、不运行实验；父已完成桌面浏览器渲染核对：37个SVG文字无出框，零外部资源，页面无横向溢出。已有验收结论沿用对应报告，不视作本次重跑。

## 1. 先把三个问题分开

- **检测**回答：哪些实体在什么位置、什么时刻接触；有限面在哪里；是否有经过证明的扫掠事件。宽相候选不是接触，也不是 CCD 结论。
- **接触模型**回答：该接触域上采用什么压力、切向牵引、耗散与材料耦合；需要真实面积、局部切线和势能，还是原生接触流形。几何曲面的 Fourier 表达不提供材料响应。
- **动力学**回答：这些载荷与关节、惯性、阻尼和外载一起，如何改变双方的同一组广义速度与构型。不能分别推进两个接触体后再拼接状态。

Kamino 是刚体接触动力学求解器；本地实验固定 Newton **1.7.0**，不能拿网页新版本接口替代。FFT 半空间接触是非局部材料/接触参考：表面一点的载荷影响其他点的位移。二者不处在同一替换层。面积压力积分也不因使用了三角片就自动成为任意材料的 hydroelastic 模型。

## 2. 源码现状：已有、HOST 实验、拟议

下表路径相对仓库根目录，行号对应上述基准提交。`src/phymeo` 下的 I0/I1 文件虽然是公共类型，当前资格仍限于 HOST 几何/代数，不表示已经接入运行时。

| 状态 | 实际入口与行号 | 可复用内容与边界 |
| --- | --- | --- |
| 已有：状态所有权 | `src/phymeo/backends/_resident_collision_plan.py:38`，`AcceptedCollisionState` | 在锁内借用 accepted bank、world/tick 与绑定；不是把各种后端自动合成一个通用设备状态。 |
| 已有：当前姿态宽相 | 同文件 `:95`，`ResidentCollisionResult` | 源码明确只报告当前姿态 broad candidates，不声称 narrow contact 或 CCD。 |
| 已有：有限检测管线 | `src/phymeo/geometry/collision_pipeline.py:80`、`:106`、`:1378` | `CollisionPair`/`CollisionDetection` 保留 witness、sweep 和特征 trace；`detect` 使用注册运动、后端宽相及受支持扫掠入口，容量超限不发布部分结果。它与 resident 接入范围不能混同。 |
| 已有：归一采样载荷 | `src/phymeo/geometry/contact_patch_adapter.py:100`、`:123`；`src/phymeo/solvers/contact_patch_loads.py:10`、`:34` | `ContactPatchSample` 的无量纲 weights 和为 1；全 patch 的载荷只加权一次。不是面积积分。 |
| 已有：resident 有限接触事务 | `src/phymeo/backends/_resident_contact_plan.py:1`、`:84`、`:114`；`src/phymeo/backends/resident_scene.py:1`、`:30` | 入步 HOST narrowphase/patch 采样与 accepted-bank 事务已有基础；`ResidentSceneBatch` 明确不提供跨独立机械树的隐式接触力共同求解。 |
| 已有：MuJoCo-TYX 行装配 | `src/phymeo/backends/mujoco_tyx/_constraint_rows.py:127`、`:154`、`:323` | 同接触点双方 Jacobian 差，原生 condim/limits/dry friction 行，`solref/solimp` 到 aref/R/D；不能把面积 stiffness 直接塞入原参数。 |
| 已有：T4 求解与实际积分 | `src/phymeo/backends/mujoco_tyx/_constraint_solver.py:140`、`:182`；`_constraint_device.py:68`；`_tree_kernels.py:535`、`:573`；`_floating_kernels.py:559` | 接触 KKT 基于 physical M；实际速度更新另用 M+h*Cjoint。新压力行必须处理这一区别。 |
| 已有：入步记录与原子发布 | `src/phymeo/backends/mujoco_tyx/_entry_storage.py:95`、`:114`；`src/phymeo/backends/_mujoco_tyx_runtime.py:914`、`:1121`、`:1135` | entry q/v/act、clock、contact 保存在同索引双 bank；候选成功后才交换 accepted bank，不用 HOST receipt 上传冒充设备来源。 |
| 已有：异构共同提交、独立观察事务 | `src/phymeo/backends/mujoco_heterogeneous.py:594`、`:613`、`:643`；`src/phymeo/backends/mujoco_heterogeneous_io.py:265`、`:334` | 物理 buckets 先 prepare/validate 再 commit；workspace 所有桶刷新完成才发布新观察帧。观察是后续事务，不能把观察失败称为物理回滚。 |
| HOST 实验：I0 面积/代数 | `src/phymeo/geometry/area_pressure.py:155`、`:178`、`:337`；`src/phymeo/solvers/area_pressure_rows.py:169`、`:242` | 已裁三角片、三个二阶精确积分点、真实 dA；共享全 DOF 的 J 与双方广义力/共同原点 wrench。没有动态求解。 |
| HOST 实验：I1 有限裁剪 | `src/phymeo/geometry/area_pressure_intersection.py:107`、`:144`、`:208`、`:294`、`:342`；`src/phymeo/geometry/_area_pressure_clipping.py` | 有界 tetra–finite-triangle 相交；从存储 FP64 值构建精确拓扑/交点，输出仍须通过 FP64 发布门。验证 shared-ID 复形、压力连续性及共面去重；保留 full/local 标记。不是通用碰撞或 CCD。 |
| 独立参考：Kamino / FFT | `experiments/contact_selection/kamino_reference.py:347`、`:384`、`:400`；`experiments/contact_selection/fft_reference.py:421`、`:541` | 固定版本刚体实验与静态半空间数值参考；独立输入、计时及物理域，不是产品中可互换的接触后端。 |
| HOST 实验：有限选型 | `experiments/contact_selection/autoselect.py:22`、`:114`、`:144`、`:211` | 五点资格、完整 registry、按输出组的误差/成本与拒绝原因；不是通用 ContactPlan，也未接每步调度。 |
| 拟议：共同 ContactPlan 与面积动态路径 | 本文第 3–7 节 | 先补 I2 的时间离散/共同响应，再接设备生命周期，最后接 resident；不以文档图替代实现证据。 |

## 3. 场景编译或初始化时选 ContactPlan

这里的 **ContactPlan 是拟议概念**，不是已有公共构造器。已有 `ResidentContactPlan` 负责有限接触事务，不等于下列多模型选型器。

一个计划至少绑定：实体与 pair/机制身份、几何/材料版本、接触模型及适用域、要求输出、误差证据、离散时间与有效动力学算子、设备能力/容量、历史状态布局及计时条件。先检查能力、物理域与精度，再在合格者中比较成本；没有证据就是不准入。初始化选定后复用；绑定变化则失效并重新检查，不在每个接触点每一步重新微基准选型。

成本应覆盖检测、材料计算、J/行装配、共同求解、数据搬运/同步、记录和发布；初始化/离线标定另列。当前实验只有公式与选择接口计时，**没有形成新 ContactPlan 的整步成本资格**。

已有标定只覆盖 R=0.02 m、E*=1e6 Pa、L=0.16 m、静态无摩擦线弹性半空间抛物面，以及 δ={50,75,100,150,200} μm **五个离散值**。五点通过不等于其间连续范围合格，也不支持其他材料、粗糙度、有限厚度、动态或压力场输出。force 标定公式和 Hertz 解析的热计时区间重叠；默认最低 median 选中谁，都不能据此宣布稳定加速。完整选择加实际预测的开销还明显大于公式本身。

新增几何已由计划 `79a15de5` 冻结：[双峰形貌数值参考](../superpowers/plans/2026-11-17-multiasperity-contact-reference.md)。采用两个 C2 紧支撑椭圆峰，非轴对称、高低不同，**不是全域凸形状**；仍是静态无摩擦线弹性半空间，24 mm 窗口不是有限厚实体。固定 256/512 两档的五个压入量，加两个独立峰对照，共最多 12 次新参考求解；数值工作已由父完成，实际结果见[双峰报告](multiasperity-contact-reference-report-20261117.md)：12个参考通过，单峰独立相加总力偏高8.06%，固定力表因40μm误差9.16%而被拒绝。本文的目标架构仍未因此成为完整运行实现。双峰受同一半空间耦合，不能把两个独立 Hertz 力直接相加当准确参考。旧单抛物面解析在此几何明确不适用，但不宣称所有数学方法都不存在解析表达。

本批拟合仅为 **force-only PCHIP**：δ=0 的解析无接触值与 20/60/100 μm FFT512 载荷用于标定，40/80 μm 只用于前瞻资格；参数先保存再读取验证点。即便通过，也只形成五个声明点的有限证据，不授权压力场、CoP、力矩、切线或储能。**总力表缺少完整 wrench 与经过验证的增量响应，不能凭 force 误差小就进入 I2 动力学。** 插值公式可导/可积不等于其导数/积分已是合格物理量；更不能把离散压力图自动解释为局部材料律。

## 4. 目标运行时：一份状态，分型接触，同一次动力学

1. **共享入步状态与所有权**：同一 world 的 accepted q/v/act、物体/关节身份、tick/time、外载和控制。检测、接触载荷和 J 必须绑定同一入步；endpoint 状态另标相位。
2. **宽相 → 有限窄相 / 受支持 CCD**：宽相只筛候选。窄相输出有限特征、接触域与状态；CCD 需要明确的运动与时间区间。没有支持的扫掠证明就显式拒绝或返回尚未证明，不能用端点重叠冒充无穿越。
3. **分型几何候选**：共同身份/时间/来源外壳下，保留不同数据类型，而不是把一切压成一个 gap：原生 manifold 的点/框架/间距；面积曲面的三角片/真实 dA/压力/梯度/来源；未来非局部材料域的网格/边界条件/耦合算子。
4. **模型与离散接触律**：在各自适用域生成法向/切向载荷、切线、势能或耗散信息；明确力 N、力矩 N·m、压力 Pa、面积 m²，以及冲量 N·s。摩擦历史与材料历史属于模型状态，不能藏在可任意复用的几何缓存内。
5. **共同 DOF / J / 质量与时间离散**：双方贡献装入同一组广义坐标与同一次动力学求解，保留双方 J 和反作用；关节、阻尼及其他合法约束共同参与。非局部算子可产生稠密/结构化耦合，不能假设必为独立标量弹簧。
6. **一致发布**：候选状态、实际接触载荷、双方响应及源入步记录一起接受或拒绝。观察帧通过其现有独立事务读取已接受状态；若观察失败，保留旧完整观察，并明确物理是否已提交，不能混搭新状态和旧力。

公共候选宜保留 world、pair/body/shape、source feature/cell/face、时间相位、法向约定及源版本；模型特有字段保持分型。空接触、合法零载荷、unsupported、容量不足和求解不收敛是不同结果。禁止截断接触后继续称完整结果。

**求解分组也要区分两类耦合。** 同一刚体或关节的惯性使不同接触共享机械响应；同一弹性材料域又可能把不同pair的压力与位移耦合起来。ContactPlan按pair记录身份、模型归属和去重，但非局部材料块应按共享材料域组织，再与机械DOF形成接触求解组。两峰试验已说明，几何接触区分开不能证明材料响应可以独立相加。独立world之间仍禁止串扰。

## 5. 两个不能省略的接线推导

### 5.1 归一 weights 不是真实面积

旧 `ContactPatchSample` 要求 Σwᵢ=1，wᵢ 无量纲；`sample_patch_loads` 中 stiffness N/m、damping N·s/m 属于整个 patch，最终力仅乘一次 wᵢ。细分采样不等于增加真实接触面积。

I0 的 dAᵢ 单位是 m²，不归一化。每片三个 degree-2 积分点能保留线性压力的合力和一阶力矩：

- F_A = Σ pᵢ nᵢ dAᵢ，F_B = −F_A。
- τ_A,o = Σ (xᵢ−o)×(pᵢ nᵢ dAᵢ)，双方使用同一参考原点 o。
- n 从刚体 B 指向柔顺体 A；g=∇p_A·n>0；J=nᵀ(J_A−J_B)。
- kᵢ=dAᵢ gᵢ [N/m]，φᵢ=−pᵢ/gᵢ [m]，fᵢ=dAᵢ pᵢ [N]。

这些是 HOST 局部法向代数，不含时间积分。I1 只接纳有限域：g≤0 明确 unsupported；共享面正面积只有两侧 g>0 且完整梯度相同才允许稳定 owner 去重。`pressure_domain='local'` 不能冒充完整压力体。

### 5.2 面积 stiffness 不能直接冒充 solref/solimp

原生行在接触 frame 下使用 J_B−J_A；I0 使用 B→A 法向与 J_A−J_B。接线必须同时转换身份、法向和力方向，不能只翻一个符号。

T4 现有关系为 `M a_phys = τ_smooth + Jᵀ f`，残差 `J a_phys + R f − aref`；R 为行 regularization，D=1/R，f 是力，不是冲量。`_constraint_solver.solve` 本身不接收 h。之后树/浮动积分以 **M+h*Cjoint** 求 `a_int`，并用 `v_next=v+h*a_int` 更新；因此有 joint damping 时，公开物理加速度与实际积分加速度不可混用。

仅在**固定几何/J、线性局部法向律、无耗散、a 确实是实际速度更新加速度**的限定推导中：

```text
v_n,next = v_n,entry + h J a
f = max(0, -k (φ₀ + h v_n,next))
R = 1 / (h² k)
aref = -φ₀/h² - v_n,entry/h
r = J a + R f - aref ≥ 0,   f ≥ 0,   f r = 0
```

这为 I2 提供可检查的离散关系，不是现成的 MuJoCo 参数映射。必须验证实际有效算子和共同响应；不能长期删掉 joint damping 来回避差异。耗散、摩擦、滑移引起的几何/压力变化与非局部耦合须另作一致离散和验收。

## 6. 防双施力、fallback 与切换

- 每个 pair 的同一物理机制只选择一条生产路径。例如同一法向机制不能同时加入 native 法向力和面积压力；某模型内部的多积分点与合法摩擦分量仍可共同存在。其他 pair 的原过滤规则不因此放宽。
- fallback 只能使用同一物理模型、同一输出合同下已验证的实现；不能因容量/不收敛把 hydroelastic 压力静默改成硬点接触，或把非局部半空间改成独立弹簧。无等价实现则保留已接受状态并报明确原因。
- 模型切换至少需要明确 warm start、摩擦/材料历史、接触身份映射和能量账本；不能只替换 force 函数。reset/restore/checkpoint 与异常必须覆盖这些状态，并记录切换版本/原因。没有合法历史迁移规则时不准运行时切换。
- 验收记录分清 entry 几何/载荷与 endpoint 状态，保存实际响应而非只存期望力；共同原点的作用反作用检查不能代替各 body-origin/COM 力矩。保持时序与源状态可追溯。

## 7. 实现顺序与退出条件

1. **先 MuJoCo-TYX 的单条 I2 面积路径**：复用 I0/I1 身份与几何端口；明确 pair 排他选择。先用零 joint damping 的有限正控核离散关系，再核非零 damping 下实际 M+h*Cjoint 与速度响应；包含两个可动体/关节、共享 DOF、接触容量和失败原子性。不得将此阶段描述为任意场景通用求解器。
2. **再接设备和生命周期**：真实 device 几何/载荷/J、有限资源、双 bank entry/state、reset/clone/checkpoint、失败后的完整保留、公共观察与 source phase。HOST 数组上传不能冒充设备现场生成。
3. **最后接 resident**：复用已有状态借用、宽相、过滤与事务；补齐跨参与体共同机械算子与隐式接触响应后才开放。不能因为已有共同 publication 就宣布共同求解已经存在。

每阶段都以独立物理/几何 oracle、作用反作用、实际 mΔv/广义响应、能量/耗散、整步计时与故障证据作准入；边界不通过就保留 unsupported。本文不新增任何计算结果。

依据：[3D 接线裁定](area-pressure-3d-integration-decision-20261116.md)、[有界选型报告](bounded-contact-autoselection-report-20261116.md)，以及上表逐项读取的实际源码。图与文字已由父源码/文本复核并实际渲染；这不是共同动力学实现验收。新双峰数值结果见独立实验报告，不能反推图中所有目标已实现。