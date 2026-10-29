# P6.5 可复用两阶段动态摩擦设计

日期2026-10-29。范围为H1的下一有限生产切片；用户已授权主agent冻结共性合同、给实现子agent明确测例/文件范围并独立验收。主agent技术裁定采用下述分阶段定律，不再逐项等待许可。本文已冻结；唯一builder dynamic_friction_design_check只读返回nothing important left，未运行数值或改文件。生产尚未实现。

## 目的、证据与方案选择

在已交付DynamicRobotNormalContactProgram的真实held重放和terminal末冲量基础上，让同一可复用程序自动解决静摩擦支撑、低μ滑动和从分离到接触的摩擦冲击，保留材料和机器人反作用、原式账本与全局事务。案例只能配置模型/输入/碰撞面，不能预先指定p/t/eta来冒充自动求解。

[候选推导](../../development/p65-dynamic-friction-candidate.md)含纯Fraction方程；[原生脚本](../../../scripts/probe_dynamic_friction_native.py)通过现有公开API在NumPy/interpret/CUDA各核对6例共18例（预设向量，不是自动摩擦）。以下三条路线已区分：

1. terminal-only：即使最终速度归零，高μ支撑每步仍漂移h²；不能支撑持续静摩擦承重，舍弃。
2. 分阶段A（本设计）：初态闭合行在held阶段使用μp，terminal使用尚未在held消费的法向容量；阶段切向功非正。静支撑和材料斜撞具有独立解析解。
3. 总末律B：总切向冲量按最终滑速决定。非平行面例能给出正held切向功而整体能量下降；它是另一离散模型，不与A混称，也不以总能量下降替代阶段语义。

选A是主agent在持续授权内的离散模型选择，不声称用户逐项选择了A。多面例中A的总切向冲量点乘最终滑速可为17/50>0；因此本产品只声明 **dynamic-robot-friction-split-v1**，不声明统一末态Coulomb或整个步的单次最大耗散。

## 支持域与公共API

```python
DynamicRobotFrictionContactProgram(
    scene, base_program, colliders, *,
    friction_by_collider, materials, budgets,
    robot_domain="robot", registry=None, inner_settings=None,
    relaxation=1.0, history_prefix="friction_contact",
)
```

- colliders为现有RobotPlaneCollider序列；旧类型的friction=0约束保持，几何对象只用于已有法向几何。**新的摩擦参数只由friction_by_collider绑定**，不删除/放宽旧类型μ0门禁。
- friction_by_collider必须Mapping，键集合恰等于collider keys，无缺项/多项。值严格builtin float、有限非负，可为0.0；owned排序副本，必须进入problem_signature。明确同一系数用于静/动摩擦，无速度相关μ。
- 所有collider的radius_m必须严格0.0，包括μ0行。这样当前仅有节点平移自由度时不会悄悄遗漏厚度切向力偶；非零厚度待后续真正端口集成，不通过加宽容差放行。
- all-prismatic机器人（含旁支），固定朝向无限平面，恢复系数0；支持任意已允许的朝向与碰撞侧，local_normal方向给出允许侧。全动态FP64距离材料域或None；原native tracing和MaterialPathLedger要求保持。可有非选旁域，由base正常推进。
- materials/budgets/registry/inner_settings/relaxation行为沿用已冻结法向合同：内层一子步，尺度和tol与外层一致，显式有限正MaterialContactBudgets；NUM-005仍为预算约束而非修复。
- capabilities只声明features=frozenset({"dynamic_robot_friction_split"})，保留实际backend/families/FP64存储与explicit history。不继承base其它未支持能力，不静默回退。NumPy FP64与实际tyx FP32计算/FP64提升，所有selected pre/post原生存储均验。
- 原DynamicRobotNormalContactProgram、InitialRobotMaterialContactProgram、FreeParticleLedger和原μ0 collider完整行为保持，不把旧能力静默替换为新律。
- 新程序拥有自己的history前缀。cold initial_history()仅返回自身字段，调用者与真实base history合并。签名覆盖定律/μ/材料/预算/几何/history，禁止隐藏全局子步计数。

## 时序、几何分类与载荷账

记节点和机器人真实native状态为z0、held后z-、terminal后z+。机器人M0/b0/J0冻结；all-prismatic保证整个步J和面朝向恒定。完整实际几何仍需独立核对，不以常量假设跳过精度门禁。

行生成：全部所选节点×配置平面，键(domain,node_index,collider.key)，域/面按稳定排序，重复或相交面合法、共享响应不能拆成互不影响的单对。

使用旧exact affine geometry/norm bounds，Lgap=min(reference_length,h*reference_speed)，b=tol*Lgap。初态或末态间距区间：lo>=-b且hi<=b为closed；lo>b为open；超深穿透/横跨分类阈值的不确定区间拒绝，不能靠history指定接触或epsilon挪开。初态closed判据由z0几何得出，不等于previous active。末态closed集由选定held候选z-确定，terminal不改配置。

每行阶段冲量SI为N s，p>=0、eta>=0：

- held world P = n*p + t（t与n正交）。在同一原状态/输入的native完整重放中，节点sum(P)/h与机器人sum(-J0.T@P)/h各输入一次。原非接触输入保持独立，不能把反作用再施一次。
- terminal world E = n*eta + tau。固定选中held输入，从同一原状态重放并在live native velocity已收敛、close之前由TerminalImpulseProgram施给定E一次；不是额外时间推进。机器人末作用点采用z-下节点在面上的当前投影；核对其J仍等于J0。
- 第二次pre的完整state/basehistory/material_path/diagnostics/sweeps/metrics必须与选中held一致；MaterialPathLedger和FrozenRobotLedger只验pre，TerminalImpulseLedger只验真实速度跳跃。不能把post塞回旧速度重建方程。
- p+eta及P+E只作总冲量诊断，不塞回held输入；不0dt、不close后调用、不host代算覆盖candidate，不用最终J1配不一致的节点反冲量。
- 每次outer子步都执行上述两阶段。所有尝试均计重放/真实launch；terminal trial执行统计已含pre，不能再重复相加。后段任何真实native/trace/terminal/close失败拒绝全帧，即使之前有best candidate也不能吞掉。

## 分阶段法向与摩擦原式

令u- = v_node- - J0*v_robot-，u+类似，u_t为单位n正交投影。

### Held阶段

共同求(p,t)满足：g1>=0，p>=0，p*g1=0。

- 初态open：t=0，严格不消费摩擦容量。
- 初态closed：C_h=μp；||t||<=C_h；t·u_t- + C_h||u_t-||=0。
- μ=0或p=0容量零，t=0；滑动时t反向且到圆盘边界，静止时任意盘内平衡量。
- 切向条件必须用真实held速度及同域材料响应，不能用最终post速度或自由速度代替。每次t变化须完整重放，不能只更改诊断。

### Terminal阶段

固定held(p,t,z-)及末closed集合，共同求(eta,tau)：末closed行eta>=0、u_n+>=0、eta*u_n+=0。

- 初态closed行：C_v=μ*eta（held已经消费μp，不能再给一次）。
- 初态open且末closed行：C_v=μ*(p+eta)（held切向为0）。
- ||tau||<=C_v；tau·u_t+ + C_v||u_t+||=0。
- 末open行eta=tau=0。初态closed但末open同样不保留任何阶段接触量；held互补已要求p=0。
- 法向与切向含非平行面/共享祖先交叉响应，不能先解normal后一次性clip摩擦就接受。容量随eta改变，盘内投影和native重算必须共同收敛。

求解器可以采用确定行序的投影GS：自由质量块W_i=w_i I+J_i M0^-1 J_i.T，法向k_i=n.T W_i n，切向正标量预条件kappa_i=trace(W_i)（只决定步幅，不是材料响应证明）。held p按g/(h*k_i)，t按u_t-/kappa_i更新并投影到μp圆盘；terminal eta按u_n+/k_i，tau按u_t+/kappa_i投影到对应容量。每项变更完整原生重放，并通过下面独立原式接受。允许等价正步幅实现，但不改定律/阈值，不用有限差分替代真实native响应。无法收敛显式拒绝，不能提高tol/预算或host修位置。

保留原outer迭代预算内的best合法trial细化，以法向+切向+间距共同score单调改进；只保存完整trial/全部载荷/审计，不只保存数值摘要。末阶段审计held摩擦永远使用选中z-，不能因调用通用_position_metrics而误用z+。达到预算可接受已完全合格的best；未知/失败不算安全。

审阅后明确候选选择语义：完整native计算返回有限、已定义的物理残差超门槛，表示该试算候选不合格，可丢弃并恢复此前完整合法best；恢复后仍须重新通过全部法则/物理/几何门槛。接受的是旧合法候选，不是接受越限状态。native/trace/terminal/close/ledger执行异常或非有限/未知计算结果仍拒绝整帧，不能回选掩盖；最终选定best本身物理越限也拒绝并回滚。成本是有限越限的后续候选可能结束进一步细化，不改变任何阈值。

## 独立精确接受门槛

继承旧法向版本所有input-once、robot/material原式、native收敛、末gap、FP32提升、world映射与成对作用点力矩门槛。新阶段不能只以投影迭代停机误差当摩擦原式证明。

对实际存储world冲量s和相对速度u，以同一存储normal n的精确有理点积计算：N=n·n>0，a=(n·s)/sqrt(N)，b=(n·u)/sqrt(N)，T²=s·s-(n·s)²/N，U²=u·u-(n·u)²/N，D=s·u-(n·s)(n·u)/N。T²/U²/D可直接精确计算；sqrt包围只用于非负范数/normal分量。以实际world而不是未施加的理想t/p核对；旋转normal误差还须旧geometry方向预算。

- 无负normal，圆盘界T<=C；maximum-dissipation residual D+C*sqrt(U²)=0。held C由实际P法向，terminal C由实际E法向或P+E法向按初态分类形成，不能由已舍入展示数替代。
- 归一化cone残差kappa*max(T-C,0)/Vref；耗散原式残差kappa*abs(D+C*sqrt(U²))/Vref²；切向/normal正交及实际world构造误差以对应节点/广义速度尺度审计。sqrt上下界的最坏端点向外取整，原settings.tolerance验收；C=0时切向量由cone界约束。不得粗暴把T²负数clip到0来掩盖实际输入，理论精确平方应非负。
- normal互补沿用单位一致的min(k_i*a,g/h)/Vref和末min(k_i*a,u_n)/Vref及独立负gap/负u_n门禁。初态open声明的held tangent必须严格零，实际构造舍入仍由上述精确映射预算限制。
- 分阶段切向功与盘容量均须报告，正切向功超原式预算拒绝；整体能量下降不免除此项。总冲量末态Coulomb只可作为对照诊断，不作为本law接受/拒绝条件。
- MaterialContactBalance以原非接触输入端点功、最终节点K/U/重力与冻结M0机器人K验显式能量预算；材料角动量按域L1、x0×held和x1×terminal分开。terminal ΔK用独立聚合质量响应及真实pre速度核对（含全部法切向量和共享点），不以观测dv自证。成对作用点的完整world力矩审计继续，不能只验normal。

## history、诊断与事务

自身history：prefix/active bool(R,), prefix/initial_closed bool(R,), prefix/phase_world float64(R,2,3)，其中phase0为P、phase1为E。initial_closed记录上一已提交步的几何分类，仅用于解释历史容量，不用来决定下一步。

有限值/形状/类型严格校验；inactive行两个world和initial_closed必须严格0/false；active行normal非负、各自盘界按保存分类/当前绑定μ可验，使用settings速度尺度，不允许拿畸形历史作warmstart。当前版本所有试算从零载荷开始，不消费历史数值作为求解初值。active历史若当前几何已经明确open拒绝；全零cold history可从当前closed几何建立。初态分类从当前几何重新计算。

每个成功子步仅提交一次base history和自身history；失败包括第2/最后子步trace篡改、terminal操作、ledger拒绝、物理预算或close异常须还原整帧state/input/history，retry=fresh。exit后不留下拉力或摩擦；重建接触不能借用此前切向量。

诊断key dynamic_robot_friction_contact，至少提供：law、row_keys、initial_closed、active、events(enter/persist/exit/inactive)、held_normal/held_tangent、terminal_normal/terminal_tangent、held_world/terminal_world/total_world、两阶段capacities_ns、pre/post相对滑速与原式残差、held_replays/terminal_replays、完整physical_budgets、material_reports；所有数组/映射owned只读。tangent与normal的展示数不替代精确认证。顶层execution累计所有原生尝试，记录实际backend/compute_dtype。

## 独立测例与验收范围

所有值SI。缺API/实际错误先RED，实施后GREEN，禁止把实现计算自己的数值作oracle。以下属于同一实施切片，最终只一次fresh code review。

1. **静支撑与滑动（无材料探头）**：h=1/4,m=1,robotM=2仅Y，初态共点静止，node重力-Y及原force+X=1，robot原effort+Y=1。μ2得到p=1/4,t=-X/4,eta=tau=0，位置/速度全零。μ1/4得到t=-X/16,vx=3/16,x=3/64。对照terminal-only最终vx0但x=1/16；生产高μ必须拒绝这种伪静止，不能只断言速度。
2. **未固定材料链新建立**：A(0,1/8,0),B(0,9/8,0),vA=vB=(1,-1,0),m1/1，robotM2Y静止，无外力/重力，h1/4,rest1,c1/16,μ1/2。自动pA3/7,tA=0，etaA1/3,tauA=-8X/21；A1(1/4,-3/56),B1(1/4,51/56)，vA+(13/21,-8/21),vB+(1,-6/7),robotq=-3/56,V=-8/21；K+U1135/882,Py=-2。B行open无载；除该例平移基座外，不声称约束X总世界动量守恒。真材料另一端变化须匹配。
3. **非平行两面区别定律**：按候选文档n1Y/μ1=.5/g10=0，n2=(-3/5,4/5)/μ2=0/g20=28/5，mass全1,h1,nodev0=(21/10,-9/5),robotV=(1,6)。自动A须p1=p2=1,t1=-.5X，pre nodev=(1,0),robot(0,5)，eta=(0,14/5),tau=0，postnode(-17/25,56/25),robot2=11/5。总t·最终u=17/50>0不拒绝，terminalΔK=-196/25。注入B的t=201/550,p1=241/275,p2=72/55，虽最终floorvt0且总能量降，held正功4221/6875必须拒绝。
4. **连续窗和步长**：NumPy静/低μ两种，T2,h1/16和1/32；静例x=v=0，滑例v(T)=3T/4,x(T)=3(T²+Th)/8，独立Fraction递推和连续x=3T²/8，误差随h减半。actual interpret/CUDA各T1/2、两h同参照，以检验真实多步；不混称设备T2。这是该离散模型厘米级粗步误差刻画，非最终递布精度。
5. **生命周期**：自由接近(initialopen)→enter→加载persist→卸载/拉离exit→inactive→再碰撞；两阶段各t/eta/容量按初态几何独立核对；分离严格0接触history，冷启动closed与旧active=false仍可静摩擦。无隐式弹簧历史力。至少2outer子步、basehistory真实演化每次一次；改变outer子步个数不得依赖隐藏计数。
6. **耦合/几何/退化**：两个粒子共享robot祖先、同节点两非平行面、材料三角（无需解析全轨迹但独立合冲量/能量/原式预算）、三维不共轴滑动、可表示刚体变换和节点/域/面顺序。正常与故障控制成对；非单位normal正常归一，零/非有限拒绝；大坐标/nextafter/不确定初态或末态、非零radius、revolute旁支、fixed节点、材料trace缺失、非法μ映射/可变输入/重复键/空节点域。空域正常且零接触，不创建伪候选。
7. **数值与事务负控制**：实际world方向/切向误差、只审latent投影、用post验held摩擦、重复容量μp、错误末作用点x0、原外功混入有效输入、跨域角动量相消、1FP64 ULP伪FP32、已有best后trace/terminal/close失败、半构造cleanup；明确注入触发和正常对照，恢复后retry=fresh。
8. **旧法保持**：旧normal/initial/free/terminal/material/SceneLoadReplay相关集不改期望；新程序μ全0在安全支持例与旧normal的最终状态在既有tol内一致（不宣称所有未决边界或迭代数bitwise相同）。实际tyx两解析例+两面/生命周期/短连续窗必须检查模式、launch和两阶段输入，不能只跑host数学。CPP仅一次有界短解析，TYX003长期仍未解决。

## 文件边界与交付

生产新增src/phymeo/solvers/dynamic_robot_friction_contact.py和至多两个私有模块_dynamic_friction_law.py/_dynamic_friction_workspace.py；允许修改__init__.py及scripts/smoke_installed.py接公共API。新tests/test_dynamic_robot_friction_{contact,physics,transaction,tyx}.py。可以复用旧geometry、ledgers和公开native API；不改旧solver/terminal/participant/kernel或现有测试期望，不为消除复制做额外重构。发现真正阻塞先给主agent原式证据再调整范围。

本阶段一个实现task交出可拒绝的完整新程序/测试，主agent另负责基准和工程。实施先精确小场景，再真实backend，再一次fresh review；不在每个内部helper重复review。最终相关/全量证据、源码外wheel和锁/CI按变化实际验证，只有新的产品修改才重验必要旧套件。性能正式记录以两小case各150、三后端串行等所有数值/review退出后执行；记录真正所有尝试与scope，不作不同精度加速比，未测Warp不称差距，不优化。

遗留/性能/tyx问题分目录，NUM005/TYX003/TYX004保持。阶段完成只关闭本摩擦无限平面能力；之后仍须有限特征切换/FK保守CCD、弯曲自碰撞、有限单爪承重、双臂与Three.js。整个P6.5标准不变。
