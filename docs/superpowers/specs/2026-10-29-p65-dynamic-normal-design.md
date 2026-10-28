# P6.5 自动法向接触建立与分离设计

状态：设计冻结；唯一builder dynamic_normal_design_check只读返回nothing important left，未运行数值；生产尚未实现。用户已授权主 agent 冻结技术合同并分派实现，无需重复请求逐项批准。下面的两阶段方法及有限支持域是主 agent 的技术选择。

## 1. 目标、范围与方案选择

交付能实际产生反作用的可复用生产程序：初态分离的动态节点/距离材料接近机器人承载的无限平面，自动建立接触、持续、分离；案例只配置物理对象和输入。至少自由探头与两节点材料链两个非递布场景独立验收。它是 H1 的有限动态子集，不完成完整 H1 或双臂案例。

选择两个分别认证的阶段：
1. 原生整步 held-force replay 求位置非穿透和单边法向载荷。
2. 同一真实末配置上的原生瞬时速度冲量消除闭合接触的相向速度。

仅偏置位置律会让末态闭合后仍保留撞向速度，不能作为冲击后状态；解析例见下文。连续 TOI 分步需要解决“缩短 h 会改变材料原生解”的一致性，目前没有通用证明。给末态冲量重新乘 h 写回配置会破坏已验收材料路径。故本切片采用明确的离散两阶段方法，不声称事件时刻精确。

支持：
- 一个 fixed-base RobotModel，**整棵树全部 prismatic**，不是只检查接触 link；固定朝向可非单位姿态。有限质量、FP64 存储；NumPy、实际 tyx interpret/CUDA，C++只做一次有界短测。
- 选中的粒子域全部 inverse_mass>0；真实 DistanceConstraints 或 None，无隐式固定节点；距离材料必须开启原生路径记录。
- 配置的粒子域全部节点与配置平面做笛卡尔候选集，不要求调用者提前选择已接触节点。厚度为显式非负节点半径；平面局部法向定义允许侧，不从速度/对象名猜侧。μ严格为0，零恢复。
- 原生 base 必须提供下述“末速度点冲量”能力。现有无接触行的 ArticulatedCouplingProgram / Tyx 对应程序支持；透明 wrapper 只能在真实转发此能力时声明。已有其他位置/速度约束若破坏所选域原式仍拒绝。
- 外层 SceneStepper 可多子步；**每个子步**都完整执行两个阶段。内层 replay 仍恰好一个子步，不需要“全帧最后子步”隐藏计数器。

明确不支持：任何 revolute（含旁支）、有限三角面/边特征切换、任意拓扑更新、摩擦、弹性恢复、物理自碰撞、机器人旋转弧 CCD。不应用仿射线段证书替代 FK 曲线。不得更改旧 initial-closed/free/material 程序的合同。此阶段之后继续摩擦/有限特征/FK 保守路径、弯曲/自碰撞和夹持；不能用本阶段代替它们。

## 2. 公共接口与所有权

新增 immutable RobotPlaneCollider：
- key: 严格内置非空无首尾空格 str；集合内唯一。
- link_index: 非 bool 的非负 Integral，转内置 int，在绑定时核对范围。
- local_point, local_normal: shape(3,) finite FP64，owned readonly；normal 非零，按现有方向工具规范化，不容许“零距离时由速度选法向”。
- radius_m: 严格内置 finite float>=0，默认0.0，表示节点中心到该平面的最小距离，单位m。
- friction: 严格内置 finite float，仅接受0.0，默认0.0；非零明确拒绝，不静默丢弃。

新增 DynamicRobotNormalContactProgram(scene, base_program, colliders, *, materials, budgets, robot_domain="robot", registry=None, inner_settings=None, relaxation=1.0, history_prefix="normal_contact")。
- materials、budgets 沿用 MaterialContactBudgets 的显式 owned 契约；materials 的 keys 即选中节点域。至少一个 collider 和一个 selected domain；空域允许产生零行。bindings 必须 particles family。其他域透传，不冒称审计了未选中域物理。
- rows 按 (domain_name, node_index, collider.key) 排序；输入 collider 顺序不改变求解或 history 含义。所有行均检测，不用 broadphase 漏检冒充优化。
- positive h/tol、scales、inner substeps=1 与 outer tolerance/reference scales 一致，其他迭代预算分开，遵守旧 replay 规则。
- Program signature 绑定 scene、base signature/capabilities、colliders、materials、budgets、全部设置及 history schema。新 feature 名 dynamic_robot_normal_split；不得复用旧 initial law 标识。
- history_fields = base.history_fields + prefix/active bool(N,) + prefix/phase_impulses float64(N,2)。冲量列依次 held_p、terminal_eta，单位N s。prefix 冲突拒绝。
- initial_history() 只返回本程序字段的 owned 零数组，调用者与 base 初始 history 合并。base history 必须原样快照/演化；每次 replay 剥离本程序 history 供 base 使用，最终只合并被接受一次的 base history 和新接触 history。
- 旧接触 history 仅事件来源，不是热启动载荷。分离行 active=False且两冲量严格置0；任何负冲量/非有限/不匹配 shape history 拒绝。初始 active=True 但几何明显已分离拒绝；fresh active=False 可从闭合状态启动。

事件定义：以子步最终闭合集合（下文原容差门槛）和输入 active 判 enter/persist/exit，其他为inactive；不把事件称真实 TOI。返回完整稳定 row key、两阶段冲量、总冲量、初末 gap、相对法速、事件和实际 replay/launch 数。readonly owned diagnostics，不保存 workspace 或隐藏物理 cache。

## 3. 关闭前原生末冲量机制

新增 runtime/terminal_impulse.py：
- PointVelocityImpulses(domain, indices, impulses, local_points=None) 是 owned immutable request；indices Integral vector，impulses finite FP64(N,3)，local_points None 或 FP64(N,3)，单位分别N s和m。验证通过真实 point port，重复 node/link 请求不得漏加。
- TerminalImpulseProgram(base_program, requests, *, registry) 是 **预先给定**末冲量的通用包装；不求接触、不改变时间步或 base 输入。requests owned tuple，同域请求合并成一次 batch（稳定顺序），不得逐行解出不同构型。
- 可选 workspace 协议 TerminalPointImpulseWorkspace.apply_point_velocity_impulses(requests)；ArticulatedWorkspace 实现：仅 prepare_velocity 后可用，按 domain 调真实 participant.apply_velocity_impulses；任何阶段失败/部分域应用失败依赖整个 workspace 丢弃，不发布半个结果。NumPy和tyx都使用原接口，禁止 host 代算后覆盖。
- ArticulatedCouplingProgram 仅在 contacts 为空时声明 terminal_point_impulses feature；wrapper 必须核对 feature 和 callable，不能私查 native.candidate。旧有接触程序行为不变。
- Terminal wrapper 完整转发 base predict/position/prepare/velocity；只有 base velocity_metrics 已符合 settings.tolerance，才在关闭前保存 owned pre_impulse_state、base diagnostics与metrics，应用 batch **一次**，保存 post state。以后 metrics/sweep 不重复施加。非有限/基础失败不能被冲量掩盖。
- diagnostics["terminal_impulse"] 包含 pre_impulse_state、pre_impulse_diagnostics、pre_position_iterations、requests、actual post phase execution delta；base history不被冲量阶段改写。计数同时保留base和末冲量新launch，不能只报告冲量前计数。
- 非零请求不允许在返回 ReplayTrial 后使用已 close 的 participant，不允许新起 dt=0 或第二个时间步，不允许“重建工作区再 predict”作为速度跳跃。
- wrapper.close 不关闭借用 base factory；每trial workspace close 必须执行且错误拒绝，最外层程序仅一次关闭 base factory。构造半失败也清理。

动态程序的速度试算可从同一原始子步再完整 replay held p，并包装当前 η 请求；**这些是候选重算，不是推进额外物理时间**。每个速度 trial 的 pre_impulse_state、base history、材料路径必须与选定第一阶段完全一致，否则拒绝非确定性/错误输入。这避免要求长期保留已关闭的 native 对象；先以正确性为主，不优化重复成本。

## 4. 原式、几何与接受条件

n 为经现有几何审计的恒定世界法向；a(q)为平面锚点。
g=(x-a)·n/||n|| - radius_m。
所有审计使用实际存储输入的 Fraction/sqrt 包围，不以展示float替换门槛。继承现有精确仿射求值、actual matmul/sum路径审计及方向归一精度门槛。预条件器可使用常规浮点；不能当接受证明。

整个机器人 prismatic ⇒ M、J、b 的冻结模型适用域明确。原生 tyx仍FP32计算/FP64存储，须验证真实转换/响应误差，不能称FP64设备计算。

### A. held-force / configuration 阶段

从原 state/inputs 为每个 trial 构造：
- 节点 force 增量 = Σ p_i n_i / h。
- 机器人 effort 增量 = -Σ J0_i^T (p_i n_i) / h。
映射、聚合及可表示积分沿用原精确一次输入账；请求力矩不得先舍入后审。作用点是初态节点向相应平面的投影；即使初态分离，纯法向 offset 平行冲量，因此成对力偶为零。固定半径同理。非零切向会产生 g0 n×p_t 偶，这就是本版拒绝μ≠0的原因。

对全部候选行求 g1>=0,p>=0,p*g1=0。迭代可用
p_i <- max(0,p_i-relaxation*g1_i/(h*k_i)),
k_i = n_i^T (w_i I + J0_i M0^-1 J0_i^T) n_i。
k仅预条件；每次改变请求必须完整原生 replay，保留材料/共享机器人/同节点多面交叉响应。

精确接受残差包括 max(0,-g)/gap_scale，负p门禁，以及 abs(min(k*p,g/h))/reference_speed（负gap另验），gap_scale=min(reference_length,h*reference_speed)。取实际法向归一后的量；world impulse不是展示幅值。迭代预算耗尽拒绝。可以沿旧受审方法在原预算内保留完整最佳合格trial以消除持续正gap偏置，不得加gap反馈修补状态/增预算/改tol。

MaterialPathLedger.require 对 **pre-impulse** 原生状态、实际 inner iterations、原输入/held p逐一认证。FrozenRobotLedger 对同一状态认证 M0Δv、q-q0=h*v_minus。原有 ledgers 不改成接受最终速度状态。初态 g0低于负容差立即拒绝；不修复深度穿透。

### B. terminal velocity 阶段

只对第一阶段 gap 区间全部落在 ±tol*gap_scale 内的行施加 η；其余严格开放行 η=0且允许仍接近。不确定跨闭合门槛的区间须细化或拒绝，不以舍入分类。正p行必须闭合。
- 节点瞬时冲量 +Σηn，机器人对应当前点冲量 -ηn，保持配置。
- 对闭合集合解 u_n_plus>=0,η>=0,η*u_n_plus=0，u为真实末节点速度减当前J1 v。prismatic范围内J1=J0须独立核对，不能推广到旋转。
- 使用实际完整候选每次算u，GS η<-max(0,η-relaxation*u/k)只是求解；同域多点先聚合一次native调用。
- 精确 residual max(0,-u)/reference_speed、abs(min(k*η,u))/reference_speed、负η门禁，与位置/held互补同时接受。
- 新 TerminalImpulseLedger 独立认证 q+/x+ 与pre配置 **bitwise不变**、未请求域和history完全不变、mΔv=实际η请求与 M(q*)Δv=ΣJ(q*)^TP；FP32存储提升、输入转换和实际增量误差在原tol下约束。不能只比较点速度或动能替代冲量方程。保存原请求，不能只看已舍入 batch。
- η只作用于速度；总冲量 p+η供诊断，但不得回填held ledger或推导位置。

### C. 物理预算与路径

总能量沿旧冻结M0机器人K +节点K/U/重力 -原始非接触端点功，使用最终速度和第一阶段位置。接触力不能混入外功抵消自己；MaterialContactBudgets保持固定的J和N m s上限。

材料角动量每域精确计算：
L_final-L_initial - Σ x0×[h(mg+F_original)+held_p_world] - Σ x1×terminal_eta_world。
按域L1累加，不跨域相消、不用浮点报告数作为门禁。可给旧私有 MaterialContactBalance 增加 default=None 的 terminal 参数，默认旧行为/结果保持，测试证明。
另记录并审计 terminal ΔK 与 η·u_minus + 1/2 η^T Wη 的质量响应；不允许超出显式能量预算的正增量，整体能量预算仍单独验收，不靠一阶段抵消伪造方程。fixed-base机器人不声称任意世界动量守恒；独立解析场景在自由可动轴核对整体动量，其他轴基座反力范围明确。

对声明的节点端点线性路径和全prismatic平面端点线性路径，g(t)为仿射；实际初末几何误差区间都>=-tol*gap_scale，则整段在同一数值包围内。这是该离散方法声明路径的平面包围，**不是**原生迭代每个内部节点轨迹、连续事件TOI或旋转FK CCD。不得使用另一未接受候选的端点。已有 feature sweep 不适用无限面，不强行凑调用。

## 5. 必须独立验收的测例

冻结NumPy tol1e-8，tyx tol2e-5，h=.25；外层 position/velocity iterations各128，inner128，relaxation1；正常预算 energy1e-6 J、angular1e-6 Nms。若解析控制受FP32预算影响，必须先证据/设计修正，不事后放宽以通过。测例normal为+Y，gravity0，robot单prismaticY质量2，普通惯量I，节点质量1，q0=vrobot0=0。

1. 自由探头 y0=1/8,vy=-1：held p=1/3，stageA xnode=q=-1/24，vnode=-2/3,V=-1/6；terminalη=1/3；末速均-1/3，P=-1,K=1/6。仅位置律的速度不满足该oracle；遗漏/重复η必须RED。
2. 两节点距离链 A=1/8 B=9/8,vy均-1，rest1,c1/16：p=3/7，A1=q=-3/56,B1=51/56，vA_minus=-5/7,vB=-6/7,Vminus=-3/14；η=1/3；vA_plus=Vplus=-8/21；P=-2,U=1/98,K+U=25/42。域全节点自动候选，B不接触但响应非零。不能预先给定p/η算作生产求解。
3. 自由探头再一步无外力：闭合保持、p=η=0；随后节点+1N外力一整步：自然分离、p=η=0、exit且history清零；继续自由步不残留拉力。初态更远正常接近但本步不开接触，η必须0，不能把所有负相对速度都约束。
4. T=2,support两h=1/16与1/32，gY=-1，free2node或compliant链保持静置有限effort平衡；另外自由探头碰撞，用h=.25/.125/.0625与一维真实非弹性TOI解比较到T=1，明确离散位置误差随h门槛及重复性，不声称本法事件精确。正常支持maxdrift NumPy1e-7m/tyx1e-5m，不增tol。
5. 两个节点共享robot、同节点两个不平行面/三角材料，整体反作用一次、rows逆序、可表示共同平移/刚体坐标变换正常对照，重复调用一致；独立线性质量系统/动量oracle，不调用生产私有求解当oracle。
6. 零gap零速法向仍来自collider；非零radius平移等效；重名keys、非法数组/类型、μ>0、revolute旁支、fixed节点、缺trace/能力、未支持输入/状态、深穿透、history冲突、预算不足、native拒绝均显式失败。其他域及basehistory真实透传，不接受假native结果。
7. 后段子步terminal应用故障、材料trace篡改、分离后history脏写、close失败与半构造故障：整帧state/history/input恢复且retry=fresh；故障前必须有已接受子步。每种故障含正常对照，不靠未触发注入“通过”。
8. actualinterpret/CUDA至少自由/链两个解析例+真实launch/phase计数；后端异常不回退。一次boundedCPP自由短例，TYX003长期保持开放。旧initial/free/material/SceneLoadReplay/Articulated定向回归保持旧合同，包含新能力关闭时原输出不变。

## 6. 实施边界和交付

允许文件：新 constraints/robot_plane_collider.py，runtime/terminal_impulse.py，validation/terminal_impulse.py，solvers/dynamic_robot_normal_contact.py及最多两个同目录私有辅助模块；修改runtime/program.py可选协议、solvers/articulated_coupling.py明确能力/调用，_material_contact_balance.py默认保持的末冲量项、__init__.py和installed smoke；对应新测试。不得改旧初态接触算法/材料积分器、SceneStepper通用时间语义、tyx pin或kernel数学。若发现必须更改边界，先向主agent报告实际阻塞。

一个实现owner按TDD推进三条相互依赖部分：native terminal机制、两阶段自动法向接触、独立物理与事务；主agent只写文档/基准，不并发改owner文件。一个fresh最终阶段review（产品+测试+计量脚本），必要问题RED修复不重复审阅。

工程：相关两CPU/Ubuntu、实际interpret/CUDA/有界CPP、公开tyx边界/ruff/docs/compile/uvpip/锁/源码外wheel；新solver需要最终全量CPU和CUDA独立验收，数值进程串行。正式性能全部数值/review退出后，三个后端按顺序各3x100，记录replay/terminal调用/launch/传输及版本；只有观察不优化。遗留/host瓶颈/tyx缺陷分别归档；未测Warp不称差距。专用分支commit/push并核对SHA/准确CI。完整H1及案例线仍保持未完成。


### 生产 ledger 的精确归一与明确物理窗

TerminalImpulseLedger(scene,before,requests,settings,context=BackendContext(),*,registry=None,history_fields=())拥有全部快照，require(after)返回只读报告。requested域仅particles（全动态）或全prismatic robot，其他注册域只验证未改变。history_fields显式用于registry clone/validation。
- 对每个particle精确算 e_v=Δv-w*ΣP_original，按reference_speed归一逐分量绝对值最大值。
- 对robot精确算 r=M(q*)Δv-ΣJ(q*)^TP_original，以同一正定M的有理solve映射成速度误差，再按reference_speed归一；全部prismatic所以无角速度单位混用。
- 输入聚合转换误差也通过同一w或M^-1映射归一，原始request乘积用Fraction保留，再与实际FP64 batch/FP32输入转换比较。不能用大请求幅值作相对分母淹没小但有物理影响的丢失载荷。
- tyx post原生速度必须严格FP32→FP64可表示；配置bitwise保持，未请求域/history bitwise保持。unknown precision拒绝。每个residual<=原settings.tolerance，非有限或类型异常转明确失败；不改候选。
- 质量与J使用当前pre配置的独立host物理求值并保留精确存储值；这是原式核对，不称三角函数或FP32全误差的无限证明。

T2 support固定：机器人gravity0、effort+2N，两个单位节点gravityY=-1，A=0、B=255/256，rest1,c=1/256，零初速；平面0。解析held p_A=2h、p_B=0、terminalη=0，全位置速度保持。h=1/16与1/32，完整T2，两模式容许漂移上限按第5节，不调tol。
free impact三个h固定相同初态，T=1真实非弹性参照位置node=q=-7/24、速度均-1/3。此一维无加速度方法端点可能恰好重合连续解；要求每个h终点误差<=NumPy1e-7m/tyx1e-5m且重复一致，不强造非零误差或声明收敛阶。后续真实材料/网格/旋转综合步长门槛仍必须另做。

末冲量时刻力矩的独立非退化控制：在free解析例保持法向数据不变，只将节点vx从0改1、x0=0。stageA/post节点x坐标均1/4，p与eta仍1/3；末冲量的z矩为x1*eta=1/12 Nms，而错误x0*eta=0。精确材料角动量缺陷减末矩后应为0；按初态位置错记eta须超过固定1e-6预算拒绝。平面为无限面且robot只沿Y运动，基座可承担未表示的力矩，不能把本控制宣称全系统世界角动量守恒。这个控制必须用于Task2时刻记账负测试，全Y一维例无法区分两个记矩时刻。

非平凡材料步长控制（解析预测，生产测试尚待执行）：仍用两节点链原初态，到T=1/2，h=1/8、1/16、1/32。无外力，真实撞击tau=1/8，碰后A+robot等效质量3，B质量1，压缩接触保持到T=1/2。令s=t-1/8、omega^2=64/3、z=B-A-1，则连续z=-(2/(3*omega))*sin(omega*s)、u=-(2/3)*cos(omega*s)，A=-s/2-z/4、B=1-s/2+3z/4，VA=-1/2-u/4、VB=-1/2+3u/4。独立离散质量方程从z=0,u=-2/3递推z_new=(z+h*u)/(1+h*h*omega^2)、u_new=(u-h*omega^2*z)/(1+h*h*omega^2)，两式都用旧值。三个h对应3/6/12碰后步，连续解位置最大误差解析预测为0.03653627/0.02227115/0.01225810m，速度0.08027827/0.03499121/0.01452745m/s。要求生产匹配独立离散参照、连续误差递减并低于预设位置0.037/0.023/0.013m、速度0.081/0.036/0.015m/s；先NumPy三h，tyx至少最细h。所有步仍保持原solver tol和物理预算；这些粗步长误差界只是离散误差刻画，不是最终P6.5精度门槛，也不声称修复NUM005。独立递推不调用生产private helper。若原生方程不等价须报告原式证据，不能调native/tol凑结果。
