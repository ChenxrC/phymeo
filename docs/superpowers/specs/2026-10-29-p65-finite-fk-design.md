# P6.5 有限凸面与旋转 FK 接触集成设计

2026-10-29。基线 f611ce2362b7d0af019e0e4c462d6904e8436f30；状态：唯一 builder finite_fk_design_check 已结束；5项重要歧义按文末裁定关闭，本合同冻结。用户已持续授权按框架合同推进；以下支持域和算法选择由主 agent 作出，不冒称用户逐项选择。实施依本合同及计划进行，不能重派同一builder。

## 目标、路线与完成边界

同一阶段交付自动有限面接触、面/边/顶点身份与建立/分离/切换、真实机器人反作用、当前构型末冲量账以及最终接受路径的保守 FK 认证。几何模块必须被生产程序调用；不能以孤立证书完成本阶段。

选原生整步重放、初态作用点 held 求积、当前构型 terminal 和最终路径审核。保持原积分器、材料路径和整体事务。事件分段暂不选：新时长必须重新积分、材料 alpha 随 h 改变，不能截取旧路径或隐藏子步。较粗步可因中途穿越或能量增益拒绝；该离散合同不保证每个可行连续问题都在任意 h 成功。

[已有研究](../../development/p65-finite-fk-candidate.md)已完成，6 个原生短例不重做。研究证明当前 native 支持旋转末冲量，却不证明一般树全步能量。首个可交付域为全 prismatic 树或一个固定原点、COM 在原点的单 revolute 转子，正半径、双面有限凸平面、mu/rest=0。它推进真实旋转几何与反作用，不完成一般关节树、有限厚度摩擦、H1 或双臂案例。后续仍按总框架路径推进，不扩大到 P7/P8。

## 公共输入、单位和支持域

```python
RobotFinitePatchCollider(key, link_index, local_vertices, *, radius_m)
DynamicRobotFiniteContactProgram(
    scene, base_program, colliders, *, materials, budgets,
    robot_domain="robot", registry=None, inner_settings=None,
    relaxation=1.0, reference_angular_speed=1.0,
    max_ccd_intervals=4096, max_ccd_depth=32,
    history_prefix="finite_contact",
)
FiniteContactTerminalProgram(base_program, requests, *, registry, reference_angular_speed=1.0)
FiniteContactTerminalLedger(
    scene, before, requests, settings, context=BackendContext(), *,
    reference_angular_speed=1.0, registry=None, history_fields=(),
)
robot_patch_sweep(
    model, q0, q1, collider, node_start, node_end, *, clearance_m,
    max_intervals=4096, max_depth=32,
) -> RobotPatchSweepResult
```

- 新 collider 位于 constraints/robot_finite_patch_collider.py。key 非空唯一，link_index 非 bool 非负整数；local_vertices 是 owned/read-only FP64 (V,3)，V>=3，循环顺序的严格凸、共面、非重复边界。用精确存储值验证共面/凸性/自交；不把凹形取凸包悄悄改变形状。零面积/共线转角拒绝。顶点 canonical 化为字典序最小起点、首个非零法向分量为正的循环方向；输入反序/循环移位物理与 canonical 特征 ID 一致。内部 fan 三角只辅助最近点，内部缝不是物理边，不生成重复约束。
- radius_m 严格 builtin finite float >0，是节点中心到有限面集合的排除距离，单位 m；无粒子转动自由度。几何为双面薄片的正半径邻域，normal 为从真实最近点指向节点的径向，不使用旧 history 或速度选择法向。不把无限支持平面附着到有限面外。面后侧同样碰撞，允许绕有限边界分离/再建立。初态/最终接受几何的零距或深穿透拒绝；有限中间试算允许距离0、gap=-rho，不构造n1，先用下述signed-support生成器推进，不能把平台free trial的零距当原生执行异常。旧 RobotPlaneCollider 和旧 normal/friction 的门禁完全不变。
- all-prismatic 整棵树含旁支；或恰一个 parents=(-1,) 的 revolute，com=(0,0,0)，local axis 是精确坐标单位向量 +/-X/Y/Z，惯量为原模型已验证的正对角惯量，质量正，joint_position 任意固定点、joint_orientation 任意原模型允许的单位四元数。这样轴惯量 I 为该轴的 inertia_diagonal，不随 q 变，COM 重力力矩为零。一般多 revolute/混合树在构造拒绝；禁止仅删旧 prismatic guard。转子所有接受路径的 q 必须在 [-4,4] rad；越域试算不被错误认证，若不能在原迭代预算得到域内合法结果即失败。
- materials 是非空 owned 域映射，值 DistanceConstraints 或 None，所选节点全部动态 FP64；原 MaterialPathLedger/native tracing 保持。每个所选节点乘每个 collider 形成稳定 row，其他域由 base 正常推进。budgets 是原 MaterialContactBudgets，仍为每子步 J/N m s 绝对上限；reference_angular_speed 是 builtin finite float>0，rad/s，进入签名。
- inner 仅 1 子步，h>0，tol/reference_length/reference_speed 与 outer 一致；每 outer 子步均两阶段，terminal 不是额外时间。backend 仅原 NumPy/实际 tyx:interpret/cuda/cpu_cpp，FP64 storage、tyx FP32 compute，features 精确为 {"dynamic_robot_finite_split"}，不继承未实现能力。
- CCD 两个预算为非 bool 正整数，intervals<=4096、depth<=32 是默认值而非允许无界计算；用户显式正整数可降低或增大有限预算，签名包含实际值。不以超时或预算耗尽冒充安全，不放大物理 tol。

## 最近点、方向与拓扑身份

每行 row=(domain,node_index,collider.key)。初态计算最近点 y0 和其局部材料坐标 b0，n0=(x0-y0)/||x0-y0||。末态重新计算 y1/b1/n1；它可能已经从面内部切到边或顶点，b0 与 b1 不要求相同。最近点候选可由已有三角算法生成；必须用精确 primal/dual 或凸集 KKT 和 FK 区间独立确认，不能只相信 floating closest 输出。

两套量分别审计：实际存储 FK/J/载荷向量，和按原存储模型精确定义的规范化旋转/FK 包络。以原 settings.tolerance、Lgap=min(Lref,h*Vref) 审计二者误差，不能用未施加的理想向量代替实际 world。距离与方向用有理点积/sqrt 包围；q/frame normalization/浮点端点不确定性必须进入包络，不能默认 libm 正确舍入。

末 gap 区间 g=dist(x,patch(q))-rho：lo>=-b 且 hi<=b 为 closed，lo>b 为 open，b=tol*Lgap；深穿透或无法分类拒绝。几何容差只复用原穿透预算，不能另添 clearance epsilon。rho>b 必须成立。

canonical feature=(kind,index)：face=(1,0)、边=(2,canonical起点index)、顶点=(3,canonical index)、inactive=(0,0)。内部 fan 缝仍为 face；边/顶点由精确局部重心/KKT active set 决定；无法在几何精度内唯一确定类别时仅对诊断标签采用更低维公共特征，不从 history 猜。标签约定不能将真实witness或normal吸附到该边/顶点；距离包围与实际world映射继续独立验原误差预算。诊断包含 own local witness/weights、距离区间、world normal、初/末 feature；每个物理 patch 只有一个 row 和容量。

## 自动求解、原生输入与两阶段原式

held 使用初态 b0/n0，并固定这些求积数据于本 outer 子步，自动求 p>=0：P_i=p_i*n0_i，完整原状态/原输入 native replay 中 node +=sum(P)/h，robot effort +=sum(-J0(b0).T P)/h，各一次。原非接触输入不改写。held 接受末几何互补 g1>=0、p>=0、p*g1=0；不是从参数表填写 p，也不是用末 J1 给 held 反力。

终末只对末 closed 行求 eta>=0，E=eta*n1，在 selected held 的完整相同输入重放中，live native velocity 收敛后且 close 前一次聚合施加。当前 J1(b1)、当前 M1；节点 +E、机器人 -J1.T E。u+=n1.(v_node+ - J1*v_robot+)，eta>=0、u+>=0、eta*u+=0。末 open 的 p/eta 必须为零（在原互补尺度审核，committed inactive history 严格零）。P+E 仅诊断，不回灌 held，不0dt、不 close 后调用、不 host 覆盖候选。

held normal 随初态，terminal normal 随末态；不能把它们混成统一末法向 law。两端 moment 使用各自 x0/y0 与 x1/y1，normal 中心力使 (x-y)cross impulse=0；正半径无摩擦不会漏掉 rho*n cross tangent 的力偶。法向增能不能仅按 held 正功裁断，必须用最终两阶段物理账。

候选生成与接受律分开。初态节点对patch支持平面有严格非零signed side时，从精确几何取sigma0；试算的正交投影在真实有限凸patch内时，生成器用g_gen=sigma0*signed_plane_distance-rho及其互补merit，允许穿过中间距离0而不构造n1。投影在patch外则仅用实际有限距离gap；初态side无法确定或恰为0时也只用有限距离生成器，不从history选side。此sign只选择求根分支，不是单面物理边界；最终接受仍是原unsigned有限距离/真实n1/完整CCD/物理账。内部fan缝不参与side判断。

允许确定row顺序有界投影迭代：正冻结预条件k00=n0.T*(wI+J0*M0^-1*J0.T)*n0，p_new=max(0,p-relaxation*g_gen/(h*k00))。它只给步幅，不假称当前有限距离导数；k00非正/非有限拒绝。最多12次确定二分减步，用完整replay后的生成器merit选进展；每次载荷改变必须重新计算真实closest，最终不接受生成器近似作证明。全部尝试计数且不增加隐藏outer时间。terminal用当前k11=n1.T*(wI+J1*M1^-1*J1.T)*n1>0及真实post u投影更新eta。共享多行完整replay，不能独立解后相加；不收敛在原有限迭代预算内明确失败。

有限双面几何中 free 候选可端点分离却途中穿越。此时必须拒绝整步并要求调用者显式减小外层 h，不能把 CCD 失败当无接触，也不保证自动找出另一离散接触根。本阶段不暗中事件分段。下面正例明确包含端点穿透可识别的自动建立；端点安全/中途穿越为必须拒绝的独立控制。

selected held 与每个 terminal trial 的完整 pre state/input/history/path/diagnostics/sweeps/metrics 必须一致。MaterialPathLedger/FrozenRobotLedger 只验 pre。所有 selected native pre/post q/v/x 均为真实 FP32 提升；一ULP host 改写即拒绝，开放/零冲量也不豁免。

## 末速度与全步物理账

新增 FiniteContactTerminalProgram 和 FiniteContactTerminalLedger 单独支持本 spec 两种 robot 域，不改变旧 TerminalImpulseLedger 的 all-prismatic 合同。旧runtime/terminal_impulse.py在velocity_metrics硬编码旧ledger，不能把revolute请求塞进它再绕过拒绝。新runtime/finite_contact_terminal.py采用相同公开workspace时序协议、owned请求聚合与完整pre证据，明确调用新ledger；仅新增adapter，旧runtime/participant/kernel不改。可复用只读聚合/克隆/正定质量工具，不monkeypatch旧模块或私调native内核。当前 q1 重新求 M1/J1；同域全部请求聚合后一次响应。精确 original request 与实际 FP32 batch conversion 分别审核，configuration/history/非请求域严格不变，速度 residual 按 prismatic m/s、revolute rad/s 独立尺度化。report owned，不能根据观测 dv 反推“期望质量”。

转子必须额外确认当前 host mass 与解析恒 I 的误差在原 equation budget 内，端点物理 K 用解析 I；all-prismatic 继续原 frozen M0。末能量用独立聚合响应计算 deltaK = generalized_impulse dot v_pre + 1/2 generalized_impulse dot M1^-1 generalized_impulse（所有节点也聚合），再与实际 post K 比较。既有 original-input endpoint work、最终 K/U/重力、按域 L1 材料角动量缺陷 x0-held/x1-terminal 与绝对 budgets 全部保留。不得用“总能量下降”替代原式/法向互补，亦不能让 CCD 替代能量。

转子绕固定轴的总轴角动量独立审核：机器人 I*omega 加所有 selected 节点相对固定轴原点的 r cross mv 轴投影，扣原外输入及材料独立缺陷；成对法向 world moment 完整三维审核。固定底座可承受其他轴反力，不声称自由系统三维机器人总角动量守恒。可复用 MaterialContactBalance 并增加转子独立 I/轴账，不改旧共享 balance 的门槛。

## 最终接受 FK 曲线证书

x(t)=(1-t)x0+t*x1，q(t)=(1-t)q0+t*q1，t in [0,1]；patch 每个顶点是完整 FK(q(t))。绑定的是最终实际接受 pre 的 x1/q1，terminal 配置 bitwise不变。不能认证上一 candidate 或机器人端点弦线后替用另一条路径。

每 row 证明 dist(x(t),conv(vertices(q(t)))) >= rho-b，b 沿上文原穿透预算。查询 robot_patch_sweep 接受明确正 clearance_m，程序只能传精确 rho-b 的向上取到最小FP64 >= 精确rho-b的表示，因此证书证明不弱于原门槛，不新增放松。结果 status="certified"/"violation"/"unresolved"，包含完整[0,1]有理分割、每段 outward distance_lower_m、见证 t/upper_distance（若违例）、实际 interval/depth 计数和输入内容指纹。非法输入构造错误；未知不伪装违例或安全。结果及数组 owned。

可实施证据路线：全 prismatic 顶点仿射，沿已有同步投影证明。单转子任意固定零帧 R0 下每个顶点是 c+A*cos(q(t))+B*sin(q(t))。对每个候选分割段[a,b]选一个固定分离方向 d，证明所有顶点的 d.(x-v) >= clearance*||d||。d 由 closest 仅用于寻找，证明全用精确有理/区间。

沿固定 d 的曲率上界为 (deltaq)^2*sqrt((d.A)^2+(d.B)^2)，因此该段标量相对投影下界 >= min(两个端点的严格下界) - curvature*(b-a)^2/8。该 projected amplitude 不能粗换成非零的全向半径从而无法证明沿轴持续等号接触。端点 sin/cos 以有理 Taylor 与显式 Lagrange remainder 包围；|q|<=4，固定80项或同样严格有界且可复核的阶数。四元数零帧可直接以二次多项式除精确norm平方构成有理旋转；axis为坐标轴，没有不明normalization。所有输出舍入向外。

一个支持方向对全部凸顶点成立即可证明整面；不必先知道时间内最近特征。每段可换方向，自适应二分；budget用尽unresolved。可用严格 primal witness 的 dyadic 时刻上界证明 violation；有限采样只可给违例见证，不能宣称安全。起/末端与持续等号接触必须有能成功的解析/投影正例；全部unresolved不能算实现了CCD。

真正 native/trace/terminal/close/ledger 异常、非有限、未知证书拒绝全帧，已有best不能吞掉。有限已定义但物理超预算试算可恢复完整合法best并重新审核全部原式+物理+同路径证书；最终best不合格即失败。不能只恢复报告标量。预算/certificate错误在第2/末子步也恢复全部state/所有history，retry=fresh，半构造资源关闭。

## history、诊断与能力

history_prefix 下 active(R,) bool，feature(R,2) int64，phase_world(R,2,3) FP64。只作记录不warmstart；inactive三个字段严格0/false。cold closed 可建立；enter/persist/exit/inactive按几何与旧active产生，persist伴feature_changed单独诊断。内部fan缝不触发feature_changed。签名包含规范几何、rho、模型域、law、材料、预算、angular scale、CCD预算与history schema；basehistory每outer子步仅提交一次，不隐藏全局计数。

diagnostics.dynamic_robot_finite_contact 至少含 row_keys、initial/final_feature、held/terminal_local_points、held/terminal_normal、held/terminal_world、gap_bounds、relative_normal_bounds、transitions、feature_changed、held_replays/terminal_replays/line_search_replays、physical_budgets、terminal_ledger、ccd。顶层 execution 统计所有真实 trial，terminal post 已含pre不双计。失败report沿原runtime空diagnostics边界，不编造失败性能。

## 独立场景与精确预定门槛

通用 CPU tol=1e-8、设备tol=2e-5，Lref=Vref=angular_ref=1；free物理budgets CPU 1e-8 J/Nms、设备1e-5 J/Nms。材料例独立给1e-3 J与1e-3 Nms（仅该固定离散模型范围，保留实际残差及步长趋势，不当NUM005修复）。任何调整必须先以原式向主agent说明，不允许为过测倒推阈值。

### A 有限平台：自动建立、内部缝、边界退出和再建立

Y-prismatic机器人M2、node m1、gravity/force0。局部正方形(-1,0,-1),(1,0,-1),(1,0,1),(-1,0,1)，rho1/8。q0=V0=0，node x0=(-3/4,1/4,1/4)，v0=(1,-1,0)，h1/4。独立Fraction oracle：p=eta=1/3，q1=-1/24，node y1=1/12，两个末y速度=-1/3，x1=-1/2。正常响应不是预设冲量输入。

连续至T2（无后续外力），node x从-3/4到5/4、z1/4，穿内部fan缝仍persist、无重复capacity；到边x1为edge，再外移gap增大、exit且P/E=0，不留无限平面支撑。h1/4、1/8、1/16逐步独立Fraction隐式Euler/终末非弹性公式与有限方形clamp距离，记录与连续事件位置误差，误差随h不增；至少2次重建相同结果。再只在首个反向步给Fx=-2/h使vx从1变-1，其后无力按反向速度共走1/(4h)步到边界并enter，再一步回面；原外功/角动量账必须通过。反向顶点/域/collider顺序与多个共享树行必须等价。

材料变体：两节点A=(0,rho+1/8,0),B=(0,rho+9/8,0)，v=(1,-1,0)，距离rest1/compliance1/16，h1/4、M2。自动pA3/7、eta1/3；q1=-3/56，A1=(1/4,rho-3/56,0)、B1=(1/4,rho+51/56,0)，vAy=V=-8/21，vBy=-6/7，x速度均1；B保持open，材料path只验pre。独立K/U/Py与两次输入反作用，不从产品诊断自证。

### B 有限转子：安全旋转接触和增能拒绝

m=I=h=1、rho1/8、gravity/force0，Z转子，局部三角(0,0,-1),(3,0,0),(0,0,1)。取alpha=1/4，theta=atan(1/10)，c=10/sqrt(101)，s=1/sqrt(101)，X=3/2，V=X/10+rho*(1/c-1)，p*=1/16。初态q=alpha、omega=theta+p*；节点x0=Rz(alpha)(1,rho,0)、v0=Rz(alpha)(1/2,V-p*,0)。自动解应得held p*，q1=alpha+theta，node x1=Rz(alpha)(X,rho+V,0)，held v=Rz(alpha)(1/2,V,0)、omega-=theta。末作用点力臂ell=X/c+rho/10，u=(-1/2+10V)/sqrt(101)-ell*theta，eta=-u/(1+ell^2)，v+=v-+eta*Rz(alpha)(-s,c,0)，omega+=theta-ell*eta。独立1x1质量、完整P/E作用点、轴L、K用原式核对。

路径gap的解析正例：未整体旋转alpha前，gap = -(1-t)sin(theta*t)+X*t*sin(theta*(1-t))/cos(theta)+rho*(cos(theta*t)*(1+t*(sec(theta)-1))-1)>=0。前两项>=t(1-t)(X*tan(theta)-theta)>0，后项由sec凸性非负；投影全在有限三角内。不是用采样认证。held deltaK=p*(V-theta)-p^2；terminal deltaK=-u^2/[2(1+ell^2)]，正常总变化<0。

同一几何将p*改成(V-theta)/2并相应改变初态v/omega，路径和末u不变，两阶段最终能量增益>2e-4 J：必须因全步energy拒绝，不能只因“held法向正功”。它是B负控制，不是第二独立场景。alpha避免复跑TYX004已知cos微角输入；严格np.errstate(all=raise)保持，设备异常不得降级。CPU/interpret/CUDA各跑正常与拒绝控制，主oracle自行math/Fraction计算，不导入研究脚本答案。

B持续窗口另固定低滑速初态：用X=1（vx=0）、V=1/10+rho*(sqrt(101)/10-1)，其他alpha/theta/rho/p*=1/16相同。它与X=3/2的冲击/增能拒绝例清楚分开，不把改初态称同轨迹。对该固定初态以1、2、4个显式outer子步推进T1，完整审核所有endpoint/gap/energy/轴L，每档须合法接受；至少2次重建结果一致。设备同T1取2、4步，重建重复一次。用独立标量closed-face方程有界求根和当前1x1末冲量作逐步oracle，不能导入产品求根/质量/geometry helper自证。

原X=3/2初态细分后会在后续已建立接触步增能，不能作为无条件通过窗口：独立标量方程得到N2第二步约+2.03464e-5 J、N4第二步约+4.47627e-6 J，CPU1e-8预算应拒绝；这些是明确失败/回滚控制。低滑速X=1的同方程得到N2后一步约-2.67521e-8 J、N4后三步约-2.52438e-9/-3.08255e-9/-3.59649e-9 J，第一冲击步约-0.0038468 J。以上是冻结前独立stdlib标量合同核对，不是native验收；实现仍须全部原生门槛和CCD。没有修改任何budget/tol，不宣称通用连续精度阶数或任意旋转滑动都可接受。

### CCD、边/顶点与事务控制

- 单转子同三角，q0=alpha,q1=alpha+pi，静止node=Rz(alpha+pi/2)(3/2,0,0)，rho1/8：端点距离大于rho，中点碰撞，certifier给严格violation，生产free候选必须拒绝且回滚。
- 在局部XY平面的三角(-1,-1,0),(1,-1,0),(0,1,0)绕Z转动，静node=(0,0,rho)、q0=.25,q1=.75：持续等号距离rho，沿Z projected curvature为0，必须certified，不能预算耗尽。
- 边/顶点距离独立：方形外node(5/4,0,5/4)最近顶点(1,0,1)，距离sqrt(1/8)；(5/4,0,0)最近边距离1/4。normal径向、无幽灵平面载荷；axis/frame/平移变换和严格interval覆盖。
- max_intervals=1 的需要细分安全曲线必须unresolved；深度预算、NaN/退化/不支持机器人、错误clearance/篡改candidatefingerprint拒绝。native后段或最后子步CCD/terminal/path/close失败全帧回滚、retry=fresh；错误J0/J1时刻、单边反力、错FP32提升和漏basehistory都有正常对照。

## 文件边界与验收策略

新collider及runtime/finite_contact_terminal.py；geometry/robot_patch_sweep.py及至多一个私有旋转区间helper；validation/finite_contact_terminal.py；solvers/dynamic_robot_finite_contact.py及至多两个私有helper。公共导出__init__和smoke_installed仅加新能力。不改旧solver/ledger/participant/kernel/runtime/依赖/旧测试期望。必要的共享修复先给主agent原式与RED证据，扩大影响选测后另裁定，不能暗中删门禁。

实施与验收是一个集成任务，不拆多次freshreview。全部新测试、两独立场景、真正设备与直接消费者选测，按[分层政策](../../development/p65-validation-policy.md)核对同快照证据并复用；父不重跑child同集。新runtime/kernel不改，不例行两CPU全量/整分支长CUDA/900计量。源码外wheel新入口与远程原完整CI保持。详细nodeids见[实施计划](../plans/2026-10-29-p65-finite-fk.md)。


## 冻结裁定与builder闭环

唯一finite_fk_design_check只读完成，无native/数值执行，结论除以下裁定外nothing important left；它不是最终code review。主agent在冻结前独立计算新场景标量方程（stdlib、非native），不重跑旧研究或旧全量。

1. R1：signed-support仅是有限投影内的求根生成器，最终仍unsigned真实几何/CCD/physics；试算零距不normalize，初/末零距拒绝。代价：不保证任意h/分支都收敛，不能以收敛失败伪称安全。
2. R2：保留X1.5冲击/能量拒绝，持续窗口另固定X1低滑速；原X1.5细h越能量预算是失败控制，不调预算。代价：本离散旋转律有可证明的接受域限制，H1一般旋转尚未完成。
3. R3：CCD目标向上舍入到最小FP64；绝不降低原rho-b。
4. R4：平台反向再进入需1/(4h)步，外力只第一步，随后真实输入/时间推进，不手改state。
5. R5：feature tie只诊断，不吸附物理witness/normal。
6. R6（主agent源码preflight）：旧TerminalImpulseProgram内硬编码旧prismatic ledger；增加独立FiniteContactTerminalProgram调用新账，保持旧runtime行为，不靠仅删guard。此adapter和ledger在同一集成产品/唯一最终review中验收。
