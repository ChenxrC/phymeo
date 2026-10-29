# P6.5 有限厚度棱柱与机器人面的摩擦集成设计

2026-10-29。基线 `16a752a87994940b0aee9b5a9751d63e90d196ad`。状态：唯一builder thick_friction_design_check已完成；R1/R2/R3重要歧义由主agent裁定并写入，本合同冻结。尚未有实现或新数值验收。用户持续授权自主技术裁定和阶段实现，以下具体方案是主 agent 的选择，不冒称用户逐项选择。

依据：[框架要求](../../development/p65-framework-capabilities.md)、[按影响选测](../../development/p65-validation-policy.md)、[已完成候选推导](../../development/p65-thick-friction-candidate.md)、[有限/FK合同](2026-10-29-p65-finite-fk-design.md)。同一纸面 probe 不重做。

## 目标、选择与边界

同一生产切片接通完整有限厚度几何、自动法切冲量、原生双方反作用、完整最终路径证明及全局事务。薄片由三动态节点构成的中面三角及正半厚度定义，碰撞体是六顶点凸棱柱，不是六个采样球。必须处理有限面在大三角内部、真实边边和部分重叠，不以孤立几何工具结束阶段。

采用既有 TriangleOffsetPointPort 的完整厚度 Jacobian、初态求积 held 重放和当前配置 terminal。节点不新增 spin；仅重心分配会漏厚度力偶，拒绝该路线。事件分段会改变材料 alpha、步长及事务，当前不引入。首版机器人只 all-prismatic，整棵树的 local joint_axis 必须为精确 +/-X/Y/Z 坐标单位轴，含共享祖先与旁支；任意固定 origin/frame 保持。有限/FK单转子的旧支持仍保留，但不自动扩展到这个新摩擦程序。一般树、转子厚度摩擦、自碰撞、弯曲和真实夹爪总案例仍后续，不勾完整 H1/H2。

允许几何/数值难例在有限预算内无法认证而显式失败；不保证任意 h、任意摩擦或任意尺度收敛。下文正常 A/B/C 和材料变体必须真实成功，不能全拒绝或替换初态。保留原能量/力矩/原式容差；没有性能优化或 graph 改造任务。

## 公共接口、所有权与单位

```python
TrianglePrismSurface(key, domain, node_indices, *, half_thickness_m)
RobotContactPatch(key, link_index, local_vertices)
DynamicRobotThickContactProgram(
    scene, base_program, surfaces, colliders, *, friction_by_collider,
    materials, budgets, robot_domain="robot", registry=None,
    inner_settings=None, relaxation=1.0, max_anchors_per_pair=64,
    max_manifold_rebuilds=8, max_ccd_intervals=4096, max_ccd_depth=32,
    history_prefix="thick_contact",
)
triangle_prism_patch_manifold(model, q, collider, triangle_positions,
    *, half_thickness_m, geometric_budget_m) -> PrismPatchManifold
triangle_prism_patch_sweep(model, q0, q1, collider, triangle_start, triangle_end,
    *, half_thickness_m, penetration_budget_m,
    max_intervals=4096, max_depth=32) -> PrismPatchSweepResult
```

- 新 surface 是 frozen/owned 数据，key/domain 为非空 builtin str；node_indices 恰3个互异非bool非负整数，保留有向顺序。half_thickness_m 为 builtin finite float>0，单位 m。绑定时检查域、全动态 FP64 storage、对应节点，重复同域无向三角拒绝。重编号/循环重排保持物理结果；反向三角只交换 s 符号和面标签，厚度体不变。
- 新 RobotContactPatch 只描述零厚度双面凸有限面，没有 radius 参数；不误用旧 collider.radius 来表达三角厚度。复用旧 canonical 多边形验证规则，owned FP64(V,3)、严格凸、共面、V>=3、无重复/共线边；可通过旧 RobotFinitePatchCollider 的验证构造并仅保留同样 canonical 数据，不能修改旧 radius>0 合同。key 唯一，link_index 在 robot 模型范围。正反序及循环移位相同 canonical 边界。
- surfaces/colliders 非空 owned tuples，按key排序；逐 pair=(surface.key,collider.key) 完整检查，不做可能漏检的未经证明 broadphase。共享节点合法，多个面请求先全域聚合，禁止每个面独立更新同一节点。
- materials 是精确覆盖 surface 域的 owned 映射，值 DistanceConstraints 或 None；所选域全部动态。budgets 沿用 MaterialContactBudgets。friction_by_collider 的 keys 必须精确等于 collider keys，值 builtin finite float>=0，零恢复。通用模型和原材料/participant 限制不改变。
- 支持 NumPy、实际 tyx interpret/CUDA/C++，FP64存储、设备FP32 compute。几何/Jacobian/有理认证在host，不能称整步GPU驻留或Graph。features 精确 {"dynamic_robot_thick_split"}。
- h>0，inner substeps=1，tol/reference_length/reference_speed 与 outer 一致。max_* 为非bool正整数并进入 signature；默认分别64/8/4096/32，实际可降低的有限预算，耗尽不得叫安全。面积/表示条件继续服从旧 offset port 门禁，不放松其1e-10尺度条件。
- b=tol*min(reference_length,h*reference_speed)，单位m；正半厚度必须大于b。严格几何计算以存储输入定义的理想规范FK和理想N为准，实际FP64 port/载荷/法向另审计，二者差不能超原预算。新的穿透指标是沿某单位方向的最小分离平移上界，不是unsigned距离非负；b沿用既有穿透长度预算，不另加epsilon。

## 完整静态几何与响应流形

A=conv{x_i +/- rho*N}，N=((x1-x0)cross(x2-x0))/norm。B是完整FK后的finite patch。理想all-prismatic FK用存储frame四元数的二次多项式/精确norm平方得到有理旋转，位置对q仿射。面/侧面/边拓扑来自棱柱，不把内部三角fan当物理边。

对非退化A及二维凸B，完整SAT轴集是A面法向、B面法向、每条A边cross每条B边；取两种方向，精确零轴去掉。对每轴d定义delta(d)=(min_A d.a-max_B d.b)/norm(d)，求最大delta，正为分离、负为相交候选的分离深度。轴/支持特征由完整顶点集合独立复核，不只取最近一个角点。轴相等/平行按规范方向与完整active feature集合合并；最大值严格并列且投影clipping合法时，先选合格patch-face normal，再选初态几何已有的同向轴（仅本步固定数据，不读history），其余按规范轴及feature key顺序。不等gap不得用epsilon制造并列，也不按速度挑选法向。并列选择不改变几何接受标准。

静态计算的可实施精确表示：在一个状态上c的分量为有理，C=c.c为正有理，六顶点及全部未归一化轴位于Q(sqrt(C))。只需针对该棱柱的私有二次根数值单元：加减乘除、严格符号和向外FP64界；C为有理平方时精确简化。比较有符号的归一化gap可以先分符号，再平方交叉相乘，不能不顾负号直接平方。所有输入Fraction保留原值，不经过float中转。构造确切等号/共面不能靠无限细分区间猜出；有理/二次域的符号不确定是失败。

给定选中轴，取A最小支持feature与B最大支持feature，投影到d的正交平面（使用I-dd.T/(d.d)，无需先浮点归一化）。精确凸多边形/线段/点求交，保留交集的全部极点；将同一投影点分别lift到两支持平面，得到r-b=delta*n及材料(B0,B1,B2,s)和robot local点。cap、side、edge、vertex均按其真实局部参数恢复。交集为空时不是一条有效接触流形：分离态返回open；相交态必须继续其他合格SAT支持轴或明确失败，不能忽略相交。未知/退化不制造冲量。

远离接触的open pair可返回空流形。对于相交/接触，必须返回一组满足支持不等式和完整clipping极点覆盖的同向接触流形；正间隙<=b亦可作为closed候选。静态报告有status、完整轴证据、gap区间、owned点/精确material坐标/feature labels、实际表示误差。物理响应normal必须经所有凸顶点的支持不等式审核，不能把任意patch法向用于斜面/边边。

held候选生成可沿初始支持侧继续深入试算，避免free trial穿过零间隙后切换成反面根；最终接受不依赖该分支选择，仍验真实完整A/B和整路径。中间有定义的有限穿透允许求根；初态深穿透、最终超b或支持特征未知拒绝。端点已分离但中途穿越由CCD拒绝，不自动寻找另一个事件分段根。

## 材料锚点与两阶段容量归属

每outer子步只在其内部建立有界anchor集合。anchor标识为(surface key,collider key,固定owned材料B与s)，不以浮动clipping角点编号继承容量，不从上一history warmstart。每个anchor在整个held/terminal计算中保持同一B/s；当前端口因节点配置重建，但没有Bdot/sdot。

几何极点的精确B/s转换到FP64端口参数时，只做明确的表示转换：凸权重非负、和为1，转换及旧port规范化误差向外计算；需要时用公共二进制权重网格得到精确和为1的可表示三元组，网格精度由FP64有效位数而非物理epsilon确定。实际port canonical weights再次构造必须幂等；不幂等或位移差超b拒绝。不把转换点冒称精确clipping极点，独立保留理想极点供覆盖/末速度审计。不同理想极点转成同一owned anchor只能合并一次；若其极点速度/映射误差不能在原尺度认证，不得静默删点。

初始流形和一次free native trial流形给出初始anchor集合；free只有试算，不提交。任一anchor的initial_closed由其初态材料点与真实finite patch几何及pair支持集合判断，不由history/发现时间判断。在初态不接触的anchor，held tangent严格0。held初态normal来自真实初态点对及支持侧：非零分离用真实有限最近点径向；精确接触用该点属于的共同支持normal。若同一anchor不能确定合法初态反力作用点/normal，拒绝这一trial，不能凭后态法向冒充初态J0。

held从原state/input重放。若当前几何显示未覆盖的penetrating极点，添加其固定材料anchor，从零冲量重新解全部anchor，最多max_manifold_rebuilds次；没有把上一trial的状态/载荷累计到下一trial。集合只增不复制，超max_anchors_per_pair明确失败。已几何合法时不为每个浮点clipping微变无休止重建。

selected held确定后，收集其末完整流形所有理想极点；没有对应固定anchor的点只增加terminal anchor，held p=0。重放输入不变，确认完整pre state/path/history/diag/metrics相同。旧anchor仅在末真实finite接触集内才是terminal active；消失的anchor不把p捐赠或复制给新点。每个p只归一个owned材料anchor，合并不得复制容量。

同一anchor初态closed：held容量mu*p，terminal容量mu*eta；初态open：held t=0，terminal容量mu*(p+eta)。新增terminal-only点因p=0，其容量就是mu*eta。inactive terminal eta/t严格0；末不接触的held anchor要求p/t在原互补尺度为零，提交history清零。该规则是有限分阶段摩擦律，不声称总P+E满足单一末态Coulomb或一般解唯一。

## 原生映射、迭代与独立物理门槛

在每个相位的配置上复用TriangleOffsetPointPort公开jacobians，J_i=B_i*I+s*(I-N*N.T)*C_i/norm(c)。held初态使用J0和机器人初态真实作用点Jrobot0：节点sum(J0.T*P)/h、robot sum(-Jrobot0.T*P)/h通过SceneLoadReplay各一次。terminal用当前J1及当前机器人点，聚合为PointVelocityImpulses，通过既有FiniteContactTerminalProgram/Ledger在native live收敛后close前一次。旧adapter/ledger不改，不用host候选覆盖原生state、不0dt。

held P=p*n0+t0，p>=0，t0.n0=0；normal互补用最终相应有限支持点gap，切速使用selected pre中固定B/s材料点的真实当前Jpre*v减机器人真实点速度，投影初态n0正交平面。Jpre与加载J0可能不同，不能把J0*vpre称真实材料滑速；这不保证任意变形的离散耗散，必须由原全步账接受。初态open t0=0；closed按下述R3精确world Coulomb原式检查（候选固定点t0=Proj_ball(mu*p)(t0-alpha*v_t)只描述迭代，不定义放宽的接受尺度）。

terminal E=eta*n1+t1，n1来自末共同支持特征；真实post u=n1.(J1*v_node_post-Jrobot1*v_robot_post)>=0，eta>=0、eta*u=0；当前切速满足容量为上述mu*eta或mu*(p+eta)的原Coulomb。对每个pair末完整流形所有理想极点独立检查法向post速度非趋近；固定配置的材料速度对B/s仿射，故同一支持normal下极点全部通过覆盖整个接触流形。不能只检查实际离散anchor而漏理想极点。

候选采用全零初始冲量、同步更新。冻结相位全world响应W_ij=Jrel_i*M^-1*Jrel_j.T，包含共享节点和机器人全部交叉项。L=max_i sum_j FrobeniusNorm(W_ij)，alpha=relaxation/L，relaxation在(0,1]。L非正/非有限拒绝；该共同标量不依赖row排列或world旋转。held normal p'=max(0,p-alpha*g_gen/h)，t'=ball(mu*p')内投影(t-alpha*v_t)，全部行同一旧trial同步更新；terminal以真实post u替换g/h，容量使用新eta。禁止逐行独立解后叠加，禁止把对称答案写进solver。

g_gen：当anchor当前投影属于选中B支持feature时，用选中有向支持平面上的signed gap；否则用真实有限最近距离，视为该anchor当前非接触。当前完整manifold仍发现其他缺失穿透anchor。候选轴可由初始侧连续选择，最终当前normal/geometry独立核对。每次更新可用最多12次共同二分alpha，完整原状态重放并比较所有行尺度化normal/tangent残差最大值；不凭单行改进接受。有限迭代使用outer的position_iterations/velocity_iterations，rebuild不重置总预算，所有trial计数。没有任意全局收敛承诺；A/B/C的规定设置必须通过。

新ThickContactMappingLedger（内部/独立validation模块）复核原始FP64请求的J.T分配、实际world合力、关于各实际点的完整力矩、功率对偶以及selected native FP32提升。不能只相信port的自审计。actual节点冲量精确聚合（Fraction/final nearest表示或确定fsum并独立误差包围）；robot广义冲量同样聚合。绝不使用单行能量和替代全聚合交叉项。

原MaterialPath/FrozenRobot只验pre；当前末质量ledger验post；全步原输入端点功、最终K/U/重力、按域L1材料角动量缺陷和budgets都保留。材料角冲量按实际节点x_i0 cross P_i+x_i1 cross E_i。额外审核Σx_i cross(J_i.T P)=r cross P及机器人相反world moment；gap*t带来的真实力矩误差也受原尺度约束，不因gap<=b而自动豁免。终末deltaK用所有聚合后nodal/generalized impulses和独立质量响应，再对真实post K。P+E仅诊断。

执行异常、非有限、unknown CCD/geometry、失败close/trace/ledger都整帧回滚，不得best掩盖。仅有限且已定义的物理超预算candidate可放弃后恢复完整合法best，再重验所有原式/物理/路径；最终best不合格拒绝。第2/最后子步故障及retry=fresh保持。所有resource关闭，basehistory每outer子步提交一次。

## 完整变形棱柱路径认证

绑定最终实际accepted pre端点，x_i(t)=x_i0+t*dx_i，q(t)=q0+t*dq，t∈[0,1]；terminal配置bitwise不变。整个棱柱是conv{x_i(t)±rho*N(t)}，不能用六条端点弦代替法向变化或六点采样代替体。

c(t)=c0+c1*t+c2*t²。每个有理分割区间先证明norm(c)>=a>0，可用分量多项式区间或固定投影下界；C1=max端点norm(c')向上界，C2=norm(2c2)向上界，则norm(N'')<=C2/a+3*C1²/a²。固定未归一化方向d上prism顶点曲率上界rho*norm(d)*(C2/a+3*C1²/a²)，all-prismatic patch曲率0。所有凸顶点pair的端点严格下界减curvature*width²/8，若均>=-b*norm(d)即覆盖整区间。这里负右端乘norm的方向必须保守：采用完整区间不等式或证明下界+ b*norm_lower>=0，不能用更大的norm_upper放宽门槛。

候选方向来自区间端点/中点SAT，只用于寻找，证明全用精确/向外界。面积常向快捷分支须证明c0/c1/c2严格共线且标量全区间正，不能仅检查端点。纯平移c1=c2=0曲率严格0，必须能认证持续等号接触。自适应二分覆盖完整[0,1]，max_*耗尽unresolved。fingerprint包含全部存储输入、拓扑/frame、rho、b和actual endpoints。

结果status certified/violation/unresolved。violation须在一个精确有理时刻证明所有完整SAT方向的最大delta<-b（即平移分离深度超过预算），不能拿一个坏方向或unsigned距离0当穿透；必要时只有unresolved也不能伪造violation。时间采样只产生有证据的违例，不证明安全。初/末/中途面积退化或未知均拒绝。所有证据owned。

## history与诊断

固定pair数K及R=max_anchors_per_pair定义schema：pair_active(K) bool、anchor_active(K,R) bool、anchor_material(K,R,4) FP64（3B+s）、phase_world(K,R,2,3) FP64。精确key/理想feature/fingerprint及clipping证据属于owned diagnostics；当前row按固定材料token字典序排序，unused slots严格0。不把slot当跨帧同一材料点，pair enter/persist/exit/inactive独立于anchor出生/消失，history仅记录不warmstart。签名涵盖canonical拓扑/模型/材料/μ/厚度/预算/迭代设置/schema。

diagnostics至少给pair_keys、两相位anchor/material/feature/normal、理想极点与实际表示误差、initial_closed、P/E/nodal loads/robot requests、真实前后相位state证据、gap/normal/Coulomb/force/moment/power/energy残差、全trial/rebuild/line-search/launch计数、CCD和转换误差。成功计数含未选trial且terminal pre不重复计；失败时沿既有report限制，不伪称公共失败部分launch遥测已支持。

## 预先固定的验收

CPUtol=1e-8，device tol=2e-5；reference_length=reference_speed=1；free absolute max_energy_gain_j/max_angular_momentum_defect_nms=1e-8（CPU）/1e-5（device）。材料变体固定1e-3/1e-3，不为失败调整。position_iterations=velocity_iterations=512，inner各1子步；默认anchor/CCD预算不变。oracle自身用Fraction/math/独立质量/裁剪和原式，不调用生产解法回填期望。

### A 双侧有限夹压

三node每m1，(-1,0,-1),(0,0,2),(1,0,-1)，rho1/8，两个独立Y-prismatic各M2、patchXZ[-4,4]、Y±rho，各真实effort向内2N；nodes g=-Z，初rest。每侧3角点anchor。

两μ=1/.25、h=.25单步：每行p=2h/3；μ1每行t=+Z*h/2，全静；μ.25每行t=+Z*h/6，node vz=-2h/3、dz=-2h²/3，jaw静，terminal0。内力不唯一；全零/共同alpha/同步算法保持对称流形的结果，不称Coulomb唯一性。独立检查全部world/nodal力矩和对称、物理sum，不允许直接初始化这些值。

CPU T=.5，h=1/8,1/16，各μ、每档2次全新重建；device T=.25，h=1/8,1/16，各μ及重建。低μ每步vz=-2t/3，dz=-(t²+t*h)/3，与连续解误差t*h/3，减半h误差减半；高μ静止。改变域/patch顺序及整体刚体变换仍等价，禁止以名字识别jaw。

### B 斜向新接触和厚度力偶

A同triangle/rho，中面初Y1/4，singleYprism M2、patchY0/XZ[-4,4]，node v=(1,-1,0)，robot0，h=.25，μ=.5，无g/effort。initialopen heldt0。

每行heldp1/5；pre nodev=(1,-4/5,0)，robotv=-3/10；nodeXshift1/4、中面Y1/20、robotq=-3/40。k=(-1/16,0,1/16)，T=sumtauX=-3/5；terminal eta=(13/80,1/5,19/80)，tauX=(-29/160,-1/5,-7/32)；post nodevx=(131/160,4/5,25/32)，所有node/robotvy=-3/5。K0=3、Kpre51/20、Kpost1905/1024，terminalΔK=-3531/5120，全步=-1167/1024，Py=-3。全接受路径只平移，底gap=(1-t)/8，须完整CCD certified。

负控制：删除J厚度块必被独立moment/mapping审计拒绝；centroid-only terminal oracle Kpost11913/6400却一角closing，必须被完整流形末速度审计拒绝；不能只看总能量下降。normal/tangent交叉响应、共享节点/robot都不得拆开忽略。

材料变体取μ0、初vx0，edge(0,2) DistanceConstraints rest=2-1/64、compliance1/16，其他同B。h.25 alpha1：独立XPBD Δlambda=-1/192，pre x0+=1/192、x2-=1/192，vx=(1/48,0,-1/48)；每角heldp=eta=1/5，末公共vy=-3/5。初U1/512、preU1/4608，额外Kx1/2304，材料Δ(K+U)=-1/768。检查MaterialPath只pre和真实native交叉调用，不能以无材料替代。

### C 完整内部patch、部分重叠退出再入

同B triangle初Y1/4，但patchXZ[-1/2,1/2]、μ0、初nodev=(0,-1,0)。patch4角均在大三角内部，三角3角全在patch外；完整clipping必须给patch4角，不能只看cloth角点。normal初冲击独立sumP=sumE=3/5、全部末vy=-3/5、robot同速；均匀四角流形对称选择每行3/20，实际材料权重独立由triangle线性方程求。重复反序/canonical/重编号/共享robot控制。

从真实接受状态开始，不手改state。首滑动步给每node Fx=1/h，后续Fx0，使vx1；h1/4及1/8，各走T2。节点/robot公共Y速度-3/5保持，Z不变；X(t)=t。独立triangle不等式与square裁剪记录full/partial/edge/empty；因square z下界-.5，最后重叠在X=4/3，X>4/3必须零残留且exit（不把某个h首次exit位置写死）。反向首步各node Fx=-2/h，此后0，共T1，X回到1，必须重建partial contact、normal/tangent均0；外力功和L账通过。两h及2次重建；device h.25整窗及一次重建。该零载荷滑移只是几何生命周期控制，不能替代A/B非零摩擦和C初非零normal冲击。

### 几何、失败、状态与后端控制

新几何测试覆盖斜面、side/edge/vertex、真实edge-edge、共面重叠/反面、窄patch内部、精确零轴、Q(sqrt(C))符号/负gap排序/近退化/范围溢出、转换位移与理想极点速度；与独立穷举原式或解析对照。失败可显式unknown但不能漏pair。正常主场景必须accepted。

CCD有纯平移持续接触、非恒N的合法分离路径、端点分离中途穿越、面积中途消失、exactFraction时刻、clearance边界、budget1未知及伪fingerprint。故障注入第二/最后outer子步的native、path、terminal、mapping、CCD、close，全部rollback/retryfresh，半构造workspace关闭。FP32提升/一ULP host覆盖拒绝，original输入不变、所有native trial精确计数。

## 文件范围与阶段验收策略

只新增以下聚焦模块，必要公共__init__导出和installed smoke新入口：constraints/triangle_prism_surface.py、constraints/robot_contact_patch.py；geometry/prism_patch_manifold.py、geometry/_prism_exact.py、geometry/_prism_clip.py、geometry/prism_patch_sweep.py；solvers/dynamic_robot_thick_contact.py、solvers/_thick_contact_workspace.py、solvers/_thick_contact_geometry.py、solvers/_thick_contact_replay.py；validation/thick_contact_mapping.py。最多这11个新模块；需其他helper先由主agent裁定，不复制修改旧solver体系。

五新test文件见计划；报告在本阶段SDD。旧offset/surface/point port、participant、kernel、runtime、terminal ledger、材料ledger、integrator、pin、CI和旧tests期望不可改。必要共享修复先报原式RED；不自行弱化旧域或budget。

实施同一集成Task1，一名实现者独占数值/index，内部TDD小步提交；主核对同snapshot证据复用，不重复新全集。完成后一次fresh全阶段review，原实现者定向修复，主闭环。按计划选测host/实际interpret/CUDA，保持阶段一次物理窗/步长/重建/故障；不跑旧两CPU全量/长CUDA/900。新映射实际CPP短程验证一次及源码外wheel由父最终快照工程完成；无新性能结论。远程完整矩阵保持，精确SHA/CI后关闭此切片。

## Builder/main裁定记录（唯一检查已关闭）

R1：已核对旧理想FK会再次除存储joint_axis的精确norm。合法非坐标轴例如存储的(1/sqrt(2),1/sqrt(2),0)引入额外根式，一般树不在同一Q(sqrt(C))。新程序及新standalone几何query均要求整树local prismatic轴为精确坐标单位轴，构造/查询时拒绝非坐标轴；固定frame仍用精确有理规范旋转，允许任意模型合法固定frame和共享树。旧all-prismatic程序支持不变。不建立多根式CAS，也不把rounded axis冒充理想FK。代价是这个切片尚不覆盖新程序的非坐标local轴，后续一般机器人合同需处理；不缩减既有功能或总P6.5目标。
R2：严格最大SAT gap并列且对应clipping合法，优先合格patch-face normal，然后初态同向轴，再canonical。C精确X=4/3时+Y允许零载荷擦边再入，而侧向(3,0,-1)会对vx=-1产生法向closing；两个几何方向都合格，必须固定一种离散法向选择，不能让实现者猜或用history/速度选择。新增二进制可表边界控制：把C的XZ几何统一乘3，patch范围±1.5，triangle XZ=(-3,-3),(0,6),(3,-3)，最后overlap位移X=4。初X4.25、vx=-1、h.25到X4，cap恒定Y接触高度、无相对normal速度/外力，held/terminal均零而末有边界流形；下一步进入partial仍零。旧C两h长窗不替换。代价是明确选择patch-face优先的合法接触法向，不声称法向cone唯一。

R3：固定每行物理接受尺度，与全局候选alpha无关。相位完整world响应对角块Wii由该相位所有节点/机器人质量映射形成；k=n.T*Wii*n/(n.n)>0，kappa=trace(Wii)>0。held用初态W0/n0，terminal用当前W1/n1。V=reference_speed，Lg=min(reference_length,h*V)。几何残差max(0,-delta_lower)/Lg；held互补残差为对pbar/g区间全部端点取max abs(min(k*pbar,g/h))/V；terminal为对etabar/u区间端点取max(0,-u,abs(min(k*etabar,u)))/V。

实际相位normal量是pbar=(P.n0)/norm(n0)、etabar=(E.n1)/norm(n1)，分别精确Fraction/sqrt包围并要求非负；latent p/eta不得替代它。new-contact容量区间为mu*(pbar_interval+etabar_interval)，先各自按本相位normal投影再以Fraction精确相加，不能调用旧combined_capacity(n1,P,E)或先float合计。closed容量分别mu*pbar、mu*etabar。每个actual phase impulse S、相位normal n、真实相位relative velocity U的切向不变量按旧_dynamic_friction_law.invariants原式计算：t²=S.S-(S.n)²/(n.n)，u²=U.U-(U.n)²/(n.n)，w=S.U-(S.n)*(U.n)/(n.n)。独立向外sqrt给tl/th、ul/uh，容量cl/ch；cone=kappa*max(th-cl,0)/V，dissipation=kappa*max_{c in{cl,ch},u in{ul,uh}}abs(w+c*u)/V²，positive_work=kappa*max(w,0)/V²。全部残差<=tol才接受；normal非负/有限/合法几何不是可放宽的残差。全局alpha/L只用于候选，新增零行不能改变这些单行门槛。补n0!=n1（例如正交相位normal）、pbar1/etabar2^-53精确容量以及新增无载荷行不改原行接受门槛的独立控制。未改旧law模块。

三项已由主agent依据原式/现代码和用户持续授权裁定，builder退出，不再派第二次builder。首版局部轴限制、法向选择及接受尺度是本合同组成部分；实施若发现反例须原式RED报父，不得静默换律/增预算。

R4（review修复父裁定）：g_gen 的“当前 anchor”明确取公开port返回的actual FP64 point，将该存储值精确转换为Fraction/Q，按初态选中轴及当前真实有限support feature做精确投影成员判断。它是实际离散端口上的候选更新输入，不替代理想棱柱的几何定义。actual与当前stored-node/N的owned B/s点表示差仍独立要求<=原b；actual在feature内时使用初态有向支持平面signed gap，可保留包围actual/owned两有定义值的向外区间；actual在外严格采用真实有限距离/inactive，不因ideal在内加入候选。禁止actual/ideal集合union、epsilon扩张、吸附、按算例切换表示和free反面轴重命名held侧。理想stored-node/N/FK完整SAT/manifold/CCD及极点速度认证不变，actual成员关系不冒称理想体成员证明；tol、预算、载荷/J/state均不变。

新增diag corner输入triangle((0,0,1),(1,0,0),(1,0,1))、square ±.5、velocity(-.125,0,-.125)、h.25、mu0，存在actual port精确投影在支持点外而当前owned理想点在内的一个ULP差异；先前512轮位置预算失败保留为此输入的候选可达性限制，不声明所有支持域点接触自动收敛。该额外未冻结的普遍收敛断言撤回，改以有界精确finite-inactive控制保留原输入；另用唯一+X patch vertex的运算精确输入做point维数自动响应正常控制。冻结A/B/C与原review side edge自动响应仍必须通过，不以此限制减免。
