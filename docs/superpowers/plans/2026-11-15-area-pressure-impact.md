# V06 双自由体法向冲击：首个力—运动闭环

日期2026-11-15，codex/exp；承接[V05有限边缘参考](../../development/area-pressure-finite-support-report-20261115.md)与[总计划V06](2026-11-15-area-pressure-contact-visual-cases.md)。本批只新增独立NumPy轴对称动力学参考；两个物体均有有限质量、由接触冲量改变运动。先隔离法向冲击的因果与能量，再扩展V05俯仰/边缘动力学，不称通用6DOF碰撞或TYX集成。

## 固定物理模型与案例

沿公共+z轴，球A在目标B上方。质量mA=0.5kg、mB=1kg；R=20mm；目标80×80×10mm，顶面zB+5mm。重力/外力/摩擦=0。初态zB=0，zA=27mm（间隙2mm），vA=-0.2m/s，vB=0；窗口0.5s，h=1ms与0.5ms。原两质量、相对速度、间隙和窗口不改。
球沿用V02已声明径向压力p_e(r)=E(1-r/R)，E=100kPa；目标为rigid-pressure表面，但有1kg惯性，不固定它的位置。深度δ=R+zB+5mm-zA；δ<=0空域。δ>0时仍调用旧sphere_plane_case的实际面积/压力积分，固定resolution=1mm，旧源码/预算不改。接触盘必须完全在有限目标顶面内，δ<=6mm；超域明确拒绝，不借无限平面补支撑。这里同轴居中盘的法向合力经过双方COM，物理对称性给零转矩；没有约束其他一般场景的转动，也不声称已实现非中心冲击。

压力耗散律p=p_e max(0,1-d*u)，u=vA-vB为法向分离速度。固定d=0与20s/m两组，不按结果标定。该Hunt–Crossley型压力乘子与[Drake compliant contact模型](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html)同形式，但本批没有执行Drake，也不称其默认时间离散。其参数含义另见[Drake hydroelastic参数](https://drake.mit.edu/doxygen_cxx/group__hydro__params.html)。刚性目标并非用很大E代替。

四条固定轨迹：elastic-h1ms、elastic-h0p5ms、damped-h1ms、damped-h0p5ms，共3000整步。先写manifest，再执行。法向方向/压力场不由输出挑选。球的绝对运动可以仍向下，但应与目标分离；是否反弹以相对速度判断。

## 时间离散与共同发布

采用固定步长kick–drift–kick分裂。每个kick在固定几何位置积分真实p_e*dA，再用中点相对速度求该kick的耗散乘子。记τ=h/2，W=1/mA+1/mB，入kick速度u：
J = τ*F_e*max(0,1-d*u)/(1+τ*F_e*d*W/2)。
vA'=vA+J/mA，vB'=vB-J/mB；u_mid=u+W*J/2。保存factor=max(0,1-d*u_mid)，以J=τ*F_e*factor作为独立方程检查；这不是碰后速度修复。

完整顺序：entry位置第一kick → 用更新后的双方速度漂移h → exit位置第二kick。一步平均接触力定义为(J1+J2)/h，不能把exit瞬时力称为推进全步的力。完整entry/mid-drift/exit状态、两个压力采样位置/积分结果、kick前后速度、J、u_mid/factor、双侧请求/实际动量变化同一StepRecord发布。任一步非法输入/越域先抛错，调用者原immutable state不改变，不发布半步。

d=0时为velocity Verlet。耗散台账每kick D_k=(τ*F_e-J)*u_mid，非负到舍入；这来自实际kick，不另补状态或能量。直接机械能仍独立从双方速度与理想势能计算，另列累计D与E+D-E0的离散误差。末端瞬时载荷和整步平均载荷可同时记录，但必须带相位，不能互相替代。

## 独立参考与预先量纲检查

连续径向场的解析总力F(δ)=πEδ²(1-2δ/(3R))，面积A=π(2Rδ-δ²)，势能U(δ)=πE(δ³/3-δ⁴/(6R))，δ<=0时都为0。解析F/U只用于oracle/诊断，不驱动接触冲量。U'=F。
约化质量μ=mA*mB/(mA+mB)=1/3kg，初始相对动能=μ*0.2²/2=1/150J；无耗散解析冲量2μ*0.2=2/15Ns，最终vA=1/15、vB=-2/15m/s。初始总动量=-0.1kgm/s，总动能0.01J。
U(6mm)>初始相对动能，所以6mm是预留的模型域界，不是可调穿透目标。该深度盘半径<15mm，距40mm支撑半边仍有余量；1mm空间分辨率按旧构造最多22径向×134角向，≤5762三角形，低于8192。0.5mm空间分辨率会超旧资源帽，因此本批预先选1mm并明确空间误差与时间误差分开，不改旧V02的已测网格。

## API与文件所有权

只NEW experiments/area_pressure/impact_reference.py，由原scene作者实现；tests/test_area_pressure_impact.py由原training-review作者独立编写，不读新实现。父负责NEW run_impact_cases.py、impact_viewer.html及本计划/报告。旧reference/friction/finite_support代码和旧测试保持。

冻结API：
- frozen AxialImpactConfig(mass_a_kg=0.5,mass_b_kg=1.0,radius_m=.02,target_half_width_m=.04,target_thickness_m=.01,modulus_pa=1e5,resolution_m=.001,dissipation_s_per_m=0.0,max_indentation_m=.006)。
- frozen AxialPairState(time_s,position_a_m,position_b_m,velocity_a_m_s,velocity_b_m_s)，五个必填float，非bool有限。
- sample_contact(config,state) -> frozen ContactSample：indentation_m（原signed δ）、relative_velocity_m_s、elastic（原PressureIntegral，world plane为当前目标顶面，origin=(0,0,0)）、instantaneous_factor、instantaneous_force_n（非负标量）。分离调用旧0深度空quad；只有正域检查深度/接触盘支持。外径必须<=target_half_width_m，不支持裁掉越界盘。
- step_pair(config,state,dt_s) -> frozen ImpactStep：entry、drifted（第一kick后漂移位置和半步速度，time=entry.time+h）、exit、first、second（两个KickRecord）、impulse_n_s=J1+J2、average_force_n、dissipated_j=D1+D2、to_dict()完整递归。
- frozen KickRecord：sample（该kick实际几何/入kick速度）、duration_s、velocity_a_before、velocity_b_before、velocity_a_after、velocity_b_after、relative_mid_velocity_m_s、factor、impulse_n_s、dissipated_j。
- mechanical_energy_j(config,state)：双方K +上述连续场U；仅诊断，不能用于返回修复状态。
所有数值输入有限非bool，质量/尺度/分辨率/E正，d>=0，0<max_indent<=min(R,target_thickness)，0<h<=.01s，time>=0；数值结果非有限拒绝。构造器验证（包括纯手建非法配置），不偷偷接受fixed/infinite mass。资源帽沿旧mesh。局部标量1D不是rigid body通用接口，qualification在runner中明确AXIAL_TWO_BODY_REFERENCE。

## 执行门与独立测试

原工程门保留：所有状态有限、maxδ<=6mm且未穿过目标厚度；全窗总动量误差<=0.01*0.1=0.001kgm/s；d=0全窗|K+U-E0|<=5%*0.01J；每d两h的峰值整步平均力/总冲量相对差<=10%。四轨迹都必须出现正载荷、压缩转卸载、最终分离且末0.1s全部样本空域/载荷0，不能仅一帧峰值算通过。
额外实现门：每kick双方冲量与保存J一致abs<=1e-12Ns；位置更新和平均力阶段绑定<=1e-12 SI；J>=0，D>=-1e-12J，样本完整作用反作用；d=20最终机械能严格低于初始、累计耗散>0；|E+D-E0|<=5%初始K（d=0同门）。弹性最终J与2/15Ns、最终速度独立oracle相对误差<=5%，damped最终相对分离速度正且小于0.2m/s。不要求任何牛顿机/训练效果。

新测试：分离自由漂移、首次入层、实际p*dA驱动而非解析F、两个有限质量的相反冲量；独立手算kick/漂移/能量；d=0对称弹性/时间反演单步；d>0固定pressure半kick标量方程与无拉力clip；共同位置/速度平移；不修改输入/冻结记录/JSON；第二kick超域的事务拒绝；质量/时间/参数/资源/非有限拒绝。独立测试只短窗/单步，长四轨迹只runner跑一次；不重跑V00–V05已绿组。不以测试数代替物理窗通过。

## 保存、视觉与停止规则

每步保存轻量完整相位状态与两个kick标量载荷/面积/COP/积分点数、源字段参数；在首次正载荷、峰值平均力、最大压入、最后正载荷等预定义事件保存对应StepRecord完整quad/pressure，事件重合可共享文件但逻辑映射保留。全窗step记录不删失败段；manifest先写且固定event规则，不挑漂亮片段。所有实际step通过后可以纯file IO选出事件，不重新求压力。
瞬时压力图只来自对应sample，平均力图明确来自两kick；视频显示真实state.time，不将每帧5秒静态切换冒物理时间。球/目标均由保存状态画出，压力盘/δ/相对速度/平均力/累计冲量/动量与能量图并列。主视频可慢放，标示0.5s真实窗口。
数值只由root唯一串行owner在fresh guard后执行，原agents仅隔离文本。环境/首静态/必要样式/新HOST/四固定窗/媒体顺序，不安装、不调pin、不远程CI、不跑产品solver。允许一次有依据实现修复，模型/参数/门槛不追绿。完整通过或明确否证均写报告；不追加更细h/更多网格循环。完成后按技能小commit/push exp，dev保持。
