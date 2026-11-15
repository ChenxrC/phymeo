# V04 分布摩擦与扭转：固定实验批次

日期2026-11-15；codex/exp基线59ff375d。状态：执行前预登记。
[总案例计划](2026-11-15-area-pressure-contact-visual-cases.md) · [上一批V03报告](../../development/area-pressure-newton-reference-report-20261115.md)。

## 范围与物理定义

只做给定压力、给定双方点速度下的面积牵引积分；不更新状态，不求静摩擦，不调用Newton/Drake/TYX或共同动力学。V03外部精度FAIL保持，本批不重复其数值格。T-smoke训练另线保持。

法向n为作用在A上的单位法向，局部相对切向速度 vt=(I-n nT)(vA-vB)。
- 充分滑动库仑：Tt=-mu*p*vt/|vt|；在vt=0选Tt=0，仅为该单值参考的约定，不代表静摩擦解。
- 正则化：Tt=-mu*p*vt/sqrt(|vt|²+vs²)，固定vs=0.001m/s；不按案例调参。此律是显式选择的平滑参考，不声称等同Drake/Newton默认摩擦实现。
- 点力dFt=Tt*dA，关于共同原点的点矩=(x-origin)×dFt；B在同点收到相反力。摩擦功率为sum(dFt·(vA-vB))，应非正。
- 法向压力用原integrate_pressure验证与积分。法向、摩擦、总力/矩分别保存；不把纯转动的零总切向力误当零摩擦扭矩。

理论依据：[Drake干摩擦建模](https://drake.mit.edu/doxygen_cxx/group__friction__model.html)给出压力层面的库仑牵引与最大耗散方向。本批的平滑律及vs是自己的固定实验选择，没有移植或运行Drake摩擦求解器。

## 固定输入与案例

R=0.01m，p=2000Pa，mu=0.5，z=0，n=+ez；A绕共同原点以omega=±1rad/s转动，B静止。
单盘中心(0,0)。双盘holdout中心(-0.015,0)、(+0.015,0)，**均绕共同原点转动，不各自绕盘心自转**。
三档(rings,sectors)=(2,16)/(4,32)/(8,64)，环形三角剖分、原三个二阶三角积分点；真实polygon面积，禁止权重归一补回圆面积。
单/双盘×两律×正反转×三档=24格，另单/双盘最细正则化omega=0共2格，总26格。
双盘各点速度用其共同原点力臂，完整保留两盘不同的合力和力矩贡献。

独立参考：
- 单盘Coulomb：Fn=p*pi*R²，Mz=-sign(omega)*(2/3)*mu*Fn*R。
- 单盘正则化：Mz=-2*pi*mu*p*omega*integral_0^R[r³/sqrt(omega²*r²+vs²)]dr。
- 双盘：在每盘局部极坐标(r,theta)生成独立64点径向Gauss×256等距中点角积分，位置加原盘心后再计算全局力臂/滑速/点矩；不套用两倍单盘公式。与三角网格/积分点不共享离散。
- 单盘正则化独立64点Gauss；零omega各摩擦量为0，法向载荷仍存在。

## 判据与停止规则

最细网格法向载荷及扭矩相对独立参考误差<=5%；最后两档扭矩变化<=5%。三档全部保存，粗档不冒称通过最细门。
每点切向性、Coulomb上界、反向转动变号、相同点力作用反作用和共同原点矩闭合、摩擦功率<=0；有限数/错误输入/容量/零压/零mu/零滑速以及刚体坐标变换由新HOST测试检查。
固定功率-扭矩恒等 P=omega*Mz（本批vB=0纯转动），误差<=1e-12W；总切向力<=1e-10N，对称工况非预置消零。
零omega不做扭矩相对误差；不以零速度未解静摩擦作为缺陷。最细双盘扭矩应超过同律两倍单盘自转扭矩，独立oracle给数值门而非只判断方向。
每点Coulomb/切向性等FP64代数绝对门1e-10（SI），共同力/矩闭合1e-12（SI）。
遇失败保留完整结果，最多一次有依据修复；不改vs/载荷/质量/网格档/门槛追绿。

## 作者与执行

复用两个原agent，隔离原worktree、保留既有untracked。scene作者只新增friction_reference.py；training-review作者只新增test_area_pressure_friction.py，后者依据本API/独立oracle编写。父写runner、viewer、报告。
所有import/数值/pytest/lint/formatter/视频编码由父唯一串行owner，fresh guard；作者只文本，不创建新助手/分支。完成后小提交到exp，dev不变；既有V00–V03不重跑。

接口：friction_reference.disk_quadrature(radius_m=.01,centers_xy=((0.,0.),),*,rings=8,sectors=64)；integrate_friction(quadrature,pressure_pa,*,velocity_a_world,velocity_b_world,mu=.5,law="regularized",slip_speed_m_s=.001,origin_world=(0.,0.,0.)) -> frozen FrictionIntegral。
字段qualification/law/mu/slip_speed_m_s/normal(PressureIntegral)，velocity_a_world/velocity_b_world/tangential_velocity_world/tangential_traction_pa，first/second_friction_force_per_point_n，first/second_friction_force_n，first/second_friction_moment_nm，first/second_total_force_n，first/second_total_moment_nm，friction_power_per_point_w/friction_power_w，to_dict()完整可JSON。
disk最多两个互不重叠盘，rings/sectors是正整数且sectors>=8；总mesh沿原8192 triangles/8193 vertices/24576 points硬限。通用积分器可消费原合法曲面quad与逐点速度，不限圆盘；无静摩擦历史/状态更新。

## 解释性可视化

俯视单/双盘，以箭头区分规定滑动与反向摩擦；固定色标显示牵引大小。并列显示“切向合力接近0，但阻转扭矩不为0”；双盘突出共同转轴与力臂。侧面示意说明A压在B上且压力是规定输入。不画伪自由动力学轨迹，视频按案例讲解且仅消费保存数据。
下一步仍为V05有限边缘/接触启闭，再V06共同状态/载荷的动力学；本批不能作为机器人支撑通过。
