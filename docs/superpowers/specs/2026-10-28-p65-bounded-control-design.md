# P6.5 H3a：有限驱动的隐式自由预测控制核心

接续已验收的五个基础切片。此设计只冻结一次子步的有界effort计算，解决[初次设计检查](2026-10-28-p65-design-review.md)中轻腕显式PD的稳定性风险；不冻结完整机器人型号、阶段状态机或接触时间积分。现有整分支审阅已经完成，新控制代码属于独立后续计划，不重新审阅旧修复。

## 语义

外部调用者提供同一快照的q、v、质量矩阵M、偏置b，M与b可以来自实际tyx查询。输入均FP64自有有限数组。控制律在**无接触的自由运动预测**上隐式求值，再以本子步保持不变的effort送给真实参与者；接触修正后的最终状态不能冒称仍满足这个自由预测控制方程。下一子步重新读取已接受状态。目标/限速/相位历史不在本纯函数中，后续需注册进SceneState.history。

对每轴定义非负kp、kd，有限非负effort_limit L；零上限表示禁用该轴。h>0。旋转轴的effort单位N·m、移动轴N，增益单位随轴定义；数组不混作无量纲物理量。

令a=kp*(q_target-q)+kd*v_target+feedforward，D=h*kp+kd。

`M*(u-v)=h*(tau-b)`，`q_free=q+h*u`，`tau=clip(a-D*u,-L,L)`。

选择该隐式自由预测，避免在小惯量下直接把显式PD接到既有半隐式积分器。它不是对接触状态进行速度/位姿裁切，不是无限质量轨迹控制，也不保证含接触、多步、限位的完整机器人稳定。

## 解法与独立判据

M必须对称正定；相对非对称度不超过mass_tolerance后显式取对称平均并记录，禁止裁切特征值。最小特征值必须大于最大值的1e-12；否则明确拒绝。默认mass_tolerance=1e-12，实际tyx测试预先使用2e-5。

D=0的轴先固定tau=clip(a)。其他轴A构成有界严格凸QP。令v_free=v-h*M^-1*b、C=M^-1，固定轴为F：

`H=diag(1/D_A)+h*C_AA`

`rhs=a_A/D_A-v_free_A-h*C_AF*tau_F`

求最小化`0.5*tau_A.T*H*tau_A-rhs.T*tau_A`，约束`-L_A<=tau_A<=L_A`。固定顺序坐标下降，每轴精确裁切一维最小值；全部耦合项保留。原始M方程和原始裁切控制律独立复核，不能只检查迭代改变量。预算耗尽或中间量不可表示抛SolverFailure。

默认effort_tolerance=1e-10（各轴自身effort单位），max_iterations=512；支持配置但本轮测试不根据结果放宽。每轮由完整解重算自由预测速度，`max(abs(tau-clip(a-D*u)))<=effort_tolerance`。原始M方程按`max(1,||M||inf*||u-v||inf,||h*(tau-b)||inf)`归一化，残差<=mass_tolerance。所有effort严格在盒内，零D/零限幅不绕过检查。

## API与数据边界

`solve_bounded_pd(mass, bias, q, velocity, target, target_velocity, kp, kd, effort_limit, h, *, feedforward=None, effort_tolerance=1e-10, mass_tolerance=1e-12, max_iterations=512)`。

返回`BoundedPDResult`：effort、predicted_velocity、predicted_configuration（均FP64自有只读）、iterations、residual_effort、equation_residual、mass_asymmetry。prediction仅为控制计算证据；应用程序必须把effort送入真实动力学参与者并审计其候选，不能把prediction写回物理状态。

本模块主机FP64。真实tyx验证用TyxRobotMath实际查询M/b，再把effort送入TyxRobotParticipant执行predict；验证launch增长、q/v/力矩上限与独立方程，完整关闭资源。没有CPU替换设备状态。暂不提供runtime控制器状态机、IK、速率限制、碰撞或关节限位。

## 验收

- 单轴无饱和解析解、正负饱和、零增益/零力矩/非零偏置与feedforward；输出所有权、原状态不变、非法输入/非有限/非正定/病态/预算/溢出拒绝。
- 非对角2×2质量的独立有界最优解：测试枚举全部自由/上下界活动集作为oracle，不复制坐标下降实现；随机小规模参数与直接控制律/原动力学方程核对。
- 轻惯量标量反例：m=1e-6、kp=2、kd=.06、h=.001，显式更新谱半径约60.97，隐式自由预测约.9677；这只证明该线性标量测试，不能泛化为真实腕/接触稳定性。
- [设计数值探针](../../benchmarks/p65-design-probes.json)另存标量谱半径、候选夹持静力容量及五个平面IK点；它们不是完整仿真或连续无碰撞证明。
- 实际两关节NumPy/tyx interpret/C++/CUDA自由预测与有限驱动验证；真实求解候选仍采用后端响应，不能直接采纳本函数预测。
- 记录小负载延迟/迭代/失败；不优化。新范围独立一次审阅，修复后必要整套/源码外wheel/准确CI，不重复旧基础审阅或P6长实验。

截止仍为2026-10-28 08:00北京时间。完整P6.5尚未达成，到点只交付实际通过的模块，不缩减综合案例门槛。
