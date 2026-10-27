# P6.5 H2b：固定构型联合接触冲量

接续H1c多域响应图与H2a局部摩擦。这里冻结一个**固定构型、已闭合接触、零恢复系数**的瞬时问题；不冻结完整场景的非线性位置load撤回、CCD响应或时间积分。没有位置修正乘子，因此本接口也不接收lambda/h，不能叠加第二份位置冲量。

## 输入与语义

`solve_contact_impulses(graph, normals, mu_static, mu_dynamic, *, velocity_tolerance=1e-10, response_tolerance=1e-12, position_tolerance=1e-9, max_iterations=200)`。

- graph为H1c算子，每边左右端当前世界点必须相同至position_tolerance米；不满足则ValueError。调用方负责真实接触侧与几何有效性，本解不替代检测器。
- normals为(k,3) FP64单位法向，从右端指向左端；相对速度为v_left-v_right，因此法向速度>=0表示分离。法向单位长度误差<=1e-12，验证后重新单位化以形成正交基。
- μs/μd各为(k,) FP64有限非负数组且μd<=μs；所有边共用给定绝对速度容差，单位m/s；velocity_tolerance在(0,1e-3]。position_tolerance有限非负。max_iterations正整数，bool拒绝。
- response_tolerance为响应对称/半正定检查的无量纲容差，在(0,1e-3]。NumPy默认1e-12；实际tyx FP32响应测试**预先指定2e-5**，速度验收也2e-5m/s，不事后根据成功与否改门槛。
- 无边可返回空冲量与各域零冲量候选；每个有边局部法向响应必须>1e-12*全矩阵最大绝对值，切向2×2块须符合H2a正定条件。不支持的退化有效质量明确SolverFailure，不能跳过该接触。

## 算法

以最不平行的坐标轴确定每个法向的两个切向，建立稳定右手正交基Q=(n,t1,t2)。在这些基下逐个单位冲量查询**完整graph.response**，得到3k×3k响应W；保留所有跨边、共享布和共享关节交叉项。记录原始W的最大相对非对称度；超response_tolerance拒绝。允许范围内显式使用Wsym=(W/2+W.T/2)迭代，记录此处理而不隐瞒；最小特征值不得小于-response_tolerance*最大绝对特征值，不做特征值截断或对角质量替代。

从总冲量p=0开始，按输入边序逐块Gauss–Seidel：

1. 当前u=u_free+Wsym*p，法向更新pn'=max(0,pn-un/Wnn)，将该增量作用到全部u。
2. 去掉本边旧切向贡献得到free_t=u_t-Ktt*pt，调用H2a解(Ktt,free_t,pn',μs,μd)，替换该边总切向冲量，并更新全部u。
3. 整轮后独立检查全图法向互补、静/动摩擦和能量。达到判据后，用graph.response对最终世界总冲量**重新实际计算响应**，再做一次相同的物理验收；候选状态也只从原快照施加该总冲量一次。

允许多接触不收敛；预算耗尽必须SolverFailure。局部解成功或某轮载荷变化小，都不能替代全图验收。返回前不公开任何部分域状态；最后域候选验证失败沿用图的原子返回语义。

## 独立返回判据

每边以最终实际相对速度u（接触局部基）及p检查，所有残差统一转为m/s：

- pn>=0；法向投影残差`abs(pn-max(0,pn-un/Wnn))*Wnn`与负法向速度都<=velocity_tolerance，等价表达零恢复Signorini条件。
- 若切向速度模<=velocity_tolerance，要求冲量处于静摩擦圆盘；锥超额乘以||Ktt||2转为速度残差。
- 若切向速度超过容差，要求||pt||=μd*pn（半径误差乘||Ktt||2），并且切向冲量与最终切向速度反向。非零半径的方向残差为`||u_t+||u_t||*pt/||pt||||`。μd*pn=0时允许零pt、非零滑动，不假称黏着。
- 所有量有限。实际端口响应的能量变化`sum(p_world*(u_free_world+response(p_world)/2))`不得大于`velocity_tolerance*sum(||p_world||)+1e-12J`；保存真实带符号能量，不改写状态/数据。
- 与实际响应不符则继续迭代或最终拒绝，不用Wsym的预测覆盖实际响应。返回记录response_tolerance、实际非对称度、速度残差与迭代数。

`ContactImpulseResult`包含states（只读域映射、各状态自有）、world_impulses(k,3)、relative_velocity_after(k,3)、contact_basis(k,3,3)、normal_impulses(k)、tangent_impulses(k,2)、regimes(tuple)、iterations、residual_mps、energy_change、response_asymmetry及response_tolerance。数组自有只读。返回的接触冲量是整个局部问题的总值，而非再叠加一次的迭代增量。

### 审阅修复：实际候选审计扩展

response成功不保证大基准速度上的实际更新可表示。认证求解要求各PointPort另提供`candidate_audit(state) -> PointCandidateAudit(initial_velocities, candidate_velocities, energy_change)`，两个速度数组均为世界坐标(m,3) FP64自有只读；energy_change为该物理状态域全部自由度的真实动能差，不是各挂接点质量能量之和。原有瞬时端口查询/交换仍可独立使用，无审计扩展的端口不能参与认证的联合接触求解。

粒子/软体按节点质量，刚体按世界姿态下的惯量，机器人按同构型完整质量矩阵与点Jacobian独立核对。tyx候选保持原有设备路径，主机FP64审计仅接受/拒绝，不提供状态或响应的替换答案。重心端口映射两个速度数组，域能量原样传递且只计一次。

每域分别核对原速和候选速度与该域实际响应，欧氏误差不得超velocity_tolerance；候选动能差与该域冲量做功的误差不得超`velocity_tolerance*sum(norm(domain_point_impulse))+1e-12J`。各域能量允许物理转移，但全图仍满足原能量不增门禁。全图再次核对速度/能量及Signorini/摩擦。两个域相同的错误不能在相对速度中抵消；未挂接自由度也计入能量。

任何候选不符立即SolverFailure，所有部分候选保留局部、原快照不变。返回的relative_velocity_after与energy_change取实际候选审计结果。空接触也验证零冲量候选，无边不能绕过验证。新门禁不放宽之前的物理门槛；明确可表示范围见[NUM-004](../../legacy-issues/NUM-004-candidate-representability.md)。

## 验收与范围

1. 单粒子闭合静态平面：无摩擦速度投影解析、分离零冲量、斜滑动静/动边界、能量不增。
2. 两个有限质量水平移动关节夹住一个有质量节点：已共点、夹爪有相向初速度、节点有向下初速度；真实动力学端口求解法向闭合与切向支撑。μ足够时节点竖直速度消除，低μ时保留可解释下滑。关节基座吸收竖直反力，独立核对广义冲量和支持反力/能量，不能要求开放系统总动量守恒。
3. 共享布/共享关节的W交叉项不能删除；全图残差、低迭代预算失败、错误法向/shape/域位置/系数、最后域失败无部分候选。
4. 固定构型实际tyx两机器人＋布域：先predict/prepare_velocity，再通过真实速度冲量形成相向夹持和向下初速度，q保持不变，组装图求解。三模式逐域实际launch增长，实际响应误差门槛如上。
5. 反作用与最终总冲量只计一次；输入状态与历史不变；记录小负载性能/迭代/失败，禁止为了收敛放宽门槛。

这些验证只是接触冲量小例，**不是**夹爪几何抓住布、持续承重、A/B交接或整场景事务。布面/厚度/力偶、离散到连续路径、非线性位置账本、弯曲/自碰撞、控制、完整重复与Three.js仍是P6.5未完成目标。
