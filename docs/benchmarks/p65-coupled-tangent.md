# A3：全耦合切向弹簧–Coulomb 核

基线 ff6efeb，HOST FP64，首批只有新 `solvers/coupled_tangent.py`、两个 `test_coupled_tangent*.py` 和本报告。旧 smooth friction、scalar return_traction、MaterialStick*、普通/机器人场景与 A2 mimic 100 g 滑移 RAW1 全不改。下列设计与输入在首次执行前冻结；本批真实执行结果见末节，最终静态验证已通过，父验收见下方补充。

现有 robot `sampled_box_contact.sample_loads` 与普通 `collision_loads._host_law` 都是速度正则滑动摩擦，零切速给零切力。仓内 `_material_stick_law.return_traction` 和 `deformable_stiction` 已有隐式弹性返回、塑性/隐式耗散和卸载储能记录，但它们的标量 mobility 不能替代机器人/刚体多站点完整耦合。`solve_friction_disk` 的有界二维数值解可以复用为全系统的块优化；该调用返回的 energy_change 在本用途仅是数值目标差，不能冒称真实动能差。普通 ContactHistory 是事件观测账本，不是切向弹性状态。

**API。** `solve_coupled_tangent(elastic_displacement_m, free_displacement_m, mobility_per_kg, dt_s, normal_force_n, *, stiffness_n_per_m, friction, tolerance=1e-10, max_iterations=256)`：位移 `(n,2)` m，W `(2n,2n)` kg⁻¹，N/kt/μ分别 `(n,)` N、N/m、无量纲；1≤n≤128，1≤迭代上限≤4096。等静/动摩擦系数是本次明确子类。W有限、对称误差≤1e-12相对尺度且PSD（仅允许有界舍入负特征，不裁剪物理W）；kt严格正，因此冗余站点仍有SPD目标。所有输入不改，输出owned不可写。

令 `A=diag(1/kt)+h²W`，`r=ξ_old+d_free`，解 `min 0.5 fᵀAf+rᵀf`，逐站点 `||f_i||≤μ_i N_i`。全无约束解合锥可直接返回；否则带全部off-block的有界块迭代，最后必须全KKT收敛，无收敛就拒绝。`d=d_free+h²Wf`，`ξ_new=−f/kt`，`plastic=ξ_old+d−ξ_new`。首批不制造状态/几何tracker；父/B负责material IDs、共转frame、候选history及与物理状态同一次发布。

结果字段冻结为 `force_n`、`predicted_displacement_m`、`elastic_displacement_m`、`plastic_increment_m`、`work_j`、`stored_energy_j`、`plastic_dissipation_j`、`implicit_dissipation_j`、`released_energy_j`、`energy_residual_j`、`regime`、`iterations`、`residual`、`kkt_residual_n`，另保有审核所需的不可变 `entry_stored_energy_j`。功用 `f·d`，弹性能 `.5 kt ||ξ||²`，塑性损失 `−f·plastic`，隐式损失 `.5 kt ||ξ_new−ξ_old||²`。要求逐站点 `work+Enew−Eold+plastic+implicit+released=0`；弹簧卸载可做正功，不能用每步摩擦功≤0的假门。N=0为open，μ=0且N>0为frictionless，两者力/新strain都为0，旧储能只计一次released，不再重复计implicit。

`result.verify_motion(actual_displacement_m, *, tolerance_m)` 是纯审核：caller必须给有限正的绝对m界，无默认；返回 accepted、actual_displacement_m、discrepancy_m、maximum_error_m、work_j、energy_residual_j，绝不改物理状态或history。核预测的hJv不自动等于旋转刚体exp-map后的真实材料点位移；真实转动留出须显示这一差异并拒绝虚假通过。任何将来场景适配都须处理实际accepted增量与全相互作用，不能直接把此核插进原先逐域独立drift而漏耦合。

**预先固定用例。** 两重复站点 `W_x=[[1,1],[1,1]],h=.1,kt=100,d_free=.03,N=4,μ=.5` 给每侧−1 N、实际预测增量.01 m，独立每点解会错为−1.5 N。单站点与旧同μ scalar函数独立比对；各向异性 `W=[[2,1],[1,2]],h=1/8,kt=64,d_free=(1/64,−1/32)` 精确f=(−5/8,7/8) N，功−37/2048 J、储能37/4096 J。另有预先构造完整KKT的滑/粘混合、站点重排和切向坐标90°旋转、减N回弹、N0/μ0一次释放、迭代耗尽/非法W/129站点/输入及输出所有权。

实际运动测试将用原公有刚体、粒子推进器与RobotPointPort：两体等反合力/动量和完整交叉W；独立粒子64步恒载至非零静牵引、8步卸载及无旧strain重入；RobotPointPort共享祖先实际响应；刚体偏心力矩产生真实exp-map与线性预测不等的拒绝留出。它们是已知法向容量下的切向子问题，不能称作新场景已自动检测box/sphere/finite-plane/robot-pad接触，完整生产适配是后续独立能力。

**Newton 1.7 本机只读观察。** 安装元数据确认1.7.0。semi_implicit/kernels_contact.py 的 eval_body_contact 仍以 norm_huber 和 min(kf*speed,μN)算速度摩擦；XPBD kernels.py 在当前两侧局部接触点的真实位置差上算friction_delta，限于增量normal multiplier，且明确避免重复加动态速度；不是本弹性history律。VBD rigid_vbd_kernels.py 有完整两体切向Delassus块、Coulomb盘和迭代切向dual，solver_vbd.py 的 rigid_contact_history（默认False）在 latest matching 下恢复投影切向乘子作数值 warm-start；sticky matching 的切向记忆由碰撞管线重放的材料锚点表示（本机第544–563行）。两者均不能未经同律核验就等同本核的材料弹性储能回执。MuJoCo原生约束路径另有锥/solver语义。本批先不加原生跨引擎跑，未来同输入会分别报告模型差异。

运动输入补充冻结：两自由刚体 m=(2,3) kg，两个并联COM材料端口，h=.1 s、各kt=120、N=4、μ=.5，外载±3 N，完整W每块为(1/2+1/3)I，首步每站−1 N，真实两体速度(−.05,1/30)m/s；另加共速boost和90°共同变换。粒子质量1 kg、h=1/16、kt256、N4、μ.5，64步外载1 N到静态牵引；2步3 N触发滑动，8步N=0保留真实速度而非清零，随后重入；每步审核实际位移与动能/外功/弹性能/塑性与两种离散损失。RobotPointPort用1 kg普通滑轨上的两个地址求全W，再施原公开点冲量，检验真实v而不宣称瞬时接口已drift。转动留出 m=1、Iz=1、r=(1,−1,0)，入步角速度0、第一接受末态角速度1/32 rad/s、h=1/8；真实姿态用原RigidSemiImplicitSolver，材料点exp-map与hJv的差必须在verify_motion中拒绝，保入态原数组。

## 父最终补充（2026-11-06）

27项功能验证之后，父Ruff通过，三个Python文件格式化前后raw AST同，最终format通过，style提交d4ebe32正常合入8fd15d3。父另完成公开A+B组合193个真实步与源码外wheel验收，边界仍是给定法向载荷的平移模型；详见[本批集成实测](p65-coupled-tangent-loaded-batch3.md)。以下保留作者交接时的原始证据状态，不代表静态或包外仍待验。

## 本批实际结果与范围

父在唯一串行执行槽绑定候选 `19751a562f1983dcd74544d08f3541d982224ea1`，作者没有执行项目命令。原始证据位于父 `artifacts/p65-postnight-batch3-20261105-01`：

| 作业 | 实测 | 结论 |
|---|---|---|
| `a3-coupled-red` | 1 failed / 0.93 s / RAW1 | 首节点确因缺少 `phymeo.solvers.coupled_tangent` 失败，保留原收据 |
| `a3-coupled-host` | 27 passed / 1.16 s / RAW0 | 两个完整新 HOST 专项通过，无数值改门或追加探针 |

27项包含22项核/边界用例和5项实际 HOST 运动用例（两体共同变换为两个参数）。实际粒子窗口连续走75步：64步恒载、2步更大外力起滑、8步卸载保持真实速度、1步无旧弹簧重入。两刚体响应及RobotPointPort均通过其各自冻结断言；前者有真实一次推进，后者只检验固定q下的真实全雅可比/冲量响应。没有新增 Newton 或 TYX 切向核执行，不将原CPU刚体推进称作原生TYX。

偏心转动的真实姿态/点位移与线性预测不等，`verify_motion` 按冻结绝对位移门拒绝；这是能力边界的成功验证，不是一般旋转接触长窗已经通过。该方法的 `accepted` 仅表示位移误差门，适配器仍须独立审核它返回的实际功与能量残差，再将真实物理候选和材料历史共同发布。

独立作者对 `19751a5` 的只读交叉审阅未发现阻塞项，已核全交叉W/全KKT、卸载及正功能量、真实运动拒绝和有限资源。本报告把角速度文字明确为入步0、第一接受末态1/32；测试原输入/期望不变。静态检查和父 A+B 组合、公共导出/包外验收另由父记录，当前不预先宣称通过。

下一能力是有界几何适配：真实法向/外载自由预测、全切向联解、各参与者同一步共同推进、实际材料位移/功审核和事务发布。当前没有自动启用 RobotScene 静摩擦，没有修复或重跑 A2 mimic 100 g 原失败窗；box/sphere/有限台面/pad 的几何身份也不由本核自行推断。