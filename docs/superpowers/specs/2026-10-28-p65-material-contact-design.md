# P6.5 初态闭合机器人—距离材料接触设计

状态：用户持续P6.5授权内的下一切片，由实施者冻结实现选择；产品尚未实现。依赖已通过准确249234a CI的MaterialPathLedger，禁止删除旧FreeParticleLedger、改原积分/容差或覆盖native候选。

## 路线及证据

考虑三路：直接换掉旧自由节点门禁并保留纯动能相对free限制（已被反例否定）；引入新守恒材料积分器（同时更改机器人/材料时间合同，本切片不做）；保留原native重放，用材料原式账本与含弹性势能/外功的独立物理预算接触外层（本次实施者选择）。

压缩双节点、c=.01、间距.8、rest1、节点质量各1、Y移动机器人质量2，h=.1：闭合冲量p=4/7，机器人及接触节点vy=-2/7、另一节点vy=6/7。总K≈.489796高于free的.444444，但K+U从2降为6/7。此处旧纯K限制会错误拒绝，不能把它直接用于材料。NumPy/actual interpret/CUDA三步长9个原生探针通过独立解析式，见[候选证据](../../development/p65-material-contact-candidate.md)。

同域材料2秒静置支撑，h=1/16和1/32：仅在|u|<=tol时停止，六个后端/步长窗口均末穿透拒绝；将最终gap一起作为收敛条件，六个窗口全部完成。不得以单步速度通过当作持续接触通过；必须每次完整replay后共同判定全部残差。探针仅normal-only实验，不是生产程序或完整物理认证。

## 接口与支持范围

新增不可变 `MaterialContactBudgets(max_energy_gain_j, max_material_angular_defect_nms)`，两个量均必须显式提供、有限实数且严格正，拒绝bool。单位分别J和kg m²/s（N m s）；它们是每子步的绝对物理误差上限，不能按本次实测误差自动扩大，也不能代替原StepSettings原式/几何/摩擦容差。完整轨迹另有累计门槛。

新增 `InitialRobotMaterialContactProgram(scene, base_program, contacts, *, materials, budgets, robot_domain='robot', registry=None, inner_settings=None, relaxation=1., reference_angular_speed=1.)`。contacts复用RobotNodePlaneContacts；materials是所选粒子域名到DistanceConstraints或None的映射，keys必须恰好等于所有contact绑定选中的粒子域。深拷贝只读；约束顺序保留，indices合法。所选域全部inverse_mass>0，FP64存储，base数学且backend为numpy或现tyx interpret/cpu_cpp/cuda；其余独立旁域原样推进。None/空约束允许作为正常控制，但旧InitialRobotNodeContactProgram的自由节点限制保持不变。

只支持原距离材料路径；base必须产出每个所选域完整material_path证据，不依据具体类型猜测或静默替代。缺记录、参数不符、隐藏后处理或额外接触导致原式不符即失败。材料约束由调用者显式传入；不从可变base私有属性反推合同。预算、materials和base signature均进入problem_signature。继承旧scene/registry/inner scales/history/close合同，构造验证错误与运行SolverFailure分开。

## 时间、载荷和迭代

保持旧步初共点求积：初态节点和局部无限平面闭合，冻结世界点、n0、J0、M0/b0/外载荷；几何和COM使用现有精确仿射门禁。节点p与机器人-sum(J0.T p)在原输入通道先精确合并，再通过SceneLoadReplay各施加一次。没有第二次kick、位置补丁或隐藏稳定化；每次trial从同一state/history/input完整重算，任何未接受trial均不持久化。

每次重放同时执行FrozenRobotLedger和所有选中域MaterialPathLedger.require（真实participant diagnostics与全局sweep数），任何内层accepted不等于外层收敛。接触速度采用v_node1-J0*v_robot1，最终真实材料速度J1*v1另名记录；不能混称。法向/摩擦圆盘与contact_residual沿用精确世界p投影合同，normal/friction/original-input/材料/最终gap都在原StepSettings容差内才接受。

GS预条件仍用wI+J0*M0^-1*J0.T，明确它只是free response预条件，不是材料精确切线；每次更新后全域native重放保留同域交叉响应。没有收敛保证，超迭代预算显式拒绝，不能减少残差门禁或增大tol。**不能仅因当前接触速度小于tol就停止；最终几何与下述物理预算也必须一起通过。** 为避免容差内的单向速度偏差在持续窗口累积，首个合格候选后可在原position_iterations预算内继续GS，直到法向/摩擦残差不再改善，保留整个最佳且原式/物理均合格的native trial；诊断计入全部重放。该迭代细化不引入gap反馈、位置稳定化、改tol或替换native状态，末分离仍按原规则允许。原始储存单位的最终gap保守下界>=-tol*min(reference_length,h*reference_speed)，初态正间隙/穿透超限拒绝。不是自动新接触/CCD，旋转末穿透仍可拒绝。

## 独立物理预算

只计每个选中域一次，不按contact行累加同一域的能量。所有下述检查直接由原始/候选存储值作Fraction与sqrt包络，不能用报告中已舍入到float的能量/动量数值代替门禁。

1. 接触求积功sum(p·(vnode1-J0*vrobot1))保持旧耗散门禁（tol*reference_speed*sum(abs(world_p))）；共点反作用矩误差保持旧门禁。没有用数值位置修正伪装接触力。
2. 冻结模型总能量 `E=0.5*v_robot.T*M0*v_robot + sum(K_nodes+U_distance+V_gravity_nodes)`。软边U=(length-rest)^2/(2c)用上下包络，硬边无弹性势能项且末本构必须通过。原非接触输入端点功 `W0=sum(F_original·dx_nodes)+sum((free_integral_robot/h)·dq_robot)`。free_integral_robot含原effort/外wrench/bias；采用真实dq，不拿接触有效输入把接触能量源从账中消掉。定义 `B_E=max(E1_upper-E0_lower-W0,0)`，要求B_E<=max_energy_gain_j。这项是冻结M0与驱动/偏置功的离散预算，不是一般多关节M(q1)真实世界能量守恒证明；诊断必须明确命名frozen_model_energy。
3. 每个粒子域实际角动量变化减去步初外冲量矩：`D_L=sum(x1×m v1-x0×m v0-x0×[h*(m*g+F_original)+requested_p])`。requested_p是合并后的原始接触请求，不是已舍入输入差。按各域L1范数求和（不让域间相消），要求<=max_material_angular_defect_nms。这项捕获材料离散角动量与设备舍入，不能把已知损失抵消后宣称守恒；不是机器人基座反力的通用封闭世界角动量证明。独立单转子/无基座该轴力矩测试另查真实整步Lz。
4. 两个预算形成StageMetrics时用`tol*positive_bound/declared_budget`保守上界，仅为接口归一：物理上限本身固定，原tol不扩大。诊断同时输出原量、单位、上限和scope，避免把归一值误当J或N。

正冲量后允许末分离，和旧步初冲量求积一致。预算可能拒绝材料原式正确但物理漂移过大的结果；正确做法是调用者使用更小步长/新算法，而非自动提高预算。fullP6.5后续仍需一般机器人能量/基座反作用、连续几何及全过程误差门槛。

## 测试冻结

- 上述压缩边三h解析p/双方速度/位置/总Py/K+U，明确h=.1时旧纯K门槛错误；same-domain材料不能用旁域替代。零/重复请求与整域合并独立检查。
- 无接触冲量但绑定初态闭合Z平面，XY旋转硬边x=+-1、v=+-Y：angular budget=.001，h=.1拒绝、h=.025接受（已知损失2-2/sqrt(1+h²)）；预算固定，不按观测调大。另用不相容硬边初态产生正能量变化，energy budget=1e-6应拒绝，尽管材料原式通过。
- 2秒支撑：gY=-1、c=1/256、rest1、初y=(0,255/256)、机器人effort=2、质量2；p每步=2h，状态静止。h=1/16、1/32；累计冲量误差<=1e-3Ns、最大位置/速度偏离<=1e-4、总能量-外功正累计<=1e-4J；原式tol NumPy1e-8/tyx2e-5，budgets分别1e-6J和.001Nms。速度-only停止负例必须保留，正例联合全部残差。
- 双节点同平面、x=+-0.4、rest1、c=1、gY=-1、robot effort2、两个contact，mu1静止（每点|pt|=.2h），mu.1向外滑动。共享域/祖先反作用只计一次，交换contact顺序一致。T2与两h；低mu的离散独立递推为s1=(s0+h*((1-d0)-mu))/(1+2h²)、d1=d0+2h*s1（T<=2尚未反向）。与原生轨迹误差<=2e-4m；连续参照d=.9-.1*cos(sqrt(2)*t)，细步最大误差<=.6粗步，粗步误差<=.02m；静态最大偏离<=1e-4。对比目的是量化一阶误差，不宣称全部网格收敛。
- 非对称小三角/单转子、同节点多平面/共享祖先多点：独立输入、线/角动量/材料能量外功与Coulomb检查，不能只复用被测solver residual当oracle。base缺路径、材料参数/compute metadata/原始请求污染应拒绝，旧free程序仍拒绝有材料域。
- 全局事务：后子步材料证据损坏、物理预算拒绝、close失败时state/history完整恢复；retry=fresh；base所有权和关闭一次，零/空域、owned诊断与配置别名。
- 实际interpret/CUDA、一个有界C++小例（TYX-003不重跑压力）、全部核心/设备/wheel/uvpip/唯一fresh最终review与TDD修复；所有数值退出后三模式各3x100串行测量，不做优化。

## 代码组织留给builder的边界

公共类建议`solvers/initial_robot_material_contact.py`，精确物理预算helper可拆`solvers/_material_contact_balance.py`；tests分别core/tyx。可抽取旧InitialPairWorkspace的几何/接触原式共用方法或加私有hook，避免复制整套生命周期，但旧free程序行为、诊断字段、门禁和回归必须不变。不能以复用为由删除FreeParticleLedger或旧能量规则；新物理规则仅属于新程序。诊断命名initial_robot_material_contact，包含材料报告、物理预算、原始p/平均力/请求积分、J0/J1、末gap、replay与实际执行计数，均owned。此设计不授权P7/P8、合并主分支或发布包。

Builder check：material_contact_design_check 返回“nothing important left”。没有待用户决定的常规实现项；代码实施尚未开始，按对应TDD计划推进。
