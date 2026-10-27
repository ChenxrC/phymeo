# P4 布料与最小体积软体 CPU 设计

承接用户已授权 P3–P5 和[集成大纲](../plans/2026-10-27-integrated-framework-outline.md)。采用 NumPy FP64，复用 P1/P2 场景、状态所有权、历史与全局事务；新设备算法属于 P6。

## 选择与边界

布料保留已有结构/剪切距离 XPBD，薄片质量与固定点不变。体积软体选择线性四面体、集总质量和 St. Venant–Kirchhoff（StVK）超弹性能量，半隐式 Euler；不把边弹簧当体积弹性体。隐式弹性/Neo-Hookean 可改善适用范围但会新增求解与材料政策，本轮不选；仅体积守恒约束不能独立提供所需剪切响应。

依据公开连续介质公式独立实现。[StVK 教学说明](https://learn.physics-simulation.org/examples/FEM_StVK.html)描述形变梯度、Green 应变、第一 Piola 应力及显式积分稳定性；本项目自行实现三维四面体、形变门禁与测试，不复制示例源码。

适用范围明确受限：无塑性/粘弹性/断裂、弯曲、自碰撞或大变形稳健性承诺。当前形变要求 det(F)>0.2 且每个奇异值在 [0.6,1.4]，越界返回失败并恢复整帧；不能为演示成功临时扩大范围。

## 数据与能量

`TetrahedralModel(rest_position, tetrahedra, density, young_modulus, poisson_ratio, fixed, gravity)`：rest_position (n,3) FP64；tetrahedra (m,4) 整数，m≥1、每单元四个不同合法点，拒绝重复无向单元及未引用节点。density>0、E>0、-1<ν<0.5；fixed (n,) 必须布尔；gravity (3,) FP64。数组自有只读，标量拒绝布尔/非有限值。

Dm=[X1-X0,X2-X0,X3-X0]，参考体积 det(Dm)/6>0；拒绝非正方向和退化（|det(Dm)|/max_edge_length³≤1e-12）。预计算自有 Dm⁻¹、volume、λ、μ、集总质量和 inverse_mass。每个节点获各相邻单元 ρV/4，固定节点 inverse_mass=0，但保留物理质量供审计。

Ds=[x1-x0,x2-x0,x3-x0]，F=Ds Dm⁻¹，Egreen=(FᵀF-I)/2。单元能量 V[μ tr(Egreen²)+(λ/2)tr(Egreen)²]，P=F[2μEgreen+λ tr(Egreen)I]。力矩阵 H=-V P Dm⁻ᵀ，节点 1/2/3 取 H 各列，节点 0 为负列和。所有节点的完整内力先计算再施加固定点边界，保留固定点支撑载荷；不得在求能量梯度时删除固定点力。

`deformation_gradient(model, position)`、`elastic_energy(model, position)`、`elastic_forces(model, position)` 为独立可测函数；输入形状、精度和物理域检查不依赖某个 solver 才生效。有限值但溢出/奇异矩阵必须失败，不能输出零力冒充成功。

状态复用 `ParticleState(position, velocity)`，外力复用 `StepInputs(force)`；`tetrahedral_family()` 注册 kind=`tetrahedra`、模型 TetrahedralModel，与普通 particles 域区分。入口和候选均检查形变域、固定点/零固定速度、全部字段与副本。没有材料历史，必须声明 history=none；未来有历史的材料需显式扩展。

## 独立程序与端口

`TetrahedralProgram` predict 从子步初态计算 f_int，v'=v+h(g+inverse_mass(f_ext+f_int))，x'=x+h v'；固定点保持原位置且速度为零。显式求解没有内部非线性迭代，不报告静态弹性平衡“已收敛”；报告积分方程残差及能量、J/奇异值范围。候选越界在提交前拒绝。

`TetrahedralPointPort` 使用节点索引和集总质量提供同一世界点冲量合同，重复节点响应汇总；瞬时冲量不改变位置，候选仍须通过四面体族验收。P5 接触位置修正后的 velocity 重建只计一次；不能重复施加位置等效冲量。

布料以单域 `ClothProgram` 正式复用已有 predictor/project_distances/residual；由统一 SceneStepper 组织子步/预算。P5 适配这些实际算子，不包装旧整帧 step 或悄悄增加时间推进。

## 冻结验证

- 单元 X=(0,0,0),(1,0,0),(0,1,0),(0,0,1)，ρ=6：V=1/6，总质量 1，各节点 1/4。E=1000、ν=0.3。
- 静止、整体平移、任意正交旋转时能量/内力接近零：能量≤1e-20、力∞范数≤1e-10；梯度中心差分 ε=1e-6，在有限仿射形变下 atol=2e-7、rtol=2e-6。完整内力合力/合力矩相对尺度误差≤1e-12。
- 均匀伸缩和剪切有正能量及可独立计算的应力/力；不能仅测体积变化。参考退化、反向、非法 ν/E/密度、重复单元、节点遗漏、形变倒置和正体积但过大剪切均有拒绝测试。
- 最小软块采用 x/z 各 3 点、y 两层（18 节点、24 正方向四面体），尺寸 0.1×0.1×0.1 m，ρ=200 kg/m³、E=500 Pa、ν=0.3；底层 9 节点固定、g=0。小加载力总量 0.05 N 沿 -Y 平分顶层，h=1/4000 s、400 步，总时域 0.1 s；要求存在有限非零压缩、所有单元保持上述域、固定点不动、重复逐元素一致。
- 两节点/多节点冲量、固定点反力、输入不变、末子步失败和完整恢复均验证。集成前独立材料测试必须通过。

记录单元/24/若干小网格的力、能量和 step 观测及稳定域；只记录条件稳定和相关精度限制，不开展积分器/性能优化。
