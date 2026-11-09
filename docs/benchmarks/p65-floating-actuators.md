# 浮根关节反射惯量与黏性阻尼：作者交接

当前状态：HOST与共享场景执行器模型已验证；见本文末尾 parent validation。下述 AUTHORING_ONLY 是原作者交接历史。

状态：**AUTHORING_ONLY / UNEXECUTED**。基于父 `666dd66`，本分支仅做源码、手推和 Git 检查；没有 import、collection、pytest、数值 probe、lint、formatter、build 或安装结果。父是唯一项目执行 owner。

## 模型与接口

`FloatingRobotModel` 新增两个 keyword-only 字段，原 positional 构造及默认行为保留：

- `joint_armature=None`：构造成长度 N 的 FP64 零数组；铰链单位 kg·m²，移动关节单位 kg。表示已经折算到关节坐标的附加惯量；传动比平方应由输入方折算。
- `joint_damping=None`：同样构造成长度 N 的 FP64 零数组；线性黏性阻力的广义力为 `-d*qdot`，单位 N·m·s/rad 或 N·s/m。

两个数组均要求有限、非负、精确物理关节数和 FP64，bytes-backed 只读且隔离原输入。零关节根只有空向量，不把根的6个速度作为标量执行器槽。

连续方程为 `M = M_body + diag(0_6,a)`、`bias = bias_body + (0_6,d*qdot)`，不改 root/child COM 质量、真实刚体姿态、惯量主轴或力/力矩映射。整个有交叉项的 M 仍共同求解。

这是有效关节惯量模型，不是额外显式 rotor body。能量增加 `0.5*sum(a*qdot²)`；不能将只从刚体 COM/spin 重建的动量冒称已包含完整电机转子的空间动量。需要真实转子/齿轮角动量时应另建实体机械模型。

## 步进与能量

`floating_mass_matrix(model,state)` 返回物理 M。`floating_dynamics_terms` 的 acceleration 仍是该连续方程在入口的实际加速度。

非零线性阻尼使用隐式更新：

```text
(M+hD) delta_v = h (applied - bias_body - D v_old)
v_new = v_old + delta_v
```

入口 M/惯性偏置保持，黏性阻尼取 v_new。全零 D 保留原 `v_old+h*acceleration` 路径，根 COM 漂移和姿态更新仍使用原函数。

新增模块接口：

- `floating_step_mass_matrix(model,state,dt)`：返回入口步进的 M+hD。
- `floating_kinetic_energy(model,state)`：使用物理 M 的完整有效动能，包含反射惯量项；不能将 M+hD 当动能度量。

定 M 的无载解析病例满足：

```text
K_old-K_new = h*v_new.T*D*v_new + 0.5*delta_v.T*M*delta_v
```

这是所选解析病例的离散恒等式，不是全部非线性接触轨迹的无条件能量定理。纯瞬时末端冲量仍用物理 M；持续力经过阻尼步进时使用相应 M+hD。父需据相位接入，不能把两者混用。

## MJCF 首域

导入器解析主默认类、命名/嵌套默认类、body childclass 与 joint class，显式 joint 属性覆盖默认值。仅接收 scalar armature/damping；根 freejoint 不继承这些标量关节槽。几何默认类的原行为保留。

非零 frictionloss、stiffness、ref，freejoint actuator 项，以及 motor 上另加的 armature/damping 都明确拒绝；不将干摩擦改名成黏性阻尼。首域仍一个 unit-gear 对称 torque motor 对应一个 hinge，拒绝重复或遗漏 motor 映射。

原 `tests/test_floating_mjcf.py` 中将 armature/damping 正值当拒绝的两个参数身份，因新能力已过时，父应调整；本分支遵守所有权没有改该旧测试。

## 已作者的验证

新 `tests/test_floating_actuators.py` 覆盖：

- 铰链和移动关节的自由根/关节完整2×2耦合解析 M 与连续加速度；
- 反射惯量只增加对应 joint 对角项，COM/几何不变，动能包含真实附加项；
- 解析隐式阻尼衰减、正耗散/离散积分项、所选共轴病例的刚体动量；
- 高黏性阻尼稳定衰减、默认零路径相同、输入隔离/不可写、非法域和根零槽；
- MJCF 默认/显式覆盖和不支持数据、重复/未绑定 motor 的拒绝。

`tests/reference_newton/robots/floating_actuator_reference.py` 是独立 MuJoCo 连续 M/bias/qacc 对照。声明的铰链/移动关节、零/非零执行器、静止/倾斜运动组合固定，不调参。它将 MuJoCo freejoint 的 origin/world-linear + body-angular 速度映射到 Phymeo root-COM/world-angular，保留基变换导数项；原生数据实际生成期望量，不调产品作 oracle。只执行 mj_forward，0 trajectory steps。脚本尚未运行，不宣称一致。

父建议队列：先本组精确文件 Ruff/format，再新测试和受影响旧 HOST 浮根/根惯量小组，随后在已配置的 MuJoCo 环境运行上述独立脚本。所有首失败/源码快照保留。

## 父/设备集成注意

父共享 runtime、exports、scene energy 和控制器修改不在本分支范围。浮根点端口原本用 floating_mass_matrix 计算冲量与能量，因此会自然接到新物理 M；scene 的 body-only 能量观察需补反射项。设备作者必须复制/上传两个数组，并计入模型 token/checkpoint 来源，不能静默丢弃。

本分支没有 G1 轨迹、29关节设备、足地支撑或舞蹈通过结论。

## 一手语义参考

[MuJoCo joint armature/damping](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-armature) 定义关节附加反射惯量和黏性阻力；其 Euler 积分使用隐式阻尼。这里仅实现线性标量项，不泛化到新版本的多项式阻尼或其它 actuator 传动。
## 2026-11-09 parent validation

The HOST implementation and shared scene integration passed 44 focused tests
(`tests/test_floating_actuators.py`, `tests/test_floating_actuator_scene.py`,
`tests/test_floating_mjcf.py`). Eight independent MuJoCo fixed-state cases
checked physical mass, bias and continuous acceleration. Three actual G1
poses additionally checked all 35 velocity coordinates: maximum mass error
`5.5511e-15`, acceleration error `3.3840e-13`, and kinetic-energy error
`5.5511e-16`. These are dynamics comparisons, not a Phymeo standing result.

The shared scene keeps physical inertia separate from `M+hD` step mobility,
including bounded drive and coupled tangent preparation. A two-body analytic
scene checks the resulting free-root angular reaction and joint decay.
Default zero damping retains the old path.

Evidence: `artifacts/p65-h1-20261109-01`, tasks 03–08. Task 07 retains an
initial reference-script import error; task 08 corrects the import and completes
the G1 comparison. The same official primitive model with the original
deployment armature/viscous damping completed a separate 20-second MuJoCo policy
reference; dry friction was explicitly zero in this matched primitive variant.
This does not replace the preserved full-mesh reference or establish policy
transfer to Phymeo.
