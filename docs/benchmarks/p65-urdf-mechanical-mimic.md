# 一个驱动和两个受力关节

`phymeo.builders.mimic_urdf.load_mimic_urdf` 从本地 URDF 读取一条实际机械联动。有界旋转关节可连接同类旋转关节，有界移动关节可连接同类移动关节。驱动和从动关节既可属于不同分支，也可串联。乘数与偏置取自原文件。该入口要求恰好一行，拒绝未知驱动、链式或循环关系、零乘数、空的可动区间及未知控制标签。普通 `load_urdf` 与专用 `load_panda` 保持各自的公开合同。

令独立位置为 `u`，原树的坐标为 `q=P u+c`。从动坐标等于 `a*u_driver+b`，速度和加速度乘 `a`。几何和材料点使用完整原树的实际姿态，点雅可比为 `J P`，质量与偏置为 `Pᵀ M P`、`Pᵀ b`。每个源碰撞分量的实际世界力先回到其原链接，再将全轴广义载荷投影为 `Pᵀ Q`。电机只对独立轴施力。从动件的受力由本次惯性平衡恢复，独立方向上的合约束力和约束功为零。URDF 指定的从动电机额定力不限制刚性机构内部的传力。

```python
from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.runtime.robot_payload_scene import RobotPayloadScene

gripper = load_mimic_urdf(path)
scene = RobotPayloadScene(gripper, **scene_options)
obs = scene.step(target_q, effort_limit=motor_caps)
print(gripper.joint_names, obs.q, obs.full_q)
print(obs.mimic_loads.actuator_effort_full, obs.mimic_loads.constraint_effort_full)
```

`scene.step` 的位置、速度、增益和电机上限均按独立轴排列。`full_q/full_v` 则是所有物理轴的已接受步末读数；`mimic_loads` 属于 `load_time_s` 对应的入步。调用 `gripper.reduce_state(full_state)` 可检查已有全轴观测是否同时满足位置与速度关系。独立位置范围是驱动上下限与从动上下限逆映射后的交集，速度上限也计入乘数。若候选跨过该适用区间，场景保留所有先前状态和载荷。本批未增加限位冲击或齿隙、弹性和电机效率模型。

## 两个独立机构

并联夹具使用两个朝向世界 X 的滑轨，关系为 `q_right=−q_left+0.004 m`。两夹爪质量分别为 0.2 和 0.3 kg，另有实际 1 kg 的竖直升降台。另一个串联机械臂在世界 Z 和随上游转动的 Y 方向联动，采用 `q₂=1.5 q₁−0.125 rad`，质量、惯量、质心及偏心施力点均不同。原始质量、速度偏置和材料点雅可比由独立几何推导。完整坐标的鞍点系统从施加在原轴与真实连杆上的力求加速度和约束乘子。[夹具 URDF](../../tests/reference_newton/robot/mimic_jaws.urdf) · [机械臂 URDF](../../tests/reference_newton/robot/mimic_elbow.urdf) · [解析控制](../../tests/reference_newton/robot/mimic_oracle.py)

本机 Newton 1.7.0 的 Featherstone 在 Warp 设备上执行联动惯量、广义力约化和全轴加速度展开。Newton 的 SemiImplicit 使用有限刚度与阻尼，XPBD 和 VBD 有各自的投影路径，这些求解器未纳入本批数值对照。本批保持 URDF 原生注册，从相同的非零位置与速度独立推进。对照只在建模时去除这条机械关联，再施加完全相同的电机命令与偏心世界力。Newton 的反力读数从其真实速度增量、全质量与全坐标力平衡重建；设备雅可比、质量矩阵和关节状态直接读自原生 API。[原生驱动](../../tests/reference_newton/robot/mimic_native.py)


## 真实夹持、抬起与释放

同一 100 g、边长 40 mm 的自由箱从有限桌面起步。指面各用四个固定材料点，法向刚度为每点 1,200 N/m，摩擦系数为 0.65。两组使用相同的闭合、上抬 50 mm、下降及张开目标，独立丝杠的力限分别为 8.0 N 和 0.35 N。从动指没有收到驱动命令。

| 物理结果 | 8.0 N | 0.35 N |
|---|---:|---:|
| 接受的 1 ms 步数 | 1,720 / 1,720 | 697 / 1,720 |
| 实际是否出现双侧接触 | 是，峰值 4.983 N / 侧 | 是，峰值 4.238 N / 侧 |
| 是否搬起 | 是，保持窗最低离台 45.886 mm | 未搬起，指面离开后桌面承重 0.981 N |
| 是否完成计划放下、松指和末窗 | 是，末窗指载 0 N、桌载 0.981 N | 联合行程边界拒步，未到该阶段 |

8 N 时，0.8–0.94 s 保持期间双指均不少于 3.997 N，桌载为零。1.29 s 放下时桌面有 1.878 N 瞬时载荷，1.62 s 两指已卸载。由每个真实指面采样力独立组装全轴广义力，使用三阶原质量矩阵与一行约束重新解鞍点系统，全部 1,720 步的最大加速度误差为 2.79×10⁻¹⁴ m/s²、机构反力误差为 8.89×10⁻¹⁶ N。两侧力的关系及其功也逐步核对。自由箱的最大冲量平衡残差低于 2.9×10⁻¹⁸ N·s。[完整窗口驱动](../../tests/reference_newton/robot/mimic_grip_window.py)

## 与原生 Featherstone 对照

Newton 1.7 直接解析相同的两份 URDF。每端分别保留和解除原生耦合，共 8 次实际推进。关节惯性、原生点雅可比、末态和偏心载荷下的全轴加速度均与独立解析比较。表内单位按两种机构分别为 m/s² 和 rad/s²。

| 构型 | Newton 机械联动时全轴加速度 | 同一原始载荷、解除机械联动 |
|---|---|---|
| 三滑轨 | (0.166664, −6.000000, +6.000000) | (0.166664, −2.500000, +8.333332) |
| 两异轴铰链 | (−3.107910, −4.661865) | (−2.648376, −7.955688) |

CPU 和 CUDA 的最大加速度误差均为 2.81×10⁻⁵，原生速度增量重建的全轴反力最大误差低于 6.4×10⁻⁶。两铰链的从动反力由独立方程给出 +0.105534 N·m，低于其 0.125 N·m 电机额定值；滑轨从动反力 −0.700000 N 超过其 0.5 N 电机额定值。早期假设两个机构反力均超过电机额定值的附加断言退出 1，原始反力、负载和模型参数保留。修订后的检查直接比对完整反力向量，并分别验证这两种额定值关系。

本次完整持载与失持窗口运行在 Phymeo HOST。Newton 两端验证的是同构型、相同偏心力与单步广义驱动，并含去掉机械联动后的实得差异。完整抓放没有计成 Newton 跨引擎验收。


弱例的最后一次双侧受力在 0.318 s，0.319 s 起持续无指面载荷，未从桌面搬起。0.697 s 的最后接受状态中，夹臂下缘比箱顶高 18.527 mm，桌面对箱施加 0.981 N。丝杠距 22 mm 联合上限仅 0.0962 mm，仍以 0.10522 m/s 闭合。两指折算质量 0.5 kg，剩余 0.35 N 全力制动需要 7.908 mm；按同一半隐式步长计算，下一位置至少 22.0083 mm。该步拒绝后，关节、自由箱、时刻和已发布载荷均保持上一状态。完整弱力窗口的原始退出码 1 与这项单独的边界核验退出码 0 都保留。

```powershell
python -m pytest -q tests/test_mechanical_mimic.py
python tests/reference_newton/robot/mimic_grip_window.py --case strong --out strong.json
python tests/reference_newton/robot/mimic_grip_window.py --case weak --expect loss-and-limit --out weak-limit.json
python tests/reference_newton/robot/mimic_native.py --device cuda:0 --out native-cuda.json
```

新增 HOST 专项 12 项、旧入口及固定工具相邻回归 4 项和最终静态检查均通过。证据目录保存 8 N 全窗、0.35 N 首次未完成全窗、末步诊断和两端原生对照。[原始物理记录和运行收据](../../artifacts/p65-night-mechanical-mimic-20261105-01/)
