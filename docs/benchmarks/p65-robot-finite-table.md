# 机械臂与有限固定桌面的受力

`FixedTableContacts` 用实际运动连杆的球或有限胶囊与定向长方体桌面求接触。输入桌面中心、三个半边长和 XYZW 朝向。法向从桌面指向连杆，两侧原始表面见证的中点是共同作用点。连杆得到关于真实质心的世界力矩，固定台面得到关于台心的等反力矩。桌面速度为零，因此功为零。调用 `evaluate_robot(model,state)` 可从模型局部的材料登记和真实入步 q/v 读出接触及 Jᵀ 广义载荷。[固支台面入口](../../src/phymeo/solvers/fixed_table_contact.py) · [有限轴段与定向箱的材料见证](p65-box-capsule-response.md)

## 两关节 URDF 的持载与卸载

使用仓库中的 `arm.urdf`，含一个转铰、一个滑轨和偏置在工具质心的 1/32 m 实体球。质量与惯量由两引擎各自从该资产读取，初始位姿和关节速度相同。固定桌面法向 (0.36, 0.48, 0.80)，初始压入 1/1024 m，刚度 2048 N/m。这给工具 2 N 法向力，静持驱动力为转轴 −0.45 N·m、滑轨 −0.96 N。支承点的台面内偏置为 1/16 m，沿 (0.8, −0.6, 0)。从静止持续作用 32 步，第 33 步撤去电机力，再推进 128 步。每步 1/1024 s。[原始 URDF](../../tests/reference_newton/robot/arm.urdf) · [两端实际推进与手算断言](../../tests/reference_newton/robot/robot_table_window.py)

| 量 | Phymeo HOST | Newton 1.7 Featherstone CPU |
| --- | ---: | ---: |
| 首步桌面所受力 XYZ（N） | (−0.7200, −0.9600, −1.6000) | (−0.7200, −0.9600, −1.6000) |
| 首步关于台心的力矩 XYZ（N·m） | (0.0600, 0.0800, −0.0750) | (0.0600, 0.0800, −0.0750) |
| 持载 32 步桌面所受冲量 XYZ（N·s） | (−0.02250, −0.03000, −0.05000) | (−0.02250, −0.03000, −0.05000) |
| 撤力首步转速（rad/s）／滑速（m/s） | −0.00032514／0.00205789 | −0.00032512／0.00205780 |
| 最后有载出步（s） | 0.0869141 | 0.0869141 |
| 接触对连杆做的总功／固定台面功（J） | 0.00064341／0 | 0.00064341／0 |
| 关节功率与真实接触点功率最大差（W） | 3.47×10⁻¹⁸ | 5.15×10⁻⁹ |

Newton 从原生球形碰撞体、固定箱、运动身体姿态与质心点速重建见证。接触经 `body_f` 对原生身体施力，`joint_f` 只施命令中的电机力，`contacts=None` 防止再叠加原生流形。固定台面单独读回反力，关节载荷仅供对照。两端 160 步的工具质心位置分量最大差 8.84×10⁻⁸ m。[Phymeo 全轨迹](../../artifacts/p65-night-robot-table-loads-20261105-01/rp-phy.json) · [Newton 全轨迹](../../artifacts/p65-night-robot-table-loads-20261105-01/rp-newton.json) · [逐时刻比较](../../artifacts/p65-night-robot-table-loads-20261105-01/trajectory-comparison.json)

2 kg 滑动胶囊在水平台面上由实际轴段的两个端部材料点共同承受 4 N。32 步静持的桌面竖向冲量为 −0.125 N·s，随后给滑轨向上 6 N，首个出步 v=3/1024 m/s，之后脱离台面。独立试片还检查圆形端帽在有限台边产生的斜向反力、处于无限平面压入高度但在有限边外的零力，以及旋转胶囊两端的不同点速和恢复力矩。[真实滑轨出步](../../artifacts/p65-night-robot-table-loads-20261105-01/10-carriage-lift/final.json) · [解析见证与台边近失](../../tests/test_fixed_table_link_contact.py)

## Panda 的独立通道

在普通入口设置 `table_link_response=True` 或 CLI 的 `--resolve-link-table`。每条 `link_table_contacts` 含实际形状、两侧表面点、世界法向与作用点、连杆质心与台心、力、力矩和功率。每行 `table_joint_load` 是此通道的广义载荷，`robot_on_table_wrench` 是机械臂施于桌面的六维载荷，`table_N` 专指自由物体与桌面的接触。已由指面处理的几何不重复登记。固定基座及未支持的移动箱形结构仍监测首次可能交叠，圆形轴深入台面实心区域时会显式拒绝。[普通 Panda 入口](../../src/phymeo/experiments/panda_cube.py)

原 120 g 长方体在两个引擎中各完成 3,600 步抓放。全窗机器人与台面接触行数为零。对各自原存档，关节状态、物体位姿和线角速度、指力和桌面力逐分量完全一致，松指后物体仍由桌面承重 1.1772 N。[Panda HOST 收据](../../artifacts/p65-night-robot-table-loads-20261105-01/04-table-phy/final.json) · [Panda Newton 收据](../../artifacts/p65-night-robot-table-loads-20261105-01/07-table-panda-newton/final.json)
