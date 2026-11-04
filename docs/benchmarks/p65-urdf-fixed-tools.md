# 固定工具随关节运动与受力

含合法 `fixed` 关节的本地 URDF 现在可直接交给 `load_urdf` 或 `RobotPayloadScene.from_urdf`。固定工具可连接于活动连杆、另一固定工具或静止基座；新的活动关节也可以安装在固定转接座上。每个活动关节仍只有一个有界旋转或移动坐标。完整惯性张量按 URDF 的惯性原点朝向旋入刚体组，偏置质量按平行轴定理合并。动力学以合成质心和主惯量工作。

原始链接名、工具坐标和材料点依然可读。即使多个源链接有相同的碰撞元素名，各自也有独立的碰撞键。各活动组按真实父子关系传力；静止固定链的碰撞点使用世界坐标，其质量不进入活动系统。

```python
from phymeo.builders.urdf import load_urdf
from phymeo.dynamics.robot_kinematics import point_jacobians

arm = load_urdf(path)
site = arm.sites["probe"]
tip_local = arm.authored_to_model_points("probe", [[0.03125, 0, 0]])
Jv, Jw = point_jacobians(arm.model, q, [site.link_index], tip_local)
position, rotation = arm.authored_link_pose("probe", q)
collision_poses = arm.collision_motion(q)
```

`collapse_fixed_joints=None` 是默认设置，有固定关节才进行上述分组。纯标量老资产默认沿用原链接局部轴。对于没有固定关节但有完整、非对角惯量的资产，可显式指定 `True` 选择主轴表示与原链接读回。显式 `False` 仍使用原严格导入器。固定工具可以省略质量，活动组的合计质量和主惯量必须有效。带控制语义的 `mimic`、阻尼或传动扩展由常规入口具名拒绝。经典 Panda 的专用入口继续单独检查已知手指关系。原导入器支持的圆柱和本地闭合 OBJ 可以随工具更新几何；N3 交互场景的结构载荷仍限定于已有反力核支持的球、有限胶囊和正交箱。本批圆柱留出验证了固定工具的真实材料位置和朝向。

两份小型 URDF 是本项目的独立实物参数试片。伸缩探头在俯视平面有两个活动自由度，基座安装、旋转臂末座、套筒和探头通过四个固定接头连接。五个有质量的原链接组成两个活动刚体。压头试片为单根竖直滑轨，附加固定压头和无质量的触点坐标。它们均通过同一个公开场景发正、反指令，向 125 g 自由球实际施力。[伸缩探头 URDF](../../tests/reference_newton/robot/fixed_tool_probe.urdf) · [竖直压头 URDF](../../tests/reference_newton/robot/fixed_tool_press.urdf)

本机 Newton 1.7.0 的 URDF 导入器构建固定关节，并可通过 `collapse_fixed_joints=True` 原生聚合。它在合并前解析包括非对角项在内的完整惯量，并应用惯性坐标的旋转；它还能够登记仿射 mimic 定义。本批 Newton 基准覆盖固定工具、接触几何材料点与偏心外载。没有将 mimic 登记外推为该驱动在每个求解器上已受验收。

首红来自旧导入器对 `joint 'mount'` 的真实拒绝。新证据分别保存模型未折叠的原始链接姿态/工具速度、真实折叠的质心/世界惯量和向工具点施力后的一步关节速度。[独立解析真值](../../tests/reference_newton/robot/fixed_tool_oracle.py) · [原生 Newton 驱动](../../tests/reference_newton/robot/fixed_tool_native.py)



## 已测固定链与主惯量

两份资产从同一场景入口发出正反各 40 个 1 ms 物理步。首步工具对自由球施加 2 N，末活动轴受到 −2 N；前向与后退命令让探头最后的滑轨位置相差 21.586 mm，自由球相差 5.931 mm。竖直压头的对应差值是 18.805 mm 和 5.943 mm。另一项试片从滑轨上移除本体惯量，仅保留固定工具的 250 g，沿轴载荷 1 N 给出 4 m/s²。无质量的整个活动组、非法惯量、普通 mimic 和带控制属性的固定关节均被拒绝。

原生合并模式出现可量化的惯量差。在第一姿态，Newton 未折叠模型的 `M[0,0]` 实测 0.317975640 kg·m²，独立五部件结果为 0.317975734 kg·m²。相同原文件经原生固定关节折叠后，两端设备都给出 0.378485560 kg·m²，超出事前界 0.000005 kg·m²。两份严格对照的首次退出码均为 1。

可以直接定位上臂这一个惯量分量。2 kg 上臂和 1 kg 肘座自身的世界 Z 惯量分别为 0.078125 和 0.030000 kg·m²。两个真实质心的平面间距分量为 0.09375 和 0.0625 m，合成质心上的惯量因此是 `0.078125 + 0.03 + (2/3) × (0.09375² + 0.0625²) = 0.116588542` kg·m²。安装版 Newton 的 `_src/sim/builder.py` 第 7084–7098 行在更新合成质心前，按子固定关节原点位移给子惯量加平行轴项。本例固定关节平移是 0.25 m，这给出 `0.078125 + 0.03 + 1 × 0.25² = 0.170625` kg·m²，原生折叠上臂实测 0.170624971。其余方向和其他模型需按各自的几何与质量单独核对。


## 同一物理步的分项对照

`q=(0.37,0.025)` 时在探头球心施加世界力 `(0.5,−2,1.25) N`，同时给两轴 `(0.125 N·m,−0.25 N)`，步长为 1/2048 s。第二姿态为 `q=(−0.61,−0.035)`。每种引擎使用自己的实际入步位置、质心力臂和角速度，原生球形状 ID 为 3。未折叠模型有 7 个刚体，材料归属刚体 6。原生折叠后为 2 个刚体，球归属刚体 1。

| 引擎与固定关节设置 | 第一姿态加速度 (rad/s², m/s²) | 第二姿态加速度 (rad/s², m/s²) | 与五个原部件的最大加速度差 |
|---|---|---|---:|
| Phymeo HOST，正确合并后的真实进步 | 2.968380, −0.678206 | 2.150833, −2.325081 | < 5×10⁻¹² |
| Newton CPU、CUDA，保留固定关节 | 2.968384, −0.678207 | 2.150818, −2.325089 | 1.56×10⁻⁵ |
| Newton CPU、CUDA，原生折叠 | 2.396362, −0.532974 | 1.702759, −2.211319 | 0.572018 |

保留原始固定链时，两端的 15 项原门槛全部通过。实际原生工具点雅可比的最大误差为 1.27×10⁻⁷，速度误差不超过 3.50×10⁻⁸ m/s，质量矩阵误差不超过 1.74×10⁻⁷。原生折叠后的形状位置、工具点雅可比、质量、质心等 10 项通过；质量矩阵、世界惯量、动量、动能、真实步加速度 5 项未通过。折叠模式的动能最大偏差为 0.00915214 J。原始保留模式中按组汇总的惯量和动量由原生各成员的实际世界位姿、质心速度及惯量后处理得到，雅可比和质量矩阵是直接调用 Newton 设备 API 得到的。

```powershell
python -m pytest -q tests/test_urdf_fixed_tools.py
python tests/reference_newton/robot/fixed_tool_native.py --device cpu --require unmerged --out cpu.json
python tests/reference_newton/robot/fixed_tool_native.py --device cuda:0 --require unmerged --out cuda.json
```

`--require unmerged` 的退出码只验收未折叠配置，两种配置仍写出全部原阈值和实际误差。默认 `--require all` 要求两张表都通过，在这组输入上返回 1。HOST 新专项 10 项、相邻兼容 7 项通过，另对两个姿态实际运行 Phymeo 物理步并验证已接受末态。最终 Ruff 与格式检查通过。[逐设备原始 JSON 和哈希收据](../../artifacts/p65-night-urdf-fixed-tools-20261105-01/) · [Phy 几何、惯量、动量与受力专项](../../tests/test_urdf_fixed_tools.py)

两份试片提交时各删除了文件末尾的一个换行符。与物理作业哈希相符的原始字节、提交版本哈希和逐文件 XML 元素一致性记录均保存在证据目录的 `original-tested-assets/` 中。
