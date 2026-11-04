# 经典 Panda 抓放自由方块

本例读取 Franka 官方描述的经典七轴 Panda 和双滑轨手。源码分发中内置从固定版本 Xacro 展开的 URDF、许可证、署名以及再生方法。公开入口 `load_panda()` 读取 XML，`asset.configuration(arm_q, opening_m)` 生成两指的对称命令，实际运动允许两指受力不同。夹持时长和目标位姿由一个预定的关节目标序列给出。[资产来历与改动](../../src/phymeo/builders/panda_assets/SOURCES.md) · [使用入口](../../examples/panda_cube.py)

```powershell
python examples/panda_cube.py --trace panda-carry.json
python examples/panda_cube.py --friction 0 --trace panda-slip.json
python examples/export_panda_cube.py --loaded panda-carry.json --slippery panda-slip.json --output panda-replay.html
```

## 算法和验收

5 cm 的方块质量为 100 g，六个自由度承受重力、接触力和关于质心的力矩。机械臂从开口 76 mm 接近，使指垫在方块上半部闭合，然后抬升、横移 75 mm、落回桌面、张开并撤走。IK 使用导入模型的末端位姿和关节限位；带重力和惯性补偿的关节控制受原资产力矩和手指推力上限约束。两指分别使用厂家内侧矩形橡胶面上的四个材料采样点。接触刚度、阻尼和滑速正则化的摩擦系数属于本例的材料配置。每个材料点把同一世界力按相反方向传给方块与实际指关节。有限桌面通过箱体的八个顶点承载。[两侧几何和载荷](../../src/phymeo/solvers/sampled_box_contact.py) · [实际积分、控制与保存](../../src/phymeo/experiments/panda_cube.py)

同一初态、控制和重力分别使用指面摩擦系数 0.65 和 0。正例要求搬运时方块最低点高于桌面 35 mm、前进超过 45 mm，双指持续受力且桌面不承担重量。松指后的位置距目标小于 12 mm，桌面承重恢复到布置的方块重量（容差 0.025 N）。零摩擦时仍须发生夹紧，但方块离台不得超过 5 mm、搬运小于 15 mm。每个接受的位姿还用箱体与桌面的轴向投影对其他厂家碰撞体做分离检查。这个检查提供已采样时刻的净空界。此场景使用 HOST 的柔顺采样接触，各种刚体多边形的瞬时硬接触另有独立测试。[搬运与失抓对照](../../tests/test_panda_cube_manipulation.py) · [通用凸面冲量](p65-rigid-polygon-impulses.md)

Newton 官方抓方块样例加载 FR3 描述。本项目的同资产几何对照将经典 Panda 的同一份 URDF 和已保存的实际关节坐标交给 Newton 的 FK，再从其两个碰撞盒独立求出指垫内面中心。该对照可发现导入、参考系或联动方向差异。两个引擎的受载抓放性能需使用匹配的动力学配置另外计量。[Newton 官方案例](https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_cube_stacking.py) · [同资产测量程序](../../tests/reference_newton/robot/panda_grasp_frames.py)

## 2026-11-04 的真实抓放

每组从同一层外初态推进 3,600 个 HOST 子步，每步 0.625 ms，物理时长 2.25 s。两组末态都由刚体积分得到。表中搬运读数取 1.35 s，双指持续施力与桌面卸载检查覆盖 0.95–1.35 s。

| 物理量 | 指面摩擦 0.65 | 指面摩擦 0 |
| --- | ---: | ---: |
| 搬运读数时方块最低点相对桌面 | +59.859 mm | −0.111 mm |
| 方块沿命令方向的实际位移 | 77.355 mm | 0.057 mm |
| 搬运期间，较弱侧指面法向力的最小值 | 5.776 N | 0 N |
| 搬运期间桌面峰值反力 | 0 N | 0.981001 N |
| 双指撤离后，末 50 ms 平均桌面反力 | 0.981000 N | 0.981000 N |
| 末态质心距指定落点 | 0.176 mm | 74.944 mm |
| 全窗指垫最大采样压入量 | 1.221 mm | 2.253 mm |

正例采样时刻其余登记实体至方块的保守最小轴向净空 5.797 mm，机械臂至有限桌面 25.872 mm。两组各自撤指后的指垫力均为零，全窗关节限位违规量均为零。抓住和失抓两次都发生了约 5.8 N 的峰值双侧夹压。这个差异来自切向载荷，夹紧命令相同。HOST 材料点解析控制与两条物理轨迹一共 3 项通过。[完整正例子步](../../artifacts/p65-20261104-panda-batch/temp-panda-pick-window/test_imported_panda_grips_and_0/motion.json) · [零摩擦子步](../../artifacts/p65-20261104-panda-batch/temp-panda-pick-window/test_imported_panda_grips_and_1/motion.json) · [原始退出与用时](../../artifacts/p65-20261104-panda-batch/panda-pick-window/final.json)
