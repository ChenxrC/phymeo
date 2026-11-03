# Newton 1.7.0 与 Phymeo 首轮模块对照

日期为 2026-11-03，机器为 Windows x86-64、RTX 4060 Laptop GPU、驱动 566.07。Newton 使用独立 Python 3.12.13、Warp 1.17.0、NumPy 2.5.4。Phymeo 使用原项目 Python 3.12.14，CUDA 行加载 tyx-compute 0.1.0a0。两个解释器分别运行同一组案例的公开入口。Newton 的 CPU 和 CUDA 预检都读到球–地面配对，并在独立自由球的实际 XPBD 步上读到高度由 0.80000001 m 降至 0.79931879 m。[环境锁](../../requirements/newton-reference-win-py312.txt) · [运行说明](../../tests/reference_newton/README.md) · [Newton 安装与驱动要求](https://newton-physics.github.io/newton/1.7.0/guide/installation.html)

## 数量与口径

35 个唯一案例 × 2 个引擎 × 2 个设备产生 140 条登记。其中 78 条执行了对应物理或导入断言，74 条通过，4 条超差。另外 32 条缺少公开入口，18 条两侧的物理量或时间定义不同，12 条超出该适配器登记的设备范围。两个凹槽壁案例各在两个设备追加了一次逐触点诊断，保留在数据文件的 `two_wall_diagnostics` 部分，计数仍为 35 个案例。选例针对已知缺口，没有测量引擎整体完成率或性能。[原始物理读数、断言与运行绑定](newton-1.7-reference-observations.json)

| 引擎与设备 | 登记 | 实际通过 | 超差 | 缺入口 | 异义 | 设备未覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Newton CPU | 35 | 26 | 2 | 0 | 6 | 1 |
| Newton CUDA | 35 | 26 | 2 | 0 | 6 | 1 |
| Phymeo HOST | 35 | 16 | 0 | 16 | 3 | 0 |
| Phymeo CUDA | 35 | 6 | 0 | 16 | 3 | 10 |

“通过”表示该引擎和设备满足这条案例列出的预期，包含预期的警告或构造拒绝。“缺入口”指登记输入无法通过该引擎的公开模型或算子表达。“异义”包括单时刻接触与区间首触、不同接触定律等。“设备未覆盖”保留适配器的范围，例如资产单位诊断只在 HOST 登记。CUDA 行中设备与 HOST 各自负责的步骤记录在读数和分方向计划中。

首轮 12 个报告保留每个案例的通过或失败。右壁与左壁的四次诊断单独收录，补全每一条接触的深度、法向、双侧见证和真实表面距离。最终五个参考脚本通过 Ruff 与格式检查；排版前后逐文件比较的 Python AST 全部相同。

## 影响开发顺序的实测

- **刚–刚。** Newton 的棱顶、坡面、近失和三种曲面原语离散检查均通过。两个可动闭合 U 形网格的空槽无接触，实际压入右壁和左壁时深度分别为 0.125 m 和 0.1875 m。追加诊断读取了每个接触，将最深 1 mm 带内的所有记录一起检查。两个设备的右壁均为 7 条并列，左右两个物体的见证点到各自真实表面的距离均为 0.0625 m，右壁法向符合的为 CPU 7/7、CUDA 6/7。左壁各有 10 个接触、9 条并列最深，法向各 9/9 符合，双方表面距离均约 0.09375 m。四组均有命中，均未满足 1 mm 物理壁面见证要求。这组观测同时保留穿深、求解器采用的点及可重建的物理边界目标。[刚–刚判据](../superpowers/plans/2026-11-03-newton-rr.md)
- **刚–柔。** 球在三角形内部穿入 1/16 m、三个节点均在外面。Newton 默认检测为 0，整面模式为 1 条三节点接触，重心权重约为各 1/3。CUDA 烘焙的网格 SDF 返回 3 点、3 边、1 面共 7 条；有限平面及高度场显示了原定的逐粒子降级。Phymeo 的整面球例、近失和镜像在 HOST 通过。另一个解析载荷例在 1/256 s 内给出布–滑块的 2 N 法向力、滑块 +1/64 m/s、三个布节点各 −1/384 m/s，总竖直动量为零。它在两种执行配置中都通过，CUDA 配置的材料映射和状态积分在 HOST，tyx 计算法向载荷。[刚–柔判据](../superpowers/plans/2026-11-03-newton-rd.md)
- **机器人。** Newton 的程序建模、URDF、MJCF、USD 在同一 R–P 机械臂构型上复现了偏置工具的坐标和速度。设为厘米的 USD 返回单位诊断；本组只检验告警。离平面 IK 的独立最小残差为 1/16 m。2 kg 滑块经过 (+4,+4,−4,−4) N 四步后的位移为 1/32 m、速度为 0。Phymeo 的程序式运动学、HOST 的 IK 和滑块步进都通过。四个格式或单位入口记为缺口。Phymeo CUDA 的三个 FK 例执行 tyx 几何与雅可比，最后的速度乘法在 HOST。[机器人判据与资产](../superpowers/plans/2026-11-03-newton-robot.md)

## 逐项结果

### 刚–刚（13 项）

| 案例 | Newton CPU | Newton CUDA | Phymeo HOST | Phymeo CUDA |
| --- | --- | --- | --- | --- |
| `RR-SPHERE-POSES` | 通过 | 通过 | 通过 | 未覆盖 |
| `RR-SPHERE-FIRST` | 异义 | 异义 | 通过 | 通过 |
| `RR-U-NOTCH` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-U-RIGHT` | 超差 | 超差 | 缺入口 | 缺入口 |
| `RR-U-LEFT-HOLDOUT` | 超差 | 超差 | 缺入口 | 缺入口 |
| `RR-U-FIRST` | 异义 | 异义 | 缺入口 | 缺入口 |
| `RR-RIDGE-POSES` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-RIDGE-MISS` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-SLOPE-FACE` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-RIDGE-FIRST` | 异义 | 异义 | 缺入口 | 缺入口 |
| `RR-CYLINDER-RIM` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-CONE-TIP` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RR-ELLIPSOID-TILT` | 通过 | 通过 | 缺入口 | 缺入口 |

### 刚–柔（11 项）

| 案例 | Newton CPU | Newton CUDA | Phymeo HOST | Phymeo CUDA |
| --- | --- | --- | --- | --- |
| `RD-SPHERE-FACE` | 通过 | 通过 | 通过 | 未覆盖 |
| `RD-SPHERE-SEPARATED` | 通过 | 通过 | 通过 | 未覆盖 |
| `RD-SPHERE-MIRROR-HOLDOUT` | 通过 | 通过 | 通过 | 未覆盖 |
| `RD-SPHERE-EDGE` | 通过 | 通过 | 通过 | 未覆盖 |
| `RD-FINITE-PLANE` | 通过 | 通过 | 异义 | 异义 |
| `RD-HEIGHTFIELD` | 通过 | 通过 | 缺入口 | 缺入口 |
| `RD-MESH-MISSING-SDF` | 通过 | 通过 | 异义 | 异义 |
| `RD-MESH-BAKED-SDF` | 未覆盖 | 通过 | 异义 | 异义 |
| `RD-ROUNDED-PLATE` | 异义 | 异义 | 通过 | 未覆盖 |
| `RD-SWEPT-INTERIOR` | 异义 | 异义 | 通过 | 通过 |
| `RD-PRISMATIC-LINK-STEP` | 异义 | 异义 | 通过 | 通过 |

### 机器人与工具链（11 项）

| 案例 | Newton CPU | Newton CUDA | Phymeo HOST | Phymeo CUDA |
| --- | --- | --- | --- | --- |
| `ROBOT-FK-ZERO` | 通过 | 通过 | 通过 | 通过 |
| `ROBOT-FK-TURN-SLIDE` | 通过 | 通过 | 通过 | 通过 |
| `ROBOT-FK-OBLIQUE-HOLDOUT` | 通过 | 通过 | 通过 | 通过 |
| `ROBOT-URDF-TOOL` | 通过 | 通过 | 缺入口 | 缺入口 |
| `ROBOT-MJCF-DEGREES` | 通过 | 通过 | 缺入口 | 缺入口 |
| `ROBOT-USD-ZUP` | 通过 | 通过 | 缺入口 | 缺入口 |
| `ROBOT-USD-CENTIMETER-WARNING` | 通过 | 未覆盖 | 缺入口 | 缺入口 |
| `ROBOT-IK-REACHABLE` | 通过 | 通过 | 通过 | 未覆盖 |
| `ROBOT-IK-OFFPLANE` | 通过 | 通过 | 通过 | 未覆盖 |
| `ROBOT-DRIVE-REVERSE` | 通过 | 通过 | 通过 | 未覆盖 |
| `ROBOT-DRIVE-NEGATIVE-HOLDOUT` | 通过 | 通过 | 通过 | 未覆盖 |

[根据本轮差异排列的开发批次](../superpowers/plans/2026-11-03-newton-reference-development.md) · [已验收的 Phymeo 公共碰撞基线](p65-collision-public-catalog.md)
