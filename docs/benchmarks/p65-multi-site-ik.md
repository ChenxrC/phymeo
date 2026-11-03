# 有界多站点 IK 与双臂跟踪

`solve_multi_site_ik` 为一个固定基座标量 RobotModel 生成关节目标。每个 `SiteIKTarget` 可指定站点世界位置，以及完整世界姿态或单根世界指向。安装点、安装 XYZW 和局部指向均相对于已有 RobotSite。单轴指向允许绕该轴自由滚转。完整姿态使用站点到世界的 XYZW 四元数；q 与 −q 表示同一朝向。[公共入口](../../src/phymeo/control/site_ik.py)

```python
from phymeo.control.site_ik import SiteIKTarget, solve_multi_site_ik
from phymeo.sensors import RobotSite

targets = [
    SiteIKTarget(RobotSite(left_link, left_offset), position_world_m=left_target),
    SiteIKTarget(RobotSite(right_link, right_offset, right_mount),
                 position_world_m=right_target, orientation_world_xyzw=right_rotation),
]
solution = solve_multi_site_ik(model, seed, targets, limits=joint_limits)
goal_q = solution.configuration
```

## 几何、权重和关节界

一次 FK 计算各目标点和安装朝向，线/角雅可比复用同一个树几何。位置残差以 m 计，姿态误差取 SO(3) 最短旋转角，单轴误差取两单位向量的夹角，均为 rad。位置、姿态/方向权重乘相应残差和雅可比行。每个正权通道须分别满足自己的物理容差才算收敛；零权通道保留实际误差读数。整体 `weighted_error` 为各加权残差的二范数。

有界阻尼最小二乘同时考虑各活动关节的可用区间。碰到边界后可将剩余位移分配给其他列，若梯度改向也可释放边界。外层用真实 FK 代价回溯，步幅采用原生关节坐标。只有活动坐标投影到给定盒内。`active_bounds` 报告返回点上达到界限的活动关节。`conflicting_pairs` 列出同一刚体上的目标违反固定点距、姿态或指向夹角等必要条件的目标对。普通局部停止和迭代预算耗尽只描述该次求解。每项 `residuals` 均对应返回配置的实测，不以优化代价替代各站点误差。

## 双臂几何和留出

两条分支共享沿世界 Z 移动的升降轴。左分支的固定安装绕 X 转动，cosβ=4/5、sinβ=3/5；右分支的固定安装绕 Y 转动，cosγ=3/5、sinγ=4/5，再绕本地 Z 轴转半圈。两端各有一个转铰、一个轴向滑轨和偏心工具，工具安装朝向也不同。直接按刚体变换写出两工具的世界真值。目标关节依次为 (1/4, atan2(4,3), 1/8, atan2(−3,4), −1/16)，种子为 (−1/8,−0.65,−0.1,0.8,0.15)。镜像试验将点和移动轴用 S=diag(−1,1,1) 变换、转轴按轴向量用 −S 变换、所有姿态用 S R S 变换。[双臂真值](../../tests/reference_newton/robot/site_ik_fixture.py)

| 独立构型 | 物理预期 |
| --- | --- |
| 一条自由滑轨同时要求 −1 m、+1 m | 权重 1、2 乘残差，平衡点为 0.6 m，两项目标误差为 1.6、0.4 m，输出明确的冲突目标对。 |
| 两根同轴滑轨有硬上界 0.05、0.4 m | 0.43 m 可由 0.05+0.38 m 达到；1 m 目标的最小剩余距离为 0.55 m。两个界上的活动关节均须报告。 |
| 单轴方向与半周姿态 | 关节位于上界时仍可沿负向转到 −π；绕工具轴的已有滚转保留。三根空间转轴的完整安装姿态单独核对。 |
| 同一站点要求身份姿态及 +Y 指向 | 绕 Z 的最佳折中为 π/4，同体的两个方向要求不能同时满足。 |

复现使用原有真实限力 PD 和停止力学，把 IK 求得的关节目标保持 1.5 s，推进 192 个物理子步。关节力/力矩上限依次为 (12 N,1.5 N·m,4 N,1.5 N·m,4 N)。跟踪试验从同一个非解种子出发，逐子步检查驱动力上限，并从实际接受的关节状态重新做 FK。[专项与执行器试片](../../tests/test_multi_site_ik.py) · [实际驱动程序](../../src/phymeo/solvers/driven_robot.py)

## Newton 的可比范围

Newton 1.7 用 `IKObjectivePosition` 和 `IKObjectiveRotation` 为两个偏心工具分别建立位置和姿态行；`IKObjectiveJointLimit` 产生越界软罚残差。求解后调用实际 `eval_fk` 和 `SensorFrameTransform` 读取两工具相对世界站点的位姿。固定支点的单轴指向可等价地通过一个单位偏心点的世界位置比较。普通移动站点的自由滚转在 Phymeo 单独验收。对于一条滑轨，在 +1 m 位置目标、+0.25 m 上界、界外残差权重 2 时，Newton 的软罚最优为 0.4 m；PhyMeo 的硬界为 0.25 m。[Newton 对照程序](../../tests/reference_newton/robot/multi_site_ik.py) · [Newton IK 公共 API](https://newton-physics.github.io/newton/1.7.0/api/newton_ik.html)

## 本机实测

| 双臂两个工具的最坏实测误差 | 位置 (m) | 完整姿态 (rad) |
| --- | ---: | ---: |
| Phymeo HOST FP64 几何解，原构型与镜像 | 3.175×10⁻¹⁰ | 4.588×10⁻¹⁰ |
| Newton CPU 几何解，原构型与镜像 | 6.644×10⁻⁸ | 1.686×10⁻⁷ |
| Newton CUDA 几何解，原构型与镜像 | 5.665×10⁻⁸ | 8.764×10⁻⁸ |
| Phymeo 真实限力跟踪，原构型 192 个物理子步后 | 2.732×10⁻⁵ | 1.463×10⁻⁵ |

两种双臂构型在 Phymeo 上均经过 11 次迭代。实际跟踪开始时，两工具位置误差为 0.8350、0.6972 m，朝向误差为 1.5773、1.4435 rad。首个 1/32 s 帧结束时位置误差为 0.8353、0.6767 m。44 个子步至少有一个执行器抵达限力，五个关节各自的峰值为 (12,1.5,4,1.5,4)。表中最终实际跟踪残差来自推进到 1.5 s 的关节状态。

Newton 1.7.0 / Warp 1.17.0 在 CPU 和 RTX 4060 Laptop 的 CUDA 上各完成 5 组求解与世界系测量。加权冲突的两端实测滑移均为 0.599998593 m；软限位两端均为 0.400002092 m，超过上界约 0.150002092 m。相同单滑轨输入下，PhyMeo 实测 q=0.25 m、位置残差 0.75 m。固定支点的方向试片两端均保留 0.625 rad 滚转。专项共 12 项通过，相邻单点 IK 和真实驱动的 22 项回归通过。最初缺少模块的退出码为 2，最终专项、两端 Newton 与静态检查的退出码均为 0。原始收据位于 `artifacts/p65-night-multisite-ik-20261104-01`。

## 复现

```powershell
python -m pytest -s -q tests/test_multi_site_ik.py
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/multi_site_ik.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/multi_site_ik.py --device cuda:0
```

当前产品求解在 NumPy HOST 上运行。在接触、碰撞净空和路径可行性没有单独验证的情况下，这个关节目标只表示站点几何和显式关节界。
