# 机器人资产的圆柱、胶囊与椭球

本地 URDF 可将 `<cylinder radius="r" length="L"/>` 导入为标准 Z 轴圆柱，半径为 r，半高为 L/2。本地 MJCF 的 cylinder、capsule 使用 `size="r h"`，h 为圆柱型轴段的半长；包含球帽的胶囊最大全长为 2(h+r)。ellipsoid 的三个 size 是局部 X/Y/Z 三个半轴。碰撞中心、安装坐标与本体惯性分别保留。两个导入入口仍返回同一 RobotAsset。[URDF 入口](../../src/phymeo/builders/urdf.py) · [MJCF 入口](../../src/phymeo/builders/mjcf.py)

```python
import numpy as np
import phymeo as ph

asset = ph.load_mjcf("tests/reference_newton/robot/solid_arm.mjcf.xml")
paths = asset.collision_motion(np.array([np.pi / 2]))
with ph.CollisionWorkspace(asset.collision_model) as workspace:
    pairs = ph.detect(asset.collision_model, paths, ph.CollisionPolicy(margin=0.1), workspace)
```

## 局部几何与实际关节坐标

圆柱和椭球在注册时保存材料中心、主尺寸及相对本体的安装旋转。调用 `collision_motion(q)` 后，中心来自实际 FK，`body_orientation_xyzw` 给出该本体到世界的初始旋转。碰撞查询将本体旋转与局部安装各用一次。胶囊沿轴取两个球心，由现有圆形碰撞通路使用这两个材料点及实际半径。[共享资产注册](../../src/phymeo/builders/_robot_asset.py) · [曲面坐标约定](../../src/phymeo/constraints/curved_solids.py)

对于 MJCF 圆柱和胶囊，`fromto="x1 y1 z1 x2 y2 z2"` 在本体局部坐标定义两端和中点。端点间距的一半覆盖第二个尺寸（若显式给出），第一个尺寸仍决定半径。胶囊的这两个端点为球帽球心；圆柱的这两个端点为端盖中心。定向的 +Z 按实际 MuJoCo 编译器与 Newton 1.7 导入源码选取“第二点指向第一点”。MuJoCo XML 网页对此方向的概述相反，两个轴对称实体的占据体和无序端点一致。实现对平行、反平行、近反平行和斜向的有效轴分别校验，并具名拒绝零长轴段、错误尺寸个数和不受支持的 `fromto` 与显式位姿组合。[MuJoCo XML 说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom) · [MuJoCo 编译实现](https://github.com/google-deepmind/mujoco/blob/main/src/user/user_objects.cc) · [Newton 1.7 MJCF 导入器](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/utils/import_mjcf.py)

## 两份实际资产及间隙

两份文件都使用世界固定基座、一个绕 Y 的转铰和一个 2 kg 子体。关节位于 (0.25,−0.125,0.25) m，范围为 ±π，子体质心为 (0.0625,−0.03125,0.125) m，主惯量为 (0.125,0.1875,0.25) kg·m²。共同圆柱在子体本地 +X 偏心 1/8 m，局部 Z 通过安装旋转指向 +X，半径 1/8 m、全高 1/2 m。MJCF 还含一个端点编写的胶囊、一个独立旋转的三轴椭球和一个仅用于显示的球。PhyMeo 注册 key 为 (世界,本体,几何序号)。固定圆柱归属本体 0，运动圆柱为 (0,1,0)，MJCF 胶囊、椭球分别为 (0,1,2)、(0,1,3)，原显示项不产生碰撞体。[URDF 文件](../../tests/reference_newton/robot/solid_arm.urdf) · [MJCF 文件](../../tests/reference_newton/robot/solid_arm.mjcf.xml)

取 z=0 的物理半空间。圆柱底端解析间隙为 1/4−sin(q)/8−|sin(q)|/4−|cos(q)|/8 m。胶囊间隙由两球心较低者减半径得到；椭球以世界竖直方向的三轴支撑函数求间隙。下表由公开碰撞查询读回，负值为穿入。

| q (rad) | 共同圆柱 (m) | MJCF 胶囊 (m) | MJCF 椭球 (m) |
| --- | ---: | ---: | ---: |
| 0 | 0.125000 | 约 0 | −0.125000 |
| atan2(3,4) | −0.075000 | −0.112500 | −0.015985257 |
| +π/2 | −0.125000 | −0.062500 | +0.250000 |
| −π/2 | +0.125000 | −0.062500 | +0.125000 |
| π | +0.125000 | +0.375000 | +0.125000 |

π−2⁻²⁴ 的姿态和经旋转、平移的固定根另作留出。真实动力学窗口给 URDF 转铰施加恒定 29/64 N·m 转矩，绕铰惯量为 29/128 kg·m²。24 个 1/32 s 已接受子步逐步用实际 q 进行 FK 和接触查询；第 19 步首次出现负间隙，末态 q=0.5859375 rad、圆柱间隙 −0.061517033 m。该窗口测的是随真实关节运动的几何查询，接触反力没有送回这次自由关节步进。[产品验收与非法资产](../../tests/test_robot_solid_assets.py)

## 后端证据

PhyMeo HOST 通过 20 项专项，另有 1 项真实 TYX CUDA 合同。后者实测一个设备 FK 核和两次碰撞候选核启动，静态曲面窄相仍按现有 HOST 几何通路测量。原有 URDF/MJCF/USD 的球盒、锚点、单位及错误输入选择性回归共通过 31 项，USD 使用既有隔离可选依赖。

Newton 1.7.0 / Warp 1.17.0 使用原生 `add_urdf`/`add_mjcf`、`eval_fk`、`CollisionPipeline.collide` 在 CPU 和 RTX 4060 Laptop CUDA 上分别执行两资产各六姿态。几何 ID 按各引擎自身注册次序读取，同时验证真实本体归属、几何类型、三维尺寸、局部轴、质心、惯量和关节限位。对于斜姿，CPU 和 CUDA 的最低实体表面接触点分别为圆柱 −0.075000077、胶囊 −0.112500072、椭球 −0.015985340 m。已产生见证的几何姿态中，最大竖直误差小于 8.25×10⁻⁸ m。URDF 两端都记录到 4 个实际接触点；MJCF 在 CPU/CUDA 分别记录到 10/11 个，使用每个形状的最低表面点比较支持深度。约零的相切形态允许不同接触数量。对预期明显分离的姿态只核对无接触，不能把解析间隙标为 Newton 实测距离。[Newton 原生读回](../../tests/reference_newton/robot/solid_assets.py)

```powershell
python -m pytest -s -q tests/test_robot_solid_assets.py -m "not tyx"
python -m pytest -s -q tests/test_robot_solid_assets.py -m tyx --require-tyx --tyx-mode=cuda
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/solid_assets.py --device cpu
.\.venv-newton-1.7.0\Scripts\python.exe tests/reference_newton/robot/solid_assets.py --device cuda:0
```

证据根为 `artifacts/p65-night-robot-solid-assets-20261104-01`，保留了旧 URDF 解析器拒绝合法圆柱的首次退出码 1。此批支持局部显式惯性的固定基座标量树和上述有限形状，不覆盖文件网格、新的 USD 曲面或动力学接触响应。
