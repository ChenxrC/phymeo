# 浮根 MJCF 与真实惯量框架首片（2026-11-09）

新增 `phymeo.builders.floating_mjcf.load_floating_mjcf`，可通过原 `ContactSceneBuilder.add_floating_urdf(..., loader=load_floating_mjcf)` 接入 HOST。首域是显式惯量、radian、单自由根和每子体一个有限 hinge，支持物理球/胶囊/盒与零碰撞掩码的视觉项过滤。

子体采用其真实惯量主轴作为内部模型框架；同步转换 joint origin/orientation/axis、COM 和碰撞局部点。`FloatingMJCFAsset.principal_to_authored_xyzw` 显式保留返回原 authored 框架的旋转。根仍保原 body frame 和独立惯量框架，不删非对角惯量、不固定基座、不裁掉关节。

独立单/多姿态与非法声明共 **10 HOST 测试通过**。含绕转后的有限胶囊端点、主轴回转、缺省角度单位拒绝和非零 armature/damping/frictionloss 拒绝。当前引擎未接入这些执行器项，因此不能在导入时静默丢弃。

本机 G1 primitive 资产（官方 `g1.xml` 的物理几何，加同一官方场景的显式 motor 声明）导入 **29 关节、30 刚体、33 碰撞形状**。在固定动作帧 0/20/200，按名称独立匹配 MuJoCo 刚体，位置、COM、authored 朝向最大误差 **1.6654e-15**。FP32 动作 quaternion 在两端同样归一化后比较。此检查是资产/运动学，不是动力学或站立通过。

主证据 `artifacts/p65-g1-research-20261109-01/13-phymeo-asset-check/` 与 `phymeo-asset-check.json`；最终测试 `22-final-mjcf-tests/`。真实策略/物理失败见[训练与人形首批报告](p65-training-humanoid-first.md)。全 M、执行器动力学、设备 29 关节容量和复杂 MJCF 机制仍待后继。
