# 有限布面上的弹性静持、转滑与释放

`MaterialStickSlipStepper` 在自由材料顶点和固定三角面的内部接触时计算切向弹力。法向层根据真实间隙测力，自由顶点受切向力，对侧三个固定节点按同一材料见证的重心权重承受反力。静持中允许弹性形变。弹性位移与不可逆滑移分别以米记录。[实现](../../src/phymeo/solvers/deformable_stiction.py) · [物理与事务测试](../../tests/collision/test_deformable_stiction.py)

## 离散力学

设 `k` 为切向刚度，`m` 为顶点质量，`e₀` 为上步接受的切向弹性位移。按外力、重力和本步法向力求出自由顶点的预测切向位移 `d_free`。黏着试解同时满足 `d = d_free + h² f / m` 和 `f = −k(e₀ + d)`。首次屈服以静摩擦上限 `μ_s N` 判定，超限时将牵引幅值降至 `μ_k N`。持续同向滑动沿用动摩擦上限，卸载进入屈服圆内时恢复弹性响应。

粒子步接受后，实际切向位移 `d` 取自顶点坐标差。静持时全部写入弹性形变。滑动时取 `e₁ = −f/k`，塑性增量为 `d_p = e₀ + d − e₁`。每一步核查自由顶点速度、固定支撑位置和下列功账。

~~~text
W_t = f · d
U   = k |e|² / 2
D_p = −f · d_p
D_i = k |e₁ − e₀|² / 2
W_t + (U₁ − U₀) + D_p + D_i = 0
~~~

`D_i` 是隐式弹性更新的离散耗散。切向弹簧卸载时可向材料返还机械功。离开法向作用层后的首次采样给出零切力，已结束站点释去的剩余弹簧储能单列。重新接触从零剪切历史开始。[原速度正则化动摩擦及其独立历史](p65-dd-material-sliding.md)

## 求解范围

一个活跃顶点在一个物理步内只能有一个固定、静止、非退化三角面的内部作支撑。多个独立顶点可以共用该面。当前按入步材料几何取力，同一顶点受多个面约束、移动支撑、边–边、面边界及跨面黏着仍需其他求解。已受载顶点穿过支撑平面的子步会被拒绝。拒绝的粒子候选保留原检查点。该入口使用自由粒子积分，整布膜内刚度、折叠与机器人抓取不在本窗口范围。

切向本构、材料窄相及状态审计在 HOST。传入同一后端的 `TyxNormalLayer`、`TyxSemiImplicitSolver` 和 `CollisionWorkspace` 后，设备分别承担法向层、粒子推进和宽相。现有 Newton VBD 对照测量的是入层冲击与动摩擦，尚无本静持本构的同输入 Newton 轨迹。

## 实际物理窗口

固定三角面与上层自由顶点间隙为 1/32 m。法向层宽 1/16 m、刚度 64 N/m，顶点同时受到 2 N 向下外力。初态法向力为 2 N。切向静摩擦系数为 0.5，动摩擦系数为 0.25，质量为 1 kg，切向刚度为 256 N/m，步长为 1/16 s。

| HOST FP64 窗口 | 接受状态上的观测 |
| --- | --- |
| 0.5 N 恒定横载，32 步 | 第一步弹性位移 0.9765625 mm。第 32 步速度为零，弹性位移 1.9530952 mm、反力 −0.49999237 N、累计塑性滑移为零。静平衡参照为 0.5/256 m。撤去横载后弹簧先做正功。48 步后反力小于 4×10⁻⁸ N，储能小于 3×10⁻¹⁸ J。 |
| 静摩擦边界及超载 | 试牵引恰为 1 N 时仍黏着。从静止直接施加 4 N 横载时，实际切力为 −0.5 N，速度增量为 0.21875 m/s。总位移 13.671875 mm，其中弹性形变 1.953125 mm、不可逆滑移 11.71875 mm。 |
| 超载步的功与能量 | 实际切力做功 −0.0068359375 J，新增储能 0.00048828125 J、塑性耗散 0.005859375 J、隐式耗散 0.00048828125 J。固定面三个节点的反力分别为 +0.125、+0.125、+0.25 N。 |
| 改质量、方向和材料行序 | 2 kg 顶点、128 N/m 刚度，面法向 (−0.6,0,0.8)，沿 (0.48,0.8,0.36) 施加 12 N。首步切向速度 0.359375 m/s、不可逆滑移 18.5546875 mm、切力功 −0.01123046875 J。两个朝向都从同一超载状态分支，持续加载保持滑动，撤去自由切向速度则重新黏着并返还能量。 |
| 离层、重接与事务 | 提升至 3/32 m 后，下一入步的六个节点切力均为零，旧历史结束。重新进入 1/32 m 开距时弹簧从零开始。两个独立顶点共用固定面可同时静持和滑动。两面同时约束同一顶点、穿过支撑平面或粒子候选速度不符均拒绝，原状态仍能重放。 |

实际 TYX interpret 和 CUDA 的每步位置、速度、受力、弹性位移、塑性滑移及功共 15 条记录一致。前 12 步施加 0.5 N 横载，后 3 步施加 4 N。第 13 步转滑，最后切向速度 0.65625 m/s、累计塑性滑移 82.0617676 mm。CUDA 完成 30 次宽相核、15 次法向层核，15 个粒子物理步均由 `tyx:cuda` 接受。法向层实际上传 99,840 B、下载 38,400 B。这组计数仅用于确认执行路径。

HOST 专项 7 项通过，interpret 和 CUDA 分别 1 项通过，专属 Ruff 与格式检查通过。基线为 `a3814dc`。原始证据包括 [32 步静持和独立递推](../../artifacts/p65-day-material-stiction-20261104-01/pytest-host/test_hold_distinguishes_elasti0/material-static-hold.json)、[超载功账](../../artifacts/p65-day-material-stiction-20261104-01/pytest-host/test_overload_slides_at_the_ki0/material-breakaway.json)、[倾斜面功账](../../artifacts/p65-day-material-stiction-20261104-01/pytest-host/test_overload_slides_at_the_ki1/material-breakaway.json)和 [CUDA 逐步轨迹](../../artifacts/p65-day-material-stiction-20261104-01/pytest-cuda/test_native_normal_and_particl0/material-stiction-device.json)。CUDA 收据 SHA256 为 `57037cb7d773c0130e8f0ba7087542f31a1aaac24239acb89f8d38f22b6dd9a1`。

~~~powershell
python -m pytest -q tests/collision/test_deformable_stiction.py -m "not tyx"
python -m pytest -q tests/collision/test_deformable_stiction.py -m tyx --require-tyx --tyx-mode interpret
python -m pytest -q tests/collision/test_deformable_stiction.py -m tyx --require-tyx --tyx-mode cuda
~~~
