# 经典 Panda 斜置长方体的偏心抓取

同一台 Panda 从静止夹一只 120 g、46 × 58 × 40 mm 的均匀长方体。两次都命令夹爪从 76 mm 闭合至 24 mm，再提起、平移 75 mm、放下和松指。物体先绕世界 +Z 转 12°，再将质心放在指垫中心线的世界 (+3, −1) mm 处。指垫中心的名义抓取高度在质心上方 12 mm。偏心来自抓取位置，质心仍在几何中心，局部主惯量为 (4.964、3.716、5.480) × 10⁻⁵ kg·m²。两份初态与命令清单逐字段对照，唯一差异是指垫摩擦系数 0.65 和 0。[冻结输入与验收](../../tests/test_panda_rotated_box.py) · [原来未旋转的强弱夹对照](p65-panda-matched-contact-law.md)

## 初始姿态、控制和目标

公开入口 `run_panda_cube(initial_yaw_rad=...)` 的零角将物体第一局部轴对准初始抓取轴。在该经典 Panda 初态，第一轴沿世界 −Y。输入 +12° 后它的世界偏航为 −78°，Phymeo 的实际初始 XYZW 为 (0, 0, −0.629320391, 0.777145961)，Newton 为 (0, 0, −0.629320383, 0.777145982)。物体质心初位 (0.309890567, −0.001000000, 0.427632052) m，放置目标 (0.384890567, −0.001000000, 0.427632052) m。目标用于计算放置误差，驱动只读取机器人自身的九个关节状态。[公开入口](../../src/phymeo/experiments/panda_cube.py) · [控制清单与每步施力](../../tests/reference_newton/robot/panda_sampled_loads.py)

该参数接受以弧度表示的单个有限偏航角，允许物体平放时绕世界竖直轴旋转。传入布尔值、非有限角或四元数数组会在求力前失败。若起步实际旋转后的任一角点超出有限台面，会在第一次推进前拒绝。例如该物体沿世界 Y 偏移 31 mm，零角时最远角为 54 mm，绕竖轴转 −45° 后约为 67.77 mm，已超出桌面 55 mm 半宽。本次未引入倾斜入场的接触初态。[输入与足迹检查](../../tests/test_panda_rotated_box.py)

## 四个独立物理窗口

Phymeo HOST 和 Newton 1.7.0 Featherstone CPU 各自从清单起点运行。每窗 3,600 个 0.625 ms 物理步，共 2.25 s。Newton 在自己的运动状态上重建八个指垫点和八个物体角点的速度与载荷，手指经真实刚体和关节受反力。这里采用相同的采样接触律，原生流形不叠加。上一批原生有限盒接触的结果另存。[采样与原生接触的区别](p65-panda-loaded-box.md) · [本批两个 Newton 窗口的原始退出与源码绑定](../../artifacts/p65-panda-rotated-offset-20261104-01/06-newton-grip/final.json) · [滑落窗口收据](../../artifacts/p65-panda-rotated-offset-20261104-01/07-newton-slip/final.json)

| 物理量 | Phymeo，μ=0.65 | Newton，μ=0.65 | Phymeo，μ=0 | Newton，μ=0 |
| --- | ---: | ---: | ---: | ---: |
| 1.35 s 物体最低点到台面（mm） | 54.783 | 54.782 | −0.134 | −0.134 |
| 1.35 s 实际 X 位移（mm） | 75.510 | 75.516 | −0.671 | −0.669 |
| 搬运窗两指较小法向力的最小值（N） | 5.405 | 5.405 | 0 | 0 |
| 全窗两指较小法向力的峰值（N） | 10.502 | 10.511 | 6.496 | 6.496 |
| 搬运窗桌面法向力峰值（N） | 0 | 0 | 1.1772 | 1.1772 |
| 最终放置目标误差（mm） | 0.550 | 0.545 | 75.677 | 75.676 |
| 非指垫结构到物体最小轴向净距（mm） | 3.757 | 3.758 | 4.646 | 4.646 |
| 机器人到桌面最小轴向净距（mm） | 20.645 | 20.645 | 20.461 | 20.461 |

原有搬运窗为 0.95–1.35 s。两端的正常摩擦箱都持续离台，放回后末 50 ms 指垫力为零、桌面承重 1.1772 N。零摩擦箱也曾承受双侧夹力，提起夹爪时箱子留在桌上；−0.134 mm 为柔顺台面的持载压入量。四窗的关节限位违反量均为零。Phymeo 的非指垫最小净距分别出现在 0.933125 s 的右指斜面和 0.6225 s 的左指斜面。净距是各物理采样姿态的保守轴向下界。[Phymeo 正例](../../artifacts/p65-panda-rotated-offset-20261104-01/phy-results/test_rotated_offset_box_window0/rotated-grip.json) · [Phymeo 滑落](../../artifacts/p65-panda-rotated-offset-20261104-01/phy-results/test_rotated_offset_box_window1/rotated-slip.json)

在 1.35 s，正常摩擦物体的真实 XYZW 在 Phymeo 为 (−0.043587, 0.034233, −0.706902, 0.705137)，在 Newton 为 (−0.043587, 0.034261, −0.706891, 0.705146)。初始条件只约束平放，后续自由刚体在偏心受力下可以倾转。Newton 两窗单步外力冲量与实测线动量增量的最大残差分别为 4.40×10⁻⁸、1.45×10⁻⁸ N·s。原始 `applied_payload_wrench` 的前三项是世界接触合力，后三项是关于实际质心的世界力矩；重力单独作用于质心。[Newton 正例](../../artifacts/p65-panda-rotated-offset-20261104-01/sampled-grip.json) · [Newton 滑落](../../artifacts/p65-panda-rotated-offset-20261104-01/sampled-slip.json)

## 复现

下面的普通入口导出正常摩擦的受载轨迹和控制清单。执行零摩擦负例时将摩擦系数改为 0 并使用新的输出路径。两份清单的 SHA-256 分别以 `a201fe83b009`、`47b6ca15904f` 开头，各自与对应 Newton 轨迹中的输入哈希相同。[命令入口](../../examples/panda_cube.py) · [7 项 HOST 专项](../../artifacts/p65-panda-rotated-offset-20261104-01/04-yaw-phy-full/final.json)

```text
python examples/panda_cube.py --box-size .046 .058 .04 --mass .12 --closed-width .024 --initial-xy-offset .003 -.001 --grip-above-com .012 --initial-yaw-rad .20943951023931953 --friction .65 --trace grip.json --inputs-out grip.inputs.json
python tests/reference_newton/robot/panda_sampled_loads.py --device cpu --urdf src/phymeo/builders/panda_assets/panda.urdf --inputs grip.inputs.json --expect carry --out newton-grip.json
```

在零摩擦 Newton 窗口用对应清单和 `--expect slip`。[冻结强夹清单](../../artifacts/p65-panda-rotated-offset-20261104-01/phy-results/test_rotated_offset_box_window0/rotated-grip.inputs.json) · [冻结零摩擦清单](../../artifacts/p65-panda-rotated-offset-20261104-01/phy-results/test_rotated_offset_box_window1/rotated-slip.inputs.json)
