# 经典 Panda 独立步进与相同采样接触律

沿用 120 g、46 × 58 × 40 mm 长方体和原来两套夹持命令。臂的八个目标时刻、九关节起始状态、1 mm 面内偏置、12 mm 抓取高度、材料、0.625 ms 物理步及有限桌面均取自冻结清单。两条试验更改的输入仅为闭合开度，强夹 24 mm，弱夹 44.8 mm。清单只含起点和控制定义。[原接触配置与两个输入](p65-panda-loaded-box.md) · [冻结用例](../../tests/test_panda_loaded_boxes.py)

## 每步的施力与读回

驱动器读取 Newton 自身入步 `joint_q/joint_qd`，用相同的 `robot_dynamics_terms` 和 `_servo_effort` 计算九轴伺服力。Newton 1.7.0 Featherstone 独立推进各刚体和关节。每侧导入指面内的四个材料点位于两条面内轴各自 ±5 mm 处。用 Newton 的身体姿态和质心线角速度得到各点的世界位置和速度。长方体另用八个顶点测试有限台面。两端都调用 Phymeo 已有的逐点弹簧、阻尼与切向函数。深度 δ>0 时法向大小为 max(0, kδ−d·vₙ)，切向为 −μN·vₜ/max(|vₜ|,0.04 m/s)。指垫单点 k=1200 N/m、d=1.2 N·s/m、μ=0.65，台面单点对应 2200、3.0、0.45。[采样力](../../src/phymeo/solvers/sampled_box_contact.py) · [独立步进](../../tests/reference_newton/robot/panda_sampled_loads.py)

每个接触在同一世界点向手指和物体施等反力，分别累加关于各自质心的力矩。固定台面的支撑力加在物体上。每个物理步清除旧力，经 `state.body_f` 注入接触，`control.joint_f` 只施九个伺服力。向 `solver.step` 传 `contacts=None`，原生流形不参与施力。记录的状态为出步状态，同一行的载荷是该步实际用过的力。另以 Newton 自身的 q/v 在 17 个时刻复查指垫位姿与速度；机器人其它结构的轴向保守净空沿用原验收程序。[冻结命令和模型设置](../../tests/reference_newton/robot/panda_loaded_box.py)

## 孤立台面控制

2 kg 长方体的三个半边长为 (1/8, 1/4, 1/16) m。底面压入 1/1024 m，四个角各承 1 N；重力为 2 m/s²。以 1/128 s 真实出步。初始切速 0.02、0.08 m/s 时，实测总切向力分别为 −1、−2 N，绕质心 Y 轴的力矩为 +0.0625、+0.125 N·m，耗散功率为 −0.02、−0.16 W。受载出步的 vₓ 分别为 0.01609375、0.07218750 m/s，ωᵧ 为 0.03750000、0.07500000 rad/s，vz 均为 0。复用相同两组穿入状态，在没有外部载荷时两次都得到 vz=−0.015625 m/s 和零角速度。四个出步独立对照手算结果，绝对容差 2×10⁻⁷。[解析与真实原生步](../../tests/reference_newton/robot/panda_sampled_controls.py) · [六维力和速度原数](../../artifacts/p65-panda-matched-loads-20261104-01/isolated.json)

## 强夹搬运与弱夹留台

同一份经典 Panda URDF（SHA-256 `2b5bd00abb4ba68e91d532d2d778da64d1c942a9b22aa7acd58dd12a713971fd`）和两份原控制清单在两个引擎各自积分。Newton 使用实际 CPU Featherstone 和 HOST 接触求值。下表的“搬运”取 1.35 s，双侧较小指力和台面峰值取 0.95–1.35 s，松指后均值取末 50 ms。

| 物理量 | Phymeo 强夹 | Newton 同律强夹 | Phymeo 弱夹 | Newton 同律弱夹 |
| --- | ---: | ---: | ---: | ---: |
| 搬运时最低点距台面（mm） | 58.268 | 58.237 | −0.134 | −0.134 |
| 搬运时 X 位移（mm） | 77.063 | 77.011 | 0.151 | 0.151 |
| 搬运窗双指较小法向力的最小值（N） | 5.407 | 5.429 | 0 | 0 |
| 搬运窗桌面法向峰值（N） | 0 | 0 | 1.1772 | 1.1772 |
| 最终目标误差（mm） | 0.488 | 0.477 | 74.849 | 74.849 |
| 松指后指面力峰值／桌面均值（N） | 0／1.1772 | 0／1.1772 | 0／1.1772 | 0／1.1772 |

两个 Newton 窗口各实际推进 3,600 步。强夹通过原有搬运、放下、净空和限位判据；弱夹双侧同时受力的峰值仅 0.353 N，保留为失抓。弱夹的微量负净距来自台面柔顺变形。强／弱的其他机器人形状对物体最小轴向净距为 4.175／6.880 mm，关节限位最大违反量都为零。物体所受世界力与实际线动量改变量的单步残差最大分别为 4.35×10⁻⁸／3.13×10⁻¹⁰ N·s。跨两窗，指面成对合力、合力矩和含台面在内的功率账残差最大为 2.53×10⁻¹⁵ N、1.90×10⁻¹⁶ N·m 和 5.56×10⁻¹⁶ W。两窗各 17 次采样的指面位置、速度分量最大误差为 2.18×10⁻⁷ m、1.88×10⁻⁷ m/s。[强夹真实轨迹](../../artifacts/p65-panda-matched-loads-20261104-01/sampled-strong.json) · [弱夹真实轨迹](../../artifacts/p65-panda-matched-loads-20261104-01/sampled-weak.json)

在原 Phymeo 三维回放的 1.350 s 画面，强夹物体悬空、双指示数为 5.42／5.44 N、桌面为 0；换到弱夹读数为 0／0 N、桌面 1.18 N。同一末时刻的 Newton 原始行，强夹实际质心 (0.384901, 0.000086, 0.486033) m、双指 5.430／5.430 N、桌面为 0，弱夹质心 (0.308042, 0.000015, 0.427498) m、桌面 1.1772 N。原画面显示 Phymeo 保存的状态。两份新 Newton JSON 同样提供完整实际 q、物体位姿、载荷、尺寸与握宽，可直接交给现有回放入口。[原图](../../artifacts/p65-load-alignment-batch-20261104-01/panda-carry-1p35.png) · [浏览器原读数](../../artifacts/p65-load-alignment-batch-20261104-01/playback-qa.json) · [回放导出入口](../../examples/export_panda_cube.py)

## 同一原生接触行单独替换切向

沿用上一组真实 Newton 轨迹的前 16 步及首次指碰。每行的见证点、两侧姿态与速度、法向六维力和力臂固定。用原生行含摩擦与零摩擦的六维力之差求切向方向及关于同一质心的力矩，把切向力大小变为 μN|vₜ|/max(|vₜ|,0.04 m/s)。逐行求和前，对一条有独立已知力臂的合成行另核对六维结果和功率。[同见证替换](../../tests/reference_newton/robot/panda_fixed_witness_traction.py) · [原生行的真实核与动量核对](p65-panda-loaded-box.md#原生轨迹从哪一步分开) · [17 个时刻的结果](../../artifacts/p65-panda-matched-loads-20261104-01/same-witness-tangent.json)

| 固定的 Newton 入步见证 | 原生接触行 | 仅更换切向幅值 |
| --- | ---: | ---: |
| 2.5 ms，四个台面点的合力 Y 分量（N） | −2.157193 | −0.002230 |
| 同一时刻的切向功率（W） | −0.001164 | −0.00000119 |
| 422.5 ms，首次单侧指碰对物体的合力 Z 分量（N） | +54.37712 | +0.70986 |
| 同一指碰的切向功率（W） | −1.82566 | −0.02453 |

这些原生见证上的差量可归因于切向幅值。整条轨迹切换为材料采样时，见证集合及法向阻尼规则也一同改变，随后两端的状态分别演化。上述两处单时刻对照没有推进替换后的下一状态。原 Newton 有限盒原生接触配置仍保留独立结果，强夹提前离开桌面且最终落点偏差 13.23 m，弱夹在搬运时抬起但放置误差 20.41 mm。此次独立步进说明同一材料采样接触律能在两个 CPU 动力学路径复现这两种夹持结果；整窗改善的各因素份额仍没有单独量出。[原生两条完整轨迹](p65-panda-loaded-box.md)

复现时用原 Phymeo 命令导出两个 `*.inputs.json`。在可导入 Phymeo 源码和 Newton 1.7.0 的环境中，将下列 `MODE` 分别换为 `carry`、`slip`，每次使用对应清单及新输出文件。接触行诊断读取上一批原生核逐行输出。[强夹运行收据](../../artifacts/p65-panda-matched-loads-20261104-01/05-strong-sampled/final.json) · [弱夹运行收据](../../artifacts/p65-panda-matched-loads-20261104-01/06-weak-sampled/final.json) · [切向对照收据](../../artifacts/p65-panda-matched-loads-20261104-01/07-identical-witness-tangent/final.json)

```text
python tests/reference_newton/robot/panda_sampled_controls.py --out isolated.json
python tests/reference_newton/robot/panda_sampled_loads.py --device cpu --urdf src/phymeo/builders/panda_assets/panda.urdf --inputs control.inputs.json --expect MODE --out newton-motion.json
python tests/reference_newton/robot/panda_fixed_witness_traction.py --inputs polymer-strong.inputs.json --rows native-authoritative-wrench.json --out same-witness.json
```
