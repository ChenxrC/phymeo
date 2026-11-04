# 经典 Panda 的自由实心球

普通抓放入口接受 `payload_geometry={"kind": "sphere", "radius_m": R}`。负载为质量分布均匀的自由刚体，机体系主惯量均为 2mR²/5。每一侧保留原有四个指垫材料点。对世界点 p 和球心 c，穿入深度为 max(0, R−|p−c|)，法向沿 c 到 p。相对速度减去同点的球速 vCOM+ω×(p−c)。作用在球上的反力在同一世界点累加关于质心的力矩。位于球心的指垫点没有唯一径向法向，会显式拒绝。[物体几何与受力](../../src/phymeo/experiments/panda_payload.py) · [公共抓取入口](../../src/phymeo/experiments/panda_cube.py)

球在水平桌面内部的真实接触点位于球心正下方 R 处，切向力可以令球转动。在有限台边和角点，使用台面实体上距球心最近的点求法向，在相应球面点求速度和力矩。起步整颗球的水平投影须落在桌面上。独立解析例同时检查斜向指面、包围盒内但球外的材料点、带自转的底部单点及有限台边。2.5 cm 球在斜向压入 5 mm 的解析例中，实际反力为 (−0.6, −0.8, 0.25) N，力矩为 (0.004, −0.003, 0) N·m；台面单点承 4 N 时的切向力为 −1.5 N，绕 Y 力矩为 0.0375 N·m。[冻结用例](../../tests/test_panda_sphere_payload.py)

Newton 1.7.0 CPU 从同一经典 Panda URDF 和只含初态、控制及材料的清单出发，注册真正的 `GeoType.SPHERE`，读取原生形状编号、半径和惯量。接触计算在 Newton 自己的入步状态上使用相同材料律，实际手指通过原生关节承反力，Featherstone 独立推进。不重复叠加 Newton 原生流形。两端每步均保存指面实际世界点和力、台面作用点和力、关于球心的合力矩、关节及球位姿、速度。旧箱体清单仍按原格式读取。[匹配接触律](../../tests/reference_newton/robot/panda_sampled_loads.py) · [已分开记录的原生盒流形对照](p65-panda-loaded-box.md)

## 首次冻结的直径 50 mm 球

80 g、R=25 mm、局部主惯量均为 2.0×10⁻⁵ kg·m²。两种指垫摩擦系数为 0.65 和 0。名义抓取在球心上方 8 mm，开度 76→24 mm。两引擎的四条轨迹各真实推进 3,600 个物理子步。预先要求正常摩擦球在 0.95–1.35 s 持续双边承载并于目标放下，零摩擦球保持在有限桌面。**首次两个应用目标均未达到**，原始退出码和稠密轨迹完整保留。[Phymeo 两个首次断言](../../artifacts/p65-panda-sphere-20261105-01/04-sphere-phy/final.json) · [Newton 正常摩擦](../../artifacts/p65-panda-sphere-20261105-01/05-sphere-newton-grip/final.json) · [Newton 零摩擦](../../artifacts/p65-panda-sphere-20261105-01/06-sphere-newton-slip/final.json)

| 1.35 s／最终物理量 | Phymeo μ=0.65 | Newton μ=0.65 | Phymeo μ=0 | Newton μ=0 |
| --- | ---: | ---: | ---: | ---: |
| 最低点相对桌面高程（mm） | −0.220 | −0.214 | −2443.4 | −2437.0 |
| 实际 X 位移（mm） | −2.665 | −2.213 | −661.1 | −660.3 |
| 末态目标误差（mm） | 83.10 | 81.82 | 12736 | 12722 |

正常摩擦球有过短暂抬离。以 Phymeo 为例，0.70／0.75 s 两侧各承约 4.72／4.14 N，0.80 s 各侧已降至约 0.25 N，0.85 s 两侧均为零。0.55、0.70、0.75、0.80 s 指面对球的实际净竖直力分别为 −2.441、+0.939、+0.789、−0.166 N，球重为 0.7848 N。球在目标搬运窗前滑出，随后由桌面承重。零摩擦球在 0.60 s 尚在桌面，已获 −0.841 m/s 的 X 速度，随后越过有限台边。机器人非指垫结构只在本例中监测净距，首个该类交叠以后的自由落体仅保留作轨迹诊断，不能计作真实无穿透运动或稳定留台。[Phymeo 正常摩擦](../../artifacts/p65-panda-sphere-20261105-01/phy-results/test_ordinary_panda_sphere_win0/sphere-grip.json) · [Phymeo 零摩擦](../../artifacts/p65-panda-sphere-20261105-01/phy-results/test_ordinary_panda_sphere_win1/sphere-slip.json) · [Newton 正常摩擦轨迹](../../artifacts/p65-panda-sphere-20261105-01/sampled-grip.json) · [Newton 失抓轨迹](../../artifacts/p65-panda-sphere-20261105-01/sampled-slip.json)

原装手掌胶囊中心线的手坐标为 z=40 mm，半径 40 mm。指垫中心的手坐标为 z=58.4+45.25=103.65 mm。对中的竖直夹具在与半径 R 的球等高抓取时，掌–球名义净距为 23.65 mm−R。直径 50 mm 时等高将与掌相交 1.35 mm，保留原球与原装几何。直径 40 mm 时同一等高姿态有 3.65 mm 名义掌净距；24 mm 闭口、每个面点横竖各 ±5 mm 仍能接触球。原料密度给出后者质量 40.96 g，三个主惯量均为 6.5536×10⁻⁶ kg·m²。按每侧 620 N/m 的驱动和 1200 N/m 的单点弹簧，等高夹持的静态半开距约 17.8 mm。上下两排的初始径向竖直分量相消，静态计算只用于冻结这个独立规格。[原手工装尺寸](../../src/phymeo/builders/panda_assets/panda.urdf)

对 50 mm 正常摩擦球，两引擎的台面载荷首次归零分别在 0.630625、0.631250 s，升到 37.11、36.51 mm 后双指分别于 0.801250、0.799375 s 完全失载，0.873750、0.871875 s 重新有台面载荷。Newton 在 0.758125 s 的径向分量向下 3.931 N；切向向上 4.531 N，最大可能向上净力 0.622 N 已小于球重 0.7848 N。零摩擦轨迹中，球心投影分别于 0.638125、0.638750 s 越过台边，边缘接触持续到 0.643125、0.644375 s。两端首个漏算反力的结构都是 `panda_link1_sc_0`，在第 1339／1340 个从零计数的入步，0.836875／0.837500 s，净距 −0.473／−0.211 mm。上表该轨迹的 1.35 s 与末态位移都发生在截止点之后。[首个结构与台边时刻](../../artifacts/p65-panda-sphere-20261105-01/sphere50-validity.json) · [Newton 独立径向及切向合力](../../artifacts/p65-ragged-seam-payload-20261105-01/newton-support-grip.result.json)

## 独立冻结的直径 40 mm 球

等高抓取，沿用相同工装、材料、闭口和搬运动作。两个引擎都完成了正常摩擦球的持续夹持和放置；**零摩擦球也被搬到目标并放下**。后者检验了“零摩擦会留台”的预期，实测将其否定。输入文件名中的 `slip` 是运行前登记的预期。

| 实测 | Phymeo μ=0.65 | Newton μ=0.65 | Phymeo μ=0 | Newton μ=0 |
| --- | ---: | ---: | ---: | ---: |
| 1.35 s 最低点距台面（mm） | 63.183 | 63.177 | 63.211 | 63.211 |
| 1.35 s 水平 X 位移（mm） | 77.040 | 77.044 | 77.026 | 77.028 |
| 0.95–1.35 s 较小指力的最小值（N） | 3.838 | 3.837 | 3.838 | 3.838 |
| 最终放置误差（mm） | 5.642 | 5.897 | 0.998 | 1.004 |
| 松指后桌面承重（N） | 0.401818 | 0.401819 | 0.401818 | 0.401818 |
| 是否达到各自预注册目标 | 是 | 是 | 否 | 否 |

四窗在搬运段的桌面法向力均为零，松指后指面力也均为零。其他机械结构与球的最小净距约 1.200 mm，机械臂到桌面的保守净距约 8.117 mm。力作用于真实球面径向所在的采样点。零摩擦时，1.35 s 的实际八点合力仍向上 0.40544／0.40547 N，而切向力为零，台面力为零，球重 0.40182 N。这种托持依赖每指两排离散材料点；该结果的验证范围是当前采样接触模型。Newton 的载荷形状号为 41，类型为 `SPHERE`，读回半径为 0.01999999955 m。[Phymeo 正常摩擦](../../artifacts/p65-panda-sphere-20261105-01/eq-phy-results/test_equatorial_40mm_sphere_wi0/sphere-grip.json) · [Phymeo 零摩擦](../../artifacts/p65-panda-sphere-20261105-01/eq-phy-results/test_equatorial_40mm_sphere_wi1/sphere-slip.json) · [Newton 正常摩擦](../../artifacts/p65-panda-sphere-20261105-01/equator-grip.json) · [Newton 零摩擦](../../artifacts/p65-panda-sphere-20261105-01/equator-slip.json) · [独立零摩擦径向力复核](../../artifacts/p65-ragged-seam-payload-20261105-01/eq-newton-support-slip.result.json)

八条真实轨迹合计 28,800 个物理子步。按运行前的目标，只有两个 40 mm 正常摩擦窗口符合预期。产品回归另核实际搬运、载荷、未受力结构和真正的放置结果。八份原始轨迹在最后的离线复核中分别保留 `actual_carried`、`actual_placed`、`registered_goal_met` 和未满足的具体条件。检查仅重读保存状态。基准过程中还复查一条原长方体搬运、四个独立盒体外力步，以及已有 CLI 入口。[原始预期逐项读数](../../artifacts/p65-panda-sphere-20261105-01/application-goals.json) · [真实球三维回放](../../artifacts/p65-ragged-seam-payload-20261105-01/panda-grasp-replay.html) · [浏览器形状与载荷读回](../../artifacts/p65-ragged-seam-payload-20261105-01/playback-qa.json) · [角冲量审计](../../artifacts/p65-ragged-seam-payload-20261105-01/eq-newton-angular-grip.result.json)

```text
python examples/panda_cube.py --sphere-radius .020 --mass .04096 --closed-width .024 --grip-above-com 0 --friction .65 --trace sphere40.json --inputs-out sphere40.inputs.json
python tests/reference_newton/robot/panda_sampled_loads.py --device cpu --urdf src/phymeo/builders/panda_assets/panda.urdf --inputs sphere40.inputs.json --expect carry --out newton-sphere40.json
```
