# 自由球与真实机器人连杆的双向受力

同一 50 mm 零指面摩擦球仍从指间滑出并离开有限桌面。球随后碰到第一臂节的有限胶囊。Phymeo 和 Newton 现在都在首次接触的同一步改变球的动量及受载机器人的九轴速度。2.25 s 接触窗内没有新的未求解结构。台面以下没有地板，球会在反弹后继续自由落体。[原来只监测结构净距的结果](p65-panda-solid-sphere.md) · [普通 Panda 抓放入口](../../src/phymeo/experiments/panda_cube.py)

## 几何、力和时刻

`SphereLinkContacts` 对任意登记次序的刚性连杆求球形载荷接触。球形连杆使用中心距，胶囊使用到有限轴段的最近点，正交长方体使用有符号距离与最近面、边或角。固支形状能承受反力，固定基座的广义载荷为零。编译资产时验证几何种类、箱顶点排列和正交性。穿入胶囊轴线或箱体的等距分离面时会具名拒绝。Panda 提供实际导入的胶囊、球和箱，两个橡胶指垫由原四点模型单独处理。机器人与桌面保留原来的结构净距监测。该通道在物理子步的起点离散检测，正间隙时不施力。[通用注册与受力](../../src/phymeo/solvers/sphere_link_contact.py)

法向 n 从连杆指向球。两侧分别计算实际表面见证，再把两个见证的中点 p 用作共同作用点。球在 p 的速度为 vCOM+ω×(p−c)，减去对应连杆在同一点的速度得到相对速度。默认结构接触材料的法向刚度为 6000 N/m，阻尼 16 N·s/m，摩擦系数 0.35，正则滑速 0.04 m/s。穿入量 δ>0 时法向力为 max(0,kδ−d·vₙ)，切向力为 −μN·vₜ/max(|vₜ|,vₛ)。球与连杆在 p 受等反力，各自根据真实质心计算力矩。Jᵀ 使用本步的 COM 线速度和角速度雅可比。Newton 的结构载荷只经两侧刚体 `body_f` 进入 Featherstone，九轴关节输入仍只含原伺服力。参考端采用同一注册几何计算见证，原生流形不参与施力。另算 Jᵀ 作为对照。[普通动力学与结构反力](../../src/phymeo/experiments/panda_cube.py) · [Newton 独立步进](../../tests/reference_newton/robot/panda_sampled_loads.py)

每行 `t` 为出步时刻，`structural_contacts`、`structural_joint_load` 和 `applied_payload_wrench` 是该步使用的载荷。结构记录同时保存形状、连杆、两侧表面见证、共同作用点、连杆质心、法向、相对点速、双方法向及切向合力、关于各自质心的力矩和功率。合扳手包含指垫、台面和非指垫连杆三条独立通道，重力另加。原观察模式可用 `structural_response=False` 或 CLI 的 `--observe-structure` 重现。旧清单未含 `structural_material` 时，参考端继续按原模式工作。

## 独立力学试片

有限胶囊轴部的 1 mm 穿入，在 2000 N/m、10 N·s/m、0.1 m/s 入射速度下应产生 3 N 法向力。切向速度 0.06 m/s、μ=0.5 时，实测球力 (3, −1.5, 0) N、球质心力矩 (0, 0, 0.07425) N·m。偏置连杆质心收到 (−0.0825, −0.165, 0.14925) N·m。两体接触功率 −0.39 W，其中弹簧储能增长 0.20 W，法向阻尼与切向滑移共耗散 0.19 W。快速分离的穿入点无拉力。独立的 2 kg 滑轨和 1 kg 自由球从静止受 2 N 等反力，1/64 s 的实际出步速度分别为 −1/64、+1/32 m/s，净线动量为零。还核对了有限端点、旋转连杆上的点速、固定基座反力、注册顺序和有向箱的材料见证。[解析与实际积分用例](../../tests/test_sphere_link_contact.py) · [六个独立控制](../../artifacts/p65-panda-structural-loads-20261105-01/13-core-final/final.json) · [四个球与箱的解析见证](../../artifacts/p65-panda-structural-loads-20261105-01/19-primitive-proof/final.json)

## 同一历史输入，首次碰触后的运动

原始输入 SHA-256 为 `d49f2f212dd38f309dc2bc8044149f5705242a2e035f9673b349727346d72e7a`。新清单唯一不同的字段是结构材料，指垫摩擦仍为 0。下表时间采用入步时刻。最后接触时间注明出步。前缀检查逐分量比较两次求解实际产生的 q、v、球位置、XYZW、线速度、角速度和时刻。[历史首次失效见证](../../artifacts/p65-panda-sphere-20261105-01/sphere50-validity.json) · [完整输入与差分入口](../../tests/reference_newton/robot/panda_structural_continuation.py)

| 量 | Phymeo HOST | Newton 1.7.0 Featherstone CPU |
| --- | ---: | ---: |
| 首碰前与各自旧轨迹完全相同的步数 | 1,339 | 1,340 |
| 首碰入步（s） | 0.836875 | 0.837500 |
| 胶囊轴部首间隙（mm） | −0.473238 | −0.211102 |
| 首步法向力（N） | 16.51878 | 14.94750 |
| 首步球力 XYZ（N） | (16.51852, 0.09398, 5.78154) | (14.94731, 0.07658, 5.23160) |
| 首步额外球冲量 XYZ（N·s） | (0.01032408, 0.00005874, 0.00361347) | (0.00934208, 0.00004787, 0.00326979) |
| 接触步数／最后接触出步（s） | 18／0.848125 | 19／0.849375 |
| 接触窗最深穿入（mm） | 1.96574 | 1.84139 |
| 球收到的累计结构冲量 XYZ（N·s） | (0.095931, 0.000571, 0.033576) | (0.094082, 0.000506, 0.032929) |
| 双方相对功率的离散积分（J） | −0.10005 | −0.09919 |

两个引擎的首个部件都是 `panda_link1_sc_0`。球首步相对于原无结构力出步多出的实际动量与 hF 一致。机器人九轴的实测速度增量与 hM⁻¹Jᵀ(−F) 的最大分量差分别为 5.28×10⁻¹⁷、1.38×10⁻⁷，Newton 的首轴实际响应约 2.15×10⁻⁶ rad/s。轨迹上的总相对功包含弹性储能变化，不能全数解释为摩擦发热。[Phymeo 逐步差分](../../artifacts/p65-panda-structural-loads-20261105-01/final-delta-phy.json) · [Newton 逐步差分](../../artifacts/p65-panda-structural-loads-20261105-01/final-delta-newton.json)

实际末端速度从只监测时的 (−0.8550, 0.0028, −15.7489) m/s 变为 (0.3441, 0.0099, −15.3292) m/s，Newton 对应 (0.3209, 0.0089, −15.3280) m/s。两边都没有搬运或放置成功。API 对未覆盖的几何和不唯一法向拒绝施力。箱形载荷保留其原结构监测范围。[Phymeo 真实态与载荷](../../artifacts/p65-panda-structural-loads-20261105-01/phy-results/test_original_50mm_ejection_no0/active-phy.json) · [Newton 真实态与载荷](../../artifacts/p65-panda-structural-loads-20261105-01/active-newton.json) · [显式历史模式回归](../../artifacts/p65-panda-structural-loads-20261105-01/09-legacy-boundary/final.json)

```text
python examples/panda_cube.py --sphere-radius .025 --mass .08 --closed-width .024 --grip-above-com .008 --friction 0 --trace active-phy.json --inputs-out active.inputs.json
python tests/reference_newton/robot/panda_sampled_loads.py --device cpu --urdf src/phymeo/builders/panda_assets/panda.urdf --inputs active.inputs.json --expect structural-contact --out active-newton.json
```


## 入步作用点与实际回放

对 50 mm 球的两条受力轨迹，从原输入和旧监测轨迹分别独立重建球的力矩、连杆质心力矩、沿物理祖先轴的广义载荷，以及球所受指垫、桌面和结构的合外力。首步之前，核对的球位姿、速度与九个关节状态在各自引擎内逐分量相同。Phymeo 首次受力的球冲量为 (0.0103241, 0.0000587, 0.0036135) N·s，同一时刻的新旧球动量差与这三个分量一致。该步第一轴的反作用角冲量约 6.78×10⁻⁷ N·m·s。[Phymeo 受力核查](../../artifacts/p65-fold-impact-and-batch-20261105-01/phy-pair.result.json) · [Newton 受力核查](../../artifacts/p65-fold-impact-and-batch-20261105-01/newton-pair.result.json)

离线回放可切换两个引擎，用“首次结构受力”或“峰值结构受力”跳转。箭头从该物理步真实的世界作用点出发，方向为球所受的合接触力。每个非零结构受力步均有对应的入步位姿，终态单独显示。浏览器核对了四个跳转的时间、姿态、箭头及力值。两端的峰值合接触力分别为 18.274 N、17.133 N。[双引擎交互回放](../../artifacts/p65-fold-impact-and-batch-20261105-01/panda-link-impact.html) · [浏览器实际读数](../../artifacts/p65-fold-impact-and-batch-20261105-01/impact-playback-qa.json)
