# 有限凸工具对可变形布面的反力

箱底只有一半覆盖三角布面时，4 N 法向合力实际通过 **0.125 m²** 覆盖矩形的中心，刚体绕质心产生 **−0.5 N·m** 的 y 向力矩。按同一几何和材料输入推进一个 `1/128 s` 物理步，三个自由布节点沿 z 受到 −1.75、−0.5、−1.75 N，刚体实际增加 +0.03125 N·s 线动量并获得 −0.00390625 rad/s 角速度。HOST FP64 与实际 TYX CUDA FP32 得到的末间隙均为 **31.346636 mm**。[HOST 原始物理步](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_box_face_reacts_at_the_fi0/convex-cloth-step.json) · [实际 CUDA](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-cuda/test_same_cloth_and_rigid_step0/convex-cloth-step.json)

## 有限材料面、边和角

在 `CollisionLoadSystem` 中登记 `hull` 和 `triangle` 各自的刚体局部顶点、自由粒子行与 `NormalLayer`，通过原 `sample` 和 `step` 读取材料载荷及实际状态。普通检测已有凸体与有限三角形的截面和扫掠。本次增加其显式邻近补丁，通过实际间隙方向投影两侧暴露特征，再用有理数剪出完整的最近点、线段或多边形。每个补丁点保存两侧原始材料编号和权重，换序用例重新按两侧实际节点重建该点。力和力矩复用原有补丁积分和材料反力映射。[测距及材料补丁](../../src/phymeo/geometry/_hull_surface.py) · [公开步进](../../src/phymeo/solvers/collision_loads.py) · [独立几何与力学断言](../../tests/collision/test_hull_cloth_loads.py)

以下四个留出使用同一刚度 128 N/m、层宽 62.5 mm。箱底在水平布面上方 31.25 mm，几何由布面的实际边界决定。

| 最近特征 | 间隙 (mm) | 积分点 | 刚体力 (N) | 质心力矩 (N·m) |
|---|---:|---:|---:|---:|
| [部分覆盖的矩形](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_finite_patch_separates_ar0/closest-patch.json) | 31.25 | 4 | (0, 0, 4) | (0, −0.5, 0) |
| [投影仅共边](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_finite_patch_separates_ar1/closest-patch.json) | 31.25 | 2 | (0, 0, 4) | (0, −1, 0) |
| [投影分离的平行边](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_finite_patch_separates_ar2/closest-patch.json) | 39.0625 | 2 | (−1.8, 0, 2.4) | (0, −0.375, 0) |
| [只剩一对最近角点](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_finite_patch_separates_ar3/closest-patch.json) | 39.0625 | 1 | (−1.8, 0, 2.4) | (0.6, −0.375, 0.45) |

后两行的水平偏移为 23.4375 mm，独立直角三角形给出 39.0625 mm 欧氏间隙。再将布面移至横向距离超过层宽时，所有节点载荷归零。原层律以每个登记形状对的有效刚度给出合力，补丁权重总和为一。因此图中等间隙的面与线承受相同法向合力，改变力臂后力矩不同。用户对网格重新剖分时应按材料模型分配每对刚度。[有限边界正负控](../../tests/collision/test_hull_cloth_loads.py)

将偏心四面体工具与布面一起作空间轴循环后，最近补丁的面积为 0.0625 m²，三个实际支撑节点为 65、9、41，第四个顶端节点的权重为零。刚体世界系力为 (4, 0, 0) N，力矩为 (0, −1/3, −5/6) N·m。另一项有切向速度的矩形覆盖试验测得 −0.3 W 摩擦功率，切向力在同一布材料点向两侧施加。[四面体材料和扳手](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_offcenter_tetrahedral_too0/tetrahedral-tool.json) · [摩擦与力臂核验](../../tests/collision/test_hull_cloth_loads.py)

## 可动布面与连续路径

HOST 使用两个共享材料边的三角面、四个独立质量和五条距离约束连续推进 12 步，共 93.75 ms 物理时间。两登记对的刚度分别为 32、96 N/m。第一步在两张面的四个积分点上分别施加 1、3 N。第二步材料面发生倾斜，两侧的有限最近补丁各变为一个点，载荷为 1.011609、3.025360 N。第 12 步分别为 0.785449、2.200619 N，四节点末 z 坐标为 −6.937、−6.188、−7.181、+2.068 mm。对每一步分别核对了双方总线动量、刚体线动量和关于质心的角冲量。[12 步节点、工具姿态与逐面载荷](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-host/test_two_connected_cloth_faces0/two-cloth-faces.json)

物理步先在实际入步姿态计算力，再以刚体参与者的真实末世界角速度与质心平移检查整段工具圆弧，并以布的三个实际端点位移构造各自的仿射轨迹。对有角漂移的凸工具，连续查询沿真实弧构造包络；转角和求证预算沿用已有凸体求证器。材料三角形在步内退化、硬核穿越或求证未决会拒绝发布整步，旧检查点仍可读取。在持续布约束场景中，连续几何描述每个已接受子步初末之间的运动，布面形状在这一段可改变。[真实角漂移](../../src/phymeo/geometry/_convex_support_path.py) · [硬核及失败发布检查](../../tests/collision/test_hull_cloth_loads.py)

实际 CUDA 资格使用单张三自由节点布的同一物理步，总计七次设备核提交，其中一次计算四行法向材料。法向工作区上传 6,656 字节、下载 2,560 字节。几何剪裁、材料分配和圆弧窄相在 HOST，真正的粒子和自由刚体状态更新在 TYX。两张相连布面的 12 步记录在 HOST 完成。[解释执行](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-interpret/test_same_cloth_and_rigid_step0/convex-cloth-step.json) · [原生 CUDA](../../artifacts/p65-convex-tool-cloth-20261105-01/pytest-cuda/test_same_cloth_and_rigid_step0/convex-cloth-step.json)
