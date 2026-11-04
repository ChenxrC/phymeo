# 真实关节端凸工具与可动布同步步进

同一个有限凸工具登记到 2 kg 滑轨和偏心转腕后，入步均从材料覆盖矩形上测到 **4 N**。滑轨的广义力为 **+4 N**。转腕工具关于连杆质心的力矩为 **−1.5 N·m**，力作用于偏心质心的另一个 **−0.5 N·m** 使转动关节实际承受 **−2 N·m**。每组另有一件登记在同一连杆上、与布分离的工具。两例各推进八个 `1/128 s` 物理步，实际关节坐标和布节点同时变化。[滑轨实测](../../artifacts/p65-articulated-convex-tool-cloth-20261105-01/pytest-host/test_actual_articulated_tool_u0/articulated-tool.json) · [偏心转腕实测](../../artifacts/p65-articulated-convex-tool-cloth-20261105-01/pytest-host/test_actual_articulated_tool_u1/articulated-tool.json)

## 公开入口与材料测量

`LinkHullTool(key, robot_domain, link_index, local_vertices, layer, *, vertex_ids=None)` 登记固体在连杆局部坐标中的顶点和材料标识。`CompliantClothContactProgram(..., link_hull_tools=(...), link_hull_operator=...)` 将工具加载到原机器人的关节质量矩阵、偏置力和布弹性求解器。`observe_hull_tools(state)` 读取端点状态，`result.report.substeps[i].diagnostics["hull_tools"]` 给出本子步骤实际使用的中点载荷。其中包含各材料积分点、双方施力位置、关于连杆质心的扳手、广义力和相对功率。[工具登记及物理映射](../../src/phymeo/solvers/cloth_tool_contact.py) · [布与机器人共同的中点推进](../../src/phymeo/solvers/compliant_cloth_contact.py)

每次从真实正运动学和连杆速度得到凸体世界位姿及质心运动，按实际有限补丁分配布节点载荷。关节反力按 `JᵥᵀF + JωᵀM_com` 映射。布半厚度 15.625 mm 以中面周围的均匀圆整半径建模。例中凸体到真实有限中面距离为 46.875 mm，对材料外层的物理间隙为 31.25 mm。每个站点记录中面点、外层最近点、凸体点和各自材料权重。接触力采用每个工具–布面登记对的有效刚度。[力、虚功与厚度留出](../../tests/collision/test_articulated_hull_cloth.py)

## 共同运动

两套资产初始关节速度均为 +0.125，滑轨用 m/s，转腕用 rad/s。滑轨第一步中点载荷 3.9375 N，四点承担反力；第八步中点为 2.487624 N，有限面变为单点支撑。末位置 0.0112804 m、末速度 0.227453 m/s。转腕第一步中点下实际有限支撑为两个点，力 4.039061 N，广义力矩 −2.524166 N·m；第八步中点为一个点，末角速度 **−0.154702 rad/s**。按转腕相对于固定轴的惯量 17/32 kg·m²，对各步观测的关节动量变化逐一核对。布节点的动量同时与中点材料内力和外层接触载荷核对。[两份八步轨迹](../../artifacts/p65-articulated-convex-tool-cloth-20261105-01/pytest-host)

额外的两片共享边、五条弹性距离约束布在六个真实步中持续向关节传力。有限边外侧的留出由真实三角边及已知法向、高度偏差独立重建间隙。实际 CUDA 资格在滑轨的一次物理步中用两个材料核分别计算入步和中点的四个积分站，法向模块共上传 13,312 字节、下载 5,120 字节。该组的机器人动力学、FK、布弹性和扫掠核查在 HOST，CUDA 执行材料层；末关节速度为 0.015625 m/s。[CUDA 收据](../../artifacts/p65-articulated-convex-tool-cloth-20261105-01/pytest-cuda/test_midpoint_uses_one_native_0/tool-material-backend.json) · [TYX 解释执行](../../artifacts/p65-articulated-convex-tool-cloth-20261105-01/pytest-interpret/test_midpoint_uses_one_native_0/tool-material-backend.json)

## 带厚度的连续间隙

每段用实际关节增量、活动关节轴和枢轴构造凸工具路径，布的三个节点分别沿各自真实初末位移漂移。按子步入口到中点、中点到末态两段检查。实际角位置 0、π、2π 的三个采样都分离的留出，在 π/2 时工具穿过有限材料三角形，程序拒绝整步并保留原输入状态。移开相同有限布面则接受整周物理步。布中面仍有正几何间隙、但正厚度已被工具穿入的静态状态也会拒绝。[逐段厚度支撑包络](../../src/phymeo/geometry/articulated_hull_clearance.py) · [圆弧、真实面与失败重试](../../tests/collision/test_articulated_hull_cloth.py)

支持的凸工具链可有多节关节。活动的祖先可全部平移，或仅一个旋转、其前方允许平移。多活动转轴、旋转下游的活动伸缩及单段超过一周的转角会显式拒绝。达到包络细分预算亦拒绝发布。与已有布结构胶囊层或背板专用解算器的混合配置尚待独立闭环验证。
