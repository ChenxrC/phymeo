# 偏心刚球绕质心旋转时的有限布面反力与连续检查

半径 1/8 m 的附着球在法向间隙 1/32 m 处向三个自由布节点施加 4 N。推进 `1/128 s` 后，刚体沿 +z 增加 0.03125 N·s 线动量，绕质心的 y 向角动量改变 −0.00390625 N·m·s。三个节点各自发生位移，终态布面倾斜，球与有限面的间隙为 0.029473865 m。HOST FP64 和 TYX CUDA FP32 均通过实际初末速度验算。[HOST 布面与刚体状态](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-host/test_deforming_cloth_and_offse0/eccentric-coupled-step.json) · [实际 CUDA](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-cuda/test_same_coupled_step_advance0/eccentric-coupled-step.json)

## 物理步及归属

以前可对偏心球采样瞬时力，但 `CollisionLoadSystem.step` 在真实刚体姿态改变后拒绝核查该步。现在自由刚体求解器发布世界系末角速度，接触系统核对实际末四元数与 `exp(h ω₁) q₀` 一致，再使用质心平移与该固定轴转角构造球心圆弧。圆弧与三节点仿射材料路径交给已有的有限凸体连续查询。此次产品改动只在物理绑定处。[原始拒绝](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/red/stdout.log) · [刚体到碰撞路径的绑定](../../src/phymeo/solvers/collision_loads.py) · [圆弧支持与首次重叠包络](../../src/phymeo/geometry/_convex_support_path.py)

受载球体质量 2 kg，体坐标球心 `(7/200,−24/200,0) m`，初始本体到世界四元数 `(0,0,3/5,4/5)`，初始世界系球心力臂 `(1/8,0,0) m`。三节点的质量依次为 1、1.5、2 kg，入步材料权重 0.2、0.3、0.5。无外力，重力为零，布三节点有不同法向初速。本步的三个 z 向接触冲量为 −0.00625、−0.009375、−0.015625 N·s。刚体入步角速度 2 rad/s，单位主惯量下的末角速度为 1.99609375 rad/s。实际 CUDA 共提交七个核，涉及碰撞宽相、材料法向、两个物理参与者。圆弧窄相和终态材料见证在 HOST，返回的见证为 `host-convex-support`。[独立力、材料权重和速度判据](../../tests/collision/test_eccentric_sphere_face.py) · [TYX 解释执行](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-interpret/test_same_coupled_step_advance0/eccentric-coupled-step.json)

## 单步中途穿面、近失与失败重放

另一组球半径 1/16 m、偏心距 1/2 m，绕世界 y 轴转半周。三角面和枢轴同作 `(1/8,−1/16,3/32) m` 三维平移。在该共动系中，球心为 `(sin θ,0,cos θ)/2`，布面高程 −1/4 m，有限面包含首触正投影。独立解析首碰时间是 `acos(−3/8)/π`。正序和换序返回同一包含解析值的时间区间，首触当时的材料位置与权重均重新核对过。初末球心的整条弦与这一有限面相隔至少 0.25 m。[正序原始见证](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-catalog/test_arc_on_translating_face_f0/arc-hit.json) · [反序](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-catalog/test_arc_on_translating_face_f1/arc-hit.json)

| 转角 | 起点 / 终点间隙 (m) | 解析首碰 | 查询的首碰区间 | 采样见证时刻 |
|---|---:|---:|---:|---:|
| π | 0.7710784 / 0.3789216 | 0.622357294 | [0.622356415, 0.622360229] | 0.622360229 |
| 2π | 0.7710784 / 0.7710784 | 0.311178647 | [0.311176300, 0.311180115] | 0.311180115 |

整周的初末球心在共动系重合，实际轨迹仍穿过有限面。完整物理步会拒绝中途触及硬核的候选，保留最近一次成功发布与调用方初态；从原检查点重试安全路径可得到相同终态。近失留出把三角面最靠近的有限边放在 `y=1/16+1/256 m`，在归一化时刻 2/3 取得精确最小间隙 **1/256 m**，宽相将其列为候选，连续检查证明整步分离。[整周几何](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/pytest-catalog/test_arc_on_translating_face_f2/arc-hit.json) · [硬核回滚、有限边近失及负控](../../tests/collision/test_eccentric_sphere_face.py)

这套入口对有偏心的附着球要求刚体求解器声明世界末角速度漂移合同，并检查返回的末姿态。未声明合同、伪造末姿态，以及耦合入口尚未支持的旋转胶囊均显式拒绝。连续几何支持本批覆盖的至多一周匀角速与各材料节点的仿射漂移；切触、超出求证预算等情况会保留未决，物理层拒绝发布该候选。[受影响的曲面相邻回归](../../artifacts/p65-offset-sphere-cloth-arc-20261105-01/curved-adjacent/stdout.log)
