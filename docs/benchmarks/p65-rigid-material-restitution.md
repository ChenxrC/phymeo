# 刚体碰撞中的材料恢复系数

对已经定位的刚体碰撞，三个公有入口 `solve_contact_impulses`、`solve_witness_impulses` 和 `step_floating_contacts` 现在均接受 `restitution` 和 `bounce_threshold_mps`。系数可为 [0, 1] 中一个数，或与所有接触行对齐的一组数。默认系数 0 延续原塑性冲量。碰撞前的闭合速率超过阈值时，目标分离速度为闭合速率乘恢复系数。阈值相等或低于阈值时用零目标；张开的行只有在整个接触网络需要时才承受压冲量。

该规则只改法向互补目标。切向摩擦仍基于真实相对滑移；报告的冲量、出步相对速度和动能变化仍从两侧实际状态检验。浮动系统先承受外力，再在原位实施冲量，使用施力后的自由速度，随后以真实出射速度推进位置。持续承重可显式用零恢复或声明抑制低速跳动的阈值。调用者仍须提交已定位的材料行，步内新生事件的定位交给几何管线。[求解器](../../src/phymeo/constraints/contact_impulse.py) · [几何见证与浮动树](../../src/phymeo/solvers/floating_contact.py)

## 两个自由球的原生参照

两个独立球半径均为 0.125 m，质量为 1 kg 和 2 kg；两球以 3 m/s 闭合并共享一非零世界漂移速度。Phymeo 使用 `detect → step_floating_contacts`，Newton 1.7.0 原生检测生成形状对并由开启恢复的 XPBD 在实际 CUDA 推进。重力为零。分别沿 +X 和 −Z 撞击，各试三种材料。对两物体各自的速度、总动量和能量都用同一解析结果校验。

| 恢复系数 | 两球各自冲量大小 (N·s) | 出射相对速度 (m/s) | 两球动能变化 (J) |
|---:|---:|---:|---:|
| 0 | 2 | 0 | −3 |
| 0.5 | 3 | 1.5 | −2.25 |
| 1 | 4 | 3 | 0 |

以上数值在两条坐标轴和两套实现中一致。验收比较速度/冲量，未要求不同离散积分的步内位置一致。另验滑动、多个不同材料行、低速抑制、张开无载、无效数据回滚和浮动关节树的真实扫掠碰撞。原入口缺参数的首项失败已保存，首个 Newton 驱动因标量零重力与 1.7 的三分量 API 不符在建模前失败，纠正输入表示后同一组物理输入通过。HOST 合同及相邻回归 67 项通过。Newton 1.7 的不同后端对恢复、滚动和扭转摩擦有各自的支持范围。[实际两体原始数据](../../artifacts/p65-newton-night-20261105-01/rebound-two-body.cuda.json) · [六组直接对照驱动](../../tests/reference_newton/robot/two_body_restitution.py) · [材料回归](../../tests/test_material_restitution.py) · [Newton 求解器材料矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html#contact-material-support)
