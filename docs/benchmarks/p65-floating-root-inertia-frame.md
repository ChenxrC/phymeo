# F3 浮根惯量 frame：分阶段验证

2026-11-07。**T1 核心已验证；T2 loader / 固定工具与 T3 场景、Newton 对照、wheel 仍待完成。**

模型新增可选 `root_inertia_frame_xyzw`，表示惯量主轴到 authored root-body 的旋转。根质量及主惯量仍使用原 RigidBodyModel；q、COM、子关节与碰撞材料坐标不变。完整树 M/b 只复合根的惯量朝向；root-only 委托原刚体积分后将姿态转回原 body frame。默认 identity 保持原路径。

## T1 实际验证

父唯一串行槽，原环境与 entrypoint guard，证据根 `artifacts/p65-generic-collision-20261107-01`：

| 命令 | 实际结果 |
|---|---|
| 137-f3-first-red | RAW1；正常输入因缺少 root_inertia_frame_xyzw 构造参数失败 |
| 138-f3-first-ruff | RAW0 |
| 139-f3-first-format / 140-f3-format | RAW1，仅新测试排版；随后 RAW0 修复 |
| 141-f3-first-host | 11 passed / 0.67 s / RAW0 |
| 142-f3-affected-host | 原 free-root 与 floating-tree 两完整文件 17 passed / 0.61 s / RAW0 |
| 143-f3-final-format / 144-f3-final-ruff | 均 RAW0 |

测试先行提交8a355b1；78c4eee 在首次执行前修正测试使用的既有 mass_matrix 字段和共同重力，没有数值调参或失败后改 oracle。最终三个源文件与新测试在141–144执行前后均未变。

独立完整 tensor 检查静止扭矩响应；两条40步真实轨迹覆盖偏心COM、非零转动、陀螺项、共同旋转和平移，每步核世界惯量Euler/Rodrigues、COM、q/v和动能。真实R子关节案例独立构造7×7 M、偏置、COM外载及加速度，包含跨根/子体耦合。另核无效frame、owned不可变输入、四元数符号、默认identity和拒步保原态。

完整轨迹各40帧，分别保存在141的两个 `test_root_frame_actual_forty_s*` 目录。没有把两条短窗称一般机器人或全场景通过。receipt `f3-core-validation-receipt.json` SHA256：
`049c433fa7f50b291c814cd2d1b5e91e2d24c3ae36c2a216bd282d0fdd7571b0`。

后续按[原计划](../superpowers/plans/2026-11-07-floating-root-inertia-frame.md)先扩合法非对角URDF root / folded fixed-tool装配，再验真实场景。原100g持稳与软布失败继续保留，未因本核心通过而改写。

## T2 实际 URDF / fixed-tool / mimic 装配

产品沿原 _mass_component 严格解析完整 COM tensor，复用 _principal 的静态合成/主轴分解；root COM直接在 authored frame求和，独立惯量frame传入T1核心。普通root-only、带子树和折叠工具共用同一root组装配，mimic直接继承，不复制parser或改关节几何。

测试先行210eb20，145真实RED为合法非对角根被旧principal_inertia拒绝。146首Ruff通过，147格式检查仅2文件需排版，148修复。149首完整新HOST **18 passed / 0.58s / RAW0**；150受影响完整 fixed-tools / floating-scene / floating-mimic **93 passed / 12.88s / RAW0**；152/153最终Ruff/format均RAW0。无数值修复、预算调整或输入调参。

新正控覆盖完整矩阵与旋转inertial-frame两种等价URDF、偏置fixed tool的parallel-axis非对角项、共同旋转/平移、原collision/sites保持，以及ordinary/folded affine mimic完整root惯量继承。原三个“合法但未支持”拒绝项退役，由上述正控替代；非正/非法triangle/NaN惯量、零/缺质量和其他未实现模式仍显式拒绝。旧错误文案的测试允许mass/inertial同义，不改变拒绝结论。

receipt `f3-urdf-validation-receipt.json` SHA256：
`17e7d31e24cef54b20e1e2bf69193516e00132bb4077e1761a7f236ff346221c`。
**T1+T2已完成；T3新的非对角根受载场景、Newton CPU/CUDA参考与源码外wheel尚待。** 旧93回归没有冒充这些新输入的交付证据。
