# 世界与接触对端的力矩阵

`ph.measure_contact_matrix(source, sensing, counterparts)` 在一次已发布的物理载荷帧或显式采样帧上读取多个物体/形状。行列沿用调用者指定的顺序，编号绑定世界。总力包含该行的全部接触；矩阵只含选定对端。切力按每个站点的真实法向分解后再求和。位置矩阵按单个力的模长加权两侧材料表面点的中点，分母在 `weight_n` 中提供。零力、无接触和跨世界格的读数均为零。体和其所属形状若在同一轴上被重叠选择，入口明确拒绝。

~~~python
reading = ph.measure_contact_matrix(load_system, [body_a, body_b], [floor, gripper])
force_on_a_from_gripper_n = reading.force_n[0, 1]
center_m = reading.position_m[0, 1]
~~~

公开类型 `CollisionLoadSystem` 绑定最近提交的入步帧；显式 `SurfaceLoads` 只能标为采样时刻。世界系下提供总力、总切力、每对端的力/切力和力权重位置，单位 N、m。转成传感器坐标或测量相对于某点的力矩，可继续使用原扳手入口。本版矩阵在 HOST 聚合。[实现](../../src/phymeo/sensors/contact_matrix.py) · [使用多个物理世界的测试](../../tests/test_contact_matrix.py)

在 Newton 1.7 的真实 XPBD 梁–地面静置/滑动窗口中，CPU 和 CUDA 各产生 32 个独立物理步。将每步实际原生求解器六维观测中的线力和两侧世界材料点交给 Phymeo，在相同接触流上逐元素对照 Newton `SensorContact` 的五张总量/成对力与位置表，绝对误差门为 2×10⁻⁵，另含 10⁻⁶ 相对容差。各自的两个远处空槽和双方符号同时核对。此项验证传感观测的数学定义；未测两套传感器的速度。[Newton 传感器定义](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html) · [CPU 实际记录](../../artifacts/p65-newton-night-20261105-01/matrix-cpu.json) · [CUDA 实际记录](../../artifacts/p65-newton-night-20261105-01/matrix-cuda.json)

首轮试图通过现有通用载荷系统推进偏心球时，连续转动附件路径在发布前拒绝。该输入保留在回归中验证拒绝后没有可读载荷；正常发布、外层事务和后续分离由已有独立双世界物理用例检查。12 项 HOST 专项及受影响的旧扳手回归通过，原始失败收据仍保存在同一目录。
