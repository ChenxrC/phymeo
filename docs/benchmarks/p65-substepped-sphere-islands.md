# 有界球岛子步与一次状态发布

普通 `TyxSphereIslandBatch.step` 在一步自由运动穿过另一球、末态分离时保留 `intermediate_crossing`。当用户希望在同一物理时长内求解相互作用时，可以指定子步数。

```python
import phymeo as ph
from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch

def ball(x, vx):
    return ph.FreeSphere(1, 0.25, (x, 0, 0), (vx, 0, 0), (0, 0, 0))

world = SphereIslandWorld(((1, ball(-1, 4)), (2, ball(1, -4))), restitution=1)
with TyxSphereIslandBatch([(7, world)], mode="cuda") as batch:
    row = batch.step_substepped(0.5, substeps=8)[7]
    assert row.accepted and row.state.tick == 1
    assert row.state.bodies[1].velocity == (-4, 0, 0)
    events = [(s.start_s, s.end_s, c.normal_impulse_n_s)
              for s in row.stages for c in s.report.contacts if c.normal_impulse_n_s > 0]
```

`substeps` 取 1、2、4、8、16、32 或 64。对可表示的 FP32 总时长，所有正片段严格覆盖原时间区间。每个片段由同一 TYX 自由积分与约束内核求解。外力用牛顿，按世界和本地球编号传入，在各正片段持续施加，重力来自各球登记值。自由位移遵循逐片段的半隐式 Euler。HOST 对每一片段复核全部球对、平面、材料点、局部能量与动量余量。

在已登记初态没有其他穿入、且实际闭合的表面存在闭向法向速度时，首先在 **零物理时长** 调用同一约束核。此记录的 `kind="initial_impulse"`。只有全部球的位置逐值保持才会采用它的速度变换。力包在这一刻的冲量为零。后续 `kind="advance"` 的几何见证、速度和冲量属于该片段通过校验的末态采样。区间内进入后，算法可能在片段末端才响应；需要更细时间分辨率时可提高预算。两端相交状态不足以确定完整运动路径时，原穿越检查继续拒绝。初态同心、不可表示的时间片、接触容量不足和能量校验失败分别保留原因。

同一外步中，选择的每个世界都从已发布状态进入私有工作数组。全部正片段通过才将该世界的 tick 加一。物理失败时 `state` 保持源对象、`stages` 为空，`failure` 给出失败片段和它的暂存诊断；`completed_advances` 是退回前实际完成的正长尝试数。可用 `keys=[失败世界]` 和新的预算/容量重试。未选世界仍占现有设备批的槽位，但其状态和上次成功观测均保持。驱动或读回异常会放弃整批的暂存值，后续从原检查点重建设备资源。原单步接口继续返回原 `SphereIslandResult`；成功调用新接口后，该世界的 `last_result` 保存含各时段的 `SphereIslandAdvance`。

## 2026-11-05 本机实测

固定旧源码 `4d254c1` 先在原恰触输入上返回 `intermediate_crossing`，随后因缺少新方法产生预期 RED。新增专项在真实 RTX 4060 Laptop GPU 与 TYX interpret 各通过 9 项。最终 CUDA 同次保留了两条原行为，之前的局部能量抵消与末态分离穿越仍拒绝，共 11 项。TYX 原生计数显示，一次初始瞬时冲量加 4 个正长片段共 15 次 CUDA 内核，15 次 H2D 共 380 字节，25 次 D2H 共 520 字节。这个计数从标准单步已创建的设备工作区开始读取。解释端 CUDA 计数为 0。

| 独立物理输入 | 真实受力与末态 |
| --- | --- |
| 初始恰触，第二球速度 (−1,2,0) m/s，两个质量各 1 kg，h=1/4 s | t=0 的法向由第一球指向 +X，双方同一点 (1/4,0,0) m，冲量 0.5 N·s，动能减少 0.25 J。末速度分别为 (−0.5,0,0)、(−0.5,2,0) m/s。 |
| 两球心由 ±1 m 以 4 m/s 相向，半径各 1/4 m，恢复系数 1，h=1/2 s | 1 片预算拒绝并保留初态；8 片时在 [1/8,3/16] s 的末端测得 8 N·s，双侧材料点都在世界原点。解析触时为 3/16 s，末球心坐标 ±1.5 m。 |
| 两球心由 ±7/8 m 以 3 m/s 相向，其余同上 | 解析触时 5/24 s 在第 4 片 [3/16,1/4] s 内，实测 6 N·s；末中心绝对值 1.000 m，连续解析为 1.125 m，相差 0.125 m。 |
| 两路径横向偏移 9/16 m、半径和 1/2 m | 完整相向穿过，保持原速度，全部 8 片均无冲量。 |
| 2 kg 球受重力 −8 m/s² 固定支撑 1/2 s | 8 片向上冲量合计 8 N·s，终点保持 z=1/4 m、速度为零。独立自由球同时受 +2 N 水平力，末速度 (0.5,0,−4) m/s。 |
| 闭合三球的恢复目标将产生 0.25 J 及两中心重合 | 各自保留 `contact_energy_gain`、`undefined_normal`。另有容量零的世界在第 3 正片段拒绝，原状态未动；只补跑该世界后，已接受邻居的状态和成功结果均维持原对象。 |

附加测试在原恰触工况同时施加 8 N 外力与 −4 m/s² 重力，核对瞬时阶段的竖直速度不变，两个 1 kg 球在 1/4 s 后各获得 1 N·s 竖直动量。另在第 3 次内部读回注入驱动故障，在故障发生的同一调用栈观察到两个世界都仍是上次成功的状态；紧接重试与独立新会话逐字段一致。

## Newton 1.7.0 原生对照

两种引擎采用相同球、输入、重力和 16×1/128 s 的读回刻度。Phy 使用一次 `step_substepped`，Newton 使用原生 `CollisionPipeline` 与 `SolverXPBD` 正步。Newton CPU 和真实 CUDA、Phy interpret 和 CUDA 都保留 17 个双体状态、原生力/冲量与完整配置。

- 恰触斜离时，四端的全部位置与速度读数逐值相同。第二球沿 +X 方向获得 0.5 N·s，双方总动能减少 0.25 J。Newton 的首个离散受力行乘 h 得 0.5 N·s；Phy 的瞬时冲量登记在 0 s。
- 第二组以 16 m/s 从 ±1 m 相向弹性正碰，四端均从实际动量变化得到第二球 +32 N·s，系统总动能与总动量变化均为零。Phy 在第 6 个片段末采样到 32 N·s。Newton 第 7 个片段的 `CONTACT_F × h` 给出 15.99999714 N·s。第 6 个保存时刻的速度相差 32 m/s，其他保存时刻速度一致。最终第二球中心分别为 Phy 1.500 m 和 Newton 1.375 m。Newton 原生离散力与恢复后的动量变化分别列出。

复现新增检验可运行以下命令。

```text
python -m pytest -q -s --require-tyx --tyx-mode cuda tests/scale/test_substepped_sphere_island.py
python scripts/observe_substepped_sphere_islands.py --engine phymeo --device cuda:0 --output phy-cuda.json --cache cache-phy
python scripts/observe_substepped_sphere_islands.py --engine newton --device cuda:0 --output newton-cuda.json --cache cache-newton
```

Newton 命令使用独立的 Newton 1.7.0 / Warp 1.17.0 环境。收据位于父集成工作区 `artifacts/p65-substepped-sphere-islands-20261105-01`。物理对照由 `f1ada3b` 脚本产生，此后仅做格式与导入排序。最终产品和原生计数专项对应源码 `3c9cf64`。`04-newton-cpu` 至 `07-phymeo-interpret` 保存四份 `observations.json`，`12-final-cuda` 与 `13-final-interpret` 保存最终专项。
