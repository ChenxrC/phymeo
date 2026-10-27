# P5 真实求解参与者接入合同

入口是 `ArticulatedCouplingProgram(scene, participants, attachments, contacts)`，
再交给 `make_articulated_runtime(program, settings=...)`。返回既有 `SceneStepper`，
它统一掌握子步、迭代预算、候选验证和全部域的提交/回滚。
新增实现均为 NumPy FP64；原 tyx 路径保留，新算法设备移植留 P6。

## 工厂与绑定

`participants` 是域名到工厂的显式映射，必须与 `SceneModel.domains` 一一对应。
工厂声明 model、StateFamily、ProgramCapabilities、point_role、problem_signature，
并实现 create_workspace(domain, state, inputs, h, settings, context)。
程序检查模型/状态族、后端/精度、角色、附件与节点索引，运行器在前后检查物理签名。
工厂是受信任代码，必须将每个影响结果的配置写入签名；不能把物理记忆藏在 Python 对象中。

内置 `RobotParticipant(model)`、`ClothParticipant(model, constraints)`、
`TetrahedralParticipant(model)` 直接使用各自原生工作区，支持机器人 q/v、节点 position/velocity。
这些适配器不调用任何整帧求解器。状态族只注册在这个运行器中，默认 registry 不变。
当前组合要求一个 articulated 域和任意数量 node 域，数学、无隐藏历史；
新的 AI 或带历史材料不能仅改能力标签接入，需扩展显式状态与独立验证合同。

## 工作区和参数单位

| 操作 | 含义和规则 |
| --- | --- |
| predict | 每子步仅一次，真实外力/机器人偏置/材料内力预测 |
| position_sweep / position_metrics | 内部约束与残差；与接触共用外层预算，不重复积分 |
| point_port(indices, local_points) | 读取当前候选，返回有所有权的世界点端口；节点不允许偏移 |
| apply_position_load | loads 为 kg·m，改变配置，速度保持；机器人用当前 M/J，节点先合并重复索引 |
| configuration / velocity | 返回原生数值副本；耦合器无需假设配置字段叫 position |
| prepare_velocity | 一次性从预测速度加配置修正除 h 重建；固定节点为零 |
| apply_velocity_impulses | kg·m/s，只改变速度；使用原 PointPort 的完整交叉响应 |
| velocity_sweep / velocity_metrics | 内部速度条件和残差，不推进时间 |
| result / diagnostics / close | 返回候选及可复制的有限数据；资源退出成功后才可能提交 |

机器人另提供 world_to_local 和 generalized_load，以当前 FK/J 把世界作用点和载荷映射到关节系统。
局部附件单位为 m、节点/连杆索引为整数。世界 SI、Y-up、FP64，与 P3/P4 保持一致。
部分工作区构造失败时由工厂清理自身未完成资源；组合器负责关闭此前已成功构造的工作区。

## 接触与独立审计

RobotSphereAttachments 保存连杆、本地球心、半径；NodeRobotContact 明确每个节点/球配对，
允许重复索引、同体多球和多域共享一个机器人。每对读取当前候选几何，以节点世界位置作为共同作用点。
位置和速度分别累计非负法向乘子；位置等效冲量只记账，绝不再次施加。
同心、不可响应的穿透、材料越域、未收敛或非有限结果拒绝整个外层 step。

contact_events 逐事件保存阶段、域/节点/球/连杆、作用点、本地点、法向、相反载荷、广义载荷、
施加前的原生配置/速度及真实增量。位置事件的载荷是 kg·m，审计冲量时除对应 h；
速度事件已是 kg·m/s。机器人诊断含 predicted_q/v。诊断数组在返回时独立复制且只读。
对非线性关节，必须用每次事件施加前的 q 验证 MΔq 或 MΔv=Jᵀload，
不能把所有事件乘最后一个 M 后宣称严格守恒。

该参考接口在每次查询建立拥有数据的端口，并重复验证几何/质量/材料域。
此轮先留性能记录，后续 profiling 再讨论候选版本缓存；不能删掉物理门禁换速度。
范围仍为离散节点/球、无摩擦、零恢复系数；见[几何边界](legacy-issues/GEOM-001-discrete-vertex-contact.md)。

接手从 [participant.py](../src/phymeo/coupling/participant.py)、
[组合程序](../src/phymeo/solvers/articulated_coupling.py)、
[接触算子](../src/phymeo/constraints/node_robot.py)和
[解析/事务测试](../tests/test_articulated_coupling.py)开始。
完整场景/反事实/回放按 [P5 规格](superpowers/specs/2026-10-27-p5-cpu-integration-design.md)另行验收，
仅通过这些接口测试不代表 P5 已完成。

## 外部驱动反事实实验

[prescribed_articulated.py](../src/phymeo/experiments/prescribed_articulated.py)提供独立实验程序：
先用 RobotProgram 计算自由 q/v，再复制为不可变的边界轨迹。
其 SceneModel 只有布料/软体动态域，`history.driver_index` 明确保存已消费的子步数；
统一运行器失败时恢复所有域和索引。边界从下一个样本读取运动，不再积分或响应接触。
每个接触仍记录等大反向载荷，位置载荷除 h 后与速度冲量一起进入 driver_contact_impulse，
driver_supplied_impulse 是外部驱动为维持轨迹所提供的相反冲量。

轨迹使用 driver_q/v 字段，审计使用 driver_trajectory_error，
不会报告虚假的动态机器人动量守恒。源轨迹步长/长度/自由方程、耗尽和运行中变更均显式检查。
普通 ArticulatedCouplingProgram 没有忽略反作用的配置。

```powershell
python examples/articulated_scene.py --scene S3 --experiment prescribed --frames 200 --output artifacts/prescribed-s3-new
```

这个命令是明确的外部驱动实验；双向路径使用默认的 `--experiment two-way`。
完整反馈差异仍需冻结的 200 帧成对运行验证，入口和 100 帧测试不替代这一门槛。
