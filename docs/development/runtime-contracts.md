# P1/P2 运行器与扩展合同

本接口已实现于 `codex/p1-p2-runtime`；完整交付状态见 [进度](../progress.md)。它覆盖统一数据、阶段调度和现有刚柔算法，机器人关节、体积材料及通用 AI 模型属于后续阶段。

## 责任划分

| 对象 | 责任与数据 |
| --- | --- |
| `SceneModel` / `DomainBinding` | 域名、状态族、物理模型；绑定后物理指纹固定 |
| `SceneState` / `SceneInputs` | 保留各域原生状态和输入；容器借用值，运行器创建完整副本 |
| `StateFamily` / `StateRegistry` | 局部注册类型、全部数值字段、入口验证和候选不变量；无全局插件表 |
| `StepSettings` | 子步、位置/速度预算、归一化容差、参考尺度、子步快照选项 |
| `BackendContext` | 后端身份与单一 FP32/FP64；不得隐式回退或混合精度 |
| `SceneProgram` / `SubstepWorkspace` | 算法、阶段计算和临时资源；历史只能进入显式状态 |
| `SceneStepper` | 子步循环、阶段预算、候选验证、独立验证、资源退出后的全局提交 |
| `SceneStepResult` | 拥有所有权的最终状态、报告；成功时可选独立子步快照 |

参数分成物理模型/程序签名、每帧输入、执行设置和后端上下文。材料、几何、约束规则放入物理配置；外力/力矩放入输入；步长单独传给 step，按状态精度量化。各域数值字段同精度，使用 SI、Y-up 和 xyzw 四元数。

## 接入一个新求解程序

1. 用 `StateFamily` 注册完整模型/状态/输入类型和全部数组字段。状态不要求名为 position，可用 q/v；额外内部字段不能遗漏。入口和候选拒绝未注册的 dataclass、实例属性或 slots 字段；只重写方法而没有增加字段的子类仍可规范化。扩展字段需注册完整的新状态族。`validator` 检查入口数据，`invariant` 检查候选的物理约束。固定体、单位姿态等内置规则不应通过新模式偷偷放宽。
2. 声明 `ProgramCapabilities` 的后端、精度、状态族、功能及 provider 类型；提供可计算稳定指纹的 `problem_signature()`。执行预算、后端和着色次序不改变物理签名；材料/接触规则改变需要重新绑定。内置联合程序另行绑定保留顺序和 dtype 的配置指纹，tyx 还绑定派生着色表；即使物理约束集合相同，修改行次序或缓存也必须构造新程序，禁止带旧缓存进入工作区。
3. 用 `HistoryField` 声明所有跨 step 数值历史的名字、shape、dtype，并放入 `SceneState.history`。工作区临时乘子可在本子步重建；若以后 warm-start 跨帧，就必须变成显式历史。程序成员只能缓存可重建数据，不能藏物理历史。
4. `create_workspace(scene, state, inputs, h, settings, context)` 获得本次拥有所有权的数据。创建失败前已申请的资源由创建函数释放；工作区返回后由运行器的 finally 调用 close。
5. 实现下表阶段。注册数据族本身不会自动决定求解或耦合规律；程序必须明确组合所需算法和作用规则，但不需要修改通用运行器。
6. `result()` 返回所有域和声明的历史。运行器验证类型、每个字段的形状/精度/有限性与不变量。物理残差不是由这些结构检查自动证明；学习 provider 必须配置同物理签名的独立 `SceneVerifier`。

| 阶段 | 允许的行为 |
| --- | --- |
| `predict()` | 每子步一次自由预测，使用 h 积分 |
| `position_sweep()` / `position_metrics()` | 一轮位置约束更新与归一化残差；不得再次推进时间 |
| `prepare_velocity()` | 从候选几何重建速度/准备冻结接触几何；位置等效冲量不能重复施加 |
| `velocity_sweep()` / `velocity_metrics()` | 一轮速度冲量更新及残差 |
| `result()` / `diagnostics()` | 候选与可复制的有限数值证据；可以保留类型化冲量账本 |
| `close()` | 释放本工作区资源；不能提交外部物理状态 |

运行器每阶段至少执行一轮 sweep，再判断收敛；这保留了原 XPBD 次序。它生成接受标志与迭代数，程序不能仅自报 accepted 绕过门禁。独立 verifier 收到额外副本，其验收残差保留在报告里。

## 事务与异常

只有所有子步及工作区 close 通过，外层结果才 accepted。末子步失败也返回外层初态的独立副本，包含全部域和历史，不返回部分成功的子步快照。报告中的 completed_substeps 只是已成功计算数，失败帧没有部分提交。

预期数值/执行失败用 `SolverFailure`，浮点溢出等也会拒绝全帧。配置或程序错误保留异常，finally 仍释放资源；调用者输入保持不变。运行器重入、关闭后调用、绑定参数/能力/schema 变化均拒绝。程序是受信任扩展，不提供恶意代码的进程隔离。

`CoupledRuntimeAdapter.last_scene_result` 保存最近返回的场景结果与计时；新调用开始即清空，抛异常时不会留下误导性的上次成功记录。适配器只转换数据，所有时间推进仍由 SceneStepper 负责。

## 点端口与未来机器人/软体

`ParticlePointPort` 和 `RigidPointPort` 提供世界位置/速度、完整端口批次的冲量响应，以及拥有所有权的候选状态。刚体固定作用点包含旋转和偏心力矩；共享同一刚体或重复索引必须一起响应，不能当作独立逆质量块。

`exchange_impulses(left, right, impulses)` 在匹配世界作用点施加相反载荷，验证双方后一起返回；瞬时冲量保留位置/姿态。当前要求两个不同状态域；同域内部接触应通过单个联合响应计算。端口不负责检测碰撞或求解约束冲量。

未来机器人端口可内部实现雅可比和质量逆响应，软体端口可内部实现插值；外部仍使用相同世界点语义。本轮实际粒子/球联合程序保留已验证的标量投影内核，没有声称机器人雅可比或四面体插值已实现。

## 可运行入口

```python
import numpy as np
from phymeo import (ParticleModel, ParticleState, StepInputs, DomainBinding,
                    SceneModel, SceneState, SceneInputs, SceneStepper,
                    FreeSceneProgram, StepSettings)

model = ParticleModel(np.ones(1), np.array([0., -9.81, 0.]))
scene = SceneModel((DomainBinding("dust", "particles", model),))
state = SceneState({"dust": ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
inputs = SceneInputs({"dust": StepInputs.zeros(1)})
with SceneStepper(scene, FreeSceneProgram(), StepSettings(substeps=4)) as stepper:
    result = stepper.step(state, inputs, 1 / 120)
    if result.report.accepted:
        state = result.state
```

刚柔场景：`python examples/coupled_runtime.py --mode numpy --frames 240 --output artifacts/new-cpu`。可选 tyx 安装后，使用 `--mode cuda`；该示例默认 NumPy FP64、设备 FP32，显式 `--dtype` 不会被覆盖。旧 `examples/rigid_cloth.py` 保留用于迁移对照和单向实验。两者使用同一个独立子步审计器。

性能限制与候选优化在 [性能台账](../performance-bottlenecks/README.md)，依赖侧问题在 [tyx_flaw](../../tyx_flaw/README.md)。当前仍保留主机残差检查和每子步资源分配；阶段墙钟时间不等于纯设备 kernel 时间。
