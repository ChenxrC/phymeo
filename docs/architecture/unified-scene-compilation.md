# 统一场景编译与可选 MuJoCo 路径

本层将资产编译、场景装配和求解后端分开。MuJoCo 编译是可选的 CPU 前端；已有 URDF/MJCF 加载器和 TYX 主线继续保留。对应[实施计划](../superpowers/plans/2026-11-10-unified-scene-compilation.md)。

```mermaid
flowchart TD
  A[URDF / MJCF / 程序构造资产] --> B[Phymeo 原加载器与公共资产]
  C[原 MJCF / URDF + includes/assets] --> D[CPU MuJoCo MjSpec.compile]
  D --> E[编译记录：MJB 快照 / 身份与坐标映射]
  E --> F[显式能力检查与资产转换]
  F --> B
  B --> G[SceneDeclaration：资产、初态、控制、有限环境、材料、world IDs]
  G --> H[scene_capabilities / compile_scene]
  H --> I[Phymeo HOST scene]
  H --> J[TYX interpret / CUDA resident runtime]
  E --> K[独立 MjModel / MjData → CPU mj_step]
  E --> L[显式 Warp：put_model / put_data → CUDA]
  E --> T[默认 CUDA：MuJoCo-TYX T0/T1/T2]
  I --> M[共同运行协议：snapshot / controls / step / reset / close]
  J --> M
  K --> M
  L --> M
  T --> M
  M --> N[ResidentVectorEnv → PolicyRunner / 训练适配]
```

## 构建与运行的边界

| 层 | 公共入口 | 负责的内容 |
|---|---|---|
| 原生编译前端 | `phymeo.builders.mujoco_compiler.compile_mujoco` / `compile_mujoco_xml` | CPU解析、defaults/include展开、推导惯量；保留完整编译物与来源；显式转换到已有资产 |
| 场景声明 | `phymeo.builders.scene_compiler.SceneDeclaration` | 拥有资产、初态、控制参数、静态几何和材料的冻结快照；多个world可有不同资产 |
| Phymeo编译 | `scene_capabilities` / `compile_scene` | 在支持域内选择HOST或TYX；保留原物理程序、碰撞身份和控制语义 |
| MuJoCo运行 | `phymeo.backends.mujoco_runtime.MuJoCoRuntime` | 选择原生CPU或独立CUDA路径；同一编译模型的多个独立world |
| 环境与策略 | 已有 `ResidentVectorEnv` / `PolicyRunner` | 消费运行协议，维护episode、动作保持、时钟和失败后的reset要求 |

`interpret` 是 TYX 的 CPU 解释路径，不表示已有通用原生 CPU 编译后端。TYX 接触路径仍可能包含 HOST 几何与力律；选择 CUDA 不代表整个管线都驻留 GPU。

当前 MuJoCo 分支消费原编译记录；本批没有实现任意 `SceneDeclaration` 反向生成 `MjSpec`。这样可以保留原生模型中的 tendon 等功能，同时防止把原生 MuJoCo 能力错误归给 Phymeo/TYX。

## 可选依赖

核心安装仍只依赖 NumPy，导入新公共模块不加载 MuJoCo、Warp 或 TYX。

```text
pip install "phymeo[mujoco]"
pip install "phymeo[mujoco-warp]"
```

本批原生CPU使用 MuJoCo 3.3.7；独立CUDA适配限定已验证的 MuJoCo 3.3.7 / MuJoCo-Warp 0.0.2 / Warp 1.17.0 组合。TYX仍按项目原独立安装流程使用，未修改其pin。

## 两条使用方式

以下场景由一个有标量关节的编译资产建立。资产的q/v已经转换到Phymeo公共坐标。

```python
import numpy as np
from phymeo.builders.mujoco_compiler import compile_mujoco
from phymeo.builders.scene_compiler import ArticulationWorld, SceneDeclaration, compile_scene
from phymeo.core.robot import RobotState
from phymeo.runtime.robot_scene import RobotCommand

compiled = compile_mujoco("slider.xml")
count = len(compiled.asset.joint_names)
world = ArticulationWorld(
    4, compiled.asset, RobotState(compiled.initial_q, compiled.initial_v),
    kp=np.zeros(count), kd=np.zeros(count), effort_limit=np.full(count, 5.0),
)
declaration = SceneDeclaration((world,))
with compile_scene(declaration, backend="cuda") as runtime:
    runtime.set_controls({4: RobotCommand(compiled.initial_q, feedforward=np.ones(count))})
    accepted = runtime.step(0.001)
    state = runtime.snapshot()[4]
```

浮根资产使用 `FloatingRobotState` 和 `FloatingRobotCommand(RobotCommand(...))`。静态环境通过 `FiniteStaticShape` 及显式材料加入声明。编译记录的 `unlowered_world_geom_ids` 必须单独处理；`.asset` 本身不是完整环境。

选择原生 MuJoCo 物理时保留原生坐标、控制和接触律：

```python
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco

compiled = compile_mujoco("scene.xml", require_asset=False)
with MuJoCoRuntime(compiled, world_keys=(4, 19), mode="cuda") as runtime:
    runtime.set_controls({4: MuJoCoControl([1.0]), 19: MuJoCoControl([-1.0])})
    accepted = runtime.step(runtime.timestep)
    states = runtime.snapshot()
```

`mode="cpu"` 使用 `mj_step`；`mode="cuda"` 默认选择新 MuJoCo-TYX，显式 `implementation="warp"` 才使用 MuJoCo-Warp。TYX的CPU解释验证用 `mode="interpret"`。缺少依赖、不支持特性或设备不可用均报错，没有CPU或Warp静默回退。控制长度必须匹配原生actuator数。含多后端选择的示例入口见 `examples/unified_scene.py`。

新MuJoCo-TYX是有限模型域的算法移植，独立于原Phymeo TYX resident。T0支持纯平移树、中心各向同性自由球、单个plane–sphere法向接触；[T1已验收](../benchmarks/p65-mujoco-tyx-rotational-tree.md)增加固定根1–8DOF hinge/slide混合树；[T2已验收](../benchmarks/p65-mujoco-tyx-floating-tree.md)增加一个原生自由根加0–29 scalar joints，nq≤36/nv≤35，保留偏心COM、非球惯量和完整根–关节耦合。T1/T2每步重算完整FK/J/M/bias，要求没有可能接触对。多接触、摩擦和G1持续支撑尚不在该新后端支持域。详[扩展路线与多案例矩阵](../development/mujoco-tyx-expansion-plan.md)。

T1/T2的 `states[world].last_step.dynamics` 提供同次设备入步body pose/COM、世界COM Jacobian、M、bias、passive、applied和smooth acceleration；`integration_acceleration`另保Euler隐式阻尼解。T2的input_q/input_v保原生不同维度，root-origin世界线速度与root-local角速度不改为Phymeo资产坐标。端点CPU观察不能替代这些入步量。

## 身份、坐标和发布

编译记录保留原body/joint/geom ID和名字、qpos/dof地址、关节参考值、主惯量坐标和COM映射。原始top-level XML、规范化spec归档与独立MJB都保存；不声称保留了所有include文件原始字节。`fresh_native_model()` 从拥有的MJB重建，之后不依赖源路径。公开的 `mj_spec/mj_model` 本身是可变对象，运行时使用独立模型，避免timestep或参数互相污染。

Phymeo浮根线速度位于根COM且在世界坐标下，角速度也在世界坐标下；原生MuJoCo自由关节qvel使用原点世界线速度与局部角速度。记录提供双向映射，不能只交换四元数元素就当完整速度转换。

共同协议统一方法与发布约定，不统一各后端的观测字段和控制schema；Env Task仍负责显式动作与坐标适配。所有快照是独立只读数据。HOST/TYX统一入口使用原 `RobotCommand` / `FloatingRobotCommand`；MuJoCo使用显式原生 `MuJoCoControl`。`step`给出完整world接受掩码。CPU原生模式支持已发布world保留的部分失败；GPU原生模式在未发布buffer完成整个批次，容量或回读失败保持全批原状态。

MuJoCo-Warp在回读前检查contact/constraint计数，因为对应版本的底层回读会截断溢出；constraint行数必须严格小于容量，覆盖上游恰满时提前返回的边界。请求控制与实际存储控制分别保留，FP32溢出在替换整批控制前拒绝。实际GPU接触字段和力存入 `last_step`；端点快照的几何/力来自明确标记的detached CPU `mj_forward`。这两者不混称整步冲量。TYX和MuJoCo-Warp各自记录实际FP32时间，不能把请求dt当作精确设备累计时刻。

## 当前支持域与后续扩展

MuJoCo到Phymeo资产转换支持单个固定树或自由根、每活动body一个hinge/slide、球/胶囊/盒子、惯量主轴与偏心COM、浮根标量关节armature/damping、对称限幅的单位直接motor。显式contact pair/exclude、禁用actuator组、复杂传动、未折叠固定子body、球关节、tendon/equality、固定根非零armature/damping等会给出明确lowering问题。`require_asset=False`仍可在原生MuJoCo运行合法原模型。

Phymeo统一场景首域是每world一个独立关节系统与有限静态环境。机器人自身互碰、多机器人同world联合设备求解、静态胶囊HOST装配和固定根碰撞形状的部分跨后端反力场景，按能力报告拒绝。此处没有通过多次独立step模拟同world共同求解。

后续优先补全场景声明到MjSpec导出、更多显式资产转换和后端能力交集，再推进更大的GPU训练world数。不能用原生MuJoCo运行通过替代Phymeo自身碰撞、软体或联合求解验收。
