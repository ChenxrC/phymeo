# 统一场景编译接手说明

**2026-11-10更新：** 默认CUDA已是独立MuJoCo-TYX，显式 `implementation="warp"` 才走Warp。T0及无接触固定根T1旋转树已完成本机验证，见[T1实际报告](../benchmarks/p65-mujoco-tyx-rotational-tree.md)。以下M1/M2和坐标/导出边界继续适用；新TYX运行源码入口是 `_mujoco_tyx_runtime.py` 与 `backends/mujoco_tyx/`，下一项为原生自由根树，不能从旧接手基线推断它已支持。

本说明整理 2026-11-10 这批统一入口。核对基线为父集成 `fab6961fe345fd4699336a8dd9ac40568bafb9cb` 及当时父工作副本；不将未提交的后续修改自动计入已验证结果。这里只做源码和文本整理，没有重新运行项目命令。

先读[架构与使用示例](../architecture/unified-scene-compilation.md)，再读[实施计划](../superpowers/plans/2026-11-10-unified-scene-compilation.md)。具体能力以 `scene_capabilities`、MuJoCo 编译记录的 `lowering_issues` 和每个后端各自的实际验收为准。

## 数据流和所有权

```text
原 URDF/MJCF 加载器 / 程序构造
    -> 原 RobotAsset / FloatingRobotAsset / affine asset
    -> ArticulationWorld + FiniteStaticShape + SceneDeclaration
    -> scene_capabilities -> compile_scene
       -> 原 Phymeo HOST scene
       -> 原 TYX resident interpret / CUDA

原 MJCF/URDF + includes / assets
    -> 可选 CPU MuJoCo MjSpec.compile
    -> MuJoCoCompilation
       -> 显式支持域转换后的 .asset / .initial_q / .initial_v
          -> 上面的 Phymeo SceneDeclaration 路线
       -> 完整 native MJB 快照 + 新建独立 native model
          -> MuJoCoRuntime CPU mj_step
          -> 默认 MuJoCo-TYX interpret / CUDA，有限T0/T1支持域
          -> 显式 MuJoCo-Warp CUDA 原生适配
```

这不是一条“MuJoCo 编译后，全部 MuJoCo 物理自然在 TYX 运行”的链路。M2 的资产转换只交付被明确支持的公共模型；完整 native 编译物是另一条可选运行路线。

`SceneDeclaration` 是现有模型的冻结装配清单，不是第二套 `SceneModel` 或新物理求解器。每个 `ArticulationWorld` 拥有独立 world ID、公共资产、初态、驱动参数、初始控制和有限静态环境。数组经拥有型快照隔离，后续修改调用者数据不能改变编译实例；编译生成的资源由各自 runtime 关闭。

M1 首域每 world 一个 articulation。多个 world 可以有不同资产和稀疏 ID；这不表示多个 articulation 已在同一 world 里共同进行设备求解。self-collision、CCD、动态环境等未支持项不会经多次独立 step 伪装完成。

## 入口、状态和语义

| 路线 | 输入与输出 | 物理与精度边界 |
|---|---|---|
| M1 Phymeo HOST | `SceneDeclaration`；`RobotCommand` / `FloatingRobotCommand`；`WorldSnapshot` | 复用原 HOST 物理与接触律，FP64 路径 |
| M1 Phymeo TYX | 同一声明和同一控制格式；`WorldSnapshot.raw` 保原设备快照 | 原 resident 动力学、FK、宽相；原 FP32 表示与 FP64 anchors；接触窄相和力律仍含 HOST |
| M2 编译前端 | 原 XML/URDF -> `MuJoCoCompilation` | CPU 原生编译、来源和坐标映射；编译成功不等于资产可转换或任一后端求解通过 |
| Native MuJoCo CPU | 原编译记录；`MuJoCoControl`；原生 qpos/qvel | 原 MuJoCo `mj_step`、接触与约束语义，不改名为 Phymeo 接触律 |
| Native MuJoCo-Warp CUDA | 原编译记录；显式原生 GPU 适配 | Warp/MuJoCo-Warp 运行，不是 TYX；其原始 GPU 步读数与端点 CPU forward 分开保存 |
| 新 `mujoco-tyx` 方向 | 独立后续实施 | 当前仍在实施，不能借用前面任何 HOST、TYX resident 或 Warp 通过记录称其已通过 |

共同的 `snapshot / set_controls / step / reset / close` 表示运行协议可以由原环境层消费，不表示状态字段和控制含义完全相同。M1 使用 `time_s` 和公共物理坐标；native 适配使用 `time`、原生 actuator/dof/body 顺序及原生坐标标签。Task 必须显式选择或转换观测与动作。

浮根尤其不能只重排四元数：Phymeo 根线速度是根 COM 的世界速度，角速度为世界角速度；原生 MuJoCo 自由关节速度涉及根原点线速度与局部角速度。M2 保留根 COM、关节参考值和地址映射，提供相应转换；转换依赖当前姿态。

M1 的 `shape_keys` 保留源 collider 到运行时 key 的映射，`collision_models` 保实际模型，`contact_receipts` 保接触发布证据。`WorldSnapshot.raw` 保 detached 后端原记录。不得丢弃 affine 的 full/independent 坐标、固定基座反作用或一对接触的另一侧来制造统一外观。

## M2 资产不是完整场景

从 [`mujoco_compiler.py`](../../src/phymeo/builders/mujoco_compiler.py) 接手时，重点核对：

- `.asset`、`.initial_q/.initial_v` 是显式转换结果，可能不存在；查看 `lowering_issues`，不能删除拒绝项后继续当作成功。
- `.unlowered_world_geom_ids` 单列未装配的环境几何。不能只拿 `.asset` 建机器人，就把原 XML 的地面、环境和接触参数也标成已导入。
- `.mapping` 保存原 body/joint/geom ID、名字、qpos/dof 地址、参考值及坐标关系；它不代表对所有 native 机制都有公共资产表达。
- `.source` 保存原 top-level XML、规范化归档和 MJB。它不声称逐字保存所有原 include 文件。`mj_spec/mj_model` 可变；`fresh_native_model()` 从拥有的 MJB 建独立模型，runtime 不依赖共享可写对象或原路径继续存在。
- tendon/equality、复杂传动、球关节、未折叠固定子体、当前公共模型不支持的固定根 a/D 等，不能因 native 模型合法就归入 Phymeo/TYX 支持域。`require_asset=False` 是保留 native 路线，不是静默降级。

M1 的实现入口在 [`scene_compiler.py`](../../src/phymeo/builders/scene_compiler.py)，专项合同在 [`test_scene_compiler.py`](../../tests/test_scene_compiler.py)。M1 已由父报告验收 HOST 13、TYX interpret 4、CUDA 4；M2 资产进入 M1 的同源短步组合，HOST/interpret/CUDA 各 1 已由父报告通过。这些是限定案例和路径的验收，不是任意 native 模型、任意环境或大规模训练的总完成证明；本文没有重跑这些组。

M3 入口在 [`mujoco_runtime.py`](../../src/phymeo/backends/mujoco_runtime.py) 和独立的 [`_mujoco_warp_runtime.py`](../../src/phymeo/backends/_mujoco_warp_runtime.py)。它们已有各自的父验证边界；后续修改和新 TYX 移植仍须分别验收。

## 发布和失败约定

M1 继续使用真实后端状态，禁止写回拟合后的 q/v。`step` 的 world 接受掩码必须完整；已成功发布的 world 留在真实末态。部分或未知发布需要 full reset，不能宣称物理回滚。全拒绝是否可重试由明确接受掩码和原后端错误约定决定；不得把未知失败补成全 False。

M1 的 selected reset 保其他 world 的控制和状态；成功恢复后才清理相应时钟和 receipt。多次编译不能共享可写状态。编译或关闭失败要保留错误 cause 和资源所有权。

TYX 对实际 h 使用原 FP32 转换，native GPU 也有自己的 FP32 时钟。检查 `last_requested_dt_s`、`last_effective_dt_s` 或原生时刻，不把请求 h 当成设备精确累计时间。二进制 h 的短步一致不代表任意 h 逐位一致。

native `last_step` 的原求解读数与 `snapshot` 的 detached endpoint-forward 数据是两份不同证据。端点力不能直接乘 h 冒充整步冲量；GPU 原始 contact/constraint 数量必须在可能截断的回读之前检查。native GPU 与原 TYX resident 的事务和容量证据也不能相互替代。

## 后续 SceneDeclaration -> MjSpec 导出还缺什么

目前 native 路线消费原 MuJoCo 编译记录，没有从任意 `SceneDeclaration` 构造 `MjSpec` 的导出器。后续应先冻结以下选择，再写开发和测试：

| 缺项 | 必须作出的明确选择 | 最小验收 |
|---|---|---|
| 完整结构与身份 lowering | 从资产结构导出 body/joint/geom、固定/自由根、惯量主轴和 COM；避免重复应用已折入模型的根变换；保源 key 与 native ID 映射 | 多姿态 FK、静态惯量和独立 M/J 对照；不能从某个姿态的 M 矩阵反猜结构 |
| 初态与坐标 | 公共 q/v、参考关节零点、根 COM/原点及角速度转换；规定是否导出 keyframe | q/v 双向往返、非零根 COM、非轴对齐惯量和非零初速；不得换姿态使测试通过 |
| 环境与过滤 | 导出全部有限静态几何、材料顶点身份、group/mask、显式 pair 排除和稀疏 world 归属 | 环境 inventory 完整；边缘/背面/互斥 pair 正负例；两侧力和力矩使用已标记参考点 |
| 接触律 | 显式选择保留 Phymeo 标定律，还是选择 native MuJoCo 律；不能只把 k/d/mu 数值塞进不同含义的字段 | 厚度、载荷—间隙、阻尼和能量的小模型标定，再做受载窗口；不同律使用工程门，不能借同名字段声称等价 |
| 驱动与被动项 | 原 bounded PD、feedforward、effort cap、armature/damping 与 native actuator/gear/ctrl 语义；明确采样与限幅时刻 | 单关节和浮根实际力/运动共同发布、惯量/阻尼例；控制格式转换不能丢失或重复外载 |
| affine/mimic 与限位 | 是否输出 native equality、保完整树还是消元；模型中的限位元数据是否成为真实约束 | full/independent 坐标、虚功/反作用、正反乘子和失败例；不得凭复制 range 偷加不同的停止力律 |
| 多 world 与后端能力 | 一个 native 模型的复制与不同资产 world 是不同问题；同 world 多 articulation 不能拆成独立 step | 稀疏 IDs、异构 world、独立初态/reset/控制；unsupported 项在创建半可用 runtime 前报出 |
| 来源、可选依赖与交付 | 导出声明 identity、版本、选定语义、native 编译物与限制报告；保持核心可选导入边界 | 无可选依赖的基础导入、源码外 wheel、小案例可重建；CPU、Warp GPU、新 TYX 各自记录实测结果 |

建议先做一个无接触 fixed tree 和一个偏心 COM 浮根的结构/状态导出，再加有限环境与显式材料策略；最后处理 affine 和更多 native 特性。每批关闭一个可调用接口或实质失败，不能用重复已有验收、扩大声明或没有新运行能力的推导代替推进。

## 本次架构文档只读审阅

核对的父工作副本已经区分 M1、M2 和 native CPU/GPU，说明 `interpret` 的含义、HOST 窄相、两种控制/坐标、独立 MJB、环境缺项以及尚未实现反向导出。未发现把 native MuJoCo 能力直接归给 Phymeo/TYX 的实质性夸大。

建议父在发布前做两处文字收窄：把“共同运行协议”明确限定为方法与发布协议，补足 Task 坐标/动作适配要求；把“完整五后端入口”改成“包含五种后端选择的示例入口”，避免被理解为所有模型/接触域通用，或 CLI 的每个分支都已有完整验收。本文未修改父文档或任何源文件。
