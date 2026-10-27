# phymeo

面向机器人、刚体接触、布料与柔性体的物理仿真引擎设计与首版开发。

**定位：** phymeo 拥有物理模型、状态、碰撞、数学/学习求解器和耦合调度；[tyx](https://github.com/ChenxrC/tyx) 拥有通用计算、编译、数组、自动微分和设备执行。

> 当前处于设计与基础实现阶段。图中描述的是目标架构，不能据此推断全部模块已实现、GPU 已通过验证或整体可微。实际交付以[进度记录](docs/progress.md)为准。

上一轮基础开发已收口：交付分支 `codex/architecture-v0` 完成计划 Task 1–15。Task 16 的原始 tyx 刚体原型保留在独立[草稿分支](https://github.com/ChenxrC/phymeo/tree/codex/tyx-rigid-wip)；本分支已选择性修复并验证刚体预测，整分支独立审阅在最终交付前执行。

当前长程目标：**一个可重复验证的刚柔双向交互场景**。在 `codex/rigid-cloth-coupling` 分支实现布片与动态球接触、联合回滚、双向/单向对照与离线回放，先 CPU 参考再验收 tyx。见[场景与验收设计](docs/superpowers/specs/2026-10-27-rigid-cloth-coupling-design.md)和[七个里程碑](docs/superpowers/plans/2026-10-27-rigid-cloth-coupling.md)。这是一项正在实施的新目标，不代表场景已经完成。

点/球几何、双向响应、联合步进与整帧回滚已实现。独立整分支审阅发现的一个速度重建问题已修复；最终版本的 CPU 基准/重复/单向对照，以及实际 CUDA 两次 240 帧均通过，设备重复的 22 个数值数组逐元素一致。与同顺序参考相比，球/布的最大位置差为 0.111/0.020 mm，见[设备报告](docs/benchmarks/rigid-cloth-device.json)。[离线回放](docs/demos/rigid-cloth.html)已更新为修复版同源轨迹；半步长和质量对照仍在复验，之后更新完整[报告](docs/benchmarks/rigid-cloth.md)并收口。

![phymeo 架构图](docs/architecture/phymeo.png)

## 设计与实施入口

| 内容 | 入口 |
| --- | --- |
| 离线 HTML 架构图 | [下载/打开 phymeo.html](docs/architecture/phymeo.html) |
| 可编辑矢量图 | [phymeo.svg](docs/architecture/phymeo.svg) |
| 物理模型、接口与算法边界 | [架构说明](docs/architecture/overview.md) |
| 两人并行分工、里程碑与测试建议 | [开发与测试](docs/development/team-and-tests.md) |
| uv/pip、私有依赖、CPU/GPU CI 与发布计划 | [环境与 CI](docs/development/environment-and-ci.md) |
| 首版可执行大纲 | [v0 实施计划](docs/superpowers/plans/2026-10-27-v0-foundation.md) |
| 已验证结果与下一步 | [进度记录](docs/progress.md) |
| 刚柔固定场景、数值轨迹与对照命令 | [复现实验](docs/benchmarks/rigid-cloth.md) |
| 双向/单向真实轨迹的暂停与逐帧对比 | [离线回放](docs/demos/rigid-cloth.html) |

GitHub 文件页不直接运行 HTML；下载后用浏览器打开即可，无网络字体或脚本依赖。

## 当前基础包与验证

已提供粒子数据合同、NumPy 半隐式 Euler、XPBD 距离/平面约束，以及布片结构/剪切网格与面积质量构建器。刚体支持自由平移/旋转、四元数与陀螺项，并新增球形边界对静态平面的无摩擦、零恢复接触。粒子与刚体均可通过共用 Stepper 验收候选并做一次同物理问题回退，刚体快照包含姿态、角速度与力矩。

`ParticleRigidPorts` 提供同作用点的粒子／刚体冲量交换、角冲量记账及每端口有效逆质量，两组状态数值失败时同时回滚。它是耦合响应算子，尚不负责求接触或约束冲量。

NumPy 支持 FP32/FP64；可选 tyx 已实现 FP32 粒子/自由刚体积分和着色 XPBD 距离/平面投影。`TyxRigidSemiImplicitSolver` 验证了 interpret、实际 C++ 和 CUDA 的平移、四元数、陀螺项与异常回滚。`TyxCoupledXPBDSolver` 已执行真实设备刚柔接触并通过完整 240 帧对照与独立重复。独立的粒子 `TyxXPBDSolver` 可选择图重放一轮投影；联合驱动暂未采用 CUDA 图，均保留主机残差检查。安装及已验证模式见[环境说明](docs/development/environment-and-ci.md)。

学习型 provider 必须配置独立物理验收器。目前 `FreeParticleVerifier` 验证自由粒子的离散运动方程；附有两系数数据回归、留出验证和错误候选回退示例。仍无球球/任意形状碰撞、关节、机器人、布料弯曲/自碰撞、摩擦、CCD 或通用 AI 求解器，详见[进度记录](docs/progress.md)。

```sh
uv sync --locked --group dev
uv run --no-sync python -m pytest
uv run --no-sync python examples/free_fall.py
uv run --no-sync python examples/hanging_chain.py
uv run --no-sync python examples/guarded_step.py
uv run --no-sync python examples/spinning_body.py
uv run --no-sync python examples/cloth_patch.py
uv run --no-sync python examples/falling_spheres.py
uv run --no-sync python examples/learned_free_particles.py
uv run --no-sync python examples/point_impulse.py
```

安装可选 tyx/CUDA 环境后，可执行 `uv run --no-sync python examples/tyx_cloth.py --mode cuda`，与独立 FP64 参考做 30 步对照。`--mode cpu_cpp` 使用实际 C++ 编译；解释器使用 `--mode interpret`。输出包含残差、轨迹差异及含同步/传输的耗时，不把小布片示例作为 GPU 加速比基准。

加 `--cuda-graph` 可启用显式 CUDA 图路径。`python examples/compare_cuda_graph.py` 在相同精度、阈值和迭代规则下逐步核对图/直接启动的状态及报告，并计量完整 step；本机小场景结果见[进度记录](docs/progress.md)，不代表其他场景或机器的加速比。

原生 pip 用户在激活的 Python 3.11+ 虚拟环境中执行：

```sh
python -m pip install --require-hashes -r requirements/dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest
```

构建与独立安装验证：`python -m build --no-isolation`，然后 `python scripts/verify_wheel.py`。后者先检查 wheel 的 Python 源码与当前 checkout 一致，再使用系统临时目录中新建的原生 pip 环境安装，需要读取公开包索引；它不会自动构建旧 wheel。CI 配置覆盖 Windows/Linux × Python 3.11/3.12 × uv/pip；工作流是否实际成功以 Actions 结果为准。

## 文档检查

需要 Python 3.11+，不需要安装第三方包。

```sh
python scripts/check_docs.py
```

uv 与 pip/venv 的隔离环境、可选 tyx 安装和后续 GPU 计划见[环境说明](docs/development/environment-and-ci.md)。

## 来源与实现原则

借鉴 Newton 的公开架构与算法论文，独立定义物理数据、接口与实现，不复制其求解器源码。对照来源和 tyx 固定提交见[架构说明](docs/architecture/overview.md)。图表样式基于 Cocoon AI 的 MIT 模板改编，保留[许可声明](docs/architecture/template-LICENSE.txt)；该声明仅覆盖模板，不为整个项目指定许可证。
