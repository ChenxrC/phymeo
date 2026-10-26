# phymeo

面向机器人、刚体接触、布料与柔性体的物理仿真引擎设计与首版开发。

**定位：** phymeo 拥有物理模型、状态、碰撞、数学/学习求解器和耦合调度；[tyx](https://github.com/ChenxrC/tyx) 拥有通用计算、编译、数组、自动微分和设备执行。

> 当前处于设计与基础实现阶段。图中描述的是目标架构，不能据此推断全部模块已实现、GPU 已通过验证或整体可微。实际交付以[进度记录](docs/progress.md)为准。

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

GitHub 文件页不直接运行 HTML；下载后用浏览器打开即可，无网络字体或脚本依赖。

## 当前基础包与验证

已提供粒子数据合同、NumPy 半隐式 Euler、XPBD 距离/平面约束，以及布片结构/剪切网格与面积质量构建器。刚体支持自由平移/旋转、四元数与陀螺项，并新增球形边界对静态平面的无摩擦、零恢复接触。粒子与刚体均可通过共用 Stepper 验收候选并做一次同物理问题回退，刚体快照包含姿态、角速度与力矩。

NumPy 支持 FP32/FP64；可选 tyx 已实现 FP32 粒子积分和着色 XPBD 距离/平面投影，后者含每迭代主机残差检查，尚未做性能优化。安装及已验证模式见[环境说明](docs/development/environment-and-ci.md)。

目前仍无球球/任意形状碰撞、关节、机器人、布料弯曲/自碰撞、摩擦、CCD 或已训练 AI 求解器，详见[进度记录](docs/progress.md)。

```sh
uv sync --locked --group dev
uv run --no-sync python -m pytest
uv run --no-sync python examples/free_fall.py
uv run --no-sync python examples/hanging_chain.py
uv run --no-sync python examples/guarded_step.py
uv run --no-sync python examples/spinning_body.py
uv run --no-sync python examples/cloth_patch.py
uv run --no-sync python examples/falling_spheres.py
```

安装可选 tyx/CUDA 环境后，可执行 `uv run --no-sync python examples/tyx_cloth.py --mode cuda`，与独立 FP64 参考做 30 步对照。`--mode cpu_cpp` 使用实际 C++ 编译；解释器使用 `--mode interpret`。输出包含残差、轨迹差异及含同步/传输的耗时，不把小布片示例作为 GPU 加速比基准。

原生 pip 用户在激活的 Python 3.11+ 虚拟环境中执行：

```sh
python -m pip install --require-hashes -r requirements/dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest
```

构建与独立安装验证：`python -m build --no-isolation`，然后 `python scripts/verify_wheel.py`。后者使用系统临时目录中新建的原生 pip 环境，需要读取公开包索引。CI 配置覆盖 Windows/Linux × Python 3.11/3.12 × uv/pip；工作流是否实际成功以 Actions 结果为准。

## 文档检查

需要 Python 3.11+，不需要安装第三方包。

```sh
python scripts/check_docs.py
```

uv 与 pip/venv 的隔离环境、可选 tyx 安装和后续 GPU 计划见[环境说明](docs/development/environment-and-ci.md)。

## 来源与实现原则

借鉴 Newton 的公开架构与算法论文，独立定义物理数据、接口与实现，不复制其求解器源码。对照来源和 tyx 固定提交见[架构说明](docs/architecture/overview.md)。图表样式基于 Cocoon AI 的 MIT 模板改编，保留[许可声明](docs/architecture/template-LICENSE.txt)；该声明仅覆盖模板，不为整个项目指定许可证。
