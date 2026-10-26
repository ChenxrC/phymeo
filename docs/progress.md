# 开发进度

## 任务约束

- 用户已授权：完成架构图、两人分工、测试/环境/CI 计划并推送 GitHub，随后按大纲实现首版架构与部分数学求解。
- 截止：2026-10-27 08:00 Asia/Shanghai（2026-10-27 00:00 UTC）。到期停止新增范围，验证、保存并汇总。
- 仓库：ChenxrC/phymeo；工作目录是本项目。不得合并主分支、不得发布包。
- 自动接续任务：`phymeo-08-00`，每半小时；截止后停用。用户没有要求每轮重复通知。

## 2026-10-27 00:20 左右：设计阶段进行中

- 已确认本地和远程没有已有提交，无用户文件需迁移。
- 已核实 tyx 研究基线 SHA `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。
- 正在生成离线图、设计说明、团队/测试计划及 uv/pip/CI 计划。
- 实现尚未开始；没有 GPU 执行或引擎测试通过的声明。

## 实施记录规则

每个完成批次追加：内容、测试命令与实际结果、提交 SHA、后端/环境、下一步。未完成的计划保持未勾选。heartbeat 与目标续跑均先读本文件和 git log，避免重复执行。

## 2026-10-27 00:40 左右：设计验证

- 独立 builder check 提出并已解决：XPBD 投影后速度重建；柔顺本构残差与几何伸长分离；回退 solver 必须匹配物理问题签名。
- 离线 HTML/SVG/PNG 已生成。Edge 无头桌面截图检查通过；SVG 文本无越界；390px 移动端页面无整体横向溢出（图在独立区域滚动）。
- Windows / Python 3.12.14：直接 Python、uv 0.11.2 和原生 venv/pip 25.0.1 三条入口均通过文档检查（7 文档、24 本地链接）。检查不覆盖外部链接/Markdown 锚点。
- venv 初次 ensurepip 受沙箱临时目录写权限影响；改用被 gitignore 的项目 artifacts/tmp 后成功。这是本机执行环境处理，不改变普通用户的 venv 命令。
- 当前无 GPU 执行；GitHub Actions 仅已配置，尚未取得远程执行结果。

## 2026-10-27 00:55 左右：Task 2 基础包

- Task 1 已提交并推送：`f39ae22f3063ce80264befc97c2ba1b86aa69195`，远程 SHA 一致。
- 新增 ParticleModel/ParticleState/StepInputs/StepReport/StepResult，浮点类型、形状、有限性、质量和复制边界。
- TDD：17 项合同测试先因缺失实现失败，实施后 `python -m pytest -q` 为 17 passed；独立 pip 环境也为 17 passed，pip check 无破损依赖。
- 已生成 uv.lock 与含 hash 的 requirements/dev.lock；Python 3.12.14 下 NumPy 2.5.4、pytest 9.1.1；锁文件为 Python 3.11 保留对应依赖分支。
- `python -m build --no-isolation` 构建 sdist/wheel；干净 `.venv-wheel`、源码树外、`python -I scripts/smoke_installed.py` 确认从该环境 site-packages 导入。
- 当前只有数据基础，无积分/XPBD/真实 GPU 能力。下一步 Task 3 半隐式积分。

## 2026-10-27 01:03 左右：Task 3 自由积分

- Task 2 已提交并推送：`f919fa3fb8414e4b0efbe1cca1af2380e320bcc6`。
- 新增共享输入验证、NumPy 半隐式 Euler 求解器以及无图形自由落体示例；固定点保持位置且速度归零；候选溢出会返回拒绝报告与原状态副本。
- TDD：18 项积分测试先因接口缺失失败，实施后全套 `python -m pytest -q` 为 35 passed；ruff 检查通过。
- Python 3.12.14 / NumPy FP64：`python examples/free_fall.py` 运行 100 步，最终高度 5.045950000000004 m，与独立离散解析解误差 4.45e-15 m 以内。FP32 也通过十步解析测试。
- 当前尚无接触或约束；下一步 Task 4 XPBD 距离约束与简单平面接触。

## 2026-10-27 01:08 左右：Task 4 XPBD 与平面

- Task 3 已提交并推送：`2dae01f05ab150f5c46dd9c719c4508c44afe0c0`。
- 新增 DistanceConstraints、Plane、XPBDSolver：串行确定顺序投影，子步内累计乘子，投影后重建速度，动态点平面接触截去负法向速度。
- TDD：33 项 XPBD/数据合同测试先因缺失实现失败；实现后补查距离与平面联立、空约束和精度边界，共 36 项 XPBD 测试。全套 `python -m pytest -q` 为 71 passed；ruff 通过。
- `python examples/hanging_chain.py`：8 点、200 步、dt=0.01 s，全部接受；最大归一化本构残差 9.96e-9，最大迭代 171；最终几何伸长 6.87e-5 m 属于允许的材料柔顺响应。
- NumPy FP32/FP64 验证；没有摩擦、CCD、自碰撞、完整刚体或 GPU 声明。固定点与硬约束冲突、预算不足均显式拒绝。
- 下一步 Task 5：固定 tyx 提交安装及公开 API 真实执行适配。

## 2026-10-27 01:22 左右：Task 5 真实 tyx 路径

- Task 4 已提交并推送：`2018111ca8860a9c0ed960cc26a551811283cd69`。
- 安装并核验 tyx direct_url 元数据固定在 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`；新增延迟导入的 FP32 TyxSemiImplicitSolver、公开 root API kernel、独立 runtime/close 生命周期。无静默 NumPy 回退。
- TDD：13 项后端/依赖边界测试先因缺失实现失败。之后 interpret、cpu_cpp、cuda 分别运行 `pytest tests/test_tyx_backend.py -q --tyx-mode=<mode> --require-tyx`，各为 13 passed；其中 6 项为所选后端合同，其他为通用配置与依赖隔离。
- 全套含 tyx 解释器 `pytest -q --require-tyx`：91 passed；独立原生 pip 核心环境：85 passed, 6 skipped（明确因未安装 tyx），pip check 无问题。uv 环境用 `uv pip check`：17 packages compatible；uv 环境不内置 pip。
- CPU C++ 实际工具链：MSVC 14.50.35717。CUDA：RTX 4060 Laptop，驱动 566.07，实测 NVRTC 12.6；可选 hash profile 为 cuda-python/bindings 12.9.10、pathfinder 1.8.4、runtime headers 12.9.79。cpu_llvm 未验证。
- `scripts/check_tyx.py --mode=cuda --require-device` 与 `--mode=cpu_cpp` 均执行非空 kernel 并通过解析检查；public-import guard 与 ruff 通过。
- CUDA 每步传输主机数组，XPBD 仍是 NumPy 参考路径；不宣称 GPU XPBD、设备常驻性能或梯度能力。没有 GitHub GPU runner 配置或远程 CI 通过证据。
- 下一步 Task 6：相同物理问题签名、能力合同、候选验收和最多一次数学回退。
