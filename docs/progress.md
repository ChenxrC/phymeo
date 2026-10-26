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

## 2026-10-27 01:30 左右：Task 6 步进事务

- Task 5 已提交并推送：`862d2568f65105f35cd9093c8c6ca3b6da3a89fe`。
- 新增 SolverCapabilities / ProblemSignature / SolverFailure 与 Solver protocol；三个已实现 solver 声明物理身份、精度、功能与梯度/历史边界。
- Stepper 为每次尝试复制状态和外力，检查候选数据与固定点，再验收报告；同问题 fallback 最多一次。全部失败返回原快照；受控域错误可回退，程序异常继续暴露。
- TDD：29 项新测试先失败，实施后补查配置变更、非法输入不重试、成功输出别名；全套 `pytest -q --require-tyx` 为 123 passed。`pytest -m tyx --tyx-mode=cuda --require-tyx -q` 为 7 passed, 116 deselected。
- `examples/guarded_step.py` 真实 XPBD 链：一次低预算失败后，高预算解在 31 次迭代达到残差 9.32e-10，原状态未改变。ruff 与公开 tyx 导入检查通过。
- runtime 仅依赖 core；共享入口验证从 dynamics 移入 core/validation。尚无带隐藏状态的学习 provider 或独立 AI 物理验收器，当前明确拒绝历史状态能力。
- 下一步 Task 7：CPU CI、安装包/锁文件一致性以及独立分支审阅。

## 2026-10-27 01:37 左右：Task 7 CI 与安装验证

- Task 6 已提交并推送：`898b193f5de847d3567b747728bd891969fad94a`。
- 新增 CI：quality、8 组 Windows/Linux × Python 3.11/3.12 × uv/pip CPU、2 组 wheel、required 汇总；所有外部 Action 固定 SHA。未配置私有 tyx 凭证、GPU runner、分支保护或发布。
- 原生 pip 公开 CPU 测试 `pytest -m 'not tyx' -q`：116 passed, 7 deselected；pip check、ruff、compileall 通过。
- 新增由 uv.lock 导出的 runtime.lock；dev/runtime 两份锁文件再次导出的 SHA256 与原文件一致。CI YAML 用独立 YAML 解析器读取成功；这不代替 Actions 执行。
- 重新构建 sdist/wheel；`scripts/verify_wheel.py` 在源码目录外的全新原生 pip 环境中只安装 NumPy 与 wheel，pip check 通过，`-I` smoke 验证 site-packages 导入、自由积分、XPBD 与可选依赖隔离。
- 下一步独立全分支审阅；尚未取得远程 Actions 结果。

## 2026-10-27 01:46 左右：审阅修正数值边界

- CI 配置已提交并推送：`19e7cd2b91996765eb8368021d304be4a346a1af`。
- 在独立 Python 3.11.15 / NumPy 2.4.6 环境运行初始 CPU 套件 116 passed，悬挂链通过；原 Python 3.12/CUDA 环境保持不变。
- 独立审阅发现两个已复现的静默数值错误：`dt²` 先溢出导致有限柔顺比值变零；FP32 归一化尺度溢出导致非零残差变零。新增两个先失败的回归测试，然后改为逐次除 dt、FP64 归一化。
- 修复后 Python 3.12 含 tyx 解释器全套 125 passed；Python 3.11 CPU 118 passed, 7 deselected；ruff 通过。审阅尚未结束。
- 电脑操作工具初始化失败（failed to write kernel assets / 系统找不到指定的路径），无法借助已登录浏览器核验远程 Actions；Git 推送正常，远程 CI 仍标未核验。

## 2026-10-27 01:51 左右：审阅修正问题签名

- 数值修复已提交并推送：`796e9de`。
- 独立全分支审阅结束：无 Critical；三个 Important/P2 分别为上述两项数值问题和 ProblemSignature 的嵌套可变列表。内置 XPBD 原先生成 tuple，但外部 provider 传 list 时可意外改变 Stepper 保存的身份基线。
- 22 项签名构造/所有权回归先失败，随后将距离和接触标识规范化为自有不可变 tuple，验证索引、有限参数及语义标识；另加 Stepper 配置变化集成回归。全套 `pytest -q --require-tyx` 为 148 passed；ruff、文档检查通过。
- 接受审阅中明确延期项：完整刚体/机器人/FEM/弯曲/自碰撞；摩擦/非零恢复/CCD；独立 AI 物理验收与隐藏状态回滚；设备常驻性能、LLVM、远程 GPU CI。它们不由本轮基础测试隐含覆盖。强行绕过不可变对象不作为安全边界。
- 正在做修复后的最终 wheel 和多 Python/安装路径复核，再结束 Task 7；持续开发任务仍有效至 08:00。

## 2026-10-27 01:55 左右：基础阶段完成，进入 Task 8

- 签名修复已推送：`347640b412963db02aa939c9749ac627fc38d6c9`；Git ls-remote 与本地 SHA 一致。
- 独立审阅者复核三个修复的原始重现与 25 项定向回归，全部通过；没有遗留审阅阻断项。
- 最终基础验证：Python 3.12/NumPy 2.5.4 含 tyx 解释器全套 148 passed；原生 pip 3.12 与 uv 3.11.15/NumPy 2.4.6 各 141 passed, 7 deselected；CUDA 7 passed, 141 deselected。重建后的独立原生 pip wheel 安装与求解 smoke 通过。
- Task 1–7 完成。远程 CI 未核验、分支未合并、没有发布包。
- 按已授权后续候选新增 Task 8 合同：独立刚体姿态/角动力学，以世界角速度、body-to-world 四元数和质心主惯量为明确边界。此刻实现尚未开始，不宣称该能力已存在。

## 2026-10-27 02:08 左右：Task 8 刚体自由运动

- Task 7 收口与 Task 8 计划提交：`f480992ff21742993b624ccd9c0da78c691dfa80`。
- 新增独立刚体模型/状态/输入、主惯量校验、xyzw 旋转数学、含陀螺项的角动力学和指数映射姿态更新；StepResult 兼容不同状态类型，已有粒子 API 保持运行行为。
- TDD：31 项缺失接口测试先失败；实现后新增近单位四元数坐标系测试先失败，再在矩阵变换中做 FP64 归一化；另验证非球形自由转动能量误差随减半步长约减半。当前刚体 33 项测试通过。
- `examples/spinning_body.py`：1000 步、dt=0.001 s，球形惯量解析姿态误差 1.34e-15 内，单位长度误差 0；非球形能量相对漂移 0.002039，明确属于一阶方法误差，不宣称精确守恒。
- CI 与 wheel smoke 已加入真实刚体旋转；待完成多环境、包与新增切片独立审阅再关闭 Task 8。刚体接触、关节、GPU 刚体和统一刚体 Stepper 尚未实现。

## 2026-10-27 02:18 左右：Task 8 完成，进入 Task 9

- 刚体实现已推送：`3b6182efa3402a5c37c1cb92b4a801e3079d3c81`。
- 全套 181 passed；Python 3.11 CPU 174 passed, 7 deselected；原生 pip 刚体 33 passed。Python 3.11 和 3.12 均在源码树外新建原生 pip 环境通过 wheel 求解 smoke。
- 独立切片审阅无 Critical/Important/需修正 Minor；另复现 101 项相关测试，以及 100 组独立随机世界惯量/四元数/全局坐标变换对照，最大差 8.88e-16。未合并主分支。
- 开始 Task 9 计划：布片网格、三角形面积质量、结构/剪切距离约束及悬挂演示；暂不包含弯曲、自碰撞或 GPU 布料。持续任务仍至 08:00。

## 2026-10-27 02:29 左右：Task 9 布片构建

- Task 8 收口与 Task 9 计划已推送：`aa6687e28fbec1abccec6987e7568c3c4d4f1f63`。
- TDD：35 项 builder 测试先因缺失接口失败，实施后全部通过。新增 XZ/+Y 网格、三角形面积集中质量、结构/剪切两组距离连接、固定点及不可表示参数/退化网格拒绝。
- `examples/cloth_patch.py` 在 FP64 和 FP32 各运行 60 步、dt=0.02 s；16 点、18 三角形、42 边，物理质量约 0.10125 kg；均全部 accepted、固定点未变且未穿透平面。
- 最大归一化残差均 <1e-6；FP64/FP32 最大迭代分别 360/361，均值约 302/步。串行 CPU 参考有明确成本，未声称高性能或 GPU 布料。
- CI/wheel smoke 和两人分工已加入布片 builder；仍待全套、多环境、包和新增切片审阅收口。
