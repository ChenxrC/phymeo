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

## 2026-10-27 02:49 左右：Task 9 审阅完成

- 布片实现与输入修复已推送：`2e8d5db`、`9222e24`。实现验证为 Python 3.12 全套 216 passed、Python 3.11 CPU 209 passed/7 deselected、原生 pip 布片 35 passed、独立源码树外 wheel smoke 通过。
- 独立审阅复现 97 项相关测试、200 组随机网格质量/拓扑/所有权探针；发现两个低优先级输入缺陷，已用 6 项先失败的回归修复。修复后布片 41 passed，审阅复核无遗留需修正项。
- 默认 400 次预算的示例只验收 60 步；两精度延长到 frame 66 均显式未收敛。新增 `--iterations` 预算选项并记录限制，不改终止容差或伪报成功。
- 开始 Task 10：tyx 着色 XPBD。新 33 项测试已因缺失实现失败，尚未声称 tyx 约束可用。

## 2026-10-27 03:00 左右：Task 10 约束内核与真机对照

- Task 9 收口提交已推送：`f9b7f0a`。输入修复后原阶段完整套件 222 passed，Python 3.11 布片 41 passed；扩大预算到 800 后，FP32 布片 180 步通过，最大实际迭代 466，最大残差仍 <1e-6。
- TDD 的 33 项新合同先失败后在解释器全过；加入并行批次/稠密系统补充验证后，全套 258 passed，Python 3.11 CPU 226 passed/32 deselected（后续还会增加接触组合测试）。
- 新 tyx XPBD 已真实执行公开内核：初始 22 个后端用例分别在 C++ CPU 和 CUDA 全过；CUDA 另通过 3 项批量/耦合/固定测试。C++ 首轮含编译约 181 s，不把该值算稳态求解耗时。
- GPU 布片 30 步通过：最大迭代 258、最大归一化残差 9.995e-6 内、无平面穿透，固定点未动；与独立 FP64/1e-8 参考轨迹最大点差 0.000754075 m。参考首次 800 次预算不足，增加到 2000 后通过，容差未改变。
- 本机 GPU 该微型场景首步约 0.234 s，后续均值约 0.584 s，计入分配/传输/逐迭代主机验收；未声称性能加速。仍待补充 C++/CUDA 检查、包验证和独立切片审阅后关闭 Task 10。

## 2026-10-27 03:11 左右：Task 10 独立审阅完成

- 实现与审阅修复提交已推送：`02f576f`、`9f796dc`。审阅发现 P2：执行阶段误用原状态，绕过验证副本，导致合法负步幅视图在 CUDA 报非连续数组错误。新增 CUDA 回归先失败后通过，执行已改用 owned candidate。
- 修复后 Task 1–10 全套 261 passed；新 XPBD CUDA 27 passed。Python 3.11 和原生 pip 核心路径此前各 227 passed，最新构建的源码树外原生 pip wheel smoke 通过。
- C++ 补充的并行/接触 4 项及负步幅回归 1 项通过，结合先前 22 项覆盖全部当前 27 个 XPBD 后端用例；LLVM 仍未验证。
- 独立审阅额外验证：256 组三维解析边在解释器/CUDA 均通过，最大误差 5.158e-7；最小 FP32 次正规参数通过；13 个分配失败位置及正常/拒绝/kernel 异常路径都释放 GPU 数组、保持输入并关闭 runtime。修复复核后无遗留需修正项。
- 已准备 Task 11 共用刚体事务步进合同；25 个测试中 24 因粒子专用门禁或缺失输入 copy 失败，1 个既有 kind 误配拒绝通过。此时尚未实现刚体 Stepper。

## 2026-10-27 03:15 左右：Task 11 共用事务实现

- Task 10 收口与下一计划已推送：`8e550c5`，远程 SHA 核对一致。
- 新增显式状态族合同；粒子与刚体共用 Stepper 的报告/签名/一次回退机制。刚体快照、候选与输入完整复制四组状态和力矩，单位四元数/固定姿态独立校验；入口不再按字段相似猜测类型。
- TDD 的 25 项新测试已全过；全套 `--require-tyx` 为 286 passed，ruff/公开导入检查通过。首次定向测试受默认 pytest 临时目录 ACL 影响，设定工作区 TEMP/TMP 后全套通过，无源码规避。
- `spinning_body.py` 已走共用 Stepper：1000 步解析四元数误差 1.333e-15 内，单位长度误差 0，非球形一阶能量漂移仍为 0.00203856。wheel smoke 也改走刚体事务入口。
- 仍待多环境/包与新增状态边界审阅；未增加刚体接触、关节或学习模型能力。

## 2026-10-27 03:24 左右：Task 11 审阅修正完成

- 实现与修复已推送：`352700f`、`73afa59`。独立审阅发现 P2：扩展子类的 `copy()` 返回缓存或自身时，可能绕过基类校验并泄漏 caller/provider 缓冲。6 项粒子/刚体回归全部先失败，随后统一用内置构造器规范化合同字段。
- 修复后全套 292 passed；审阅复核原重现、新 6 项回归，以及 copy 直接抛异常的子类探针均通过，确认入口/候选不再虚调用子类 copy，四组刚体状态与力矩保持所有权。无遗留审阅问题。
- 实现提交此前已通过 Python 3.11 核心 252 passed/34 deselected、原生 pip 运行时 57 passed/1 deselected及全新 wheel smoke；修复后的多环境/包复核将随收口记录。
- 按已有球体接触候选规划 Task 12：只处理以质心为球心的球形刚体与静态 y-up 平面，固定无摩擦、恢复系数 0；不包含球球/通用碰撞/CCD。
- 修复后的 Python 3.11 和原生 pip 3.12 核心套件各 258 passed/34 deselected；重新构建并在源码树外用全新原生 pip 环境通过 wheel smoke。Task 11 收口完成。

## 2026-10-27 03:36 左右：Task 12 球/平面接触实现

- Task 11 收口与接触计划已推送：`718db00`。Task 12 首批 37 项测试先因缺接口失败，实现后通过；再加入陀螺/力矩保持、真实数学回退及微小穿透回归，当前接触 41 项通过。
- 新增 SphereShapes、完整半径/平面/定律签名和 RigidSpherePlaneSolver；仅处理 COM 球心、y-up 静态平面、无摩擦和恢复系数 0，位置稳定化不注入弹开速度。
- FP32/FP64 落球示例各 1000 步、dt=0.002 s，最终全部法向速度为 0、最低间隙为 0、固定球未移动；旋转解析误差分别 2.20e-7 / 1.333e-15 内。
- 自检发现通用数值设置的 NumPy 标量 promotion 会把微小 FP64 残差除法或比较降精度，导致误接受。新接触与已有 XPBD 各 2 项回归先失败，再把设置规范化为有限 Python/FP64 标量；XPBD 修复单独推送 `64e65d8`。全套当前 335 passed。
- CI 和 wheel smoke 已加入落球/接触；仍待多环境、安装包和新增接触切片独立审阅。
- 延伸上一轮所有权审阅时，发现绕过 Stepper 直接调用数学 solver 的旧入口仍虚调用 state.copy；数值失败也可能回传别名。补充 23 项直接入口/跨状态族/失败快照回归后，core 验证器按类型规范化，失败路径调用内置复制实现；当前全套 358 passed。

## 2026-10-27 04:00 左右：Task 12 极端尺度审阅修正

- 接触实现已推送 `3743e4c`；独立审阅复现两个 P2：大半径消掉小平面偏移、最小正穿透归一化成零。新增 8 项接触/NumPy XPBD/tyx host residual 回归全部先失败，再加入补偿求和与共用正残差下溢保护。
- 修复后全套 `--require-tyx` 366 passed；真实 CUDA 38 passed/328 deselected。Python 3.11 核心、原生 pip 3.12 核心均为 328 passed/38 deselected。ruff、公开导入边界和文档检查通过。
- 修复前后 Task 12 wheel 均在源码树外用全新原生 pip 环境通过安装与求解 smoke；新增 tyx 下溢回归在 C++ 后端也通过。不改变接触模型、残差容差或迭代预算。
- 独立窄范围复核：116 项接触/XPBD 测试通过；8,700 组间隙与 Fraction 精确计算一致，408 组有限接触高度为最小不穿透表示值，156 组归一化符合精确商舍入或正数下溢保守规则。两项审阅问题关闭。

## 2026-10-27 04:05 左右：Task 13 可选 CUDA 图实现与初次测量

- Task 12 数值修复已推送 `4053504`；Task 13 计划基线 `cbf117e`。首批 21 项测试先因缺少接口失败，再实现严格 `cuda_graph` 开关、显式 CUDA 图和本步资源生命周期；补充资源构造失败与非有限路径后共 25 项通过。
- 真实 CUDA 全套 391 passed；默认 interpreter 全套 374 passed/17 个明确 GPU 专用 skip。配置错误不要求导入 tyx，核心无可选运行时依赖。公开导入边界、ruff、文档检查通过。
- RTX 4060 Laptop / driver 566.07 / Python 3.12.14 / NumPy 2.5.4 / pinned tyx 0.1.0a0。`examples/compare_cuda_graph.py --steps 30` 使用 16 点、42 边、8 色，dt=0.02、tol=1e-5、最多 400 次，实测最大 258 次；交替图/直接顺序，每步位置、速度和完整报告相同。
- 首步 direct/graph 为 0.2259/0.0437 s，可能含编译与缓存不对称，不用于比值。其余 29 步完整步均值 0.5535/0.09057 s，中位数 0.6381/0.09710 s，p95 0.7314/0.12824 s；该次均值比 6.11。包含分配、传输、本步图捕获、逐轮 host 验收与清理；没有锁定 GPU 时钟/功耗，属于单机小场景测量，不是通用性能承诺。
- 图路径与独立 FP64 参考的 30 步示例也通过：最大残差 9.99412e-6，最低高度 0，最大点差 0.0007540744 m。Python 3.11 和原生 pip 3.12 核心各 336 passed/55 deselected；源码树外全新 pip 环境的 wheel 安装与 smoke 通过。
- 独立图边界审阅通过：25 项真机测试；额外探针覆盖同一 solver 依次 3/17/5/3 个粒子、变化质量/力/gravity/dt、负步幅状态、首次 replay 前捕获不修改缓冲、同签名 Stepper 数学回退、finalize 后异常的实际资源释放和后续复用。无阻断项。
- 默认图开关关闭，无跨步驻留或设备残差归约；远程 Actions 状态仍未核验。

## 2026-10-27 04:20 左右：Task 14 学习候选独立验收

- Task 13 已推送 `a2fbd2c`，远程 SHA 核对一致。Task 14 计划基线 `42ba0e8`：增加显式 mathematical/learned 能力、CandidateVerifier/VerificationReport、Stepper verifier 门禁与所有权隔离，以及无约束自由粒子 FP64 物理验收器。45 项缺失接口回归先失败后通过。
- 小型两系数线性回归在 64 个训练场景上拟合、64 个独立留出场景上全部接受；残差约 1.16e-15，位置与数学标签差约 2.24e-16 m，损坏系数及域外 dt 均成功数学回退。只是数据回归接入演示，不是通用 AI/神经网络或性能成果。
- 初版全套真实 CUDA 436 passed；Python 3.11/原生 pip 3.12 核心各 381 passed/55 deselected。但独立审阅发现 FP64 大基值与中间下溢会造成假零参考误差，因此未将该初版作为收口证据。
- 新增 3 项审阅回归先失败：1e16 位置或速度隐藏 0.5 误差，以及 w*f 下溢后被大 dt 放大的有限真实响应。改为展开运动增量的补偿方程残差；进一步回归覆盖非零次正规乘积丢精度后被放大约 0.05934 m 的情况，按实际 underflow 信号保守拒绝。精确次正规结果保留；当前 verifier 50 项通过，示例仍通过。
- 独立窄复核确认所有上述问题关闭，并验证精确次正规及二倍缩放不被误拒绝。修复后真实 CUDA 全套 441 passed；Python 3.11 和原生 pip 3.12 核心各 386 passed/55 deselected。公共 CPU CI 已加入回归训练/验证示例；远程 Actions 仍未核验。
- 实现已推送 `6cbdd5c`。包验证第一次误用旧 dist wheel，新增 smoke 因缺少 provider_kind 失败；此前写入“包验证通过”的进度结论过早，在此更正。随后显式重新 build 当前源码，源码树外干净 pip 安装及新增 verifier smoke 通过。为避免再次误用旧包，新增 Python 源码内容/文件集合一致性预检，4 项匹配、陈旧、缺失和多余文件测试先失败后实现；该检查不取代实际安装。

## 2026-10-27 04:40 左右：Task 15 同作用点耦合端口

- 防陈旧 wheel 检查已推送 `6c40385`，Task 14 台账收口真实 CUDA 全套 445 passed。Task 15 计划基线 `f1b3f36`，首批 33 项缺接口测试先失败，再实现 ParticleRigidPorts、ImpulseLedger 和双状态原子结果。
- 新增同作用点 J/-J、关于刚体质心的反向角冲量、世界／体主惯量响应；固定约束的载荷保留在 ledger。逐端口有效逆质量只表示自响应，多个共享对象端口的交叉项没有被省略成“全局解耦”承诺。
- 多点累积、子类 copy 所有权、混合精度拒绝及极小惯量直接比值测试后 37 项通过；全套真实 CUDA 482 passed，Python 3.11/原生 pip 3.12 核心各 427 passed/55 deselected。随后补两个退化场景先红后绿：固定刚体 K 和零冲量不应计算无用、可能溢出的力臂；当前端口 39 项通过。
- 偏离质心冲量示例：FP64 线／角动量误差约 1.11e-16 / 4.58e-16，相对点速度响应误差 1.11e-16；FP32 分别约 8.43e-8 / 2.72e-8 / 4.71e-8。只是瞬时交换算子，无碰撞检测、冲量求解、位置积分或多速率时间累计。
- 独立审阅发现有效质量提前构造倒惯量会在零／小力臂时无谓溢出；两个回归先失败后改为 Gram 分解，当前端口 41 项通过。独立复核另测每种精度 40 组共享端口随机场景（每组 7 粒子/4 刚体/30 端口），以及修复后 50 个随机姿态 K 响应；K 最大误差 1.60e-14。无遗留审阅问题。
- CI 和 wheel smoke 已加入端口；修复后真实 CUDA 全套 486 passed，Python 3.11/原生 pip 3.12 核心各 431 passed/55 deselected。重新 build 当前源码后，39 个 Python 文件与 wheel 完全一致，源码树外全新 pip 安装和端口 smoke 通过。ruff/文档/公开导入边界检查通过。
