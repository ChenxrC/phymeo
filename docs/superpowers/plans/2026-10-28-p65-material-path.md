# P6.5 XPBD材料路径 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline with TDD. Keep the authorized dedicated branch and one final fresh review.

**Goal:** 让原生NumPy/tyx材料求解暴露可核对的逐边投影证据，不改变原物理结果。
**Architecture:** opt-in临时workspace记录(count,18)逐sweep帧；NumPy在原操作旁记录，tyx由traced kernel直接写设备操作数据。关闭时原路径不变，diagnostics复制所有权。下一阶段ledger再独立认证，当前不放行材料接触。
**Tech Stack:** Python3.11+、NumPy、现有固定tyx公开API、uv/pip。
**Spec:** [已builder check设计](../specs/2026-10-28-p65-material-path-design.md)。

## Global Constraints

用户授权持续P6.5，不重复审批常规实现；先关闭上一精确head的CI。当前产品aef9b257/118文件源码hash da5b54842e3256ef20b6eb5d27bfdcded77543f656330603f2ce7975d5989123。不得改原XPBD/numpy/tyx算术顺序、tol、积分器、旧FreeParticleLedger限制；不可host覆盖或用最终lambda伪造路径。该计划没有新的物理求解器完成声明。C++长期TYX-003无修复不重复压力。性能只观测，不优化。提交push专用分支，不merge/publish。

## Review Focus

- 默认关闭的真实kernel/传输计数不能变化；traced/plain候选与lambda要bitwise一致。
- trace验证必须在任何状态修改前；无效dtype/shape/只读目标不得留下半修改。
- 零边、None、两固定点/零denominator、coincident方向都必须记录实际分支，不能含未初始化值。
- 连续sweep的帧独立存储；只读快照及trial/substep之间无别名或未注册history。
- 不把记录存在当成材料守恒认证；保留NUM-005反例、实际compute精度/转换边界及未记载的额外load声明。

### Task 1: 两条native路径与诊断合同

**Files:** Create src/phymeo/constraints/distance_trace.py、src/phymeo/backends/_tyx_distance_trace.py、tests/test_material_path.py、tests/test_tyx_material_path.py；Modify constraints/distance.py、solvers/cloth.py、coupling/participant.py、backends/tyx_node_math.py、backends/tyx_participants.py。
**Interfaces:** spec中的record_material_path关键字；project_distances(...,*,trace=None)，traced的/count,18/写入原生有效delta与方向。诊断material_path schema/columns/frames/predicted_position/predicted_velocity/alpha/h严格按spec。helper提供帧shape/dtype/writeable验证与owned快照，不生成物理状态。

- [x] 写constructor bool/签名、native projector单边/两边逐sweep、开关bitwise等价和默认无额外诊断，运行预期missing keyword/API RED。
- [x] 实现NumPy记录与统一帧helper；零denominator先写原始/最终相同和delta0。显式预测副本来自native原结果，Expected NumPy GREEN。
- [x] 为actual interpret/CUDA写同样单边、多轮共享边、固定/coincident/empty与数据所有权RED，随后实现新traced kernel和TyxNodeMath可选分支。关闭时保持旧kernel原调用。Expected非零actual completed_launches，traced/plain状态和lambda逐位相同，记录不是host反算。
- [x] 扩展非法trace目标在修改前失败、两次diagnostics隔离、原输入不变、完整SceneStepper最后子步/关闭失败全恢复并重试=fresh；trace只属临时workspace。先RED后最小修复。Expected新旧参与者相关回归通过。提交。

### Task 2: 独立物理证据与工程验收

**Files:** 扩展上述tests；新建scripts/benchmark_material_path.py及阶段acceptance/engineering/review/性能记录；更新installed smoke、progress/continuation。
**Interfaces:** Task1 material_path帧；现有NUM-005和probe_xpbd_material_balance.py作为独立原式反例，不把该脚本当求解器。

- [x] 单边解析记录全部18项；用独立逐边方程核对实际position和lambda变化，再累计配置载荷/h对照实际速度重建；共享双边最终lambda*末梯度仍误差>.05且真实路径匹配。Expected普通尺度误差符合原backend预算。
- [x] 旋转硬边h=.1/.05/.025核对Lz=2/sqrt(1+h²)，原反例保持；软材料同时统计K与U，不把释放应变能当增能。actual interpret/CUDA与有界C++短例，C++长循环未验收明确记录。
- [x] 单次fresh最终review，Critical/Important一轮TDD修；必要全量CUDA/core311/nativepip312、interpret/短C++、ruff/docs/公开导入边界/compileall、uvpipcheck、build及源码外wheel。Expected命令全部读结果，失败不计通过。
- [x] 数值测试退出后，用同一小材料负载、相同初态、开关两配置各3x100，NumPy→interpret→CUDA串行计量并保留每次计数/失败；不是完整时域性能。不做优化。Expected真实路径及bitwise前提持续成立。
- [x] 保存源码hash/工程/NUM-005范围/性能/续作，提交push、核对准确SHA/CI并更新既有ACTIVE自动任务。完整P6.5继续，不关闭全goal。

执行状态：Task1完成；独立物理证据完成，单次fresh审阅1Important经5RED修复。记录API已实现，生产共同账本/联合接触未实现。修复后的全量工程、源码外wheel与1800次串行观测已收齐。准确69b866a的CI40671540301与Documentation40671540325均success，最后交付框已闭环；不重复旧探针或已关闭审阅。
