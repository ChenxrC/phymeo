# P6.5 仿射特征整段净距认证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development；用户要求明确合同/测例后交实现子agent。TDD；本独立阶段仅一次fresh最终code review及必要修复，不另加重复task/branch review。

**Goal:** 证明声明仿射运动的点/三角和边/边整个时间区间保持正净距，或明确返回违例/未决，供可复用accepted-path verifier使用。
**Architecture:** owned FP64端点定义精确有理轨迹；有界时间细分、同步相对顶点投影的精确整段证明；候选最近点只提供可行权重，真正证明不依赖舍入采样距离。纯函数不可变分区报告，薄事务测试消费真实候选端点。
**Tech Stack:** Python3.11+/NumPy、Fraction；无新依赖，host计算，actualtyx只生成端点。
**Spec:** [冻结设计](../specs/2026-10-29-p65-linear-sweep-design.md)。

## Global Constraints

- 只支持(1,3)/(3,1)/(2,2)特征，四组FP64有限端点；clearance_m严格内置float、有限正，按实际存储精确值，不接受bool/int/FP32。
- 归一化t∈[0,1]；max_intervals正Integral非bool，max_depth Integral非bool且0..52，返回内置标量和tuple。预算按取出区间计数，左优先深度遍历。
- certified必须整段距离>=clearance；violation必须实际有理可行见证严格小于clearance；unresolved不允许被集成层当安全。
- 分区完整覆盖[0,1]，早停/预算耗尽不能漏掉pending；候选权重精确归一后作用于真正有理轨迹。不得用float平方下溢、舍入中点或closest.distance当证据。
- 不改旧closest/静态证书/材料接触/SceneStepper/tyx pin；不假装支持FK旋转、TOI、接触反作用或生命周期。完整H1/P6.5仍后续。
- 原e63a1bf准确两CI已成功；不重做旧模块/审阅/观测。一个最终review，性能只观测，正式计量等待所有数值/审阅退出。

## Review Focus

- 次正规clearance平方不能先在FP64下溢为0；整数或高精度阈值不得偷偷转换成不同物理请求。
- 中点/候选浮点舍入或closest KKT容差不能产生假的安全/违例；真正有理见证与同步端点证明独立重验。
- 违例早停及预算耗尽仍需完整区间覆盖；接触等号与起终点违例不能被遗漏，不冒称首次TOI。
- 退化/最近点候选失败保守处理，未知RuntimeError不吞；轴/顶点对候选不能作为静默物理或后端替代。
- 报告不可变标量/tuple，不保留方法字符串数组或可变区间别名；薄验证拒绝unresolved并在末子步与close后全局恢复。

### Task 1: 生产整段认证与独立物理路径测试

**Files:** Create `src/phymeo/geometry/linear_sweep.py`（公开校验/报告/有界遍历）；若证明与候选部分需独立组织，允许create `src/phymeo/geometry/_linear_sweep_math.py`（仅本模块内部精确算术/候选），不拆新公共通用库。Create `tests/test_linear_sweep.py`、`tests/test_linear_sweep_transaction.py`；modify `src/phymeo/__init__.py`公共导出与`scripts/smoke_installed.py`。不能改其他生产/旧测试语义。
**Interfaces:** Consume旧closest公开函数只作候选；produce `linear_feature_sweep(first_start, first_end, second_start, second_end, *, clearance_m, max_intervals=128, max_depth=20) -> FeatureSweepResult`，字段严格按spec。

- [ ] 写缺API RED（测试体内断言公开API存在，不以collection error代替RED），然后加入板穿越/安全/等号滑动解析例RED。最小实现输入/冻结报告/精确同步方向证明，真正运行确认GREEN。
- [ ] 写两根独立杆件穿越/安全对照、错过根中点的t=.25板穿越、max_intervals=1未决RED，加入有界左优先细分。先冻结128/20正常预算，不因看到结果放宽。
- [ ] 加入安全但必须细分的转向杆解析例（spec第5项），独立验证distance²>=.5而r=.5；预算1未决、128/20必须certified，不能靠始终未决满足保守性。
- [ ] Review Focus逐项RED：最小正clearance/ULP中点、类型/shape/NaN/Inf、深度0/52及计数不超预算、初/末违例、所有pending分区覆盖、共同运动/互换/置换/可表示变换、退化/近平行/候选失败/异常传播、owned结果与失败重试。独立oracle用解析或单独有理几何推导，不调用生产私有证明作oracle。
- [ ] 薄SceneVerifier用before/after真实多域端点认证，正常接受、末子步violation与unresolved拒绝、验证通过后close失败；state/history/输入全恢复、retry=fresh。至少两子步，不改变运行时或覆盖物理状态。测试侧负控制去掉拒绝应RED，防止注入未触发。
- [ ] actualinterpret/CUDA各短例生成原生端点并断言后端执行计数和实际位置；标记tyx，host证书不称GPU CCD。相关新模块/旧surface_features、feature_distance_bound/事务/SceneStepper一次定向回归，记录实际命令/版本/RED/GREEN。
- [ ] 补公开导出和installed smoke中板穿越+安全对照（源码smoke不冒充安装）。ruff/diff检查、自查文件范围，提交产品测试；写task report给主agent。源码外最终wheel由Task2主agent负责，Task1不重复旧全量/长期CPP/正式性能/最终审阅。

### Task 2: 单次审阅、工程验证与后续接入

**Files:** 新stage acceptance/engineering/review与PERF、可选小型serial benchmark；更新progress/continuation/framework双线状态。不修改Task1算法以省性能。
**Interfaces:** 消费Task1API；输出版本化工程/性能/准确Git与CI证据，继续后续生命周期/机器人FK路径，而非勾完整H1。

- [ ] 主agent核对spec覆盖与任务报告后，连同计量脚本仅派一次fresh最终code review；重要问题先独立RED再修，必要复验，不另派第二次review。
- [ ] 两CPU环境相关回归、Ubuntu新host、实际interpret/CUDA短集成、ruff/docs/边界/compile、uvpip/锁导出一致及最终源码外wheel。无新增tyx kernel，不重跑旧1950CUDA/已完成静态距离900观测；准确最终远程CI承担全分支CPU矩阵，不将选测称全量。
- [ ] 所有数值/review探针退出后，三个独立host路径（简单安全、途中穿越、安全细分）各3×100串行计量，记录status/区间计数/实际源与脚本hash，排除fixture。未测GPU/Warp不作比较，不优化。
- [ ] 更新框架线“声明仿射路径的净距认证”、案例线“尚未接入动态接触/机器人旋转”，记录遗留与支持边界。commit/push专用分支、核对远程SHA/准确CI，然后继续厚度侧/法向/接触身份history/自动建立分离与FK保守路径。不合主分支/发布包/扩P7P8。

计划自查：spec的接口/精确阈值/整段证明/候选/分区/预算/退化/两独立场景/事务/后端均落入Task1，工程观测与交付属于Task2。builder一次nothing important left；Task1/2尚未开始。
