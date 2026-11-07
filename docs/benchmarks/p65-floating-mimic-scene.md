# F2 浮根仿射 mimic：T3 共同场景候选

2026-11-07，**产品与完整窗口 UNEXECUTED**。作者从 tests-first `60a98335a2c5c3155e6c1cec6e6c0860787bddbb` 继续，仅新增本报告与 `src/phymeo/runtime/_floating_mimic_participant.py`。未 merge 父、未改冻结测试/例程/资产/物性/门，未执行 import、collection、数值、pytest、static、build 或设备命令。

父已将 tests-first 导入 `9326aa1`，124 首 RED 为 **1 failed / 0.66 s / RAW1**：既有 standalone +1 N、原行 −4/7 N 先验先通过，随后准确在 `make_floating_registration` 拒绝 `FloatingMimicRobotAsset`，不是 missing import。该记录保留，不重跑。T2 原23新项与37共享兼容已由父接受；这里不将其算作 T3 通过。

## 接入与维数

新内部接口为 `FloatingMimicParticipant(address, registration, occupied)`，消费父原 `FloatingRobotRegistration` 的 wrapper 资产。父维护原登记验证、factory 类型派发、builder 说明和根导出；本片不复制 registry 或场景。子树未合父抽取的 `_resolve_floating_drive` 等共享接缝，不能把当前子树独立可导入误报为已验证。

participant 的独立状态 q/v 为 7+k/6+k，完整物理状态为 7+n/6+n，`velocity_count=6+n`、`body_count=n+1`。资产、初态、命令与观察均按原机制复制/冻结。只构造一次原 `FloatingRobotParticipant` 作完整 tree 的只读适配：它提供 full M/b/J、poses、实际材料 owner/F1 authored-frame 转换和唯一一套 key mapping。完整树登记中的零控制元数据不会被用于 prepare/integrate；不创建额外物体或自由度，不重复占用 collision key。

## 一次物理候选和真实载荷

`prepare` 在同一入步展开完整状态，用原完整几何适配读 M/b/J，再以 `S=diag(I6,P)` 投影 M、bias 和已声明 COM 外载。控制直接调用父原 `_resolve_floating_drive`，保自由根 Schur、限幅与 `feedforward=None`/显式零的区别；接触由场景随后在同一入态统一采样，未塞进 PD 默认前馈。

`integrate` 要求每个真实 body 的 COM wrench、完整 `6+n` 接触载荷和零 fixed-root 接触行；用同一入步完整 J 核 `contact=Jᵀ*wrenches`（逐分量 rtol/atol 各2e−10），不一致或非有限即拒绝。外部 COM 力矩与实际接触 COM 力矩分别计一次，独立电机仅从 k 维 effort 进入。唯一物理推进是已验 `asset.step(...)`：约束加速度、已约束末速度、原浮根漂移一次；只读适配器的 prepare/integrate 从不调用。

观察的 `contact_effort=Sᵀ*contact_effort_full` 为6+k；`contact_effort_full` 仅本次碰撞，`applied_effort_full`/`acceleration_full` 为6+n，drive为k。`mimic_loads` 保 T2 语义，其同名 contact 字段包含除重力外的全部非电机载荷，包括显式外部 COM 力矩；不能和 scene 的碰撞专用字段混记。原行反力、各行/总入步约束功与独立审核的根六维约束余额由 T2 回执原样保留，gravity/惯性仍在 bias。

适配器没有内部 accepted state、时钟或事件计数。所有候选、拥有的观察、末态 geometry 和事件 staging 仍由现有 `MultiRobotScene` 在成功后一次发布；本模块抛错不触碰上一接受态，合法重试不消费 episode。

## 冻结验收和交接

[案例册 T3](../development/p65-floating-mimic-casebook.md) 与 `60a9833` 三文件保持原字节：公开 wrapper 正控；四档真实自由根 PD；普通非 mimic +X 滑轨机器人和 right_pad 的真实2 N 接触、反序与双侧材料；第二参与者限位拒步/晚期事件异常/返回所有权；原 F03 T=1/4 s、h=1/2048 与1/4096、反序登记和轴循环平移四窗。手工计划13测试实例，实际 collection 与结果由父记录。

公开例程 `examples/floating_mimic_contact.py` 保存完整逐步 public reading、full/independent 状态、材料/作用点、COM 力矩、约束回执与时间，初始完整 geometry 及逐帧真实 poses；另保实际冲量、线/角动量、动能和 full 平衡诊断。失败保存全部已接受前缀和异常，退出2，不调整原输入使窗口通过。四窗计划2560子步是待验工作量，不是已跑结果。单步共同点力矩与功率核对不等于有限步角动量/能量精确守恒，实际残差单列。

父优先新增短节点，再跑冻结完整窗口及真实受影响共享组；作者未发起任何执行。新增两文件待父集成/验证与格式处理。当前没有 T3 HOST/CUDA 性能数字或 GREEN 结论。

此候选不增加 floating pads、自碰、连续碰撞响应、变形参与者、设备求解或新摩擦律。原100 g mimic在1.048 s滑移3.005689 mm的持稳 RAW1不变；本片的物理自由根约束与无摩擦被动受载窗口不能替代该失败。[原范围](p65-force-events-filtered-batch2.md)
