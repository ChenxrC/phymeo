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

## 父实际共同场景结果（2026-11-07）

当前状态：**T3_HOST_VERIFIED / 共享入口审阅完成；新组合wheel待验**。父导入9326aa1/e484b48后，独占接通原FloatingRobotRegistration、MultiRobotScene派发、builder说明与公开exports。无数值缺陷修补或夹具/容差变更。

- 125/126首静态仅2处import排序、5文件排版；127/128仅style修复，原记录保留。
- 129首次完整T3：**13 passed / 17.83s / RAW0**。首公开入口、PD4、普通机器人互碰2、三种事务/拥有数据、四窗全部门及协变/步长细化均通过。
- 130受影响共享：**13 passed / 0.66s / RAW0**，覆盖普通root-only、普通/固定mimic混合、双浮根、未实现模式拒绝及原双机器人共同sphere。
- 131/132最终Ruff和7文件format-check均RAW0。原T2新23+兼容37不重跑、不并作本批新数。
- 原RD完成父4共享hook与participant/外层事务的只读接口审阅，未发现阻断；父完整阅读新源码与测试，各自记录，不以审阅代替执行。

四份实际完整gzip在129的明确basetemp中，分别512/1024/512/512帧，共2560子步；失败字段全部null，全部到0.25s、有begin/end且末步卸载。h=1/2048首/末正载荷时刻0.00439453125/0.06591796875s，峰值10.11380374689702N；h/2首0.004150390625s、末相同、峰值10.113857954737142N。反序相同；轴循环和平移的峰值10.113803746897332N。没有重置中间状态或缩短窗口。

最大总线动量残差：h及反序约1.29720e-6 N·s，h/2约6.48591e-7，变换约1.29720e-6，均小于原5e-4门。实际有限步角动量残差分别约1.11559e-8/5.55939e-9/1.11559e-8/1.17810e-6 N·m·s；平移后的角动量残差还含位置叉线动量误差，不声称该量跨原点不变或精确为零。逐步完整力余额、双侧作用点矩/功率及原mimic行约束功按冻结门通过。

每份保存完整reading/initial/end/entry几何、输入质量矩阵与bias诊断、全材料地址/接触力矩/约束行与事件时间。四文件压缩bytes分别548599/1085161/548606/677238；父原rawexit/源before-after及四完整流读取收据为 `artifacts/p65-generic-collision-20261107-01/f2-scene-validation-receipt.json`。未重新调用物理/生成器取统计。限定HOST浮根仿射机制与共同场景，原100g静摩擦失败、软布两初态放台及GPU浮根仍开放。


## 新组合 wheel 正式限域接受（2026-11-07）

产品fdfa586，133构建RAW0；wheel SHA256 `3BB31FEFC4B66A0CCB45B6500EE2C22566C4D1F5E2566CBCD8C31C4F03D37D7D`，1180542 bytes。18个变更product文件已与解包字节核同。运行cwd及PYTHONPATH均在源码外Temp新目录；实际270个phymeo模块来自该wheel。

134原RAW1保留：F2全部512步/接触/作用点矩/完整mimic余额/事件断言已通过，随后C3最终命中列表错误。脚本把原小盒子组合半宽3/32的[28+delay]误用于这里的两球组合半宽1/8；实际逐步FP64几何、q/v和候选断言均先通过。135只修正独立列表为world7=[27,28]、world24=[28,29]，补完整rotor保存，RAW0；读取并复用原F2完整gzip，未重跑F2物理，未改wheel、源代码、几何或工程门。此失败是新包装验证脚本问题，不改写为首次全绿。

源码外实际F2到0.25s，首/末受力0.00439453125/0.06591796875s、峰值10.11380374689702N，末态卸载，最大线动量残差1.2971987494374595e-6 N·s。实际CUDA双rotor各40步、正负恒矩及delay0/1，q/v、当前有限球AABB、完整候选/tick/epoch与原计数通过；动态包络上传0，分配不增长，两个buffer均使用。完整状态/几何/输入/载荷/事件以及80个world-step设备记录保存在 `artifacts/p65-generic-collision-20261107-01/f2-c3-wheel-results`，父收据 `f2-c3-wheel-acceptance.json` SHA256 `876536F7B6971FBA39CA38610822E7C9876BBB10498D4B0761AB6A981EDC9E11`。

本片正式接受：HOST浮根仿射mimic与原共同场景、CUDA单R接受态到当前姿态宽相。不是浮根GPU、多关节驻留、设备窄相/CCD、持续摩擦或整个Newton功能对齐；原100g mimic和软布目标失败继续保留。当前执行槽释放，后继按原多案例计划继续。
