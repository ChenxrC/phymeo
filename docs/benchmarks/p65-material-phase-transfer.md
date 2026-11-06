# 材料相位分页：C0 测量协议

状态：**TESTS-FIRST / UNEXECUTED**。父基线 b63bb895796899355d769911e06486d4ccb90ce6。作者只新增协议、测试和明确 NotImplementedError 的可导入脚本 stub；全部项目执行由父串行负责。尚未实现传输优化、MaterialForceFrame 或默认切换。

本批只研究现有 TyxMultifaceContactBatch 的同相位分页。原 TyxNormalLayer 每页128行，传入13列、读回5列；每页 CUDA 实际上传6656 B、一个kernel、一次显式sync、下载2560 B，未用尾行也传。entry/midpoint/endpoint是三个依赖真实物理状态的时相，不能合并求力时刻。WorldContactEngine 已有一次同步的完整链，不在本批范围。

## 预先冻结的输入与预算

- 物理矩阵：W1/W32 × 2/8面，另W3异构（2面受载、8面受载、8面布沿z移−0.5 m分离）。复用已验 multiface_load_cases.make_world，不改原构造器、质量、材料、控制或几何。非连续world键先冻结，重复本地ID由world区别。
- 所有物理窗h=1/256、4步，从同一初态恢复，复位在计时外。每档1个cold、2个warm、3个formal；另1个独立未插桩同输入4步窗作完整输出对照，共最多28物理步/档。短资格可缩小，但不能取得成本门资格。
- 每档最高180 s，原始JSON编码总量最高32 MiB；在启动runtime前验证预算。新完整payload直接写全新 `.frames.jsonl.gz` 无损sidecar，每个窗口完整保存，不挑帧、不按输出挑案例。达到任一预算立即保留已完成原始记录并标失败/未完成，不裁剪成complete，不自动扩大预算，不删除任何旧证据。
- 独立算术行协议为0/127/128/129行：固定有限FP32二进制分数，含分离、零权重、法向、摩擦/阻尼及正则化滑速。期望五列由独立标量算式给定。这些行不是物理轨迹，不用于15%门。

## 冻结的成本门与完整性

计时whole是四步返回完整SceneStepResult的墙钟，包含几何、两侧力散回、积分、验证和HOST报告。序列化、文件IO、离线审核与复位在外。caller设备API与HOST装配分项；跨线程仅沿共同时间线分配排他区间，挂起世界时间另存不相加。剩余工作标residual，均不冒称GPU核事件时间。

主门分子只计同一相位第2页起的upload/kernel-submit/explicit-sync/download排他墙钟，第一页面不凑门；全部页API占比另报。P−1页API时间仍含真实工作与等待，称**候选成本上界**，不是已证实可消除时间。只有完整正式CUDA物理任务、真实S>128、原始计数一致和完整输出一致，且该上界/正式whole≥15%，才允许后续实现对照。冷编译、单页、interpret、输出不一致及短资格均不放行。门通过不等于收益或默认切换。

保存每相位S/P、world/face/node区间、全部13列输入和5列实际输出，及真实native/layer计数。既有phase区间不是持久材料身份。完整物理state/report按原schema保存；同输入插桩与未插桩逐字段比对，仅允许预先声明的report.substeps[*].timings差异，不规范化力、位置、接受状态或失败原因。source HEAD与依赖源SHA前后绑定；实际Phymeo加载路径/设备签名随收据保存。

## 父执行入口与当前缺口

首RED节点：`tests/scale/test_material_phase_transfer_profile.py::test_cost_gate_uses_full_task_and_excludes_cold_compile`。新脚本可正常导入；期望在summarize_windows明确抛出NotImplementedError，证明成本协议缺口，而非路径或环境失败。等待父有效RED后再实现profiler。

正式CLI冻结为 `--mode cuda|interpret --worlds 1|32 --faces 2|8 --expected-head <sha> --out <全新绝对json>`，异构使用 `--case heterogeneous --worlds 3 --faces 8`；可传全新绝对 `--cache-root`。输出主JSON保存摘要、输入绑定和gzip sidecar SHA。普通错误/设备不支持不得skip或fallback。

尚无本批实测值。旧W4×8的0.60/162.36 ms只是既有归因，不能外推到本批真实S>128任务，也不会重跑旧验收套来充数。
