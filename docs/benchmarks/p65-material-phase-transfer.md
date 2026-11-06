# 材料相位分页：C0 测量协议与交接

状态：**IMPLEMENTED / AUTHOR-UNEXECUTED**。父计划基线 b63bb895796899355d769911e06486d4ccb90ce6；tests-first 9ba776ab88d08aeb10e5b64c6fde3539900d942b。父 `03-c-protocol-red` 已记录有效 RED：1 failed / 0.12 s / RAW1，正常导入后进入 `summarize_windows` 的 NotImplementedError，前后源码快照一致。此后作者只实现四个 NEW 文件内的测量工具和专项，未运行 import、测试、静态检查或 profile。后续验证由父串行完成；本批没有性能结果、优化实现、MaterialForceFrame 或默认切换。

原 TyxNormalLayer 每页128行、13列输入及5列输出；源码路径每页 CUDA 上传6656 B、提交一个核、显式同步一次、下载2560 B，未用尾行也传。这是源码事实，实测将分别核对 layer、原生调用和 TYX runtime 计数。只有法向/摩擦行律在设备上；几何、材料映射、积分、力矩、双方力散回和完整报告仍在 HOST。entry/midpoint/endpoint 依赖不同物理状态，不能合并求力时刻。已有 WorldContactEngine 不在本次范围。

## 固定输入和预算

- 物理矩阵为 W1/W32 × 2/8 面，另 W3 异构（2面受载、8面受载、8面布沿 z 移 −0.5 m 分离）。复用已验 `multiface_load_cases.make_world`，质量、材料、控制、原几何不改；非连续 world 键与重复局部 ID 在输入清单中冻结。
- h=1/256，每窗4步且从同一初态恢复；复位在计时外。每档1 cold、2 warm、3 formal，另重建独立程序和工作区执行1个未插桩同输入参考窗，总计最多28物理步。短资格可缩小，但不能取得成本门资格。
- 每档最高180 s，在安全边界检查，不打断 native 调用。完整原始 JSON 编码总量最多32 MiB，摘要最多2 MiB；完整记录直接无损写入全新 `.frames.jsonl.gz`，记录 SHA、编码字节与实际压缩字节。达到预算保留此前完整记录并标失败/未完成，不挑帧、不自动扩大预算、不删除旧证据。编译缓存独立存放，不把其占用冒作上述输出上限以内。
- 算术行协议为0/127/128/129行，固定二进制分数输入，含分离、零权重、法向、摩擦/阻尼及正则化滑速；五列期望是独立给定的标量结果。它不是物理轨迹，不能打开成本门。

## 计时、观测和成本门

`whole_ns` 是返回完整四步结果的墙钟，包含原几何、双侧力散回、积分、验证、HOST 构造及插桩残余；序列化、文件 IO、离线审核、复位与 hook 安装在外。嵌套调用沿同一墙钟投影为排他区间；挂起世界栈的等待另报，不叠加成墙钟。记录 upload/kernel-submit/explicit-sync/download 与 HOST 装配、几何、状态复制、发布及未分类部分，均为 API/HOST 墙钟，不称 GPU event 核时长。每相位另保存实际 S/P、world/face/node 区间、完整13列输入/5列输出、有效 FP32 wire 行和计数；尾页仍按实际整页字节记。

主门分子只计同一相位**第2页起**的上述四类 API 排他墙钟，第一页面不凑门；全部页 API 占比单列。P−1 页仍含实际计算与等待，称**候选成本上界**，不是已证实可消除时间。仅完整正式 CUDA、实际 S>128、各计数闭合、与未插桩完整输出一致，且上界/正式 whole ≥15%，才允许后续实现对照。冷编译、单页、interpret、输出不一致或短资格均不放行；门通过不等于加速或默认切换。

完整 accepted state、history、substep state、report 和材料记录保留在 gzip 中。各窗完整 canonical payload 的 SHA 与独立参考窗核对，唯有预先声明的 `report.substeps[*].timings` 值归零且保留字段名；不删除力、位置、状态或拒绝原因。全窗口实际计数保留，未插桩路径不伪造实测同步数，另列 `source_expected_synchronizations`。晚容量/设备调用失败保留已做工作、旧发布状态/阶段 hash 和自动重试为 false；修复后的显式新运行使用原输入。离线审核失败也保存完整失败窗，若该窗超原容量则保留原错误及归档不可用原因，不截断物理记录。

## 父执行入口

先协议组（不导入项目运行物理），再原律边界及短物理组；作者未执行以下命令：

```text
python -m pytest tests/scale/test_material_phase_transfer_profile.py -m "not tyx"
python -m pytest tests/scale/test_material_phase_transfer_profile.py -m tyx --tyx-mode interpret --require-tyx
python -m pytest tests/scale/test_material_phase_transfer_profile.py -m tyx --tyx-mode cuda --require-tyx
```

首 RED 节点为 `test_cost_gate_uses_full_task_and_excludes_cold_compile`。真实后端节点为 `test_original_law_protocol_keeps_all_fields_and_independent_forces`（4行数）、`test_short_real_profile_preserves_full_outputs_and_actual_phase_rows`、`test_failed_profile_keeps_partial_work_and_retry_uses_original_state`（晚容量/设备调用）。父为各命令提供全新 guard、basetemp/cache 和明确 PYTHONPATH，不重跑旧物理套。

正式 CLI（父填写同一冻结 full SHA 及全新绝对路径）：

```text
python scripts/profile_material_phase_transfer.py --mode cuda --worlds 32 --faces 8 --expected-head <40位冻结SHA> --out <全新绝对路径/profile.json> --cache-root <全新绝对cache路径>
```

五档为 `(1,2,uniform)`、`(1,8,uniform)`、`(32,2,uniform)`、`(32,8,uniform)`、`(3,8,heterogeneous)`；末档追加 `--case heterogeneous`。默认次数为上述1+2+3+1窗。`--mode interpret` 和 `--steps 1 --warm-windows 0 --formal-windows 1` 只作短资格。实际 dense 档若未达 S>128 或资源上限，只保留结果，不调整物理输入制造门。

源码绑定清单为全部 `src/phymeo` 跟踪文件、本次 script/cases/test/report、原 `tests/scale/multiface_load_cases.py` 以及 `pyproject.toml`；核 full HEAD、清洁状态、逐文件 SHA 前后一致。实际所有已加载 Phymeo 模块必须位于绑定树的 `src/phymeo` 下，CUDA ordinal、原始设备名和 driver version 随收据保存。主输出同时记录完整命令、cache 路径、sidecar SHA 和停机原因。

旧 W4×8 的既有归因不外推到本次真正超过128行的任务；没有本批实测前，不给预期速度比或新增传输路径。
