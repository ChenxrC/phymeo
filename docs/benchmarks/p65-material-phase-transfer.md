# 材料相位分页：C0 测量协议与交接

最终状态：**C0 五档实测完成，成本门未通过，默认不变**。最终数据与初始失败链见末节。以下为作者初次交接时的协议记录：**IMPLEMENTED / AUTHOR-UNEXECUTED**。父计划基线 b63bb895796899355d769911e06486d4ccb90ce6；tests-first 9ba776ab88d08aeb10e5b64c6fde3539900d942b。父 `03-c-protocol-red` 已记录有效 RED：1 failed / 0.12 s / RAW1，正常导入后进入 `summarize_windows` 的 NotImplementedError，前后源码快照一致。此后作者只实现四个 NEW 文件内的测量工具和专项，未运行 import、测试、静态检查或 profile。后续验证由父串行完成；初次交接时没有性能结果；本批始终没有优化实现、MaterialForceFrame 或默认切换。

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

## 父串行验证与最终实测（2026-11-06）

正式测量源码为 **3e6dff494d898ad19a3a3695d1de0c368b511805**。五档均为全新进程/cache 的实际 `cuda:0`、NVIDIA GeForce RTX 4060 Laptop GPU，driver version 12070；每档1 cold、2 warm、3 formal、1独立未插桩参考窗，7窗各4步。各 `profile.json` 记录 `status=complete`、`source_verified_after=true`、`complete_output_match=true`，且7窗完整 canonical output SHA 各自相同。这里独立复核了已保存摘要、SHA 字段、设备和原退出收据；没有再次运行求解器或另作数值审计。

证据根为 `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01`。下表站点 S 和页数 P 为一个4步窗内按时间排列的12个物理相位：首相位 / 其余11相位；总页数及额外页数是每窗合计。最后两列分母均为3个正式窗的完整任务墙钟之和，不是观察调用或单核时长。

| 证据/真实配置 | S：首 / 其余 | P：首 / 其余 | 总页 / 额外页 | 额外页 API / whole | 全部页 API / whole |
|---|---:|---:|---:|---:|---:|
| [41 W1×2面](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/41-c-cuda-w1-f2/profile.json) | 4 / 2 | 1 / 1 | 12 / 0 | 0% | 12.31175% |
| [42 W1×8面](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/42-c-cuda-w1-f8/profile.json) | 16 / 8 | 1 / 1 | 12 / 0 | 0% | 6.97364% |
| [43 W32×2面](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/43-c-cuda-w32-f2/profile.json) | 128 / 64 | 1 / 1 | 12 / 0 | 0% | 0.61080% |
| [44 W32×8面](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/44-c-cuda-w32-f8/profile.json) | 512 / 256 | 4 / 2 | 26 / 14 | **0.15468%** | **0.91856%** |
| [45 W3异构](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/45-c-cuda-w3-f8-mixed/profile.json) | 20 / 10 | 1 / 1 | 12 / 0 | 0% | 4.87411% |

以下所有时间单位为 **ms/完整4步窗**；暖态范围是2个 warm 窗，正式范围是3个 formal 窗。p95 按冻结的 nearest-rank 规则，在仅3样本时就是最大值，不构成长尾分布结论。cold 包含该运行的首次准备/JIT 成本；单独冷参考窗只用于输出一致性，不与暖正式窗相除做加速结论。

| 配置 | cold | warm 范围 | formal 范围 | formal p50 | formal p95 |
|---|---:|---:|---:|---:|---:|
| W1×2面 | 918.4844 | 29.9098–33.5877 | 28.7837–33.9063 | 29.5008 | 33.9063 |
| W1×8面 | 675.4907 | 47.9836–49.5146 | 46.8433–56.9762 | 47.5836 | 56.9762 |
| W32×2面 | 1205.1423 | 765.1119–793.2932 | 665.8019–855.3818 | 749.8775 | 855.3818 |
| W32×8面 | 2366.5128 | 1522.2134–1629.7751 | 1462.6435–1873.9045 | 1532.5130 | 1873.9045 |
| W3异构 | 622.0374 | 95.7152–102.0266 | 89.7497–96.1094 | 94.8937 | 96.1094 |

W32×8面是唯一 `eligible_formal_cuda_dense=true` 的档：3正式窗 whole=4869.0610 ms，额外页 API=7.5316 ms，全部页 API=44.7252 ms。候选上界只有0.15468%，远低于冻结15%门；五档 `measured_cost_gate=false`。W1的小档控制虽全部页占比较高，却没有可合并的额外页；W32×2面首相位恰128行，同样没有额外页。由此**不授权分页传输优化、不新增 MaterialForceFrame、不切默认**。这些任务的世界数、面数和实际 S 不同，表格不是速度排名，也不是 GPU 全链驻留或跨引擎性能比较。

每窗 W32×8面实际26次 kernel/显式sync、H2D 173,056 B、D2H 66,560 B；其余四档每窗各12次、79,872 B、30,720 B。layer 与原生计数均通过原审计，inactive 尾行仍按整页实际字节计。W3异构分离世界仍真实推进但无受载材料行，没有删掉该世界来缩小分母。

### 初始失败、修复与资源记录

- `03-c-protocol-red`：正常导入后明确 stub RED，1 failed / 0.12 s，RAW1。
- `27-c-protocol-first`：12 passed 后人工时钟 oracle 失败，1 failed / 0.43 s。原 oracle 将85–90错误算作 caller await；真实 await 仅0–20。父 `28` 只修期望为 world30、idle5、caller25、law20、kernel20及注释，不改时钟输入或分配算法。`29` 完整27 passed / 0.52 s，RAW0。
- `30-c-interpret-qualification`：正常5节点先过，晚容量失败的记录又用整数键 dict 调用了仅接收字符串映射键的 fingerprint，掩盖原异常（1 failed、5 passed / 1.36 s）。父仅将两处 state mapping 转为 `tuple(sorted(mapping.items()))`，保留整数 world ID 和 typed state，不改旧 fingerprint 或物理。`31` 两故障节点2 passed / 1.09 s；`32` 全部7 interpret passed / 1.04 s；`33` 全部7实际CUDA passed / 2.48 s，均RAW0。
- `34` 初次4项导入顺序问题与 `35` 初次格式失败均保留；`36/37` 修正后 `38/39` Ruff及3文件format通过。`40` 证明 profiler raw AST 一致，cases/test仅连续import顺序变化，无物理/协议数值改动。
- `41–45` 均RAW0，每档 gzip 固定8条完整记录（1输入清单+7窗）。JSON编码/实际gzip字节依次为 405,547/76,404；1,089,271/213,569；10,103,839/679,289；32,264,838/4,535,185；1,557,069/306,908。最大档仍在32 MiB原始编码预算内，没有触发资源停机，也未扩大预算或删除旧证据。

父 `46-c-saved-evidence` 已另作仅文件流审核，RAW0：五份 gzip 的实际 SHA、字节与固定8 records均匹配，5档共40 records；每个物理窗完整4步/12相位，完整输出 hash 一致，无求解器重跑。摘要见 [material-profile-summary.json](C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch6-20261106-01/material-profile-summary.json)。B作者对 C 脚本的独立文本审阅无阻断。

复现绑定仍是测量时的3e6dff4及各 profile 内完整逐文件SHA；本次仅追加报告，不改上述源快照。作为精确定位，profile脚本SHA256为 `50de8a4c1ae0502414b57f16ebc30639d50ee27fbcbaaad4f487a8f34b36baea`。父 `47 build`、`48 outside-wheel` 均RAW0，但 wheel验证的是本批A/B新增产品功能；C仅新增仓库测量工具，不能把该wheel验收记成C再次设备测试。本轮C按冻结成本门结束该分页优化分支，保留原路径。
