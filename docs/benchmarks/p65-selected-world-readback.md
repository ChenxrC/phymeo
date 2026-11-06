# 按世界选择观测：C0 受控成本入口

状态：**UNEXECUTED measurement candidate**。作者仅按冻结计划编写，未执行 import、collection、pytest、格式检查、设备探测或 profile；尚无成本结果，也未达到新增 readback plan/kernel 的实现门。本报告不沿用旧完整观测、凸包缓存或 Newton 的倍率。

基线为 `2027f721a0d1937de6f0e89ec869b6c03951f625`。tests-only 提交为 `beb323b6ff08ba9372ef443e91b123967fdb8c46`；父已在 `artifacts/p65-top-three-batch5-20261106-01/01-c-profiler-red` 保留首个真实 RED：1 failed / 0.19 s / exit 1，明确断言尚无 C0 测量工具。后续 GREEN 和正式计量待父唯一串行槽。

## 冻结输入与任务

新入口 [profile_selected_world_readback.py](../../scripts/profile_selected_world_readback.py) 每进程只测一种公开输出。`snapshot` 保存完整 `WorldCheckpoint`，包括世界/拓扑、int tick、位置/速度、实际上一拍力、延迟历史、held control；同时保存同次原生 `_read` 返回的全部12列FP32状态和int32 tick，不再额外读设备。`kinematics` 保存原世界键、维度、int32 tick和六列FP32观测。两个接口不互算速度比，不以六量摘要替代完整检查点。

| W | K（独立进程） | 模型 |
|---|---|---|
| 1 | all，等于1；不重复测同档 | 一个自由球 |
| 32 | 1、8、all | 自由球/一步延迟滑轨交替，各16个 |
| 128 | 1、8、all | 自由球/一步延迟滑轨交替，各64个 |

上表对 `snapshot`、`kinematics` 各执行一次，共14个正式 CUDA 进程。interpret 只用于新工具资格，不能开启性能实现门。世界键固定为 `101+17*slot`；子集按注册逆序选取，all保原注册顺序。每次仍真实推进全部 W 个世界，只观察 K 个。

- 自由球：质量2 kg，半径1/8 m，初位 `(slot/8,-slot/16,2)`，初速 `(1/4,-1/8,0)`，重力 `(0,0,-8)`，无接触/外力。
- 滑轨：真实单关节 `DrivenRobotProgram`，质量2 kg，轴+x，零重力，`kp=kd=0`，限力4 N，目标/目标速度为0，held feedforward为2 N，一步真实延迟；初始q、qd、历史均0。
- `h=1/128 s`。单次冷物理步之后，5预热窗＋20正式窗，每窗32步；连续推进到tick801，窗间不复位、不换输入。W1没有滑轨，这一边界明确保留。
- 独立解析审核在计时外进行。第n步球的z为 `2-n(n+1)/4096`、vz为`-n/16`；滑轨q为`n(n-1)/32768`、qd为`(n-1)/128`，第一步实际驱动力0，随后2 N。均为该窗口可精确表示的dyadic输入；非零误差即失败，不用产品输出反算参考。

## 计时、计数和门

`whole_ns` 为逐步 `step + 当前接口观察 + 保留返回引用` 的窗口墙钟；同步沿旧入口原样发生，不补新同步。计时外另记录 import、案例构建、批工作区构建/初始上传、控制设置和逐窗离线审计。冷窗包含首次实际提交/JIT；新cache目录防止把旧磁盘缓存叫冷启动，但不假定操作系统或驱动为全新状态。

同一窗口内独立记 `step`、`observation`。每阶段细分 `_launch`提交API、显式`_sync`、`_read`下载API、`_upload`上传API、HOST构造。snapshot的构造边界为整个`WorldBinding.checkpoint`；kinematics为结果dataclass构造。各自剩余时间叫 `host_other_ns`，含键验证、HOST选择/数组拷贝、字典和包装器管理，**不是纯构造函数时间**。另记循环保存/派发残差；排他项必须非负且精确加回whole。

所有阶段是HOST调用墙钟。异步提交包含编译/绑定/提交开销，下载可能含隐式等待；均不叫GPU kernel事件时间。包装器会扰动成本，因此本结果是归因诊断，不作为新路径的正式加速比。

每次真实读回按返回数组的`nbytes`计量，并对照batch与TYX原生计数。interpret的`read_bytes`是HOST复制字节，`d2h_bytes=0`。CUDA状态读取期望为：kinematics每次52W B；snapshot子集每次52K B，全量52W B。物理步的接受标志另有4W B，不藏入观测列。snapshot子集的K次read_world提交/同步单列；全量及kinematics无此观测核。全部原生计数字段也归档，冷启动额外运行时活动单列，暖态不许有不明差值。工作区分配计数仅表示TyxWorldBatch所持设备缓冲，不冒称整个驱动分配统计。

正式统计报告20个窗口的p50/p95、总world-substeps/s、selected-world-observations/s、各阶段占whole和observation的比例。前者分子是`W×640`，后者是`K×640`；不把少读世界说成少推进世界。

实现门分母固定为正式32步的推进＋同种观察全任务 `whole_ns`，包括公开HOST构造、排除JSON和离线审计文件IO；分别看观测下载、观测提交＋同步各自占whole的比例。另保观察调用子段分母的比例供归因，但不能用极短观察子段的高占比放行。只有真实CUDA、K<W、完整5/20×32采样且其中一项≥15%，才给父提示`measured_cost_gate=true`。两个不足15%的子项不相加凑门；短资格、interpret和全量档一律不获资格。脚本从不自动改变产品或实现核，父仍审核实际归因。若无充分成本依据，本节点结束并保留数据。

## 来源和输出绑定

正式CLI冻结名为 `--select`、`--backend`；旧提案名`--selection`、`--mode`是兼容alias。必传完整`--expected-head`。运行前后核HEAD、被执行脚本、新专项、pyproject及全部tracked `src/phymeo`文件的路径和SHA256；拒绝dirty/新增未跟踪产品源、错误HEAD、末态源码漂移。CLI还核实际加载的Phymeo路径确为脚本所属树的`src/phymeo`，记录解释器、Phymeo/TYX版本及路径；CUDA名称/ordinal来自实际状态缓冲与同设备driver查询。

`--out`、`--cache-root`必须为全新绝对路径；同名`.frames.jsonl`也必须不存在。每窗完成后在计时外追加一条含完整真实观测的raw行，JSON收据保存各窗摘要、输出SHA和raw文件SHA。失败保留之前完成的raw窗并写`status=failed`，不覆盖旧结果；源码末检失败也不能发成功收据。没有后台进程和子进程矩阵调度，14项均由父guard串行调用。

PowerShell示例（**尚未执行**；需在最终clean提交后填本次HEAD）：

```powershell
$source = 'C:/Users/38347/.codex/worktrees/p65-world-scale/phymeo'
$evidence = 'C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-top-three-batch5-20261106-01'
$env:PYTHONPATH = "$source/src"
$head = (git -C $source rev-parse HEAD).Trim()
& 'C:/Users/38347/Documents/ChatGPT/phymeo/.venv/Scripts/python.exe' `
  "$source/scripts/profile_selected_world_readback.py" `
  --worlds 32 --select 8 --kind snapshot --backend cuda `
  --expected-head $head --cache-root "$evidence/c0-cache/w32-k8-snapshot" `
  --out "$evidence/c0/w32-k8-snapshot.json"
```

替换W、K和kind形成上表14项，输出/cache路径逐项不同。正式项保默认5预热、20正式、32步；如需首个小CLI资格，可显式 `--warm-windows 0 --formal-windows 1 --steps 2`，该收据被标为不具备性能门资格。

## 新专项交接

[test_selected_world_readback.py](../../tests/scale/test_selected_world_readback.py) 只验新增测量工具的真实混合状态、完整历史输出、实际字节和排他归因、CLI/源码/文件所有权、异常后包装器恢复；不重跑旧物理尺度矩阵。它使用仓库`tyx_mode` fixture，父可以只将设备节点在CUDA再次执行。

```text
python -m pytest tests/scale/test_selected_world_readback.py --tyx-mode interpret --require-tyx -q
python -m pytest tests/scale/test_selected_world_readback.py -m tyx --tyx-mode cuda --require-tyx -q
```

实际结果待父填写。作者当前交接为UNEXECUTED，不宣称GREEN、真实CUDA、成本门通过或性能收益。

