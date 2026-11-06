# 按世界选择观测：成本归因与显式读回实测

**实测结论：snapshot 的 W32/W128、K8 计划有重复收益；W1 snapshot及全部kinematics档更慢，默认入口完全保持。** C0在 `fbdee826f335c135f237f1c861f12d99cdddbfa2` 完成14个CUDA配置；C1在 `c5bdff90a76da5ba49c7a9468a7d42d17a7ba895` 完成18个无插桩配对进程，全部RAW0、每lane801tick、完整公共输出相等、资源关闭归零。功能和源码外wheel组合验收也已通过。作者仅文本/Git，项目执行全部由父串行完成；这里不作整体快于Newton的结论。

基线为 `2027f721a0d1937de6f0e89ec869b6c03951f625`。tests-only 提交为 `beb323b6ff08ba9372ef443e91b123967fdb8c46`；父已在 `artifacts/p65-top-three-batch5-20261106-01/01-c-profiler-red` 保留首个真实 RED：1 failed / 0.19 s / exit 1，明确断言尚无 C0 测量工具。父后续证据为23-c-first-interpret（20 passed / 1.75 s）、24-c-first-cuda（3 passed / 2.02 s），以及25-c-base系列14项完整CUDA计量。

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

C0复现命令形状（原14项已测，无需重跑；若另行授权复现需填所用clean HEAD）：

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

上述C0新专项的实测结果见开头，已绿组不重复。本节之后是由成本门授权的独立C1增量，作者不自行执行。



## C0 实际成本门与 C1 增量

父归档目录为 `artifacts/p65-top-three-batch5-20261106-01/25-c-base-<kind>-w<W>-k<K>/profile.json`，完整raw帧在旁。14项均`status=complete`和801tick；8个CUDA子集都通过全任务15%成本门。例如：

| 实际配置 | submit+sync / whole | download / whole |
|---|---:|---:|
| snapshot W32 K8 | 0.601557 | 0.196421 |
| snapshot W128 K8 | 0.591968 | 0.197155 |
| kinematics W32 K1 | — | 0.201247 |
| kinematics W128 K8 | — | 0.192942 |

以上是父保存的C0插桩归因比例，不是新计划的提速比。以真实门为依据，C1优先减少snapshot的K次提交/同步；kinematics仅确认减少读回字节；下方C1结果表明这不足以抵消新增提交/同步等成本。

公开首红由父保存在`32-c1-public-red`：2 kg自由球旧step和解析状态通过后，`prepare_readback`缺失，1 failed / 1.16 s / RAW1。父共享薄hook提交`933079625fa84744e51ec53edfb1246310bafddc`，作者新增两个模块：

- [计划与资源所有权](../../src/phymeo/backends/_world_readback_plan.py)
- [实际TYX选择核](../../src/phymeo/backends/_tyx_world_readback_kernels.py)

```python
plan = batch.prepare_readback([628, 101, 611], kind="snapshot")
reading = batch.readback(plan)  # 真实owner校验；外实例拒绝且没有设备动作。
reading = plan.read()          # 同一batch便利入口。
plan.close()
```

计划拥有冻结keys/kind、三个设备缓冲（索引、结果行、ticks）和两方向的kernel句柄；每次从当前active接受缓冲一次gather、一次显式sync、两次实际下载。索引只在创建上传，HOST checkpoint逐行构造仍保留，不称为GPU公共记录发布。kinematics读回28K B，snapshot读回52K B；索引另4K B，常驻设备缓冲分别32K/56K B。空选择不分配、不提交、不同步、不触空batch未存在的runtime。

所有操作保持原batch锁。snapshot读取当前held controls与入步历史、实际力；局部restore/reset或数值拒步后照当前接受状态读取，不保存旧active指针。返回数组/tuple与计划scratch独立；创建中途失败关闭已经分配的buffer，保累计实际分配/上传计数，但回滚live bytes且不留下半成品计划。读回失败遵循原`_device`恢复，不改物理状态和tick，旧读数仍有效。

父的`device_allocations`保持累计次数；`device_owned_bytes`和新增`readback_plan_owned_bytes`均为当前CUDA缓冲拥有量，单独plan.close即扣除，batch.close先关闭所有计划再释放旧缓冲。新增readback索引上传/成功读取/失败读取计数单列，`readback_plan_live_count`由集合给出。plan有独立stats，手动与batch关闭均幂等。编译代码缓存由TYX Runtime管理，不计成计划的三个数据缓冲。

## C1 无插桩同任务配对（已实测）

新CLI `--compare-plan --first original|plan`。每个进程建立两个独立相同初态，original使用旧公开方法，plan使用显式计划；两路各cold1＋5预热＋20正式32步，均801tick。首顺序按三独立进程original/plan/original，窗内继续交替先后，以配对的同tick完整public输出逐项比对。配置为W32K8、W128K8、W1all各snapshot/kinematics，共18个配对进程。默认入口不切换。

正式配对循环**不安装PhaseProbe**，避免旧K次调用的包装器成本夸大收益。只计原`step + 同种公开观察 + 保留输出引用`；计数快照、构建/索引上传、完整序列化和离线独立审核在外。C0诊断只作定位，不能把它的插桩时间直接与C1无插桩数据算倍率。

原生/batch计数和新增`PLAN_COUNTERS`分开保存。每窗核完整公共帧相等及同一public SHA；不比较不同wire形状的C0 `output_sha256`。C1不再拦截设备读回，raw文件保存全部实际公共输出，原C0原生wire文件保持原样，不额外下载或复制整包资产。raw JSON使用紧凑分隔符，减少磁盘占用；所有新cache正常保留，不清理旧证据。报告各路绝对p50/p95/吞吐、每窗比值、p50比值和总吞吐比值，分母明列，保留W1可能退化。

新命令在上方固定源码/解释器/证据根约定上增加：

```text
--worlds 32 --select 8 --kind snapshot --backend cuda --compare-plan --first original
```

三次任务使用不同全新out/cache，首序original、plan、original。六配置各三次共18项，54/55/56系列全部RAW0；57汇总检查通过。执行设备为实际 `cuda:0`、NVIDIA GeForce RTX 4060 Laptop GPU，driver记录12070。源码在整批前后绑定同一 `c5bdff90a76da5ba49c7a9468a7d42d17a7ba895`。

新增功能组的复现入口（初组已验，关闭异常随后单独重验；无需再跑已绿组）：

```text
python -m pytest tests/scale/test_selected_world_readback.py -k readback_plan --tyx-mode interpret --require-tyx -q
python -m pytest tests/scale/test_selected_world_readback.py -k readback_plan -m tyx --tyx-mode cuda --require-tyx -q
```

本组包含公共双缓冲、多世界重排/32步零稳态分配、跨实例、空/非法选择、held/history与局部重放、第二次下载故障、局部数值拒步、中途分配失败、两种完整输出无插桩配对和CLI计数命名。作者只交源码与精确节点，不重跑C0的20个已绿实例或14配置。

## C1 实际延时、吞吐与范围

[18项汇总](../../artifacts/p65-top-three-batch5-20261106-01/comparison-summary.json)及其54/55/56逐任务`profile.json`保留完整数字和raw帧。下表是三独立进程的最小–最大范围，延时单位为**每32步完整任务的ms**；比值在各轮分别计算后取范围，分母是同轮计划p50。吞吐使用每轮实际`W×640 / 正式whole总秒数`，不由p50反推。两路逐帧同tick、世界键、维度、完整checkpoint或六量输出相等，独立解析位置/速度误差均0。

| 同种观测任务 | p50：旧→计划 ms | p95：旧→计划 ms | 三轮p50旧/计划 | 世界子步/s：旧→计划 |
|---|---|---|---:|---|
| snapshot W32 K8 | 47.151–53.736 → 10.092–11.535 | 60.380–66.385 → 11.524–13.646 | 4.658–4.815 | 18,434–20,866 → 80,384–98,393 |
| snapshot W128 K8 | 49.100–49.587 → 10.552–11.142 | 53.170–57.420 → 12.161–13.395 | 4.443–4.699 | 81,963–82,714 → 365,840–383,640 |
| snapshot W1 all | 5.516–6.132 → 7.509–8.105 | 7.045–8.397 → 8.515–9.868 | 0.728–0.757 | 4,933–5,642 → 3,805–4,217 |
| kinematics W32 K8 | 6.490–7.742 → 7.602–9.415 | 8.686–9.748 → 9.324–11.418 | 0.822–0.900 | 127,922–147,107 → 107,082–128,180 |
| kinematics W128 K8 | 7.142–8.117 → 8.299–10.051 | 7.574–10.051 → 9.275–12.652 | 0.808–0.861 | 491,842–580,691 → 391,604–489,730 |
| kinematics W1 all | 5.670–5.919 → 7.501–8.489 | 5.844–6.783 → 8.644–11.133 | 0.697–0.756 | 5,371–5,626 → 3,641–4,150 |

三轮每项的20个正式窗均有相同计数关系；以下bytes仅为**该观测的D2H**，不含两路都保留的每物理步4W B接受标志。launch计入物理推进与观测，不能当作纯观测核数。

| 任务 | 旧→计划观察D2H B / 640步 | 旧→计划实际launch / 640步 |
|---|---:|---:|
| snapshot W32/W128 K8 | 266,240 → 266,240 | 5,760 → 1,280 |
| snapshot W1 all | 33,280 → 33,280 | 640 → 1,280 |
| kinematics W32 K8 | 1,064,960 → 143,360 | 640 → 1,280 |
| kinematics W128 K8 | 4,259,840 → 143,360 | 640 → 1,280 |
| kinematics W1 all | 33,280 → 17,920 | 640 → 1,280 |

snapshot K8的提交/显式同步减少，实际字节不变，出现重复端到端收益。kinematics字节减少但新增一次gather和显式同步，所有已测档的p50、p95及总吞吐均退化；W1 snapshot也退化。C0的15%门只证明值得试验的归因，不能保证收益。因此保留显式opt-in，不切默认、不扩大到尚未计时的K或一般接触场景。这里推进的是自由球/延迟滑轨，HOST仍构造公共记录，不称完整GPU驻留物理或跨引擎提速。

## 功能、首错与源码外wheel

- C1功能：35 interpret **14 passed / 2.69 s**，36 CUDA **13 passed / 4.60 s**，无数值修复。37/38首style的1项import、4项format失败原样保留；39/40处理后41/42全0，43确认4个Python文件non-import AST及scoped imports一致。
- 关闭异常：44保留有效RED。首计划清理回调抛错时，旧hook跳出循环，第二计划未关；父以`1ca80f6`统一ExitStack关闭回调修复。仅新增留出重验：47 interpret **1 passed / 1.15 s**，48 CUDA **1 passed / 1.68 s**。49 lint0、50首次format1保留，51/52最终0，53 raw AST相同；最终测量源码为`c5bdff9`。A独立复核确认该finding闭环，无新阻断；没有重跑C0或整套旧绿组。
- [58 wheel构建](../../artifacts/p65-top-three-batch5-20261106-01/58-integrated-wheel-build/)基于父正常集成`dddb612dcfbbbf9305279ff62214ac9d17095055`，RAW0；wheel为1,107,756 B，SHA256 `8a04551901d5ccc884dc9442ea4d2f1ebfa526b4e06c7d523fa137ab5ee1cc4a`。
- [59源码外组合验收](../../artifacts/p65-top-three-batch5-20261106-01/59-outside-wheel-integration/) RAW0，cwd为`Temp/phymeo-top-three-wheel-dddb612-20261106-01`，所有Phymeo模块的`__file__`均在解包wheelroot。C用真实CUDA、3个world、非连续选择`[503,91]`验证两kind计划的4个接受tick、旧完整snapshot同状态、空选择与close owned bytes归零；同次还覆盖A反序登记40步9门和B04真正朝向变化的独立首触`asin(7/8)/3`。这是源码外解包wheel的组合验收，不是再次全套、安装依赖或全GPU窄相验收。
