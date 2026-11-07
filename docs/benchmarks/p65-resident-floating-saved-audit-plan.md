# C9 浮根驻留保存审计计划（作者未执行）

状态：`AUTHORING_UNEXECUTED`。只新增标准库消费者 `tests/scale/resident_floating_saved_audit.py`；不导入 Phymeo、NumPy、Newton、pytest、生产者或参考求解器，不推进物理，不重新求 M/J/FK/动力学。父独占最终审阅、静态及实际审计。当前正在运行的 460 未读取；本设计未打开任何输出 payload。

## 预声明成员来自源码

权威是父当前 `test_resident_floating_step.py`、`test_resident_floating_collision.py` 的函数/参数/_save 声明，以及 `resident_floating_reference.py` 原输入。代码 `membership()` 固定全部节点、pytest 参数顺序和保存相对路径；发现哪些文件不能决定需要哪些案例。

| 组 | device 节点/每模式 | gzip/每模式 | 原保存语义 |
|---|---:|---:|---|
| P03 公共首节点 | 1 | 1 | 真实 HOST 前提、完整初态/模型、实际一步 |
| P01–P04 七窗口 | 7 | 7 | 每例 40 步，同入步独立参考与完整状态 |
| 共同变换/boost/大平移 | 4 | 4 | 原四配置各 40 步 |
| P05 n=0/1/9/16 | 4 | 4 | 每个完整 B/D/Q、M/J，各 40 步 |
| 固定根工具聚合 | 1 | 1 | 真实 URDF、40 步 |
| delay 0/1 × cap .125/8 | 4 | 4 | 每个 4 步，外载当前、仅驱动延迟 |
| W1/W4/W32 | 1 | 6 | 四种 W1，W4/W32 各 40 + tick19 后 21 重放 |
| 普通/affine/free 混合 | 1 | 1 | W4 全 40 步 |
| 晚故障 warm0/1 × 三边界 | 6 | 6 | 原完整 before/target/retry/reference |
| 非法恢复 | 8 | 8 | 有效 before/retained；坏请求未落盘 |
| 三种候选域拒绝 | 3 | 3 | before/recovered；即时拒步快照未落盘 |
| 低预算与正常预算 | 1 | 2 | retain 原态 + 独立 2 步完整正常窗 |
| owned / 巨大 anchor / 局部恢复 / π 邻域 / 分配失败 | 10 | 10 | 1+1+5+2+1 节点，原各自有限记录 |
| FTZ 算术 | 5 | 5 | 三有限实际结果；两非有限真实异常文字，不能伪造 NaN 通过 |
| P07 五几何 × 40 步 | 5 | 5 | 所有原材料、当前有限 S/C/H bounds |
| P07 warm W1/W4/W32 | 3 | 3 | bootstrap2；暖态40仅 bounds/ticks/counters；entry/final 完整状态 |
| root 固定工具 / branch 过滤 | 7 | 7 | 1 个40步 + 6 个静态过滤声明 |
| 非法 owner/source / 浮根 contact 拒域 | 11 | 0 | 原测试没有 `_save`，只同源 pytest 收据 |
| 未绑定旧接触 / bounds事务 / close / child固定工具 | 4 | 4 | 2步混合、有限事务、两plan关闭、40步原工具材料 |
| **总计** | **86** | **81** | step56→62文件；collision30→19文件 |

HOST 独立锚点6 + 非法声明9 = 15，全部无 `_save`。审计不把它们虚构成轨迹。旧 helper 的 445/443 只能列 `historical`，不能填补当前成员。当前算术 459 的5+P01及460其余80，只有实际结束且来源绑定一致后，才可用显式白名单的并集覆盖 interpret86；未来 CUDA完整组或明确拆分补集按相同规则。不根据成功输出自动挑组，不重复计已完成节点。

## 有界运行合同

CLI：`python -I -B <审计脚本> --manifest <绝对JSON> --output <尚不存在的绝对JSON>`。作者不执行此命令。manifest schema 为 `c9-saved-audit-v1`，含 `required_modes`（host/interpret/cuda 的显式全集需求）、`required_external`、`runs`。每 run 明确：

- `kind`: pytest / native / wheel / historical；`mode`；已结束 `directory` 和原作业 `config` 绝对路径；原 `head` / `raw_exit`；**完整**原路径→SHA256的 `source_sha256`。
- pytest：按固定 `membership()` 的完整 node ID 声明 `selected`（执行顺序）、`completed`（成功前缀）、`failed`（null 或首错节点）。`payload_roots` 显式把已完成有save节点及首错有save节点指向相对于run的原tmp根；`basetemp` 是原pytest保存根。正常文件名/子路径由固定map决定。首错仅允许 `failed_payloads` 列出该节点原本可能保存的文件子集，作为不计完成的失败前缀；无文件就是空列表。
- 真实pytest args 的入口/`--deselect`/mode要与白名单吻合；本消费者不支持无法直接解释的 `-k`。原 stdout 最终 passed/failed 与白名单相符，首错 `FAILED node` 必须存在。不能用 stdout 反推权威集合。需覆盖的 mode 不允许重复成功节点；完整来源快照逐run独立核，允许显式后续补集。
- `historical` 必有 `reason`，只保存原收据和源关系，不读它的payload、不提供当前完成覆盖。
- 外部 native/wheel：另加 `payload_directory`（相对原run）；native mode为cpu/cuda:0，wheel为interpret/cuda。`required_external` 显式命名如 `native:cpu`。它们的两个固定文件名由代码决定，不能选择部分case。

每份 source 在 start/final/config 全覆盖且前后SHA相同，原 job-local snapshot 实际字节也要匹配；不读取当前WIP代替历史来源。device成功组必须包括新的 `_tyx_floating_numeric.py` 与 `tests/scale/_floating_numeric_kernel.py`，加原两专项按实际选择与独立reference；父为每job提供其真实完整 SHA 清单。旧helper通过不能混入新helper集合。

限制：每压缩文件64MiB、展开256MiB、累计展开2GiB；元数据16MiB；最多8192个输入/库存项、JSON深度96、最多64个显式run。逐文件读/解压/检查/释放，不全体解压落盘；保留压缩原件、原大小与SHA，并记录展开字节。标准库JSON会暂存单文件对象，256MiB是字节上界，不冒称同量RAM；大W32若超界明确报缺口，由父审定分组，不自动采样或提高上限。

## 可检查与不能补造的边界

完整窗口核原 input、所有world身份、原 topology、local/world Q、D、四块held、完整history、实际drive/generalized、B体几何、D×D M及B×3×D双J、entry/exit时间、保存reference/残差与累计计数。仅对已保存值作原门比较；19→21重放逐完整state/geometry/diagnostic/energy相等，不删路径或字段来制造相同。几何保存原 model/bindings/vertex IDs/revision，实际 bounds 包住已保存独立原材料bounds；注册对和候选只允许同world、原完整key。暖态明确不要求中间q/v读回，核40帧的原计数不增、双bank与末态原材料，不能用oracle补成device snapshot。

非法恢复的坏q/v/NaN未保存、domain失败即时状态未保存、只读写保护异常与callback实际次数等依赖绑定源码中的原pytest断言；脚本只核实际保存部分及同源原退出。scan不会把 `{nonfinite: ...}` 当有效物理值。FTZ的overflow/0÷0只有原异常收据，三underflow为实际有限数组；不重新运行数值表达式。

native独立入口核七声明各40完整step、真实shape/joint/body metadata、FREE父锚基底与世界COM转换数据、完整原M/J、原输入力与post-RNEA raw/actual solve-origin/decoded COM同存、所有原named checks、pin版本及原Newton源snapshot；绝不把normalized-Euler与exp-map混为一个轨迹门。wheel入口核W4 40+21与两个bank的原完整输入/拒步/重试、P07显式unsupported、关闭字节、实际loaded module origin和归档SHA/安装成员回执。离线不重新安装/遍历原机器安装目录，其来源验证是保存的同run消费者结果。

输出仅 `saved_records_consistent` 或 `incomplete_or_inconsistent`，始终 `new_physics_pass_count=0`。不把保存审计另加到86/15或native物理通过次数中。当前无审计执行、无静态检查、无新增物理结果。

## 父执行前观察格式补足

父全文读完原1107行消费者后补足两项输入读取：source snapshot中的wheel等二进制仍全bytes/SHA绑定，但不当UTF8源码解码；外部native/wheel的payload根须与原命令行 --output 的显式绝对路径完全一致，允许合法源码外目录，不搬动输出或改原summary路径。均未改物理、预声明成员/门或payload字段，实际审计和静态仍待父串行槽。导入原874ff199时两文件Git blob相同，工作副本仅LF/CRLF差异，已独立核规范化全文相同。
## 父已结束首错与必要补集的观察规则（2026-11-08）

460已实际48pass后在原exact-pi节点失败；原节点保存发生在assert之后，
当次没有该失败完整payload，manifest明确failed_payloads为空。468保原输入
重现并保存完整失败；469修复后的两个pi节点和470四个异惯量/耦合原节点
属于改变gyro源码后必要的定向验证，不能重新标记460源码或冒同hash整套通过。

审计现在允许显式 qualifies_completed 映射：键必须恰为本run与此前已成功覆盖
节点的交集，每个值必须有具体源码变更验证理由。仍全核该run真实CLI、
原成功前缀、完整payload/源/退出；最终coverage用集合，只计每逻辑节点一次。
未声明重复、凭空声明资格或用失败填覆盖都会拒绝。禁止为了绕过此规则重跑已绿节点。

476原剩余31项在6pass后停止于warm W1的固定根计数名tree_diagnostic_reads。
纯浮根storage没有登记该固定根字段；原snapshot/kinematics公共计数和全部
floating_state/fk/diagnostic读回计数均保留，直接索引实际字段，不以get默认0掩盖。
暖态测试先保存真实40步再做观察断言；479只运行其尚未完成25项。
476原失败发生在保存前，不能补称同次完整warm证据存在。
这些是测试/保存观察修正，不改变产品积分、原物理输入、40步或工程门。

477对两个观察文件lint通过，478保留缺一个空行的format-check失败，
补空行后480 format-check通过。实际保存审计仍未运行。

## 父实际保存审计结果（2026-11-08）

原作者未执行状态保留为历史。父在最终实际记录上运行495，RAW_EXIT=0，
状态 saved_records_consistent；20个原始任务、7284个实际文件绑定、192752739
解压原字节已核，覆盖HOST15、解释86个独立身份和最终CUDA86个身份，
含native CPU/CUDA与源码外wheel两端。解释重复节点按明确源码修复资格记录，
不将跨源码分组声称为最终同源整套通过。审计新增物理步为0。

首次485、488、492的RAW1与源码快照全部保留：原读取器分别错误假定
无保存HOST必有basetemp、混合接触diagnostic单数字段、native配置路径必绝对、
关闭计数名resident_collision_plans，以及移动工具帧的state必为world映射。
修正依据原测试/原公开统计/原工作目录；单移动工具检查点、geometry与diagnostic
按其完整实际结构核验，并显式核原独立points/low/high包围，不修改payload。
无保存HOST仅在固定membership确实无任何文件时允许缺目录，device缺记录仍拒绝。
source集合先按原receipt cwd解析配置相对路径，before/after与snapshot/hash门保持。
每个payload失败现在附原node与逻辑路径，避免缺少定位信息。

最终reader SHA256 d3880a247255f70b93d4156d92c7795ff6b9d17dc32e35825192ca38bbac47e2；
493 Ruff与494 format均0。完整审计结果
[495/audit](../../artifacts/p65-generic-collision-20261107-01/495-c9-saved-audit-reader-v4/audit)
SHA256 7b0a831eaac34e2036a5c1a0770c0dda650e05d5e185dab4c74e7e3a1c353fe2；
原manifest SHA256 095b3e963ad2d39d03bfa427d208cfbc6730713bf82a6aeee7c5155549e7f646。
未重跑任何物理窗口或生产者，未补造首失败缺失payload，未删除原失败。
