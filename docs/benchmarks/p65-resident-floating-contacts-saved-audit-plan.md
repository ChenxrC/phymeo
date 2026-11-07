# C10 保存审计：固定身份与纯作者计划

状态：**AUTHORING_ONLY / UNEXECUTED**。这是离线 observer，不运行生产者、物理、几何、M/J、独立参考或任何项目 import。父唯一串行执行，C9/C10 物理接受与本审计是独立状态。

依据父 `d75cdaf4be0ce751f6fd27b39b4d4982f4460689` 的 C10 合同末节；作者测试固定于 RD `d7bed1556fe630f5ace96bc7c251f4cd4e056fae`。原生 driver 使用本树 `5c959c8868e215e8eb6208173fa45882b9a658f0`，外部 wheel consumer 使用同一 RD 提交。未读取任何运行中的输出，也未按输出选身份。

## 预声明 membership

两个专项每 backend **76 device 节点、61 个 gzip**；独立 HOST 节点 **0**。F01 的 HOST patch/2 N/M 前置在该设备节点内，不能另加一次 HOST 通过。脚本 `membership()` 枚举完整原测试名、参数、pytest ID 和相对保存名。

| 测试族 | device 节点 | gzip | 固定内容 |
|---|---:|---:|---|
| F01 | 1 | 1 | root-only、HOST 前置、一次真实步 |
| F02 | 3 | 3 | child/motor/root，各 40 步 |
| F03 | 4 | 4 | 原序/反序/cycle/boost，各 40 步 |
| F04 | 1 | 1 | 非零角速、阻尼、真实弧，40 步 |
| F05 | 3 | 3 | COM 偏心、principal frame、cycle，40 步 |
| F06 | 1 | 1 | 实际 folded URDF 工具，40 步 |
| F07 | 6 | 6 | hull/cap side/end/miss/CC/非盒 HH，各 2 步 |
| F08 | 5 | 5 | N9/N16/body15/root+body16/共享祖先，40 步 |
| F09 | 1 | 1 | 连续载荷、分离、重入，40 步 |
| F10 W 矩阵 | 1 | 6 | W1×4/W4/W32，各 40+21 重放 |
| 容量两 bank | 4 | 4 | warm0/1 × world503/6 |
| delay/cap | 4 | 4 | delay0/1 × cap1/8或8，各4步 |
| 真实 late fault | 6 | 6 | upload/candidate/receipt × warm0/1 |
| restore stale | 2 | 2 | 两 bank、真实 candidate 后 restore |
| allocation/global cap | 1 | 1 | 实际 scratch 关闭和合法重试 |
| owned/foreign/close | 2 | 2 | F01/F03、重建完整一致 |
| old mixed | 1 | 1 | ordinary/affine/scalar/未绑定 floating，40+21 |
| pair filters | 6 | 6 | default/same-body/mask/excluded/parent 两类 |
| invalid graph/material/source | 11 | 0 | 原测试断言，不伪造记录 |
| builtin bool | 4 | 0 | int/NumPy bool/str/None |
| invalid floating body | 3 | 0 | (N,body)=(16,17)/(15,16)/(0,1) |
| old body bound | 2 | 0 | ordinary/affine，不虚构 body16 |
| huge-anchor source | 1 | 1 | 原输入、真实 broad、拒绝与 retained |
| candidate source | 3 | 3 | shape/group/world 原 packet、变化、重试 |

W4 显式 world 顺序 `(503,91,1207,6)` 分别 F01/F02-motor/F08-N9/F08-N16；W32 另 28 个 world 为 `10001+97*i`，输入按四例循环，仍保存每个 world 全行。反向注册不删除 owner/world 字段。所有步长原 `1/1024 s`；不重算物理。F07 只把独立首几何锚与“已上传 wrench 的独立响应”分别核，不能冒全窗几何 oracle。

## 审计输入与边界

CLI：`python -I -B SCRIPT --manifest ABS --output ABS_NEW_DIR`。manifest 固定 schema `c10-contact-saved-audit-v1`，声明 required_modes、required_external 和有序 runs。每 run 必须给原目录/config/head/raw_exit/完整 source_sha256；pytest 给 authored selected/completed/first failed 节点及 payload_roots、basetemp，首失败存在的 payload 子集另列。不同源码的通过范围独立列示；必要重复 completed 节点须明确此前 run 与修改原因，unique coverage 只计一次。输出只能为全新目录。

原 start/final、command、stdout/stderr、RAW_EXIT、before/after/source snapshots 全部绑定。所有源码 bytes（包括二进制 wheel）哈希保留，二进制不当 UTF-8 解码。payload_root 绑定原 CLI；外部 --output 可以在 runner 日志目录外，不能复制后改路径。枚举保存路径只检查固定 membership 的遗漏/多余，不决定应有集合。

逐 payload 检查完整原输入/模型/命令、N/B/D/Q、held/history、local_origin/world position、同次 diagnostics/M/J/full body wrench、actual geometry/sites/material/forces/work、真实 tick 和 capacity retain/retry。只比较已有独立 oracle 与实际字段，不求新动力学/接触。40+21 原字段完整相等；epoch/counters 独立单调，不删除物理字段作“规范化”。

保存缺口明确：20 个负控无 gzip；若 late fault、容量或 owned 测试只保存 before/retry、原断言中间 retained/不可写行为未落 JSON，本审计不能重新证明该瞬间，绑定同源 pytest 结果并注明。F07 没有独立滚动几何真值；巨大 anchor 负控没有正常物理轨迹。失败节点可能在保存前失败，保日志及真实前缀，不补 null 数据当完成。

Native 每端 F01/F03 各40，完整 raw contacts/shape/material/filter、FREE state/COM velocity、body_solve_origin/raw RNEA/decoded wrench、full M/J、独立轨迹、checks 与 pin source 均保存；零 matched 外力。失配和 observer/IO 失败保持原状态，不冒 native 通过。Wheel 两模式分别 W4 40+21、两个真实 bank capacity/retry、最多2 alignment 步，以及 F03 40；逐行完整物理记录、安装成员/loaded origins/关闭资源。F03 不额外声称有 replay/capacity。

资源上限：每 gzip 64 MiB 压缩、每 payload 256 MiB 解压、全部累计 2 GiB、8192 文件、JSON 深度96；metadata 单件16 MiB。只在内存有界读解压，不落解压副本，不删除证据。原文件 SHA/压缩与原 bytes 均记录。未执行 observer 不报告 pass；最终返回 audit coverage 与原生产者状态，不能替代物理验收。

## 交接 schema 与执行前注意

- 本批 `--output` 是 **全新目录**，固定写 `audit.json`；C9 老 auditor 的同名参数是新文件，本脚本明确不沿用那一输出容器。全部 evidence 只读。
- Run 基础字段：`id/kind/mode/directory/config/head/raw_exit/source_sha256`；`directory/config` 为绝对原路径。config 的相对 `files` 从原 `start.cwd` 解析，不能从 auditor 当前目录解析。完整原 source 集合与 config、start、final、snapshot bytes 一致。
- Pytest 附加 `selected/completed/failed/payload_roots/basetemp`。`selected` 用代码固定 `.node` 字符串按原 CLI 顺序，CLI 可文件/函数/参数节点及明确 `--deselect`；`-k` 仅接受完整字符串 `not test_f01_public_root_contact_keeps_full_six_coordinate_response` 和 `test_f08 or test_f09 or test_f10`，其他表达式拒绝；`completed` 是实际前缀，`failed` 无错时 null。`payload_roots` 明确每个会保存的完成/首失败节点的原 test tmp 路径。首失败的 `failed_payloads` 只能是该节点固定文件名子集，可以空。完整 gzip inventory 仍不能有漏报/多余。20 无保存节点无 payload_root。
- `qualifies_completed[node]={prior_run,reason}` 只用于确有历史完成的必要复验；每次原记录全验且按原 source 集合单列。未重复的节点不填该键。不同 source 的 union 不命名同源通过。
- `kind=native|wheel` 附加 `payload_directory`（绑定原 CLI 绝对 `--output`）。native `mode=cpu|cuda:0`，wheel `mode=interpret|cuda`。native 再给 `native_source_sha256`，完整13份 pinned 源的相对名→SHA；它们与 native 自带 `source_snapshot`/before/after 对齐。成功 native 固定两个 gzip；成功 wheel 固定一个 gzip。失败运行明确已有 `failed_payloads` 子集，只保存原状态，不增加通过覆盖。
- `kind=historical` 可登记原前置/观察器错误，须 `reason`；如存在原 gzip，用显式 `payloads` 逐件有界读取，覆盖计数为0。不能用此类型把 required 节点补绿。
- 最低成功源码集在 `REQUIRED_SOURCE` 明列：本批两专项/两个独立参考、实际 test helper、contact plan/bridge、浮根 packet/projection、原 C9 storage/kernel/bounds 与最终 `_tyx_floating_numeric.py`。F06 要包含真实 URDF；mixed/old-body 包含旧 affine 事务 helper。父仍在原 config 完整列真正依赖，其余也逐字节绑定，不只验这个最低子集。
- 数据边界：partial node 的所有 bytes、字段与 nonfinite wrapper 原样保存并类型扫描，不施成功门掩盖原失败；不会把 `_audit` 未返回的半行补成完整 oracle。成功记录比较已有保存真值、实际帧及完整重放；不重新算 M/J/几何/轨迹。正式 producer 修正若改变 schema，必须按确切源码调整 observer并保首错，不能按输出删除字段凑过。
- 成本为逐文件 JSON 解码/结构与已有数值比较、原 byte/hash/ZIP成员读核；逐件有界读取、不收集全批 payload；W矩阵跨文件比较只留完整世界帧的摘要指纹。无物理核/全量解压落盘。wheel 当前安装文件只读核与原归档/保存成员一致，任何漂移报告observer不一致。

本文件与脚本尚未执行、静态检查或格式化。没有新增物理 pass；由父之后独占队列运行。作者不修改既交 C9/C10/source/tests/native/wheel，也不读取其运行中的输出。

## 执行前只读审阅修正

RD 指出合法 F06 的 `ResidentFloatingTreeBinding.from_asset` 使用 `topology=None`，真实来源由 `_source_shape/_model_token` 保存。初稿一律要求非空 topology 会误拒该既有输入；这是 observer schema finding，没有产品执行失败。现分别核：直接安装的完整 topology 必须等于同 world 入步 checkpoint；资产绑定的原 shape 只能替换声明 key，其余字段逐项相同，原 model token 必须等于该 checkpoint 的完整物理 model token。保留原材料 IDs/local points 与实际 frame 检查，不通过移除 provenance 门修复。旧 ordinary/affine 资产若采用同来源协议，仍核其真实 model/graph token，不从 K 坐标猜全体。

另两项执行前 finding 一并修正：相对 `--basetemp` 与 `config.files` 一样按原 `start.cwd` 解析；每个 payload root 独立匹配原 selected 顺序下 pytest 的 node.name 非词字符替换、30 字符 prefix 和同 prefix 递增 ordinal（无保存节点也占 ordinal）。该规则来自原解释器安装的 `_pytest/tmpdir.py:282–287` 与 `_pytest/pathlib.py:make_numbered_dir`，只适用于本合同的新 basetemp、正常串行 pytest 次序；不从输出挑目录。F03 另核保存的 binding tuple 原始正/反注册顺序：`CollisionModel.shapes` 会规范排序，不能以它代替未规范化的注册输入。两参数目录交换会在目录身份或原 binding 顺序处被拒，未改 fixture/物理门。

第4项修正是完整公开 map coverage：按完整 checkpoint 的实际类型确定树 world，geometry 和已保存 diagnostics 必须覆盖所有树（不要求 scalar 有 FK）；summary 必须覆盖全部 state。receipt 按真实 binding 的 world 集合核，初始 tick0 允许空；wheel 新容量 plan 的拒绝 world 仅保此前真实已发布的 receipt，不能把其他 plan 的 receipt 填进来。额外核 wheel 入步、初始和末态公开 map，避免中间行缺一个普通树的 FK/diagnostic 被 receipt 内部数据掩盖。本轮仅观察器文本修正；四项均未执行，原物理模型/门/数据不变。

父实际分组 CLI 对齐：上述两个 `-k` 完整字符串是严格白名单，分别按作者 `Member.function` 排除精确 F01 函数，或保留 `test_f08`/`test_f09`/`test_f10` 前缀。接受独立参数与 `-k=` 拼写但不改表达式内容；多表达式、未识别表达式均拒绝，不 eval、不导入 pytest、不按输出决定选集。过滤后的 authored 顺序仍用于原 tmp prefix/ordinal 与成功前缀绑定；76/61 总身份不变。此为单独执行前 observer 兼容修正，既有四项来源/覆盖修正未重写，仍未执行。

RD 对第4项的限定复核发现：虽然已保存的 diagnostics map 会核全树覆盖，mixed 前19个未参与重放的行仍可整段缺字段。现仅在 authored mixed 的全部40个 forward、21个 replay 行显式要求 `diagnostics`，再复用完整 map 校验。依据原 producer 每行均保存该公开读回，不能依赖后21行的重放比较来证明前19行；其他本来不保存独立 diagnostics map 的 schema 不变。此窄修正仍为纯作者、未执行，也未读取任何实际输出。
