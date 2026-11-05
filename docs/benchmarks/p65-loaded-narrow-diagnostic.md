# C3 同次受载窄相诊断

本批只新增诊断驱动、协议测试和本文，不修改产品、原 C1 driver、原独立物理 oracle 或依赖。来源起点为 `ff6efeb07143cb2b3eb671a48d26d4ad6bd2e06c`。父在唯一串行槽执行；作者没有运行项目。

## 已有证据与待解问题

C2 原九档已完整同值验收，但不同时段的墙钟结果混合：free W128 变快，loaded/birth 及无跨世界过滤行的 W1 均变慢。因此不能仅凭这组先后测量认定过滤复用造成受载回归。本批不重跑九档，也不读取或重新哈希旧大帧。

当前源码可定位到重复工作，但尚无新实测占比：

- `src/phymeo/geometry/collision_pipeline.py:608` 的 hull–sphere 分支直达 `static_convex_support` 或 `affine_convex_contact`，没有使用本函数的 `prepared_hull` 缓存。
- `_convex_support.py:341` 在每次外部球–凸包查询中构造 `HullShape`；`_convex_hull.py:81` 再次转换精确坐标、校验拓扑维数、求真实边界和材料图。固定箱的入口及路径端点查询会重复这些构造。尚不能把这些校验直接删掉，任何未来复用都须绑定实际注册材料及姿态。
- 同一函数 342–371 行搜索真实面投影或真实边，372–408 行计算距离界、检查世界分辨率并发布材料见证。这条路径通常绕过通用 GJK `_closest`，不能用 GJK 时间代表它。
- `_convex_support_path.py` 的 `_Window.sample` 再查静态见证，`clear_on/run` 验证完整时间区间。`CollisionLoadSystem.step` 的真实入口受力采样、积分和 `_checked_path` 均保留；没有把端点采样替代路径证明。

## 冻结 A/B 与输入

A 为**本次同一产品源码中保留的 full-dispatch**，仅在调用范围把 `_complete_pairs` 指向现有 `_detect_pairs`；B 为当前 filtered-dispatch。两边都先执行原 envelope、真实 backend classify、容量与几何检查。它不是整版旧二进制，也不测旧版本工作区构建成本；两边都创建当前版本工作区。

固定顺序为 loaded W1、birth W1、loaded W32、birth W32。每档有 untraced 和 traced 两个独立窗口，各有 A/B 两个独立公开 ContactScene，固定 `H=1/128 s`、第 1–16 帧，使用原 C1 种子、物理律、容差及独立 oracle。每帧次序交替 AB、BA，即整个调用序列 ABBA；不根据测量结果重新排序或挑选帧。两场景只在单线程上轮流执行，完整输出对齐同一个物理步。

loaded 为真实 4 N 平衡滑动；birth 为原有限箱棱线进入正余量层，第 9 帧入口恰为零压缩，第 10 帧首次正力。16 帧覆盖前后。W32 仍是 64 个真实形状、每次完整 2016 个 pair，其中 1984 个跨世界 pair；重复本地 body ID 和镜像输入保留。

## 测量边界与完整性

未插桩时长从 `scene.step(H)` 前到紧接的 `scene.geometry()` 读回后，单列 step 和 geometry 时长。物理输入状态、每步真实返回 frame、完整 geometry、contact_history 和新的 contact_events 原样 canonical 编码；公开字段排除列表为空。history 生成发生在 step 内，history/events 属性读取和 oracle、归档在计时区间外，范围明确，不把它们称为已计入的公开调用。

各变体逐步对比所有 canonical 字段及类型，包括物理时刻、路径、力、两侧材料、过滤行、新增未知字段。每一侧独立通过原 scalar oracle。全部应用与实际 TYX 计数保留；只有明确的四个累计/差分统计时钟允许不同：

- `geometry.envelope_ns`
- `geometry.broad_ns`
- `geometry.midphase_ns`
- `geometry.narrow_ns`

比较只把这四个值设为 0，保留其键；禁止按 `time`、`ns` 或前缀泛删。每次保存前后完整 stats 和 native stats，原生全局绝对值只作原始记录，逐调用增量必须相等。核数、字节、候选、材料、实际力与状态不得不同。

每边初次调用和后续 15 步分列，完整 16 步 p50/p95 也原样保存；首次编译、首次接触调用和逐 runtime 资源初始化可能影响首调用，不能冒称稳定吞吐。该单次小窗口的目的是排除粗大的时间段混淆和定位调用工作，不据此作跨引擎或整版速度排名。

两条公开完整流使用原 C1 gzip 增量 codec：第一帧完整，后续保存无损变化；本次每条新流同次完整解码一次，校验每帧与 whole logical 摘要。旧矩阵不参与解码。traced 与 untraced 还必须逐帧同值、非墙钟计数相同，未因插桩改变路径或状态。

## 独立 trace 窗

诊断使用调用线程的 `sys.setprofile` 和仅 `_exterior_sphere_hull` 的两个源码行锚点。锚点按原行文本唯一匹配并保存绝对文件路径、SHA 和行号；不改产品函数体，不包装数学函数。不允许已有其他 profiler/trace，结束或失败均撤销两个钩子与分派替换。

记录包含构造 HullShape/边界/MaterialChart、面边搜索、材料权重、投影后距离界及见证完成、区间支持界与路径验证、分派、复制、真实设备接口等。每段保留 call count、inclusive 和**互斥**时间：按最深活动区间分配，全部 exclusive 加上明确的 caller 残项严格等于该次公开计时边界。inclusive 子调用不重复相加。

`projection_distance_material_finish` 混合距离计算、分辨率门与材料输出，不声称是纯校验；Python device 接口 spans 是 API 墙钟，`pure_gpu_device_ns=null`。全部 trace 时长含插桩扰动，与未插桩时长分开，不拿它作为正式 A/B 墙钟。每帧上限 100000 spans，超限明确失败；每变体报告 traced/untraced 总墙钟比，超过预声明 20 倍时标 `attribution_usable=false`，仅保留调用次数和完整原始证据，不据受扰动占比挑优化。

额外的内部差异也不删除：每步 A 比 B 多构造的 `_pair` 数必须为 `2 × different_world_rows`（入口和路径各一次），B 的 `_complete_pairs` 调用 2 次，A 为 0。所有 HullShape/真实投影/材料与路径证明函数次数必须一致。此处是当前四档的独立拓扑计数，不是一般模型的过滤公式。

## 执行与容量

父先运行协议（原首 RED 已保存），再运行固定 CUDA 小矩阵。示例均在本树、已选项目解释器、fresh guard 后，`<HEAD>` 必须是实际集成/候选 clean HEAD，`<新目录>` 必须从未存在：

```powershell
python -m pytest tests/scale/test_loaded_narrow_protocol.py -q -p no:cacheprovider --basetemp <全新绝对临时目录>
python scripts/profile_scene_loaded_narrow.py --backend cuda --expected-head <HEAD> --out <父artifacts全新目录>
```

HOST/interpret 是显式支持的协议诊断模式，本批正式小矩阵只跑上述一次 CUDA，不追加其它规模或模式矩阵。脚本依次输出四档自然边界，全部 256 个新公开步（4 档 × 2 窗 × 2 变体 × 16 步）。不启动子进程或线程，不占用第二个执行槽。作用域替换不是可并发库 API；该脚本进程内不得另有业务检测线程。

仅作排程估计：按既有 W32 成本量级并为 Python trace 扰动留余量，预留约 3–8 分钟；这不是新实测。流式 gzip 与固定 64-shape 上限预计在数十 MiB 量级，预留 256 MiB；原始 logical byte 数、实际 gzip byte 数均在收据中保留。每步只持有两个当前 canonical 帧及当前 trace；不累积完整大帧，不截断返回字段。达到 span 上限、来源变化、字段/计数不等或几何拒绝即保存失败报告并停止，不调物理输入或容差。

每档输出 `pair.json`、两条 `*.json.gz`；traced 另有 `timelines.jsonl.gz` 原始 spans；总目录保存 `manifest.json`。metadata 包括每边初态、rigid rows、原输入 seeds；所有三项原工具、两项新 Python 工具和已加载产品源码在前后分别绑定，必须稳定。

## 本批验证状态

- 首测试提交：`1c36161b1e2518f8275f0bbd6100afd9a18accfb`。
- 父真实首 RED：`c3-protocol-red`，RAW1，1 failed / 1.29 s；原公共 4 N 与独立 certify 先通过，再因诊断 driver 缺失断言失败。
- 驱动、负控与本文为待验候选。作者无项目执行；尚无本批性能结论，也未选择或实施新的产品优化。
