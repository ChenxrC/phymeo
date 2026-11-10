# B3：分页受载接触消费设备 patch（冻结实施）

状态：AUTHORING_GRANTED / PROJECT_EXECUTION_PARENT_ONLY。依据父当前 744d569d6028783f188d9cc7a4ded272a1625936；三个既有 K1 未跟踪文件未动。子树 3405cbf50fe476f876a5977503df8e0695699d01 clean。下方父裁定授予明确范围纯编码；仍不授子 agent 执行。

## 最小范围与接口

建议实施，现有五元素 commit 和三个 geometry hooks 足够，B2 adapter、共同 mechanics/物理核、求解器均不必改。

1. 作者 MOD src/phymeo/backends/_resident_contact_bundle.py。
2. 作者 NEW tests/scale/test_resident_device_contact_bundle.py。
3. 父 MOD src/phymeo/backends/tyx_world_batch.py，仅 prepare_contact_bundle 参数转发与说明。

公开参数沿用 B2：narrow_phase="host_patch"；device_patch_site_capacity=None。device 模式 None 表示每页 4096，显式整数 1..4096 对每个非空页各自生效；HOST 模式拒绝显式 device cap。参数须在任何页分配前验证。原 per-page candidate_capacity、per-world site_capacity 保持，不另设全局 site 桶或扩页硬界；首片无需再增加逐页不同 cap API。

## 构造、不可变结果、一次提交

- 先按 C1 完成全部 broad/material/topology 注册，再分配各页与共同 mechanics。之后在 batch lock 内为每个非空 _ContactCollisionPage 创建一个 ConvexQueryPlan(batch, page, site_capacity=cap)。该 page 已是正确 ResidentCollisionPlan owner，不复制注册。新 query 自动登记为 page 的 dependent。晚 query 构造失败经原 bundle.close 关闭此前页/query/mechanics。只保留原“全部 broad 注册先于分配”保证，不冒称后续 query 分配/编译也能全部预检。
- 在 bundle 模块新增 frozen ResidentDevicePatchBundleConsumption：owner、bundle registration、topology_generation、state_epoch、active；按声明页序的 page_consumptions（原 B2 DevicePatchConsumption，真正空页为 None）；预构造 page_queries tuple；不可变聚合 samples。保留各原 consumption 的完整 raw/source、differences、patch_differences，不造伪 ConvexQueryResult 或共用 registration。聚合前严格检查 key 不交叠，禁止 dict 覆盖。新增只读 last_device_patch / last_device_queries，均只展示已提交证据。
- _stage_contact_geometry 遍历既有 attempt["pending"] 的 (page,page_result,pending_layouts)，确认与 bundle result 对应项身份相同；调用该页 query._stage_query((page_result,pending_layouts,src))。返回 broad/pending 必须仍是原对象；require_current；epoch==attempt epoch、active==src、完整 sorted row pairs==该页 candidates。每次调用 patches_from_device 时，entry frames 必须只取该页 _page_worlds：原 adapter 明确要求 frame key 集合精确等于 query.shape_keys，不能把整个 bundle entry 传入。
- 所有页转换完成后才返回一个 .samples 聚合，共同 mechanics 仍每 world 装配一次、每 storage 求解一次。原 diagnostics["device_patch"] 保存该 aggregate；原五元素 commit 已有最后一项容纳它。_check_contact_geometry 验 aggregate owner/registration，逐页 query.require_current（原 source/layout/shape revisions/tick/epoch/bank 全保留）。锁内顺序 scoped loans 足够，不保留借用 view，也不重跑 broad phase。
- _publish_contact_geometry 只发布已构造 query 结果和赋 aggregate；不能验证/转换/分配/launch/sync/readback。原 _publish_query 仍提交所有页 pending layouts。一页乃至末页失败、late finish 或最终 source 变化，必须使全部 state/tick/epoch/bank、broad/convex last_result、成功计数和力收据不变；attempt/work/failure 计数可增加。成功后 query 仍是入步 epoch/bank 的历史证据，绝不标成终点 current。

## 空页、容量、关闭

- 真空页不建 query，保普通 empty broad result，page-aligned narrow 项为 None。非空 shape 页即使无 allowed/detected pair，也运行合格 query，返回空 samples；不能把 UNKNOWN 当空接触。
- 无页或全空页时，原 _begin_attempt 在 geometry hooks 前返回，因此 last_device_patch 为 None，last_device_queries 为预置全 None tuple；batch 仍只推进一次普通 free step。默认 HOST 不创建任何 query。
- 资源按页：W256 的两个 128-world BB 页各需 512 sites，而不是将单 query 扩成 256 worlds。原 shape/pair/GJK/EPA/memory 界不变。device cap/UNKNOWN/RESOURCE/ABSENT 导致整批异常，不 fallback、不截断；cap 等值成功。逐 world 的旧 ContactSiteCapacityExceeded 仍仅拒该 world，保其旧 state/力收据，其余 world 同一次提交可推进；四个真实 BB sites 全计，包括零力点。
- raw→effective 一阶矩与全部 B2 转换门逐页完整保留。queries 由各 page 的 dependents 所有；bundle 原 ExitStack 尝试关闭所有页和 mechanics，不能关闭 batch，也不再重复手动拥有 query。测晚构造失败清理、全部 closer 都被尝试、正常重复 close/batch 继续。现 close 原语没有通用失败资源重试保证，本片不暗中承诺或重构它。

## 新固定案例库（父唯一串行执行 interpret/CUDA）

| 新例 | 必须保留的独立验收 |
| --- | --- |
| 单页对 B2 | 同 input/control/load 连续三步，完整 state/力力矩/来源/差分及四站点权重 once；不同 owner token 按结构对应，不跨实例比 identity。 |
| W129、W256 全真实 BB | 每 world 2kg slider，原 H、非零入步速度/驱动/外载，连续三步，每 world 四 site；独立 F、v、q 公式、双侧 force/torque、moment 门；每 storage 一次 mechanics/solve，非每页各求解。 |
| world/page 反序 | 稀疏 IDs、batch 插入与 page 顺序反转、world→page 划分交换；逐 world 物理结果与 keyed 全 query 集一致，各自原页序仍保存。 |
| 四 layout 跨页混排 | ordinary/affine/floating/floating-affine 各两个真实受载 world，分成两个混合页，每种 layout 在两页都有；完整 M/J/load/work 独立审计，不能替成四个单 layout 页。 |
| 最后页失败 | 先成功，再末页 cap/未知几何或 source/moment 转换故障；另 late finish/precommit fault。全部旧 publication/layout/state 保留、费用诚实、原输入重试成功。 |
| partial reset / world cap | reset 选定 world 后原 epoch query 全失效；新步正确取 reset 与未 reset 数据。另单 world cap<4 仅该 world 拒步保旧收据，其余推进。 |
| HOST/empty/ownership | 空页穿插、全空/零页、无候选非空页、默认 HOST 等价、cap 等值与少一、末 query 构造失败关闭此前资源、重复 close 后 batch 可用。 |

只跑新矩阵及直接受影响的少量 HOST-bundle 兼容例，不重跑已绿完整 C1/B2。fixture、物理预算和转换门不放宽。两轮无进展诊断后给最小反例并停止扩范围。

这是世界分页与 TYX 设备几何的组合；表示转换验证、力律装配与完整物理诊断仍有 HOST 工作。不是全 GPU force，也不是 MuJoCo-TYX/GPU PPO 或无限规模支持。

## 父冻结裁定（2026-11-11 04:45）

B2 新增完整矩阵已在父 681261d6 源通过：解释 67 / 80.94 s，CUDA 67 / 313.08 s；相邻默认空批和 HOST bundle partial reset/capacity 解释2、CUDA1亦通过。一阶矩转换门已完整补足，原两处 import/format 与测试属性误名的失败保留。B2 保存47件审计尚待单槽执行，不能冒称已经完成，但它不阻止本片隔离纯编码；实际B3验收前须完成该前置审计。

父接受上文全部接口/资源/事务定义，沿用原 mtyx_t2_source 实施。其 clean child3405cbf5 在原 worktree 新建隔离分支 codex/p65-paged-device-contact-20261111，基于父 cbfc6f21；旧分支/ignored报告保留。只 MOD _resident_contact_bundle.py 与 NEW tests/scale/test_resident_device_contact_bundle.py；父独占 tyx_world_batch.py 四参数行/必要说明。先新测试再实现小提交，不碰 B2 adapter、共享 mechanics、物理 kernels、bundle 既有 HOST 测试或三旧 K1 WIP。

作者保留完整预声明矩阵和保存证据路径，在交接前给出手工节点清单；支持性声明必须区分 authoring 与实际执行。可复用既有独立物理 oracle 与 serializer，不能 import/run 项目来生成预期；不得使用 raw host patch 来偷换 device 模式。四布局混页必须真实有非零入步速度/驱动/外载和双方反应。新增聚合类型留在 bundle 模块，由公共属性/收据访问，本片不额外扩展顶层 exports。

父目前独占 run93 G1完整v2轨迹，子作者禁止 import/collection/test/数值/离线probe/lint/formatter/build/install/profile；只源码文本、file IO 和小 Git。06:30 停止新范围，当前任务目标此前交作者清洁提交及未执行标记，父全文增量 review 后再排新矩阵和源码外公共 smoke。无新助手、重复旧 reviewer 或远程CI；失败按具体原因报告，不改预算/fixture/物理初态。这个组合不等于全GPU力学装配或MuJoCo-TYX路径扩域。
