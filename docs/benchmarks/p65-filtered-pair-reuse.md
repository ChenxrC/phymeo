# C2：普通碰撞查询的完整过滤记录复用

本批只减少普通 `ph.detect` 对不可变过滤行的重复 HOST 分派和记录构造。完整全局 pair 表、实际后端分类、所有公开返回字段及原始 C1 输入、物理 oracle、基准程序不变。性能结果须由父在最终集成源运行原九档后填写；这里尚不宣称收益。

## 输入不变的依据

`CollisionModel`、`CollisionShape` 和 `CollisionKey` 都是冻结登记，排序后的 shape 表、world/body、group/mask 和 excluded 对固定。工作区按原优先级为 code 1–4 构造不可变 `CollisionPair`：different_world → same_rigid_body → mask → excluded。这些行没有几何 witness、运动 trace 或可变材料数组，允许后续返回 tuple 引用同一行。

每次查询仍完整执行 `_envelopes` 和实际 HOST/TYX classify。仅当本次后端返回的 code 与登记模板一致时才复用。code 0、code 5 以及任何不匹配的 code 保留原窄相；不会根据模板覆盖实际分类。AABB 的 code 5 取决于姿态，且 hull/mesh/plate/tetra 的某些合法性检查原本即在 code 5 下执行，不能缓存成静态 separated。

返回仍包含全部过滤行和原 global order。每次新建外层 list/tuple 后按 global slot 回填动态行；先前已发布的 tuple 不改写。容量超限不发布，事务放弃和设备故障继续沿原路径处理。

## 分批与设备边界

筛选动态 slots 后仍按原全表的 4/128 行窄相批次边界提交 features，防止把原先不同批次的运动特征意外合并。原紧凑/world 专用查询的默认分批路径保留。没有修改 TYX kernel、pair 表、上传/下载内容或容量。

这是一项 HOST 分派优化，不是新的 GPU 窄相。原所有 pair 的分类和读回仍发生；完整结果组装及所有实际接触几何仍需执行。工作区创建时增加一次模板构造和存储，原基准单列的 build 时间保留这项成本。

## 专项和实际状态

- 冻结测试提交：`65a35e346a92beebaa47068657ae4a1f576eb87b`。
- 父实际 RED：1 failed，1.06 s。独立 touching 双侧点/ID/法向、全 21 行顺序、AABB miss 与过滤优先级先通过；失败处是第二次真实公共查询仍把 code 1–4 送入 `_detect_pairs`。
- 产品及几何专项候选：`e012423936235e800727f7e59b326589d31e6ca2`。作者未执行项目；HOST、interpret、实际 CUDA 与比较协议由父串行验证。
- 负控覆盖 AABB 运行时 5↔0、RR/RD/DD 与重复本地 ID、过滤对象的非法运动、分离时的非法几何、容量重试、事务撤销、实际分类不匹配、设备故障和原运动 feature 分批次数。
- 本文后续只据父真实收据补绿测、源码和九档结果，不推断 W128 未测细项。

## 完整收据比较协议

运行新的只读比较程序：

```powershell
python scripts/compare_scene_scale_frames.py --before <C1收据根> --after <C2收据根> --out <全新比较JSON>
```

两边目录均保留 `matrix-{free,loaded,birth}-w{1,32,128}`，每档 cold/warm 32 步。比较程序复用旧 C1 已接受的同次全解码验收；新驱动同样在本次运行中全解码验证自己的 payload。默认比较每帧完整逻辑 SHA、whole logical SHA、manifest 绑定、archive 首行输入元数据、计数及物理时刻；摘要相同即完整 canonical payload 相同，不再次解码旧 32 帧。只有出现新的某步摘要差异才调用原无损 decoder，沿流定位首个完整字段差异并明确失败。每窗记录比较帧数及额外解码数（成功匹配时为 0）。公开输出排除列表为空，不按 time/ns 名称泛化删字段，不截断跨世界过滤行。

仅以下测量墙钟允许不同；键和统计 count 均保留：

| 位置 | 允许不同的值 |
| --- | --- |
| 每步 manifest 行 | step_ns、geometry_snapshot_ns、public_to_end_ns、oracle_ns、archive_ns |
| 每步 counters | geometry.envelope_ns、geometry.broad_ns、geometry.midphase_ns、geometry.narrow_ns |
| manifest | declaration_ns、started_utc、ended_utc |
| window | build_ns |
| public_statistics / step_statistics | total_ns、p50_ns、p95_ns、minimum_ns、maximum_ns；先与原 32 个时长重算值一致 |

原生 kernel/传输/分配/参数包计数、应用计数、候选/接触/材料数、力、状态、全部路径和 history 均不得不同。新增、删除或改名字段会拒绝。源 HEAD/绝对路径属于另行核验的来源元数据，两边原始绑定完整保存；每侧前后必须稳定。原 benchmark/profile/cases 三工具 SHA 必须跨两版相同，实载入产品文件仅允许 collision_pipeline.py 改变。比较脚本自身另存路径和 SHA，并核对读回结束后未变。

父将先只集成 C2 再运行原九档，不混入本批其他功能的共享改动；完整同输出门通过后才解释测得的 wall clock。

## 父本批专项核验

2026-11-05父唯一串行槽：HOST 31 passed / 1.10s；interpret 2 passed / 1.11s；CUDA 2 passed / 1.85s，均RAW_EXIT=0。两设备组只选择本批两个标记tyx的用例，31个HOST例按相同文件单独完成，非漏测。

首Ruff check通过；首format-check指出3文件排版，父保存原三文件并格式化，随后独立AST逐文件严格相同，最终Ruff和format-check均0。证据位于父 artifacts/p65-postnight-batch2-20261105-01 的 c2-* 各收据。未重跑同语义已绿组。实际九档矩阵与性能仍待后续收据。