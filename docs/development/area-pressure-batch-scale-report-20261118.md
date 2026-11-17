# I3h：5/20/32 world 同输出整步成本（2026-11-18）

本批完成固定 CUDA 多规模比较，**3 个案例通过**，没有扩大公共 32-world 上限。5 world 首次即通过；20 world 首次在构造时因夹具漏传 expanded-batch 显式资源预算而失败，尚未开始仿真；补上既有预算参数后只续跑 20/32，均通过。原失败和 guard 记录保留。没有改变几何、h、质量、材料或物理门槛。

## 固定范围和实际输出

四 model、三 bucket；按原五 world 模板顺序复制，32 是模板前缀，不是相同占比的线性扩展试验。每档 25 连续步，第一步单列，随后三个各 8 步暖窗。每步交替先运行 HOST/mixed；计时包括当前 FK、几何、传输、共同求解、统一观察刷新和读回。完整状态、几何、观察数组与目录逐项逐字节一致；比较本身在计时外。

| world（area/native） | 首步 HOST / mixed s | 后24步 HOST / mixed s | 本次 mixed 暖窗减少 |
| --- | --- | --- | --- |
| 5（4/1） | 1.64056 / 1.45414 | 3.25649 / 2.94129 | 约9.7% |
| 20（16/4） | 1.97214 / 1.61511 | 9.14864 / 8.17278 | 约10.7% |
| 32（25/7） | 1.80741 / 1.85473 | 14.28162 / 12.66143 | 约11.3% |

每档三个暖窗的 mixed 时间均低于 HOST，但这仍是同一机器、同一组小型模板的一次固定比较；无通用提速保证，不与 Newton 速度作结论，默认仍 HOST。模型/scene/workspace 初始化独立记录，先后次序和 JIT/cache 会影响初始化，不能将其差值当稳定性能优势。

所有 template-world305 副本在 input tick20 首次产生非空面积与正力，峰值4.305981636 N；native 世界实际接触峰值2.450003624 N。不是把末端状态反推成旧输入载荷。

| world | 全25步原候选 / 严格证空 / 设备复核 | HOST裁剪回退 | launch | 上传 / 下载 bytes |
| --- | --- | --- | --- | --- |
| 5 | 400 / 320 / 80 | 0 | 25 | 76800 / 371200 |
| 20 | 1600 / 1280 / 320 | 0 | 25 | 76800 / 371200 |
| 32 | 2500 / 1995 / 505 | 0 | 25 | 76800 / 371200 |

固定 query workspace 峰值75264 B，close后live_bytes=0。20/32 每条比较路径独立64 MiB registered CUDA payload预算，实测各304648 / 474988 B，reserved与unknown均0。预算仅覆盖已注册数组载荷，不包含context、JIT/cache、allocator、HOST、外部controls/workspaces，也不把它与query workspace简单相加冒充总显存。5沿原无显式预算路径，未重跑补计。

## 验证和证据

证据根：`artifacts/area-pressure-batch-scale-20261118-01`。
- 05：1 passed、20构造失败；5 cost在fixed-scale-basetemp，保留部分20失败文件。
- 08：2 passed、1 deselected，55.70s，RAW_EXIT0；expanded-scale-basetemp保存20/32完整cost。
- 01 Ruff0、02首format-check1、03format0；06外来执行guard拦截无RAW_EXIT，07预算修正后format0；09最终Ruff0、10最终format-check0。
- 04也是执行前guard，不算pytest失败。一次本地文本替换在写入前拒绝，随后用明确LF规则完成；不涉及产品执行。

三个cost SHA256：
- 5：39ab511e37aa849775401a4c7c2792d93a25e533f7dacbca786ecf259e1de898
- 20：260fee6a9b06c938307e9fd2b0eb79660573534e7cfdb68d840cbcb2f701ec41
- 32：26d7722fa7ca290bf7911a473de84b7107bacdc7f0ce5f703bbb533c90d4616c

5通过后仅新增>16世界预算支路和报告字段，5物理路径未改、未重跑。20/32最终源通过；测试不是所有比例/模型/长时稳定性结论。80仍未支持、未执行，未标skip或用分批冒充。

下一节点按[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)完成显式 area body-pair 任务载荷 reader；避免继续扩大同类基准代替功能接线。
