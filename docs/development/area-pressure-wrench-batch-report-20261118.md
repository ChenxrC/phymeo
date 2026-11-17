# I3m：同帧多物体接触力矩，一次读回

2026-11-18，codex/exp 源码01afd760；8新节点（7 HOST＋1 CUDA）和8受影响旧HOST节点全部通过，Ruff/format通过。dev不变。

`BodyContactWrenchReader.measure_many(frame, requests)`接收1..1024个有序 `BodyContactWrenchRequest`，一次工作区读回、一次完整面积ABI/目录验证，返回原顺序的全部measurement。单measure也走该路径，保旧结果。重复请求可用，参考点为独立不可改tuple；全部world/body/group/参考点先验证，坏末请求零读回。读回后的相关矩缺省或数值溢出会整次拒绝，不返回前缀或留下跨帧缓存。

```python
from phymeo.control.body_contact_wrench import (
    BodyContactWrenchReader, BodyContactWrenchRequest,
)
reader = BodyContactWrenchReader(workspace, area_identities=identities_by_model)
queries = (
    BodyContactWrenchRequest(41, "object", (0., 0., 0.), ("support-body",)),
    BodyContactWrenchRequest(41, "support-body", (0., 0., 0.), ("object",)),
)
object_load, support_load = reader.measure_many(current_frame, queries)
```

参考点仍由调用者绑定同一入步，不自动称COM；返回只有接触力/矩。导出尚无历史的world返回None，与有效但零接触的measurement区分。native保原力偶，area保已求解力、真实dA及材料/sample身份，不再乘面积，也不伪造geomID。不是零HOST读取的策略特征接口。

实际CUDA使用原I3j三world短场景，前两步＋paired restore后一步，共3次step；7个查询的many与singles逐字段完全相同，公开readback与area validator计数均1对7。选择reset使该world回到无历史，其他world保持；旧frame拒绝，恢复和重放一致。没有重跑I3l450步窗口，也没有把7倍少读回等同7倍整步加速。

HOST独立手向量覆盖双方同参考点反作用、分组/余下接触、细分面积不改变力矩、顺序/重复、caller数组所有权；末请求元数据/真实矩缺省/溢出/完整ABI坏值、foreign/stale、invalid/empty history与无跨帧缓存均已验证。

证据 `artifacts/area-pressure-wrench-batch-20261118-01`：03新完整8pass/20.15s/RAW_EXIT0；04旧HOST8pass/1.22s/RAW_EXIT0（1个旧CUDA有意不重跑，新CUDA覆盖已改批量接口）；05/06静态通过。源码执行前后SHA相同，只有执行前排版，无数值失败/参数调节。同次CUDA完整gzip151159B，SHA256 acd5ad4f57561be0dbe26decb708494536f5b5dd0ba356ae2470ac2cbe6e0bcd，清单在validation-summary.json。

下一继续[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)的I3n资产可携带声明，再按完整当前路线选择真正未覆盖的场景/接触功能。