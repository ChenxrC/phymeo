# I3f：可证空候选与跨世界批量接触查询

2026-11-18；源码提交 f8d902ea45b50825f299f14f456cb62cac5f8922。codex/exp 本地，dev 未改。属于[至 07:10 的计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)。

## 做了什么

TyxAreaPressureIntersector 新增 frozen AreaPressureIntersectionRequest 和 intersect_many(tuple)。先完成整批身份、当前输入网格和原始候选容量检查，再排除严格分离的候选。严格 AABB 分离或某个 exact tetra 半空间对三角形三顶点全部为负，是空交集证据；贴面、贴边和未证空不被略过。剩余候选按固定 workspace capacity 分块，全部成功后一次更新不可变结果快照。旧 intersect、last_result、last_routes 保留单次/末请求语义，新增 last_results、last_batch_routes。

single/heterogeneous scene 从所有当前 owner 收齐 world/pair 请求后统一求交，物理仍使用既有全批事务。查询诊断不是物理接受结果：后续物理失败时 scene.last_geometry 和 accepted state 仍保留。每 query 的原始 cell×face 不超过 workspace capacity；全批原始候选不超过 2048，证空不绕过容量门。空 tuple 合法，无设备提交并发布空批。当前仍是 HOST 精确准备、设备拓扑提案、HOST 复核的混合路径。

## 验证

12 新节点＋56 受影响兼容节点，共 68 passed、0 skip：

- 静态 interpret：18 passed / 3.88 s；CUDA：18 / 3.34 s。
- 连续场景/成本、clone、末 query 故障、OPEN 准入：8 / 37.46 s。
- 原场景协议及异构当前状态/事务：24 / 55.49 s。
- 6 Python 文件最终 Ruff / format 均通过。

新测试包含有正面积的五候选按 2+2+1 分块、独立解析面积/合力矩、原 HOST 字节结果、严格 AABB/斜平面排除与贴边反例、末请求非法公开字段、第二块真实设备读回后故障、末 query 发布准备后故障、容量/类型/关闭/空批及恢复重试。初次 guard 遇外来 pytest/tsc，因此未执行；后自然释放。首 Ruff 三个 import 排版问题已修复，原日志保留，无功能测试失败。

## 同输出成本

固定原 5 world / 3 bucket / 12 步；每步包含 FK、当前几何检查、上传、裁剪、精确复核、共同求解及统一观察读回，不含模型/资产/workspace 初建。HOST 与 mixed 的完整物理状态、几何、接触输出及观察逐字节一致。

| CUDA 本轮 | HOST | mixed |
| --- | ---: | ---: |
| 首步 / s | 1.5327364 | 1.4782849 |
| 后 11 步总计 / s | 1.8306440 | 1.7464525 |

mixed 的 192 原始候选中，156 个严格证空、36 个设备拓扑精确复核成功、0 HOST 裁剪回退；提交次数 48→12，上传 18432→4608 B、下载 89088→22272 B，workspace 仍 9408 B。这里的箭头对照 I3e 同一固定案例的工作量，不是跨运行耗时比较。

本轮 mixed 暖运行约低 4.6%，只有短窗口，且绝对耗时较前次波动明显；不据此声明稳定加速。interpret 后 11 步 HOST 7.66127 s / mixed 7.56088 s，同样不作为设备性能结论。CUDA mixed 查询总计 0.76850 s，其中准备 0.65739 s，设备/传输 0.02885 s，精确重建 0.03183 s；分项不含全部发布开销、属于整步的子集，不重复相加。默认仍 HOST。

## 下一步

当前 scene 构造完整世界 mesh 后，查询准入立刻重构并再次验证同一网格。I3g 将用内部原始声明在统一 preflight 中只构造一次；公开输入仍重验，实际 stored FP64 世界几何也仍完整验证。HOST 和 mixed 同时受益。完成该项后再做固定小/中规模整步比较，不移植新的材料模型、不把本结果叫全 GPU 碰撞或训练效果提升。

证据：artifacts/area-pressure-batched-candidates-20261118-01，包括原 guard、命令、stdout/stderr、RAW_EXIT、validation-summary.json 与 scenes-basetemp 两个 complete-step-cost.json。旧 I3e 原证据不修改、不重复归档。
