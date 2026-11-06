# 单次布–杆求力的输入共享

状态：**候选已编写 / 作者未执行**。父计划10856e531dced5589496a7e1187068bf7aad5d19；tests-first 3d2f1449583fcdf3dccec4266a6c276948de67fb。父 `01-c-valid-red` 已真实 RAW1：原有限两面上 −2/+2 N 正控先通过，随后既有 ClothLinkContacts 缺 `local_reuse` 参数失败。没有用缺新模块导入作为 RED。以下功能、专项和性能均待父串行验证。

上批 W32F8 三正式窗的排他 HOST 几何2,032,520,600 ns、材料装配1,243,796,200 ns、whole4,869,061,000 ns只是选点证据；不是本批收益，也不是GPU核时长。额外分页0.15468%的关闭结论保留，不再优化分页。

## 限定变化

新私有 `ClothLinkEvaluation(contacts, state, geo)` 只属于一次 `evaluate()`：拥有当次三角位置、按原材料ID排序的节点速度、每条 capsule 的端点/COM和 `jl@v`、`ja@v`。各侧 CollisionMotion 按需只构造一次。`sample(link_index,face_index)` 仍调用原精确静态函数；`relative_velocity(link_index,face_index,ids,weights,point)` 返回原 arm、relative，保持 `weights @ node_velocity − (linear + cross(angular,arm))` 的原运算次序。原法向/摩擦 law、逐站加力、joint effort、材料地址、完整记录均由旧路径继续负责。

父维护的开关为 `ClothLinkContacts(..., local_reuse=False)` 和 `CompliantClothContactProgram(..., cloth_link_local_reuse=False)`。默认关闭，未配置材料而打开会拒绝。对象不在 contacts、world 或 workspace 上保留，不跨 entry/midpoint/endpoint、几何变更或失败重试缓存；没有近邻匹配、弹簧历史或新材料记录体系。路径守卫 `_witness/check_path` 原行为保留，本项不扩大连续检测能力。

专项使用两种网格分辨率、非连续节点与面顺序、两连杆共享祖先、共享布节点；独立重建材料作用点/轴点、法向力、机器人点速度、反力矩及功。另验实际四步变形、质量改变的加速度/动量、坐标协变、几何/速度更新及单双站点变化、实例隔离、拥有数据、默认关闭、退化/轴穿面、晚阶段坏law回滚/修复重试。TYX专项对照完整实际步骤及原生页数/传输；不把HOST几何称作设备驻留。

## 父验证和最小性能协议

作者只改4 NEW：`src/phymeo/solvers/_cloth_link_evaluation.py`、`tests/test_cloth_link_evaluation.py`、`scripts/benchmark_cloth_link_evaluation.py`、本报告。两旧求解器文件由父负责。父可先运行该新test的 `-m "not tyx"`，再 `-m tyx --tyx-mode interpret --require-tyx` 和 CUDA 同项；每次新guard/basetemp，不重跑完整旧套。测试含独立 A/B 收据协议负控，记录缺失/不同输出/来源/计数/并发或插桩均拒绝比较。

benchmark复用上批固定 `material_phase_transfer_cases`/`multiface_load_cases`、完整序列化与限额工具，不改输入或复跑五档。只 W1F8、W32F8；每档3独立配对，次序 scalar/reuse、reuse/scalar、scalar/reuse。每个lane独立进程/cache，1cold+1warm+3formal，每窗从原初态恢复后4步；无 PhaseProbe。whole包含原几何、真实TYX材料律、积分/验证和完整公共返回，reset、离线审核、JSON/文件IO在外。5窗完整输出及history保存在新gzip，唯有既有substep timings值规范化；SHA、实际原生/layer计数、设备、source文件前后绑定保留。每端180s安全边界、32MiB原始JSON、2MiB摘要；失败在预算内存完整尝试并保原异常，绝不自动重算或改预算。

CLI（父填写合并后冻结的full SHA与新绝对路径）：

```text
python scripts/benchmark_cloth_link_evaluation.py run --worlds 32 --implementation scalar --pair 1 --mode cuda --expected-head <full-SHA> --out <absolute-new-lane.json> --cache-root <absolute-new-cache>
python scripts/benchmark_cloth_link_evaluation.py summarize --profiles <12-absolute-lane-json-paths> --expected-head <same-full-SHA> --out <absolute-new-summary.json>
```

第二端用 `--implementation reuse`；按1/2/3轮顺序分别AB/BA/AB，W1同样执行。summary要求全12端、同物理输入/完整输出/来源/设备、实际相同计数、非重叠进程时间线；分别给每端原始正式wall、p50/p95与每秒world-substep、每轮同输出比值。3样本nearest-rank p95只是最大样本。冷态另列，不与暖态相除造速度。父另由guard保证整个主机项目队列串行。

尚无本批实测收益。若无端到端收益，保留有限可选能力/工具并如实报告；即使某档获益也不自动开默认，不外推至Newton、全GPU求解或其他规模。
