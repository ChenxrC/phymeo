# 面积压力接触：2026-11-18 至 07:10 执行计划

起点：codex/exp d385760d；北京时间约 00:25 开始，07:10 结束。dev 不变、不 push；原 root 为唯一项目执行 owner，原两 agent 可并行只读审阅和隔离测试编写。全部数值、import、pytest、静态检查和构建串行，执行前 fresh guard。三个既有未跟踪文件保持。

## 当前依据

当前状态/FK → HOST 有限接触面 → 共同 TYX 动力学 → 统一观察已经打通。I3e 混合路径由设备提出裁剪拓扑，HOST 精确复核后发布。固定 5 world / 3 bucket 的 12 步 CUDA 比较中，后 11 步 HOST 0.9153 s、mixed 0.9268 s；192 对有 156 次 HOST 回退。尚无提速依据，不能把设备 kernel 运行称为全 GPU 碰撞或性能胜出。

## 顺序与分工

| 节点 | 难度 | 交付 | 验收 / 决策 |
| --- | --- | --- | --- |
| I3f，候选筛选与批量提交 | 中高 | 可证明空候选、跨 world/pair 请求批次、固定 workspace 分块，single/hetero 接线 | 几何/压力/状态与原 HOST 同输出；贴边不漏；末 query/chunk 故障不发布部分结果；容量和 lifetime 明确 |
| I3g，减少不变资产重复准备 | 高 | 根据 I3f 分项成本选择，建立 owned 资产的准备与失效机制 | 不信调用者 private caches；当前位置仍重算；reset/restore/clone 与材料身份保真；有实际成本收益 |
| I3h，固定多规模成本比较 | 中 | 同模型、同物理输出、同观察范围的小/中规模矩阵 | 包含 FK/几何/传输/求解/读回；报告初始化与稳态；无收益保留 HOST 默认，不调案例求绿 |
| 后续功能节点，按余量选择 | 中高 | area body-pair 观察的应用读取，或当前表示的通用场景入口 | 必须补真实缺口；不同时引入 FFT 材料、摩擦和另一响应体系；完成独立验收后提交 |
| 收尾 | 低 | 更新报告、续作入口、当前架构/差距概览 | 06:35 起优先关闭在途批次；06:50 后不新启预计无法在截止前结束的大组，07:10 汇总 |

root 负责共享源码和串行执行。multitask_scene_review 只读检查数学/事务/资源；multitask_training_review 在原隔离树编写下一批新增测试，父审阅并导入后执行。不得重启已验收旧组或创建新的周期任务。

## I3f 固定合同

新增 immutable request 和 intersect_many(tuple)。每 query 保留原 workspace capacity 的总 cell×face 上限；整个请求批次总 pairs 不超过现 MAX_CANDIDATE_PAIRS=2048。全部请求先检查身份、原输入网格与容量，再进行设备工作；非空候选按固定 workspace capacity 分块。严格 AABB 分离或 tetra 的某个 exact barycentric plane 对 rigid triangle 三顶点均小于零，才可证明 empty；贴边/未证空结果沿原 exact_host/reject。不存在静默截断、概率丢候选或改变物理误差门。

全部 query 成功后同时发布 last_results / last_batch_routes。旧 last_result / last_routes 保留最后一个 query 的接口语义，诊断 counters 不作为物理状态。single 与 heterogeneous scene 先从当前 owner 搜集请求，再整批求交；物理仍使用原全批事务。不得缓存旧状态或跨 world 混用材料。

## 避免循环

1. 每一批先固定语义、测试和比较范围，再执行；已通过同源码不重复运行。
2. 功能正确但整步无提速：定位最大的实际成本，只做一个有依据的改进再比较；结果仍无收益就记录边界，转向功能缺口。
3. 不为通过修改质量、几何、步长、预算或精度；不以理论证明代替实际工程功能，也不为不可观察的精度无限扩展证明。
4. 测试失败先保留原始证据、定位并修复对应原因；不能通过新增大批近似重复测试掩盖失败。
5. 到期只汇报实际完成；未完成节点保留明确起点，不标全部 Phymeo/P6.5 完成。

相关依据：[I3e 报告](../../development/area-pressure-device-topology-report-20261118.md)、[当前续作](../../development/p65-continuation.md)。

## 01:00 执行前调整与下一验收配置

- I3f 已完成：f8d902ea；12 新＋56 受影响兼容节点全部通过。156/192 原候选得到严格分离证据，4 个 world 查询合并后每步一次设备提交；默认仍 HOST。
- I3g 已进入验证：内部 raw 声明在本步统一构造一次完整 mesh，HOST/mixed 同时去重，公开 mesh 输入仍全量重验，不跨步缓存。
- I3h 在运行前读到公共 MAX_WORLDS=32，因此固定为 **5、20、32** world（80 不受支持，不执行、不标 skip、不分批冒充）。不扩容量或改 model/h/material。按原五世界顺序循环复制，32 使用前32条，保存完整 world→model 清单；每档只比较自身 HOST/mixed 相同输入，不以不同模板占比推断线性扩展。
- I3h 固定 CUDA、capacity=64、25 连续步（首步＋3段各8暖步），每步交替路径先后，比较完整状态/几何/统一观察字节。模型声明的共用准备耗时、各批次/scene/workspace setup、整步含读回时间分别记录；不设速度通过门。若仍无稳定收益保留 HOST 默认，下一功能节点为显式 area body-pair 载荷 reader，同帧消费和checkpoint身份；不把 reader 接通说成 Env 或训练已经接通。
