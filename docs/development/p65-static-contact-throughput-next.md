# 静态接触采样吞吐：下一批边界

**RESEARCH_NOT_IMPLEMENTED，2026-11-10。** 本文仅据父源码 `b622fdf680c5c6c0b7aa130473a915b2ae2e8f68`、原 K1 未提交计划及一份父已完成的单帧 profile 作只读核对。没有运行项目、导出、数值 probe、测试或性能测量；新接口均为建议，未冻结实施合同。三个原 K1 WIP 保持原样。

## 这次证据支持什么

父 [N10 记录](../../artifacts/p65-connections-20261110-0800-01/78-loaded-frame-host-profile/result.json) 从保存的真实受载末帧读取 34 个 frame、14 个 pair、14 个 site，物理推进步数为零；重建的 wrench 与保存值逐值相同。该次 cProfile 约 0.748 s，其中 14 次 `_hull_round_patch` 下的 168 次 `segment_minimum` 累计约 0.603 s，14 次 `HullShape` 构造约 0.055 s。它指出一个可定位的静态几何工作量，**不能给出整段运行的 wall-time 占比、未插桩吞吐、GPU 收益或与 Newton 的速度排名**；也不能据此称缓存是主要瓶颈。

## 源码路径与可复用部分

[contact_patch_adapter.py](../../src/phymeo/geometry/contact_patch_adapter.py) 的混合 round/hull 采样每次构造 `HullShape`，遍历凸体面的三角扇，并为每个三角临时构造 `_FacePath` 与 `_Pose`，随后调用 [segment_minimum](../../src/phymeo/geometry/_capsule_triangle_sweep.py)。这里复用的是模块中的**静态最短距离助手**，没有启动 swept CCD。该助手保留两个端点对面的投影、穿面候选、所有三角边的 segment/segment 候选，以及全部等距极值；最终发布一个点或完整最近区间的两端，保证材料地址与接触 patch 的含义一致。当前轴穿入任意凸体的分支仍会拒绝，不能在优化时暗中改成支持。

不过，静态调用仍通过 [_FacePath](../../src/phymeo/geometry/_triangle_pair_sweep.py) 建立常数时间多项式和面积有效性检查，再通过 [_axis_features](../../src/phymeo/geometry/_round_sweep.py) 的零速度输入建立仿射参数、时间范围和特征记录。它们是复用实现带来的工作；并非每个静态候选都在执行 CCD 根隔离或二分。现有通用采样器、HullShape 和上述助手中，没有已成形的 prepared/static 公共输入协议可直接打开。

[box_link_contact.py](../../src/phymeo/solvers/box_link_contact.py) 已有精确静态 segment/OBB clamp 分段极小值 `_closest_span`，并另有轴在 OBB 内部的处理；它只适用于已验证的真实 OBB。任意八顶点 hull 不是 OBB 判据，材料地址与 containment 语义也不能靠顶点数替代。[table_box_patch.py](../../src/phymeo/solvers/table_box_patch.py) 的特定桌面 hull 缓存同样不等于一个通用、可失效的 prepared C/H 路径。

## K0、K1 与力消费之间的实际边界

| 层 | 当前源码事实 | 尚不能宣称的能力 |
|---|---|---|
| K0 accepted source | [_resident_convex_source.py](../../src/phymeo/backends/_resident_convex_source.py) 精确限定 `type(binding)` 为 `ResidentTreeBinding` 或 `ResidentStaticBinding`；[_device_convex_query.py](../../src/phymeo/backends/_device_convex_query.py) 还要求 `accepted.tree.specs`，按普通 tree 的 rows/groups 和 anchor 编译 source。 | **不支持 floating / affine / floating-affine source 输入。** 驻留动力学或接触宽相能处理这些类型，不使 K0 自动具备其布局。只删 guard 会错误读取 accepted pose。 |
| K1 query patch | [prepare_convex_query](../../src/phymeo/backends/tyx_world_batch.py) 的 `site_capacity` 已是实际可选参数；默认不分配 patch，正容量分配 patch buffer/kernel。解析路径核 bilateral 材料重建、skin、gap、权重与 source qualification；有真实 `_tyx_convex_patch_kernels.py`。 | K1 不是完全空白；但生成合格 query patch 不等于物理受力消费，也不扩展 K0 的输入域。 |
| 原 K1 WIP | [原计划](../superpowers/plans/2026-11-08-device-convex-patch.md) 与两个未提交测试/参考文件限定零物理步 query 几何、point/line/face、材料、权重、矩与资源状态。计划明确排除 full reduced solve、finish/publication、friction、CCD 及 floating/affine 扩域。旧执行摘要只作历史，不覆盖父最新验收。 | 不把未提交计划标题当成当前 CUDA 接受，也不修改其原案例库或借它宣布 K2 已通。 |
| 现行 force bridge | [_resident_contact_bridge.py](../../src/phymeo/backends/_resident_contact_bridge.py) 仍先调用 HOST `contact_patch_sample`，全部 site 计入容量，再调用力定律；零力 site 也保留，并记录双方 COM wrench / static reaction。 | K1 的 `skip_host_patch_search` 是 query qualification 字段，当前不意味着这一 bridge 已消费设备 patch。 |

原 K1 三个 WIP 是 `docs/superpowers/plans/2026-11-08-device-convex-patch.md`、`tests/collision/test_device_convex_patch.py`、`tests/reference_newton/rr/device_convex_patch_reference.py`，本研究不改它们。

## 最多两个后续节点

### 节点 A：通用 HOST 瞬时 round/convex 采样路径

先从当前热点做一个边界清楚、可独立比较的改动：将静态 segment/triangle 与 segment/segment 候选生成从时间结构中分离，保持 exact-rational 判定、全部等距候选、平行最近区间、端点/边/面材料权重、零轴退化和原失败分类。共享低级代数可以，不能用端点采样、单一最近点或容差合并来替代完整 patch。

可同时设计不可变的 prepared 静态凸体描述：绑定原 shape key、world、revision、材料 IDs、原点字节及 topology；保存已验证 hull 面、边、三角扇和静态三角数据。首批只允许真正不变的 `ResidentStaticBinding` 几何复用。移动 hull 不能按 shape ID 复用 world 顶点结果；旋转、状态 epoch 或坐标重建可能改变精确几何，应重新验证或走原路径。缓存容量、生命周期、owner、关闭与失效都必须显式有界，不能全局无界缓存。静态准备只是节点的一部分，不能用消除约 0.055 s 构造工作来承诺同时消除约 0.603 s 的极小值工作。

验收以多几何族为主：球/零轴与凸体；胶囊端点、侧面、倾斜面、边和顶点；平行线 patch、跨三角扇的同一最近区间、共面细分、近触两侧和全部零力 site；交换双方、稳定 ID/存储置换、旋转和平移协变；多个 pair 共享静态 hull；动态 hull、revision、owner 或拓扑变化时正确失效。比较完整 sites/地址/权重/维度/normal/gap/资源与失败类别，再比较原力定律下双方力矩，保留原预算与字段。OBB 快路仅在独立验证其域与地址映射后复用。先通过差异验证，再单独测几个保存姿态族的未插桩耗时和工作计数；不重跑已绿同源整组。

### 节点 B：accepted source 扩域与设备 patch 消费门槛

先补明确的 source adapter，而不是绕过 K0 限制：分别对 ordinary、floating、affine、floating-affine 的真实 accepted 布局定义 pose gather、浮根平移/旋转与 full/reduced 坐标关系；绑定 owner、graph/topology generation、state epoch、active bank、world、tick、shape revision 和原 source 精度费用。复用已有受控 source 的方法必须由实际布局证明，不能仅按继承关系接纳。

首批门槛仍为零物理步 query：各布局采用同样的多几何族、独立解析姿态和材料重建；测错误 owner、stale epoch、reset/close、跨 world、容量与 unresolved/unsupported/source-precision 路径。原 K1 query 验收由其 owner 沿原案例库闭环，不在这里重派或替换。只有输入域与完整 patch qualification 都通过，才另冻 K2 的受力消费：保持原 force law、双边力/矩、权重与能量、single publication 和晚故障语义；所有 site 在力求值前计容量，零力 site 不能删；设备解不完整时不得伪装成已消费。K2 的多步受载/摩擦验收需另列，不能由零步 query 或 G1 单案例代替。

## 优先级变化与不变部分

这次证据使节点 A 成为一个较近、可测的 **HOST 静态几何**工作项，并使节点 B 的 source adapter 成为任何 floating/affine 设备 patch 主张的前提；并未证明 GPU 路线或缓存一定更快。此前[驻留规模研究](p65-resident-contact-scale-research.md)的整世界分页 bundle 仍针对 W256 的 shape 注册容量障碍；它没有被单世界单帧 profile 否定。两者解决的层不同。先分别给正确性与独立性能证据，再决定实施先后；不抬原 cap、不改接触定律、不以 G1 特判换吞吐。
