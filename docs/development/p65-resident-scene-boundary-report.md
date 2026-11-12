# S1/S2 ParticipantScene 设备边界来源：作者交接

状态：**CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**。基线 `79e3e0892f364c3681d642f14df23f946c7617e8`，分支 `codex/p65-scene-boundary-20261112`。作者全部代码 **UNEXECUTED**：没有 import、collection、pytest、数值 probe、lint、formatter、compile、build、install 或 profile。父共享接口与后续实际资格另以父原始收据为准。

## 本片提供什么

公开入口由父接到 `ResidentSceneBatch.prepare_boundary_query(model, bindings, *, limits, memory_limit_bytes)`；结果仍是既有有限边界距离 packet，`source_kind='resident-scene-rigid-device-v1'`。当前真实架构是同一 `ResidentBatchTransaction` 下的一个 `TyxWorldBatch`，participant 映射为 private mechanical world，四 physical storage 共享同一个 runtime/accepted epoch/bank。没有把 SceneBatch 伪装成另一个 TyxWorldBatch，也没有新建或复制几何、BVH、narrow、积分器或接触求解器。

唯一新增产品文件 `_resident_scene_boundary.py` 实现已冻结的五个内部 hooks：`compile_registration(plan, model, bindings)`、`borrow_source(plan)`、`validate_source(plan, view)`、`logical_ticks(plan, view, accepted_ticks)`、`require_current(plan, result)`。构造参数为原 scene。它强持 scene 和原 layout，使用真实 service 的 `borrow_boundary_source(scene_owner, registry_fingerprint)`；父负责 loan/reentry、公共转发、QueryPlan 的 hook dispatch、设备公共 frame 转换和原资源栈。

注册逐 participant 检查 logical key → mechanical ID → actual slot → family/row/full physical spec 的一一关系、完整 model/topology/affine graph 与原 anchor；逐 shape 保留 logical CollisionKey、原 point/facet/revision、原 wrapper 和 asset provenance。原四 direct constructor 的完整 physical topology 与 M1-A 四 factory 来源都复用原 admission。静态材料必须在 `static_body_ids` 声明，不能抢占动态 body。未调用仅支持旧 convex 域的 `layout.bind_geometry`。

不同 participant 的 physical anchor 可以不同。每 logical world 选择 canonical first participant 的原 FP64 anchor；动态 shift 是 native anchor 减 canonical anchor，static shift 为零且原 static 点已相对此 anchor 打包。HOST 只处理原注册常量和读取证据检查，不计算当前 FK/最近点后回填设备。父先从 actual accepted FK gather，再用设备 `anchor_shifts` 转换。packet 保存 `native_poses`、原 `anchor_shifts_exact`、实际 FP32 常量/上下界/差值，避免已舍入 common pose 掩盖平移误差。

空几何注册仍绑定 scene 的所有 logical worlds 和全部 participant ticks，包括没有任何 shape 的 participant；不同 participant 的 tick 不一致拒绝。active loan 中的 adapter 校验不调用 `batch._check_open()` 或 `service.locked()`，避免错误触发同线程写重入门。来源结束后 lease 失效，不能留作 live FK 指针。

这是独立查询：查询失败保留旧 query publication，并不撤销先前已提交的物理步。旧 packet 在后续 physics/reset/restore 后仍 stale。资源完全沿用父 M1 lease/DeviceResources；PENDING 只允许 close 重试，UNKNOWN_RELEASE 保 owner/runtime/scene 的强引用且不再次 free。没有宣称 device-native deforming M2、CCD、solid inside/outside、持续接触身份、载荷接入或曲面 patch。

## 精确作者范围与提交

相对基线恰六 NEW，父旧 shared files 没有由作者改动：

| 文件 | 用途 |
| --- | --- |
| `src/phymeo/backends/_resident_scene_boundary.py` | 来源准入、logical/native 映射、loan、ticks/current 检查 |
| `tests/reference_newton/rr/resident_scene_boundary_reference.py` | 原 XML/OBJ/四 family 参数、独立 rigid FK 与资源字节式 |
| `tests/collision/test_resident_scene_boundary.py` | 连续运动、四来源、heterogeneous worlds、工具/物体、reset/restore |
| `tests/collision/test_resident_scene_boundary_protocol.py` | 注册/容量/原子发布/重入/ticks/UNKNOWN 协议 |
| `tests/installed/check_resident_scene_boundary.py` | 源码外公共两臂消费者，独立有限 segment–triangle oracle |
| 本报告 | 边界、预登记、交接 |

提交顺序：`49de1ec2` 三文件 tests-first；`76bb8894` NEW adapter；`312523c2` 根据执行前父 review 将 HOST release 夹具的空 participants 改为一个合法 ordinary participant（几何仍空，未放宽原 participant 域）；`d69c943c` NEW installed consumer。后续父 style 不倒灌到这些历史作者字节。

| 文件简称 | 作者工作 SHA256 |
| --- | --- |
| adapter | e13fd29d5434a72672be7dc3a9057d870e5cf5d79d3a4580e7399452dc8d21e4 |
| reference | 801d6f0d70d9a2792f53a7597ea8600b96643c12709c76d65a0b58774aaa4428 |
| motion tests | c85df5554d7f20134f927dfa7e1ee49b688e2c5d0406b2a336cc6474a5b3cc7a |
| protocol tests | 3de1bd53500ebb4bbd95362d84e098384457d8b4ccfb3fda75c5967b935d081b |
| installed consumer | 46abf3c17368edc7bb158cd2b3c9e4cad0164a32e3a13390ec16f4f26d0ff557 |

## 预登记案例与首轮顺序

手工枚举，**UNCOLLECTED**。每个 TYX 节点按同一固定输入分别由父执行 interpret/CUDA；不把枚举数当通过数。

| 节点 | 各模式数 | 必须证明 |
| --- | ---: | --- |
| four_families_cross_participant_three_steps | 8 | 四 family × asset/direct；两个真实运动 participant、不同 anchor、完整原面/round 材料 |
| heterogeneous_sparse_worlds_and_original_local_ids | 2 | 两 sparse worlds、每 world 四 family、注册正反序、原 local IDs 重复但不得串 world |
| two_arms_fixed_tool_free_object_and_static_surface | 2 | affine/floating-affine 两臂、原 fixed tool 一次变换、实际自由物体、声明 static mesh，identity/cyclic rotation |
| registration_order_preserves_logical_geometry | 1 | 同输入两独立 rebuild，按原 logical geometry 完整结果相同 |
| partial_reset_restore_and_next_real_step | 1 | 只 reset/restore 选定 world，另一 world 不变，下一真实 step |
| empty_geometry_still_binds_all_logical_ticks | 1 | 无 shape 仍查所有 participant 的同一 logical tick，零设备几何字节 |
| no_host_current_geometry_as_query_input | 1 | 禁止 service.geometry/_read_packets 向查询提供当前点 |
| registration_rejection_precedes_allocation | 8 | missing/unknown participant/body/group/family/topology/revision/static 声明；合法错误对象到 registration，禁止 allocate |
| exact_and_one_less_limits_are_whole_query | 3 | memory/candidates/groups 完整恰满与少一，所有候选先完整 prepass，不能 partial narrow |
| query_failure_keeps_prior_packet_but_not_prior_physics | 4 | 最后 BVH/meta/common-frame/packet 失败，先前物理已前进、旧 packet 保留、正常重试 |
| scene_writers_are_blocked_during_actual_source_loan | 5 | step/reset/restore/controls/close callback 不能重入 |
| participant_without_geometry_cannot_have_a_different_tick | 1 | 无 shape 的真实 participant 读回 tick 被改，完整拒绝并保原发布 |
| wrong_scene_and_full_provenance_cannot_be_current | 1 | 另一 scene owner、原注册/anchor/address/source 身份拒绝 |
| filters_use_validated_bodies_and_remain_complete | 1 | 原 same-body、same-family、mask、excluded 全 pair 表 |
| TYX 小计 | **39** | 主文件 16，protocol 23 |

另 **2 HOST-only** `test_host_unknown_release_strongly_retains_scene_and_runtime[before/after]`：真实 scene/plan/ledger 所有权图，假的有限 buffer free 在副作用前/后失败；独立 siblings 清理、24 未知 live bytes、旧 publication、runtime/scene 强留存、close 不重复 free。它们不是实际 GPU 几何结果，不重复旧 M1 的完整 CUDA free 故障矩阵。

建议父首轮：两个 HOST 所有权节点 → ordinary/direct 三步 → 四 family 与异构 frame → protocol 完整剩余 → CUDA 同固定矩阵；原 M1/M1-A 默认 adapter 路径仅选受共享分支直接影响的 direct/static/empty/current/close 邻居，已有相同未变绿组不无条件重跑。全部计数以首次 collection 与原 raw receipt 为准。

## 独立 oracle 与资源

reference 不导入产品；原 F0 双 sibling slider、1/2 kg、mimic −1/+0.004、真实 root/关节非零速度、h=1/1024 均执行前固定。四 family 真实 checkpoint 只提供 accepted q/v，原材料与 rigid transform 用独立闭式重建；候选完整性/三角距离/材料 witness 沿既有 COL2/M1 独立 exact reference。没有产品 gather/closest-point 自证。原 distance≤1e-4 L、witness≤2e-4 L、normal≤2e-3 不变，L 为原 pair local edge/axis/radius 长度，不含世界平移或 participant anchor 差。

相对原 M1 每 shape 新增 36 persistent bytes（FP32 S×9 shift center/lower/upper）和 48 query scratch bytes（FP32 S×12 native pose）。预登记：

```
Bpersistent = 68 S + 36 N
Bglobal = 112 S + 36 N
Bgroup(F,R) = 40 F + 48 (2F−1) + (52 + 4 ceil(F/30)) R
Bpeak = Bpersistent + Bglobal + sum(Bgroup) + 1060 sum(P)
```

精确资源例 S=2,N=5,F=4,R=1,P=4，两个不同真实 participants。空几何持有完整 logical source 元数据/tick readback但无 geometry device buffer；budget=1 合法，0 仍按原 API 拒绝。原 shape/source/provenance 数组由父 `_nbytes` 计 HOST array bytes；非 Python RSS/驱动峰值，不能把 registry 中 bytes 的 Python 开销说成零成本。

全部保存原 input/registration/wrapper/常量差值、实际 physics 前后/返回、source/raw BVH/pack/narrow/typed packet、所有 attempt prefix 与失败。序列化 registry 中原 bytes 与 ndarray 的完整 dtype/shape/bytes，不丢 NaN 原表示；实例对象用本记录 opaque identity。

## 安装消费者

`check_resident_scene_boundary.py --mode interpret|cuda --installed-root ABS --output freshABS`。自包含原 URDF/OBJ、两个 sparse worlds(17,83)，每 world ordinary + floating-affine 两个真实 arms。ordinary 原 asset mesh 对另一个 participant 的原 asset sphere 与 direct finite capsule；实际不同 anchors，全部原 facet/local IDs 保留。三连续物理步预计 6 logical worldsteps / 12 participantsteps，然后 selected reset/restore 和一次新 query；共 4 queries / 64 rows / 3 stale 拒绝。

公共数据和原资产原文构造 oracle，不读取 binding 私有来源作期望。有限距离 oracle 独立比较端点→三角、段→三条有限边、穿面候选；保原距离/witness/normal门。每次物理调用前、实际返回后、snapshot 后分别持久写 journal，query outcome 在断言前保存，失败/close 同样保前缀；一个失败查询不声称 physics rollback。所有 loaded Phymeo module origins 必须来自指定安装目录，禁止源码 tests/examples imports。完整 JSONL.gz 与 summary 保存，源码外执行仍待父。

## Parent source qualification — 2026-11-12

Root serial qualification completed on the integrated source: run 014 has 41 passed (39 TYX interpret cases and 2 HOST release cases), and run 015 has 39 passed / 2 HOST-only deselected on CUDA. Both RAW_EXIT values are 0. The first run 013 stopped at a TYX lowering error because the newly added frame-translation kernel contained a Python docstring; changing that statement into a comment preserved the arithmetic and allowed both full backend selections to complete. Initial style findings and their raw sources are retained.

The tested scope is logical participant ownership, real accepted device FK, common-frame finite boundary queries, and their transaction/resource behavior. This is query support, not new contact-force response or continuous collision detection. Source-external wheel qualification is still pending for Batch A. Evidence: `artifacts/p65-three-track-a-20261112-01/{013-scene-interpret-first,014-scene-interpret-repair,015-scene-cuda}`.
