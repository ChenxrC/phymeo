# 转动 mesh–round 的逐采样面候选

状态：**FOCUSED_HOST_INTERPRET_CUDA_VERIFIED**。2026-11-07 父已完成新专项和四个旧hooks集成；本阶段共享回归和源码外wheel仍待A/C就绪。下文作者交接合同保留为历史，其UNEXECUTED描述由末节实际记录更新。

基线 `2684ba9c6664ffb84125a902ac1be93ac5320595`。tests-first 为 `39fd8b9`，扩展 HOST 合同为 `585b714`。父的 `artifacts/p65-generic-collision-20261107-01/03-b-first-red` 已保存有效 RED：绑定 `585b714`，1 failed / 4.08 s / RAW1；真实中途穿入、独立首触区间和完整输出对照先通过，失败仅是旋转分支未调用面级候选（`calls` 为空）。未重复旧几何验收或旧性能组。

原计划 HOST 文件名与已有网格–网格测试冲突。父裁定本批使用 `test_rotating_round_mesh_candidates.py` / `test_tyx_rotating_round_mesh_candidates.py`；旧 `test_rotating_mesh_candidates.py` 保持原样。

## 范围与原有义务

显式 `CollisionWorkspace(..., mesh_midphase="bvh")` 扩展到既有单转胶囊、单转网格–球/胶囊、双独立转轴及移动枢轴的原支持域。默认 `all_facets` 保持。每个实际旋转距离样本通过既有 HOST 或 TYX BVH 取得原始面索引；有限距离、材料见证、包含和连续时间判定仍是 HOST。TYX 标记仅表示真实面包围盒 refit/traverse，不表示全设备窄相或整条物理链驻留。

`MeshRoundArcCandidates(mesh, selector).sample_axis(ends, *, radius)`：

- 输入是一份已验证 ClosedMesh 的初始材料坐标系，以及绑定同一 owner、顶点顺序、定向面和当前坐标的 MeshRoundFaceSelector。
- `ends` 为同一时态的一点（球心）或两点（有限胶囊轴），保留现旋转采样器给出的有理坐标，不先降精度。全顶点到有限轴的最小精确距离给最近边界距离上界；复用原 selector 的向外 FP32 盒、候选类型/范围/空集检查。
- 返回排序唯一的原 facet 索引。必须包含全部最近并列以及半径内边界；半径内没有面时仍须返回最近面。每次轴变化重新查询，不将前一步索引或时间索引当作身份/全程安全集合。
- 主查询保留 rotate_enclosures 误差、最终存储舍入误差、速度上界、first_bounded_contact 原整段排除式及时间/空间/样本容量。存储后的轴与之前有理轴不同则重查，不能借用端点弦或已采样候选集合排除未采样运动。
- ClosedMesh.inside、全顶点 farthest/大圆体包含支持、原横穿与擦触判断仍全量。空候选是具名未决；实际设备坏回读在原 callback 边界关闭工作区，旧 last/layout 不发布半份结果。恢复必须新建 owner 后重试。

本批不新增 kernel、时间 cell 批候选、材料历史、接触力律或求解器，也不更改默认入口。

## 父维护的最小接缝

父独占 `collision_pipeline.py`、`_rotating_capsule_mesh.py`、`_rotating_mesh_sphere.py`、`_dual_turn_mesh_round.py`：

1. 三个转动 query 构造器接末尾 `*, face_selector=None`。有值时创建一次 `MeshRoundArcCandidates(mesh, face_selector)`，仅在本 pair 查询期间存活；默认不创建。
2. 两个 `_minimum(ends)` 用 `sample_axis(ends, radius=self.radius)` 缩小原 facet 循环；球的 `_point_nearest` 同样只接该原面子集。原有限距离、下界排序和 break 判据不变；双转轴继承的 `_minimum` 消费相同字段。
3. 三个 `_pose` 中实际生成公开材料见证的 `MeshRoundQuery(..., face_selector=...)` 沿同一 selector。仅用于准备原 triangle charts 的构造不需要筛面。
4. pipeline 允许既有合法转动 mesh–round 建立当前 selector，并传给相应 query；现 callback/owned binding/异常关闭协议不变。无支持路由不得仅靠候选包围宣称有 CCD。

作者只新增四个文件：`src/phymeo/geometry/_rotating_mesh_candidates.py`、上述两个测试文件、本报告。父后续接缝尚须在同源资格中验证。

## 冻结测试与真实设备边界

HOST 新专项覆盖：双轴中途入层/球/移动枢轴/杆身首碰；单转胶囊端球全程净空和单转网格；108面实际精确距离输入变少；凹槽及解析全程净空、初始内含/大球包网格、孤立擦触；材料重排/反序/轴循环/镜像；角度与样本资源、坏拓扑/跨世界过滤；精确原面 oracle、同样本全部最近并列、逐时态重查、错 owner/坐标/面绑定、坏轴/空候选；容量与调用者事务中止、同 key 新姿态 refit、原输入重试。

设备新专项分别使用仓库 `tyx_mode`：真实双转移枢调用；五个共址但隔离 world（非连续键、重复局部 ID，含108面）与同设备 all_facets、HOST 的完整公开输出比对；三种真实 `tree_hits` 异常/形状错误/越界 bit 后原发布保留、工作区关闭及重建重试。禁止 `BoundsTree.host_hits`，保存实际 face counts、refits、完成核、传输字节/次数和同步；所有双侧材料点在同观测时态独立重建。

首 RED 节点：

```text
python -m pytest tests/collision/test_rotating_round_mesh_candidates.py::test_turning_pair_queries_real_face_candidates_after_independent_hit -q -x
```

最终资格入口（由父串行，不代表作者已执行）：

```text
python -m pytest tests/collision/test_rotating_round_mesh_candidates.py -q -x
python -m pytest tests/collision/test_tyx_rotating_round_mesh_candidates.py -q -x
```

第二个入口按父标准环境分别选择 interpret/CUDA。实际收集实例数、运行结果和源码绑定以父收据为准。当前没有性能结果；每采样一次设备提交可能更慢，功能正确不等于加速，不更改原 CCD 门或默认开关以换取吞吐。

## 父实际验证与集成记录（2026-11-07）

父基线d673a1f，作者最终4392249；新增4文件和父4旧hooks共同验证。父只对候选做import顺序/排版修正，首次Ruff 11项I001及format 9文件均保存，最终check/format通过。对面表反序被构造器规范化的无效负控，在运行前改成真实同quad对角线翻转，先证合法ClosedMesh再测试绑定拒绝。

- 原首RED：03-b-first-red，raw1，实际first-hit前提成立而face callback为空。
- 挂接后首GREEN：06-b-first-green，1 passed / 2.88s / raw0。
- 完整新HOST：13-b-host-complete，39 passed / 44.78s / raw0。
- 真实interpret：14-b-interpret，5 passed / 29.50s / raw0。
- 真实CUDA：15-b-cuda，5 passed / 22.35s / raw0。

上述三组完整运行的7份source before/after/current均相同。所有已通过同源组不再重复。五world用例各保存完整expected/actual公开结果，并在本次测试内逐项核等；CUDA实际299面查询、299 refits、598完成核、324048上传bytes、53960下载bytes和299次同步。没有HOST BVH fallback。包含候选容量拒绝、三种设备坏回读、原发布保留/关闭/新owner重建。

原RD作者对父四旧hooks做纯文本交叉审阅，未发现新问题；复合旋转、stored轴重查、全量包含/远点/横穿、时间/误差预算和默认all_facets均保留。完整stage的另一作者审阅与组合wheel仍待后续，不在此处宣称已完成。

收据在 artifacts/p65-generic-collision-20261107-01；b-qualified-source-output.json汇总同源绑定、真实输出文件SHA及计数。功能门通过，不宣称性能胜过旧枚举或Newton；每采样提交的成本保留为后续实测问题。