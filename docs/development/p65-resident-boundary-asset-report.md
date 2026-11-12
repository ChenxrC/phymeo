# M1-A 原资产边界来源：作者交接

状态：**CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED**。作者分支 `codex/p65-boundary-asset-20261112`，基线 `b66504f3c82c868fddf3830bb0f9b2260bd3fd7b`。本报告记录作者交付，全部作者代码均 **UNEXECUTED**；没有执行 import、collection、数值、pytest、Ruff、formatter、build 或安装。父在独立集成树执行的结果以其本批原始收据/总结为准，不能倒灌为本分支已验证。

## 范围与当前行为

本片把四类原 `from_asset` 材料来源接到既有 M1 的 accepted FK → device bounds/BVH → 原有限 triangle distance 核。入口、查询、margin、limits、资源预算与原 source epoch/原子发布/lease 都不增接口。不是新的几何核、当前 HOST FK 上传、ParticipantScene 多 bank、变形 M2、CCD、闭体内外或载荷接入。

准入为唯一新产品文件 `src/phymeo/backends/_resident_boundary_asset.py`，公开调用仍是原四 binding 的 `from_asset` 与 `TyxWorldBatch.prepare_boundary_query`。私有函数：

```python
admit_boundary_asset(shape, binding, *, physical_spec, owning_spec)
# BoundaryAssetAdmission(group_index, local_points_m, provenance)
```

检查 exact 四 binding family、真实 physical/owning spec 类型与对象归属、原 source shape 全部字段、原完整 FP64 model token、两个 affine graph、实际 body group 和完整有限原材点。返回独立 bytes-backed FP64 原局部点与 immutable provenance；不做 FK/最近点/设备工作，不重复 factory 的 fixed-tool frame 变换。原 factory 的 topology=None 仍保留，父 source definition 使用已经验证的 physical spec topology。

provenance 八字段为 schema、binding_family、source_shape、model_token、graph_token、physical_topology、group_index、local_points_m，schema 为 `phymeo-boundary-asset-admission-v1`。原未重映射 asset key 与本次 scene key 均保存。新增数组位于原 `_source_base` 递归计数路径，不称 HOST 零成本。

父独占并实施三处共享增量：两个基础 `from_asset` 工厂只在局部允许集合中加入 triangle/mesh，及 M1 source 对 asset 分支调用本 adapter、保存 provenance。作者没有修改这些文件；旧全局 `_FINITE_SHAPES`、旧凸体 compiler、四 storage/kernel 全部不动。真实 URDF/MJCF OBJ mesh 保完整原面；不把 box/hull 改成 mesh。程序 triangle 明确来自原 RobotAsset/LinkCollision，不声称 URDF 新增 triangle 导入。

## 提交顺序与精确作者文件

| 提交 | 变更 |
| --- | --- |
| d277dc91 | NEW 已冻结资产准入合同 |
| ba3114fa | tests-first：NEW 独立原 XML/OBJ/frame reference 与 NEW 多案例 tests |
| d3de4cd1 | NEW 注册准入与原 provenance adapter |
| ebc3287d | 负控逐项失败 stage/reason、分配前门与原状态保留 |
| 2c48960a | 按实际父 guard 顺序区分两个 affine graph 的错误文本 |
| 8bfab631 | NEW 自包含源码外公共消费者 |
| 3372e5f2 | 消费者原资产公开数据期待、物理步执行前/返回后/后状态分别留证 |
| 49e60d4a | 消费者完整四面 BVH stop node=2F−1，未误作 valid 字段 |

作者相对基线恰六 NEW：

1. `docs/development/p65-boundary-asset-contract.md`
2. `tests/reference_newton/rr/resident_boundary_asset_reference.py`
3. `tests/collision/test_resident_boundary_assets.py`
4. `src/phymeo/backends/_resident_boundary_asset.py`
5. `tests/installed/check_resident_boundary_assets.py`
6. 本报告。

前五文件作者工作字节 SHA256（父后续 style commit 应另外绑定其 source，不改写这些历史字节）：

| 文件 | SHA256 |
| --- | --- |
| contract | 10658463f342992fb26c0d1f66e376e6873b41562586ae1b86810c7c3b2b23fe |
| reference | 5a17350d27297b7f49e8f375c54907c60bd249abc64b63256bfa25f602e1b605 |
| tests | cbdb54c4260fec76ed7492e4e69a451c7a219bc6b0ba1b0fdb461501b250700a |
| adapter | 3b04d2bd71b96482b2f3ac75f6221d51fa60b57abb676970b66821dbd08c06fb |
| installed consumer | fcd695d956bb52c49f55033e8b183c47f5322a67dbd486ecd2f074603f0d7d83 |

## 预登记节点（手工枚举，UNCOLLECTED）

| 函数 / 参数 | TYX 各后端节点 | 内容 |
| --- | ---: | --- |
| asset_surface_round_three_real_steps：4 layouts × 3 variants | 12 | plain OBJ、fixed-tool 一次 frame、program triangle；原材点与真实 accepted q 的独立有限几何 |
| four_asset_families_sparse_worlds_reordered_and_reset | 1 | worlds 83/17/107/31，混合四 family、反 registration/storage 次序、三步、selected reset/restore 与下一实际步 |
| real_mjcf_capsule_material：fixed/floating | 2 | 原 MJCF capsule 两轴端点；fixed 原 OBJ，floating 当前支持的 hinge capsule + 既有 static surface |
| wrong_public_asset_model_rejected_before_allocation：4 layouts | 4 | 公开另一资产的原模型错误，注册前完整拒绝 |
| wrong_original_graph_rejected：2 affine layouts | 2 | 原 full model 相同、mimic graph 不同；分别保父既有前置 guard 顺序 |
| asset_material_and_installation_rejections：9 faults | 9 | revision/radius/IDs/winding/none-direct/body-alias/hull 注册；missing-face 为 source constructor；source-key 为 factory |
| asset_resource_failure_preserves_publication_and_retry：4 faults | 4 | 全候选 exact/one-less、memory 少1、最后 meta/packet 故障、保旧结果与恢复重试 |
| asset_original_filter_table：3 filters | 3 | 同实际 rigid body、mask、excluded，全 pair 表保留 |
| 小计 | **37** | interpret/CUDA 同一矩阵，不把参数枚举当实际通过 |

另有 **2 HOST-only**：`test_factory_mesh_does_not_open_old_convex_compiler[False/True]`，证明两基础 factory 的 mesh 准入没有开放旧凸体 compiler。无额外 CUDA-only 故障节点，也不重复旧 UNKNOWN_RELEASE 全矩阵。

所有注册负控先构造合法 wrong-source 对象，再进入相应 `raises`，并禁止 M1 `_allocate` 被调用；三面 closed mesh 确实在 `CollisionShape` 自身被拒绝，明确不计作 adapter 拒绝。公开错误 source key 在 factory 被拒绝。未要求抵御 `object.__setattr__` 恶意伪造 private 字段，该对象安全边界不在合同中。

## Oracle、资源与失败证据

reference 只导入 pathlib/NumPy，保预登记原 OBJ tetra、原 XML、F0 物理数据、fixed-tool 的已知 R/P、独立 quaternion 与 sibling slider pose。世界材料由真实 checkpoint 与独立原素材重建，有限候选/距离沿已验 M1 独立参考，未调用产品 FK、collision_motion、gather 或新 adapter 作 oracle。floating MJCF importer 的范围没有扩大为 mesh importer。

所有 h=1/1024、原质量/惯量/几何均固定。原工程门 distance≤1e-4 L、witness≤2e-4 L、normal≤2e-3 保持；L 为原 pair local length，不含 world 平移。near-parallel/zero-normal/tie 与原 kernel 资源门不变。

无新增 device buffers/launches。仍为：

```
Bpersistent = 32 S + 36 N
Bglobal = 64 S + 36 N
Bgroup(F,R) = 40 F + 48 (2 F - 1) + (52 + 4 ceil(F/30)) R
Bpeak = Bpersistent + Bglobal + sum(Bgroup) + 1060 sum(P)
```

asset exact 预算负控固定 S=2,N=5,F=4,R=1,P=4，使用执行前既有独立 formula，不能从输出峰值倒填。provenance 新 FP64 数组由原 `_constant_bytes`/packet `_nbytes` 另外计真实 HOST array bytes；不是 Python RSS 或 GPU 驱动峰值。全部实际设备 FK/gather/BVH/pack 与父 PENDING/UNKNOWN 关闭语义原样复用，本次不复制或改该资源实现。

保存完整 source/输入/packet/raw/注册 provenance/物理前后状态；错误 raw prefix 与旧 publication 保留。负控保存其实际阶段/文本，但没有查询的 constructor/registration 负控不伪装产生物理轨迹。

## 安装消费者

`tests/installed/check_resident_boundary_assets.py` 自包含原 XML/OBJ、两 sparse worlds 7/4096、ordinary factory source、三真实连续步。每 backend 预期 6 worldsteps、3完整queries、24 facet rows，两个先前查询在下一物理步后 stale 拒绝。四布局/MJCF/affine graph 完整矩阵留在源测试，不在包外重复。

消费者只导入 installed public Phymeo API，不导入仓库 tests/examples。证明期待取自原 public asset.collisions/model，而不是 binding 私有字段自比；元数据 token 只按已公开 schema 从原 full model fields 解码。有限 point-triangle 参考独立比较 interior projection 和三条 closed edges，保全部原面和原材点。HOST 几何仅用于检查结果，绝不作为 device 当前几何输入。物理步前记录 before/h，返回后立即记录原 result/计数，再独立捕获 snapshot；查询失败与关闭失败均保存原始异常和 attempt prefix。

CLI（仅父安装验证时执行）：

```
python ABS_OUTSIDE/check_resident_boundary_assets.py --mode interpret --installed-root ABS_SITE --output FRESH_ABS_OUTPUT
python ABS_OUTSIDE/check_resident_boundary_assets.py --mode cuda --installed-root ABS_SITE --output OTHER_FRESH_ABS_OUTPUT
```

输出 complete `resident-boundary-assets.jsonl.gz` 与 summary.json，原数组 dtype/shape/bytes、原 float bits、全部 installed module origins/hash、来源 XML/OBJ 与查询计数均保存。消费者/cwd/output 必须在 checkout 外；全部已加载 Phymeo modules 必须来自明确 installed-root。旧 editable 源路径不被白名单绕过。

## 父后续最小资格与限制

父唯一串行 owner：先新2 HOST，再 `test_resident_boundary_assets.py -m tyx --tyx-mode interpret` 全37与CUDA全37；首静态/必要排版、原失败和最终 source 同次收据由父保留。作者不运行这些命令。

原 direct/static 表示被共享 source 分支触及，建议仅旧 `test_four_layouts_three_real_steps` 的四 layout × static/dynamic 八节点/后端作为受影响接缝；旧 convex mesh 拒绝已由本批两个 HOST 覆盖。其余原几何核/BVH/资源全矩阵没有新行为，复用既有接受证据，不无条件重跑。任何父新增具体 finding 再定义有依据的受影响节点。

完成 authoring 不是正式功能接受。ParticipantScene logical world/participant 多 bank、M2变形源、CCD、有限载荷响应、完整 closed-solid 判断均未由本片获得。父接线+原始运行/保存/包外检查最终通过后，才能按本合同有限来源范围接受。

## 作者交接期间的父串行进度（消息回报，不由作者重跑）

父已回报 HOST-only 两节点通过。首完整 interpret run024 在13节点通过后，固定根 MJCF capsule 的独立参考地址顺序发生首失败：原 fixed importer `_fromto` 使用 z=(start−end)/length，而 `_robot_asset` 以 center−/+half 生成顶点，因此其原 material IDs 对应 **end,start**；floating importer 直接保 fromto，对应 **start,end**。设备发布的两轴端点完整，不能把静态几何轴的相同集合误说成产品丢点或错误运动。

父仅在 NEW reference 增加固定根显式 `FIXED_CAPSULE=((0,.5,.5),(-.25,.5,.5))`，固定根测试改用该原 importer 地址约定；floating/CAPSULE、原 XML、产品和原工程门不变。首失败原 raw 保存，已通过13节点不重跑；剩余24 interpret 和完整37 CUDA由父继续。此增量不在本报告上表作者历史 SHA 中，最终验证必须绑定父修后 commit。作者不编辑已移交的三源文件，不凭这条回报宣称完整矩阵通过。

## 父源码资格：2026-11-12 当前批次

状态：**SOURCE_QUALIFIED / INSTALLED_PENDING**。父共享接线及最终 source 以本节所在提交为准；上文作者 UNEXECUTED 与旧 SHA 是历史，未倒灌为测试结果。

- run023 两个 HOST 节点通过；interpret 原37节点分 run024 已过13、run027 已过5、run028 剩余19完成；run029 CUDA完整37通过。已过同源节点没有重复执行。
- run027 的第二个首错仅是 floating-affine wrong-model 负控的期望文本：原 full graph token 包含 model，既有前置检查先报“boundary floating affine graph differs”。测试按原 family 的准确拒绝阶段检查；产品检查顺序、模型绑定和物理门不变。
- 受影响旧 direct/static 八节点：run035 interpret 8通过，run037 CUDA 8通过。run036 的 basetemp 错写为未创建的035-cuda目录，pytest setup报WinError3、没有物理步；原日志保留，037只修运行目录。
- 新三文件首Ruff/format失败、import排序及排版均保留；最终检查通过。消费者首排版失败亦保留，run033/034消费者与修改后测试静态检查通过。fixed-capsule参考修正、floating-affine错误文本修正均未改原XML、几何、产品或精度门。
- 证据位于 artifacts/p65-parity-motor-asset-20261112-01，独立源码/消费者审阅见 [审阅记录](p65-resident-boundary-asset-source-review.md)。作者与审阅槽已释放；父仍持唯一串行构建/安装验证槽。

本节只接受四类资产到原M1设备查询的源码资格；统一wheel及源码外消费者尚待完成。不包括ParticipantScene多bank、M2、CCD、接触力或闭实体内外。
