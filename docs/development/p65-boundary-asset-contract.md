# M1-A：资产材料来源准入合同

状态：**接口与案例预登记；待父裁定/作者实现/串行验证**。基线 `b66504f3c82c868fddf3830bb0f9b2260bd3fd7b`。本片把原资产材料接入已经验收的 M1 accepted FK→device bounds/BVH/pack→有限边界查询；不新增几何算法、物理推进或 ParticipantScene 多参与者银行适配。

## 1. 已核实的缺口与源码证据

- `_resident_boundary_source.py:204–214` 只接受 direct 完整 topology，随后明确拒绝 `_source_shape`。四类 `from_asset` 恰设置 topology=None，同时保存原 FP64 `_model_token`/`_source_shape`；两个 affine 家族另外保存原 graph token。把 None 换成当前 checkpoint topology 会丢失这条来源证明，不能作为实现。
- `_resident_tree_binding.py:86–126` 与 `_resident_floating_binding.py:62–94` 已做原 collider/shape membership、固定工具 authored→physical frame 一次转换、真实 group 归属。affine 两 factory 分别委托它们（`_resident_affine_binding.py:22–31`、`_resident_floating_affine_binding.py:35–48`）。新准入不重复该 frame 变换。
- 资产 **确实保有原 mesh，不是只有 hull**：`builders/urdf.py:153–159` 读取 OBJ；`_robot_asset.py:135–140,164–177` 保留 `mesh`、原 triangles 与原局部 vertices，并用 ClosedMesh 检查原闭面。MJCF `builders/mjcf.py:196–216,254–255` 同样支持 OBJ mesh 与 capsule。box 是 hull，不能把 box/hull 偷换成 mesh。
- 阻断 mesh 的是两基础 factory 使用 `_FINITE_SHAPES` 的准入；父只在这两个 factory 把允许集扩大为旧集合并 triangle/mesh。**全局 `_FINITE_SHAPES` 与旧 convex compiler 的有限凸体门不改**。mesh 因而可走 M1，仍不能假称走旧凸体查询。程序化 RobotAsset 可合法包含一个原 triangle；这不是 URDF triangle importer 支持声明，原 mesh 至少四面约束不变。
- `ResidentSceneBatch` 使用 `ResidentBatchTransaction`，后者把 ParticipantKey 映射成内部机械 world（`_resident_batch_transaction.py:42–54`）；`_resident_participant_layout.py:437–505` 的 logical world/participant/body bijection 与当前 M1 的一个 world 对一个树源不同。不能直接暴露 `_service._batch` 丢 logical owner。ParticipantScene→M1 另立后继，不在本片修改 bank/lease。

固定官方 Newton 1.7.0 本机参照位于父 `.venv-newton-1.7.0/Lib/site-packages/newton/_src/`：`sim/builder.py:7891–7912` 分别记录 shape_body、shape_transform、shape_source；`:13348–13349` finalize 原 body/frame 数组；`sim/model.py:728–730,821` 保原 source 与模型寿命内 mesh 对象。采用的是资产材料/局部frame/动态body来源分离的通用接缝，不声称复制其完整 mesh/SDF/contact pipeline，未执行 Newton。

## 2. 精确公共调用与内部适配

公共 API 不增参数：

```python
binding = BindingType.from_asset(asset, original_key, key=scene_key, world=world)
shape = dataclasses.replace(original_shape, key=scene_key)
plan = batch.prepare_boundary_query(model, bindings,
    limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=budget)
packet = plan.query(margin=margin)
plan.require_current(packet)
```

原四 BindingType 与四物理树对应，不通过 subclass 推断 family。新私有接口冻结：

```
admit_boundary_asset(shape, binding, *, physical_spec, owning_spec)
    -> BoundaryAssetAdmission(group_index, local_points_m, provenance)
```

返回 frozen dataclass；local_points_m 为独立 bytes-backed immutable FP64 `[N,3]`；provenance 为递归 immutable mapping。函数只处理 `_source_shape is not None` 的 asset 分支；direct/static 不调用，保持旧 source definition 字段原样。

准入在任何 M1 device 分配前完整检查：exact 四 binding type；key/world 一致；kind 是 M1 原 triangle/mesh/sphere/capsule；`shape == replace(binding._source_shape, key=binding.key)`（包括原 IDs、radius、revision、所有 facets、group/mask 等全部 shape 字段）；原 `_model_token == physical_spec.model_token`；affine `_graph_token == owning_spec.graph_token`，floating-affine `graph_token == owning_spec.graph_token`；真实 physical group/bodies 范围、原 point 数、完整有限 FP64 local points。owning affine spec 与 physical spec 必须是同一已借用世界的 spec.physical；不能用另一个同尺寸 graph/spec 替换。

provenance 固定字段：`schema='phymeo-boundary-asset-admission-v1'`、`binding_family`、`source_shape`（保原未重映射key）、`model_token`、`graph_token`（非affine为None）、`physical_topology`、`group_index`、`local_points_m`。不替原绑定伪造 topology。父 asset definition 记录 `admission='asset'`、该 provenance，definition.topology 用已核 physical_spec.topology；原完整source binding/hash自然包含新增证据。原 points 继续由同一 M1 pack/upload路径处理，实际设备 FK不回到 HOST重算。

旧 factory 未独立保存原 asset/collider 对象；正常 factory 产物中的 immutable FP64 points 是保留的 authored material。此合同检查公开错误模型/graph/shape/facet/来源，不把 Python 私有字段可被 object.__setattr__ 恶意伪造的对象当成安全边界，不凭缺失字段声称防篡改证明。

## 3. 文件责任与依赖顺序

作者 NEW：

1. 本合同 `docs/development/p65-boundary-asset-contract.md`。
2. `tests/reference_newton/rr/resident_boundary_asset_reference.py`：预登记原 OBJ/XML/material frame literals、独立 q→pose 与原有限几何参考。
3. `tests/collision/test_resident_boundary_assets.py`：tests-first，完整新来源矩阵/负控。
4. `src/phymeo/backends/_resident_boundary_asset.py`：仅准入与immutable provenance。
5. `tests/installed/check_resident_boundary_assets.py`：公共安装消费者，自含原 XML/OBJ。
6. `docs/development/p65-resident-boundary-asset-report.md`：实际作者交接/待验证限制。

父 MOD：`_resident_tree_binding.py`、`_resident_floating_binding.py` 仅 factory shape-set；`_resident_boundary_source.py` 仅 asset 调 adapter/定义 provenance，direct/static 原表示保持。如 factory 必须修改错误文本或局部集合，不能连带开放旧 convex compiler。无新增 export，内部 admission 不冒公共 shape 几何 API。

顺序：合同→独立 literals/tests commit→admission实现→父三处窄接线→父文本核→唯一串行新矩阵+原 direct/convex有限邻居→源码外消费者。新代码不依赖 ParticipantScene 或 M2，不等待其 bank适配。

## 4. 固定多案例与 oracle

采用既有 M1 F0 两个 sibling +X slider 原物理数据（root2kg，driver1kg，follower2kg，原惯量、joint origins±1/4、mimic -1/+0.004；h=1/1024，非零入步速度），完整四 family。用新明确的几何资产，不按输出调参：OBJ tetra 四点 `(0,0,0),(0,2,0),(0,0,2),(-1,0,0)`，四个原闭合 oriented faces；sphere radius1/8、局部center `(-1/4,1/2,1/2)`。surface装driver，round装follower，身体101/202；同一实际body的多shape必须保同body ID。根的原非identity旋转与平移保留。每个正控3真实连续步，不以手写两帧替代。

- **A01**：四family × plain/fixed-tool（8）。真实 URDF→OBJ mesh 与 sphere 均从原 source key 经 factory 准入。fixed-tool 采用质量0 authored frame的已知非identity固定旋转/平移，保物理group质量与惯量；独立按 XML固定变换一次构造原世界材料，检查不漏转/不双转。每个mesh所有原 facets全保；查询候选按完整独立 AABB检查，包含共享边/远facet按实际固定输入的结果，不强制所有facet都成为候选。
- **A02**：四family 程序化原 triangle 材料（4），明确通过 RobotAsset/LinkCollision 数据构造，保原 source IDs、registered material 与 physical group。禁止把不闭合三角伪成mesh。
- **A03**：原 MJCF capsule轴来源（至少普通/浮根两个入口）；普通MJCF可同时原OBJ mesh，若当前浮根MJCF mesh importer不支持则明确用已验direct/static surface，不能暗扩importer。所有原capsule端点/方向/radius被保留。
- **A04**：混合四family sparseworld，重复原asset/localIDs重新映射world/body/shape，storage次序变换，3真实步；不按全局ID合并材质。另保合法mesh+round同一真实group过滤、mask/excluded的完整pair表；跨world不得生成候选。
- **A05**：公开错误来源/准入负控：四family wrong original model；两affine wrong graph；wrong shape revision/radius/point IDs/facet orientation/missing face；wrong source key、physical group/body别名、unsupported原hull/curved、无asset且topology=None。每个标明失败在factory还是M1注册，不允许异常后仍分配/launch。
- **A06**：新asset source下 selected reset/restore后旧packet拒绝与下一真实步；exact/one-less全候选capacity与原解析内存预算；最后raw/meta或packet构造失败保旧结果且原raw前缀保留，恢复真读回后成功。只覆盖新来源与共同发布接缝，不重跑原全部UNKNOWN_RELEASE矩阵。

测试文件提交时给最终参数节点表，不能把上述组数当pytest执行数。原M1距离1e-4L、witness2e-4L、normal2e-3、conditioning/zero-normal/ties门全部沿用。独立 oracle 从原 XML/OBJ字面、实际accepted checkpoint q/v与固定frame推世界材料；不能调用 product collision_motion/FK/gather/admission 的输出定义期望。完整原有限segment/triangle Fraction参考可复用，只读已接受reference，不用新producer自证。全raw/source/registration/provenance/连续运动在任何行为断言之前保存。

## 5. 内存、生命周期与通过/停止

不新增device buffer、kernel或launch：原S/N/F/R/P与同一个注册几何完全相同，M1 owned预算式不变。新增 provenance 的 immutable FP64 arrays在原递归HOST constant/packet nbytes中计入；不称零成本或GPU峰值。实际 device字段、source epoch、完整facets/候选、PENDING close-only、UNKNOWN_RELEASE强保留及原batch runtime lease仍用已有机制。admission失败必须在首M1资源分配前；已有plan/父物理银行不变。

安装消费者每backend至少两个稀疏world、真实导入OBJ+round资产、3真实步/完整原来源链、旧证据失效、所有加载Phymeo module来自安装目录；不导入仓库tests/examples，不用HOST当前点重新上传。

成功：四factory实际原来源+原mesh进入同一M1设备链，两后端多案例/负控/恢复通过，原direct邻居不变，旧convex对mesh仍明确拒绝，安装消费通过。代码完成不能叫功能通过。失败先分factory/来源映射/数值/计费/发布，不改原geometry/门/预算追绿；一个有证据的通用修复仍不缩小同一问题则保留最小反例停止本片。

ParticipantScene→M1后继必须另冻 logical world/participant→actual row/frame 的 accepted loan、共享close/UNKNOWN隔离及同worldcandidate过滤；M2要真实deformable producer bank。它们不能靠本片资产准入自动获得，CCD/力响应也仍另域。