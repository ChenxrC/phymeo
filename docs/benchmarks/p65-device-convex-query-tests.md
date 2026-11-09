2026-11-09 父最终状态：默认 K0 查询已限域验收，interpret/CUDA 各142、284完整记录、新wheel与440产品成员同源；见 [正式报告](p65-device-convex-query-acceptance.md)。下文保留作者历史检查点。可选 K1 patch 未验收。

# K0 设备凸体查询：tests-first 固定案例与保存合同

**AUTHORING_ONLY / UNEXECUTED。** 父权威计划为 `docs/superpowers/plans/2026-11-08-device-convex-query.md`（317cd41 及 source/wire/loss 补充）。本作者未 import、collection、数值执行、pytest、lint/format、编译、构建或运行 Newton。计数来自明确源码参数表，不冒称已 collection 或通过。父在这份作者检查点之后的执行记在计划 §12：每端 142 个身份有 pytest 通过记录，含后补的 10 个 touch；142 份 gzip 未留存，K0 未验收。本文件仍是作者检查点，不是那份执行记录。

## 1. 精确四 NEW 与执行分组

- `tests/collision/test_device_convex_query.py`：125 个固定身份。
- `tests/scale/test_resident_convex_query.py`：17 个固定协议身份，其中 W32 两个后置。
- `tests/reference_newton/rr/device_convex_query_reference.py`：独立 NumPy/stdlib 有限几何声明与参考；本轮不是 native Newton driver，不导入 Phymeo/TYX/Newton、不调用旧 HOST query 生成期望。
- 本报告。无 OLD 产品、既有测试/报告、顶层导出修改。

| 组/测试函数 | 参数身份 | 每 backend 数量 | 文件 stem |
|---|---|---:|---|
| `test_k0_family_gap_witness` | SS/SC/SB/SH/CC/CB/CH/BB/BH/HH × sep/touch/overlap | 30 | K0-Q01-{family}-{phase} |
| `test_k0_finite_features` | end_box/cap_side/cap_end/cap_oblique/cc_parallel/cc_skew/cc_end_end/h_slant/h_vertex | 9 | K0-Q02-{id} |
| `test_k0_frame_and_identity` | 10 族 × reverse/cycle/tilt/permute/cycle_reverse_ids | 50 | K0-Q03-{family}-{variant} |
| `test_k0_near_zero` | SS/CH/HH × pos-far/neg-far/pos-near/neg-near | 12 | K0-Q06-band-{family}-{label} |
| `test_k0_domain_classification` | 下表完整16声明 | 16 | K0-Q06-domain-{id} |
| `test_k0_api_preflight` | bool-gjk/gjk-zero/epa-193/vertex-three/face-129/pair-negative/memory-zero/duplicate-reverse | 8 | K0-API-{id} |
| `test_resident_convex_protocol` | 下表完整17协议 | 17 | K0-Q08-{id} |

合计每 backend **142 节点/142 完整 gzip**；140 核心 + W32 sparse/dense 两个后置。每节点只写自身 `{stem}-{mode}.json.gz`；不按输出选择身份。纯 query 部分101，域声明16，协议17，额外 preflight8。构造/源拒绝没有 device result 时保存 None 或未到达阶段，不能伪造零 raw 或计作成功 query。

首检查点 fdc3c77 的四原身份顺序为 `SS-sep, SS-overlap, CH-sep, CH-overlap`，其余26个 Q01 身份续列，首四不重复计。首有效 RED 节点：

```text
tests/collision/test_device_convex_query.py::test_k0_family_gap_witness[K0-Q01-SS-sep]
```

父已报告588有效RED：1 failed/1.27s/RAW1，先原 literal + HOST有限patch + 真FixedRobotTree/broad成功，准确停在缺 `prepare_convex_query`。该事实为父回执，本作者未执行或读取 evidence；不重新跑 RED。

## 2. 原物理与独立参考

S半径.05；C真实有限轴±.5、半径.05；B为±.6/±.1/±.1八点；H保持原五边形柱10点。三 gap=.02/0/−.01；Q03保持原置换、3/5–4/5倾转与大整数原材料 ID。非轴安装使用真实普通树 joint orientation，动态binding保group-local原点；没有将动态pose伪成HOST上传当前world几何。

声明参考先以独立有限支撑面和literal核 gap。通用 stored-input 参考从已核 source 的原上传点/半径与真实pose重建，两者差异单列；参考用有限端点/点线面投影、完整支持三角面枚举、边边距离和polytope SAT，不使用GJK/EPA产品或其返回normal/features来生成预期。原HOST patch仅作为既有前提/差异口径，不能成为新结果唯一oracle。

参考FP64内部判断使用1e-12面侧/命中舍入及固定小量退化判别；它们只处理本固定有限尺度下参考构造，不改变产品的物理门。验收仍是 gap1e-4 L、surface/closure2e-4 L、normal2e-3、单位长度2e-4、权重64u；L由声明的真实形状直径与相对origin独立算，不能用产品自报尺度放宽。

20个非零Q01必须RESOLVED且双侧真实材料/normal/gap过门。10个相触及近零内带可明确未决；外带正确符号。K0只一个有限见证，不把它当完整patch/归一线面载荷，不声称K1/K2或任何物理推进功能已实现。

## 3. 16个域声明身份

`coincident_ss`, `deep_center`, `deep_offset`, `identical_hh`, `zero_axis`, `flat_h_far`, `duplicate_xyz_far`, `bad_address-repeat_id`, `bad_address-missing_point`, `nonfinite-nan`, `nonfinite-inf`, `rounded_h`, `h_capacity-v64`, `h_capacity-v65`, `origin-rebase`, `origin-loss`。

- zero_axis保两个不同原ID与同坐标，必须解析device解，参数/权重1/2；不能拿UNKNOWN或新构造拒绝充正控。
- v64保完整32边柱；v65加原面内点，不能削点。flat/duplicate/bad-address/nonfinite必须先完整source拒绝，即使远处。
- deep/重合保不唯一/包含原因；禁止clear或伪造成功唯一法向。
- rounded H保持radius=.01；S中心(0,0,.17)，只验明确unsupported，不将radius改0。
- rebase原SH(.02)加(2**20,−2**20,2**20)，同源表示可用时完整原门；精度不足必须明确原因。
- loss按父补充：B动态原8个local点/真实安装(2**60,0,0)，S静态r=.05/center=origin+(0,0,.17)，原可表示几何gap=.02。记录local与world展开丢x半宽两份输入；原完整有限源拒绝或SOURCE_PRECISION均允许，不得RESOLVED。此是原loss身份补全，不增查询或改变负控。

## 4. 17协议身份与原调用量

| 身份 | 封闭调用量与独立门 |
|---|---|
| w1/w4/w32 × sparse/dense | 每身份2次query，共12；3P真森林，q=(0,.5,1)或(0,.09,.23)，每world0或2候选；不构造跨world pair |
| filters-samebody/mask/excluded | 每身份1，共3；只world54原SC被过滤，实际7对，samebody必须同一真实group |
| epoch_bank | 初query、advance后旧结果require_current拒绝、新query、restore后旧结果拒绝、恢复query，共5尝试；1实际h=1/1024步，1restore，S位移1/32768 |
| pair_capacity | cap2成功/cap1真实required2拒绝/新cap2重试，共3；旧broad容量足额，不混旧帽失败 |
| budget_one | tilt-HH原−.01，一次；真解可通过，否则预算/未决；不能预设必须失败或声称分离 |
| fault-raw_nan/raw_bad_id/late_record | 各真实成功/真实返回后破坏或晚构造失败/同输入重试，共9；无manual throw替代raw validator |
| owner_close_rebuild | 成功/真另一batch跨owner拒/closed拒/fresh成功，共4；broad和batch关闭依赖plan、live0、累计alloc保留 |
| construction_failure | 第二次真实_allocate返回并登记后抛，实际释放、batch仍能读原accepted态；0 query/0物理步 |

协议共37 query/currentness尝试（其中epoch中的2次是明确旧结果验证）、另1构造失败；仅epoch有1物理步和1restore。W32=96shape/416原点、允许域96pair、dense64实际候选；原world键17+37j，body11/29/71、shape3/5/7、跨world复用材料IDs。宽相候选回HOST再上传如实记，不冒device compact pairs。

故障接缝是父/RD确认的 `_allocate(*args,**kwargs)`、`_read_raw()`、`_build_result(*args,**kwargs)`。raw故障保原数组与只改一项后的完整字；坏ID取非本shape的2**31−1。late_record先调用真实完整owned构造后抛MemoryError。所有故障保原last_result/broad.last_result、完整物理checkpoint、owned旧结果和同输入retry，不改产品caps。

## 5. 完整记录与精确协议核对

每gzip包括该身份固定输入、有限形状/ID/半径、原绑定与完整checkpoint、独立声明/actual-upload参考、每尝试全部raw与公开结果/source/stats、before/after、失败原因/未到达stage、关闭后资源。ndarray同时存dtype/shape/完整C序bytes_hex及可读values；NaN/Inf保原数组字并显式非有限标记，不靠JSON NaN扩展。精确bare object仅编码两字段opaque过程内标记，不声称可恢复owner identity；实际owner判断仍调用runtime公开接口。

source核父§7所有必填表：完整model/ordered bindings、world_keys/真实FP32 anchor扩宽、shape rows/groups、原ID/offsets、两套local点与frame、点量化差、三半径表、真实pose[S,12]/stamp[S,3]、过滤后allowed_pairs。static点在HOST声明端减实际anchor；dynamic原local与真实storedR相乘。stamp第三列与实际batch全局slot比较，不能混shape_world_rows。

wire逐列核raw到public无暗改，所有active ID与所属shape一致；unused ID=-1/weight0，非logical padding meta与numeric0/ID−1。未知enum/未写logical行/非法validbit不接受；invalid gap/normal/witness只零占位并保False。公开数组bytes-backed写保护，旧owned结果在后续查询/失败/restore后不变。

查询完整成功才写completed=True；unexpected失败仍在finally保原输入和完整已发生prefix，重抛原异常。预期源/API拒绝另明确分类，不能作为device-resolved；构造前失败不编造raw。保存IO失败仍可能遮蔽原异常，父runner原stdout/exit/sourcebeforeafter是最终独立绑定，不伪称物理回滚。

暖态只在两次query周围核无新增device allocation/static topology upload；不在其间snapshot或补当前HOST FK。记录实际candidate read/upload、pose gather、full raw/source readback及缓存成本。非benchmark，不给速度收益。

## 6. 待父串行验证与限制

此版本只有文本自查，没有collection、格式检查或数值通过结论。父将按既定分组实际验证，首四之后不得因未知/失败而调整原几何、门或预算；所有首错需保留。新source及公开协议若有准确不匹配，先给最小fixture/interface修正，不静默改变物理预期。

本矩阵不新增空model/空显式请求独立身份，也不宣称其已验证；相关实现合同仍须遵守，父可按后继授权补独立边界。没有原生Newton实际执行/参数拟合；后置native与源码外wheel/保存审计另授，不能以本参考模块名字冒原生通过。
## 7. 用户停止后的作者检查点

父通知用户已清除长期goal，本作者立即停止继续扩展，仅保存现四NEW。当前为 **CHECKPOINT / UNEXECUTED / FINAL_TEXT_REVIEW_INCOMPLETE**，不是 CODE_READY 或最终验证交接。上述142固定身份/文件名已落为初稿，父保预配membership；没有collection/import或正控执行。首588有效RED仅复用父原回执。

剩余工作：三Python最终逐行文本自查尚未全部收束；完整oracle/source/protocol逻辑与最新产品schema尚待独立审阅；所有格式/静态/实际interpret/CUDA与保存完整性验证均未运行。所有现有输入、门、代码和报告原样保留为检查点，不据未执行源码宣称142节点可通过或已通过。恢复工作须由用户/父另授，不自动开启后继。本段停在作者检查点。父随后的执行记在计划 §12，不把本段改写成 142 已通过。