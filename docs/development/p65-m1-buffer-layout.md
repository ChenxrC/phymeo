# M1 S0：固定 buffer ABI 与字节预算

状态 **TESTS_FIRST / UNEXECUTED**。本文件把已冻结公共面和17组案例变成数组合同，不是执行测量。父 grant 为 3498b5fd。memory_limit_bytes 沿既有 ConvexQuery 的 **本 plan owned TYX arrays** 语义；interpret 的相同数组也计费。借用物理 bank 不归本 plan 所有，HOST 常量/下载/不可变 packet 另列真实 nbytes，不能冒成零成本或 GPU 峰值。

## 1. 索引与同源 frame

S=按 CollisionModel 原规范顺序的 shape 数，N=完整原材料点总数。每 group 是一个 surface 与同 world 所有 eligible round；F=原 facet 数，R=round 数，T=2F-1，K=ceil(F/30)。triangle 的唯一原 facet ID=0；mesh 原 triangles ordinal 是 stable facet ID，上传 permutation 使用明确映射，不重编号。round ordinal 是 group.round_keys 原规范顺序。sphere 原单中心 ID保留，打包时重复该中心形成零轴，不新增材料点。

shape_meta 列为 layout(0..3)、physical row(-1 static)、physical group(-1 static)、batch world-slot、pointbegin、pointend。static 仍有实际 world layout/slot，只不读 FK。每一 layout 从父 _borrow_boundary_collision_state() 的相应 accepted physical view 借 meta/fk/ticks，floating body_extra=1，另两 layout=0。不需要不存在 layout 的伪 device 占位数组。

各 world 公共 anchor 为原 physical spec.anchor（FP64 注册常量与实际设备 anchor绑定），dynamic FK 是该 anchor 相对世界轴坐标；static 原世界点在注册时一次减该 anchor。每 pair 再由 **设备** triangle首点选 frame，五点相对之，不能 HOST 下载坐标算当前 bbox/pack再上传。原 point/radius FP64→FP32 字及转换delta完整保存。注册 source geometry 不是当前理想FP64 FK；实际当前字来自 accepted physical bank。

## 2. 每个 owned 数组（未列者不得偷偷分配）

所有 i32/f32 都是4B，零长度不创建TYX对象，不以max(1,n)偷加空占位。borrowed数组、Python元数据不在下表，但各自来源/host nbytes在stats/evidence单列。

| 阶段 | 名称 / dtype / shape | 字节 |
| --- | --- | --- |
| persistent | shape_meta i32[S,6]；local f32[N,3]；local_bounds f32[N,6] | 24S+36N |
| persistent | radius_bounds f32[S,2]，列为 RN32(radius)、outward-upper(original radius) | 8S |
| query全局 | pose f32[S,12]；stamp i32[S,4]，列(valid,tick,world-slot,physical-row) | 64S |
| query全局 | points f32[N,3]；point_bounds f32[N,6] | 36N |
| 每组 BVH/pack索引 | faces i32[F,3]（全局point rows）；round_shapes i32[R] | 12F+4R |
| 每组 bounds | leaf_bounds f32[F,6]；query_bounds f32[R,6] | 24F+24R |
| 每组 BVH | nodes i32[T,6]；order i32[F]；node_boxes f32[T,6] | 48T+4F |
| 每组 traversal | params i32[R,3]；bvh_output i32[R,K+3] | 12R+4R(K+3) |
| 每组P候选 pack | selected i32[P,2]（round,facet，HOST只编排整数）；geometry f32[P,5,3]；radii/scales f32[P]各一；owners i32[P,2]；frames f32[P,3]；pack_status i32[P,2] | 104P |
| 每组P候选原核raw | candidates f32[P,18,11]；valid i32[P,18]；summary f32[P,16]；meta i32[P,7] | 956P |

pack_status=(valid,attempt_generation)。generation仅1..2^31-1，达到边界明确拒绝，不回绕。全P由完整BVHbitsets决定；不按几何status/输出force挑子集。原核owners存group round/facet row；typed发布映射回原 IDs。M1 group.bvh_raw保完整K+3，最后三列为nodevisits/leaftests/escape；escape必须等于nodes[root,2]。旧COL3 K+2表示不改变。

树拓扑只按原facet IDs平衡分割（可用 BoundsTree 的全零layout boxes），从不为建树读取当前几何。全部node refit都由本次device leaf_bounds产生。树输入表按原IDs映射验证；pending layouts只在全query成功后公开。数据生成/清空没有另加scratch；TYX局部标量不算可外部owned数组。

## 3. 峰值与空/失败路径

    B_persistent = 32*S + 36*N
    B_global = 64*S + 36*N
    B_group(F,R) = 40*F + 48*(2*F-1) + (52+4*ceil(F/30))*R
    B_prepass = B_persistent + B_global + sum(B_group)
    B_peak(P_1,...,P_G) = B_prepass + 1060*sum(P_g)

所有组的bounds/BVH/raw与pack保持存活至该次query完成；不靠隐式复用减公式。先核 B_persistent 再首分配；query先核 B_prepass，再gather及所有组BVH；核所有几何capacity和 B_peak 后才分配 **任何** narrow/pack缓冲。相同实际P下预算=所需可通过、少1拒绝；不得用运行峰值倒填测试门。constructor分配失败最高为已成功分配项的prefix和；prepass失败≤B_prepass；post-count拒绝峰值=B_prepass；完整narrow失败≤B_peak，绝不把失败计数回滚。正常query后live_bytes回到B_persistent，正常plan.close后0；free异常后保失败buffer真实size计数，不能假0。

empty model S=N=G=P=0，所有上式0；返回完整空schema/filtered表，零launch/owned arrays/runtime lease，无None→伪device占位。非空模型但无eligible groups仍验证source/gather，G=P=0；有groups但P=0保BVH证据、无narrow launch。候选上限全query8192，各group原512vertices/128faces/64rounds、groups32，仅收紧。

HOST bytes分三层准确记账：authored/constant metadata数组（含FP64）、本次每次真实download返回nbytes、递归immutable packet中不共享的array storage。常量本身在HOST、设备常量、readback三者不可算成同一份；摘要不声称此数是进程RSS/驱动峰值。opaque metadata对象开销不伪造为精确数。owned_limit不覆盖Python保存证据或借用physicalbank，这与既有resident query预算一致；device-source不是零往返。

## 4. bounds与校验，不新增精度门

local_bounds由原FP64常量的外向FP32两端定义，local为RN32实际使用字；static常量先相对已绑定anchor。gather center沿现 accepted FK 的 mul_rn/add_rn顺序；point_bounds按相同FK字对local_bounds逐项选乘积端点，并调用既有 _tyx_resident_revolute_bounds._lower/_upper作工程外向余量。static直接复制注册两端。leaf bounds是完整三顶点interval min/max；query bounds是完整两轴端点interval min/max，再外向扩 radius-upper+margin-upper。margin的实际FP32上界作为kernel scalar参数，绑定原scalar与转换字，不省计source证据。不下载当前点构造boxes。

每次完整raw读回后，HOST仅核finite/order/包围实际gathered点、margin/radius绑定、IDs/stamps及原核已发布候选的材料重构，不生成最近点、缺候选或改witness。原距离/witness/normal门、conditioning与32u方向带均不变；这些是工程源表示检查，不称形式化IEEE/global碰撞证书。过大相对坐标/不可表达pair在pack_status中拒绝，不能加绝对位置到L放宽门。

## 5. S0的内部测试接缝与父hooks

父hooks固定 _register_boundary_runtime_lease(lease)->token（lease.runtime is batch._runtime）、_release_boundary_runtime_lease(token)、_quarantine_boundary_runtime_lease(token,ledger)。persistent与query各有自己的DeviceResources/runtime lease；query全部数组归同一query ledger，成功依次free后正常runtime-lease.close注销。未知free由globalledger强保留lease→batch/runtime，向父quarantine注册；PENDING可close重试，UNKNOWN不可重试。借用source只走父新 _borrow_boundary_collision_state；scope内不再调用默认batch._check_open，防callback重入step/restore/set_controls/prepare/close。

为测试有限真实故障，plan保以下私有稳定接缝：_borrow_source() context、_validate_source(view)、_download(array,label)、_build_packet(...）、_close_query_resources()；还有 _persistent_resources/_query_resources。_download label前缀为source/ 或 group:<ordinal>/（如 group:1/bvh、group:1/meta、group:1/summary）。生产数据仍真实计算，负控只在对应一次readback/发布/释放处注入错误；不是mock成功几何。

source view沿旧convex相同只读字段风格：accepted/owner/registration/world_keys/shape_keys/shape_revisions/bindings/world_layouts/shape_rows/shape_groups/shape_layouts/sources/_lease；_validate_source每次query在进入dependentkernel前完整检查。source.gathered_points/axes使用CollisionKey→只读array mapping，world_anchors使用world→FP64tuple；source.stamps包含rows i32[S,4]与poses f32[S,12]。stats至少有memory_limit_bytes/persistent_bytes/prepass_required_bytes/peak_required_bytes/live_bytes/peak_owned_bytes、allocations/uploads/uploaded_bytes/downloads/downloaded_bytes/launches/completed_launches/synchronizations、host_constant_bytes/host_downloaded_bytes/packet_array_bytes；输出source与packet保前述完整ABI。

M08/M10保合法mesh同body按rigid安装过滤；CollisionModel同body triangle+round输入继续拒绝、mesh<4原面继续拒绝。新tests不能复制一面伪mesh绕过。额外空输入/precise-byte/重入负控属于S0资源协议补足，最终报告将逐项给实际节点数，不冒原49恒定。


## 6. S0 tests-first 节点清单与测试接缝补足

手工参数枚举，**UNCOLLECTED / UNEXECUTED**：
- 主文件共27：M01/M02 8；M06 1；M04 6；M05 3；M07 2；M03 3；M09 1；M14 3。
- 协议文件共用36：M10 10；M11 1；M12 6；M13 4；empty/filtered/zero-candidate 1；callback reentry 5；M08 1；注册准入5；latest/full-source/owner 1；prepass/narrow allocation-prefix 2。
- 额外 M15 HOST 4（before/after × query/batch-close），M16 CUDA 1。63个共用节点 + 4 HOST + 1 CUDA；不是原49的未说明扩充，也不是实际pytest数量。M17安装消费者后续独立编写。

原公共 memory_limit_bytes 必须正 builtin int，故empty模型用预算1但owned字节严格0；0仍拒绝，不能为empty silently改变原公共标量合同。M12两world各3triangle点+1sphere点：S=4,N=8，两组(F,R)=(1,1)，P各1；exact预算直接由本文式计算，不先运行观察峰值。

有限资源注入接缝增补 _new_query_resources() 和 _allocate(ledger,shape=...,dtype=...)；全部生产资源仍通过这些真实方法。_close_query_resources在未知free后通知父quarantine，不再调用父公开_check_open。query与batch.close故障路径都保旧last。M15只fake native array/free（明确不称CUDA几何），实际M1 plan、DeviceResources、父注册/close路径不替换；exercise()退出后只剩weakref检查，闭包/monkeypatch上下文不能假保活。M16仅一次真实free后人工抛错，不再调用真实runtime.close，因为父其余物理buffer仍依赖它。

attempt.counters保完整 launch_labels（逐次已提交kernel字符串，如 group:0/narrow），用于核全容量检查发生在任何narrow前；缺字段不能当“未执行narrow”。source/stamps下载必须保存完整源字并允许严格验证tick/row/slot。PENDING/UNKNOWN和query正常失败不可混为一种状态。

同次保存使用完整dtype/shape/bytes，含原始非有限字标签；公开opaque owner只在内存按is核。测试source/world frame oracle从真实accepted checkpoint与原独立参考重建，不调用新gather或新BVH作oracle。reference仍只NumPy/Fraction；未执行任何代码来选择fixture、预算或容差。
