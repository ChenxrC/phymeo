# M1 预登记案例库：resident 刚性设备来源的有限边界查询

状态：**PREDECLARED_CASE_LIBRARY / 新测试尚未编写、收集或执行**。2026-11-12。依据父源码 6f539171、已接受 COL2/COL3 及[原 M1/M2 合同](p65-device-mesh-source-next-contract.md)整理；只补具体案例和顺序，不改变合同、不授权项目执行。下面 17 组供下一 agent 直接 tests-first，公共 ABI 仍由父 owner 冻结。

本片是 **同一 TyxWorldBatch accepted 刚性状态 → 设备当前顶点/轴 → 设备 BVH 候选 → 原 COL2 距离核 → 一次只读发布**。仅 sphere/capsule 对 triangle/mesh，四树及明确静态注册。M2 变形 bank、CCD、接触力、摩擦和曲面 B-point 均不在范围；旧 COL2/COL3 全套不是默认重跑清单。

## 1. 已有构造与独立参考

| 代号 | 可复用路径/符号 | 准确用途 |
| --- | --- | --- |
| F0 四树 | [test_device_convex_family_sources.py](../../tests/collision/test_device_convex_family_sources.py) 的 declaration、independent_last_pose；[test_device_convex_sources.py](../../tests/collision/test_device_convex_sources.py) 的 LAYOUTS/ASSET | 原四布局、URDF、q/v、根姿态、稀疏世界；不复用 curved query 产生的 witness |
| F0 资产 | [floating_mimic_jaws.urdf](../../tests/fixtures/urdf/floating_mimic_jaws.urdf) | 两条 +X sibling slider，driver 安装 x=-1/4、follower x=+1/4；mimic q_f=.004-q_d、v_f=-v_d。原质量、惯量与限位不改 |
| F1 转动/控制 | [test_resident_device_contacts.py](../../tests/scale/test_resident_device_contacts.py) 的 fixture；[test_resident_tree_dynamic_contacts.py](../../tests/scale/test_resident_tree_dynamic_contacts.py) 的 _case/_build | D03/D04/F04/A02 的真实 tree/command。只 plain batch.step，不调用旧 contact solver 或 contact audit |
| F1 参考运动学 | [resident_affine_reference.py](../../tests/reference_newton/robot/resident_affine_reference.py) 的 oracle；[resident_floating_contact_reference.py](../../tests/reference_newton/robot/resident_floating_contact_reference.py) 的 case/base/world_points；[resident_floating_affine_reference.py](../../tests/reference_newton/robot/resident_floating_affine_reference.py) 的 case/expand/mechanics | 从同次实际 accepted q/v 独立重建 body pose，不用新 gather、产品 FK 或产品 query 作 oracle |
| F2 有限边界 | [test_triangle_distance_dispatch.py](../../tests/collision/test_triangle_distance_dispatch.py) 的 flat、mesh_scene、two_worlds 及 filters/capacity/late-failure | 原有限 feature、cube/凹体、原材料 IDs、过滤与发布语义；换新设备来源，不重做全部 COL2 数学核 |
| F3 独立距离 | [device_triangle_distance_reference.py](../../tests/reference_newton/rr/device_triangle_distance_reference.py) 的 oracle、make_case、local_scale | Fraction active-face 最小化 [0,1]×triangle simplex；不调用产品 18-candidate helper 或旧 HOST 碰撞查询 |
| F4 资源/故障 | [test_device_triangle_distance_protocol.py](../../tests/collision/test_device_triangle_distance_protocol.py) 的 fake exercise/context、CUDA post-free fault；F0 的 source-loan negatives | 复用有限故障位置和验证方法；新断言针对 M1 共享 owner/runtime，不以旧 ledger 测试重新跑绿充数 |

F0 固定布局顺序 ordinary、affine、floating、floating_affine。世界 IDs=(83,17,107,31)，每个 layout 的基础节点放四世界，index=0..3；构造/存储顺序可反转。原输入完整保留：

- q=(index+1)/512，v=(index+1)/1024；完整 scalar q=(q,.004-q)、v=(v,-v)。普通树使用两坐标；affine 使用原 reduced 单坐标，不在 HOST 替产品展开来驱动设备。
- H=1/1024；gravity=0、kp=kd=0、effort_limit=1、delay=0。公开构造是 load_mimic_urdf/load_floating_mimic_urdf、DrivenRobotProgram/FixedRobotTree、AffineRobotTree、FloatingRobotTree、FloatingAffineRobotTree、TyxWorldBatch。
- floating root position=(2+index/4,-3,1/4)，xyzw quaternion 依次为 (0,0,0,1)、(.5,.5,.5,.5)、(0,.6,0,.8)、(0,0,1,0)；root linear velocity=R(1/128,0,0)、angular velocity=0。固定根没有虚构的 root pose。
- 每次实际 batch.step(H) 后 query，连续 3 步，步间不 reset q/v。保存入步/末态与真实步结果。actual stored root quaternion 归一化副本仅用于独立 FK，不能写回状态。

## 2. 第一个 tests-first 提交固定的新增材料几何

不要依赖 query 输出反推 fixture：

- driver-local triangle T_x=((0,0,0),(0,2,0),(0,0,2))，vertex IDs=(11,37,91)。这是 F2 原 T=((0,0,0),(2,0,0),(0,2,0)) 的 cyclic rotation。
- follower-local sphere 中心 a=(-1/4,1/2,1/2)，radius=1/8；capsule 端点 a、a+(1/4,0,0)，IDs=(11,37)，radius=1/8。球只保一个原材料 ID；核内部零轴不得冒成两个原球材料点。
- triangle key=(world,101,0)，sphere=(world,202,0)，capsule=(world,203,0)。sphere/capsule 是同一 follower 上两个只读资产，round–round/same-body 等完整过滤记录仍保存。
- 固定根 driver/follower physical group 为 0/1，浮根为 1/2；必须由实际 topology/asset 证明核对。M02 triangle 装 driver、round 装 follower；M01 triangle 固定在已声明初始 driver 世界姿态，round 仍来自真实 follower。
- 静态坐标允许由上述初始 literals 的 R/t 一次计算。禁止每步下载当前 pose 再上传静态点。新 boundary binding 类型由父冻结，不能把 triangle 冒充 hull 塞入旧 convex binding。保存真实 checkpoint.topology；floating-affine 继续绑定原 tree.inputs 零 drive 所得 graph_token，不手造 reduced/physical 图身份。

M02 的独立解析 oracle 为：

    x_driver = t + R(-1/4 + q_d, 0, 0)
    x_follower = t + R(1/4 + q_f, 0, 0)
    d_axis = 1/4 + q_f - q_d
    triangle_witness = x_driver + R(0,1/2,1/2)
    axis_witness = x_follower + R(-1/4,1/2,1/2)
    normal(axis -> triangle) = -R e_x

原 3 步域 d_axis 为正，capsule 最小点为首端点，第二端点更远 1/4。distance 是轴/中心到 triangle 的距离，gap=distance-radius；不是闭体内外判断。这些是代数期望，未运行产品；真实输入若不满足预声明域必须保留失败，不能根据输出换 fixture。

## 3. 17 组有限节点

数字是手工预登记参数身份，未 collection。除 M15 仅 HOST、M16 仅 CUDA 外，源码节点拟两后端各一次。节点内同时注册或固定次序查询 sphere/capsule，不再隐含乘出另一套形状参数矩阵。手工合计为共用源码 49 节点/后端、额外 HOST 4 与 CUDA 1，安装消费者各 1；这不是实际 collection 或通过数。

| 组 / 节点数 | 固定输入与已有入口 | 必须观察及独立门 |
| --- | --- | --- |
| M01 / 4 | 每 layout 一个 F0 四世界节点；static triangle + moving sphere/capsule，margin=.5、3 真步 | actual entry v 非零、末态与 round 材料坐标变化；epoch/tick/active 对 accepted bank；F0 独立 FK+F3，不能反复初态上传 |
| M02 / 4 | 同 F0，triangle/round 分别随 driver/follower，margin=.5、3 真步 | 双方材料点均动、相对运动不只是公共平移；上述解析 d/witness/normal、完整 ID/bary/t；query 不改物理 q/v/epoch |
| M03 / 3 | F1 的 D03 ordinary、F04 floating、A02 floating_affine；原 command、H、3 步。首/末活动 body 安装本节固定 T_x/round local 点，margin=1 | D03 v=(1/128,-1/256)、motor[0]=1/16、force[-1,x]=1/32、torque[0,z]=1/64；F04 force[0,y]=1/32、torque[0,z]=1/64；A02 v[0]=1/256、drive[0,2]=1/16、同 F04 force/torque。其余原数据不动。独立 FK/F3 核真实 R 变化、全候选；不要求任意安装 pair 都接触/承载 |
| M04 / 6 | F2 flat 的 front/back/edge/vertex/through/outside 各一节点，含 sphere/capsule。原 T/axis/radius/IDs；用 F0 ordinary 初始 pose 把 round 固定安装到 body-local | front/back distance=1/4；edge=sqrt(1/8)、vertex=sqrt(3/16)；through=0、normal=None；outside margin=.01 空带，其余 .5。保 capsule 有限轴，F3 逐行核 |
| M05 / 3 | F2 mesh_scene：shared_pruned、concave、closed_inside；原 sphere center/radius。mesh static，sphere 经 actual body gather | cube center(1,1,1/4)、margin=.2 保底部 2 原 facet/剪其余 10；凹体 center(1.25,1.25,.5)、margin=.2 保原 4 候选且 min=.25；cube center(1,1,1)、margin=1 得 12 行 distance=1，同节点 margin=.01 空带仍不叫 separated |
| M06 / 1 | sparse worlds (83,17,107,31) 各一不同 layout，反序 batch/model/bindings，重复 local IDs；M02 资产，3 真步 | 无跨世界候选、遗漏/重复；原 key/facet/axis → packed row 双射；四 layout 实际都被消费 |
| M07 / 2 | (a) M05 shared cube 的 vertex/facet 存储置换，稳定 material/facet IDs 不变；(b) F0 floating 初始 root t=(2^24,-2^24,2^24)、quaternion=(.5,.5,.5,.5)，原 M02 local 资产、margin=.5；F3 的 caller-origin=0 只作独立参考表示 | 按完整 identity 对齐距离/witness，不能坐标去重；大平移先相对 frame，保实际 producer anchor/pair origin/relative 字，不虚构 M1 caller-origin 参数；L 不随绝对位置增大。不强求小位移在巨大绝对 FP32 坐标可见 |
| M08 / 1 | F2 原 filter test 的 5 shapes/10 unordered pairs：triangle、body20 两 sphere、mask0 sphere、capsule，原 excluded pair | 完整 reason 保 excluded/same_rigid_body/mask/same_family；原 eligible candidates=2；non-applicable 不偷偷 HOST fallback |
| M09 / 1 | M02 ordinary 四世界真实步后 query；守卫只包产品 query，oracle 在外 | 禁当前坐标 HOST 重建、COL2 HOST _prepare/closest、HOST bbox；记录实际 launch/transfer/bytes。允许整数 bitsets 编排与事后完整 readback，不谎称零 D2H |
| M10 / 10 | 基于真实 loan 分别注入 owner/world/epoch/active/layout/registration/topology/point-ID-map/expired/standby 错误 | dependent kernel 不消费坏来源，旧 last/accepted bank 不变；负控可替换 seam，positive 不得手造合格 loan |
| M11 / 1 | M06 成功后 snapshot、selected reset(17)、新 query、restore 保存的 accepted checkpoints、再 query+1 真步 | 每次旧 require_current 拒绝、新包匹配实际 epoch；未选世界按原语义保持。只用 snapshot/reset/restore，不虚构 batch.clone；不能恢复未提交候选 |
| M12 / 6 | F2 two_worlds，world7/4096 每组单 triangle/单 sphere，原 translation (2^20,-2^20,1)、margin=.5；groups/candidates/vertices/faces/segments/memory 六参数 | 所需 2/2/3/1/1；typed limits exact 通过、少一拒。几何容量在全 bounds/count 后、narrow 前核；memory 从冻结数组 ABI 字节式预登记 exact/one-less，在相应 buffer 分配前核，不用测得峰值定预算；无 partial publish |
| M13 / 4 | M12 正常 last 后，在末组注入 BVH readback、COL2 readback、source stamp 校验、immutable packet 构造失败 | 全 query 一次发布，旧 last/pending-layout/epoch 保持；实际 counters/attempt raw 留存；解除故障同 accepted 输入重试，不重置物理掩盖半发布 |
| M14 / 3 | F3 near_parallel_conditioning、zero_area、through_face 原 literals | 保原 CONDITIONING/DEGENERATE/RESOLVED-zero、全部 candidate/ties，zero normal=None；不删未决行成 empty band，不 HOST 修点 |
| M15 / 4 HOST | F4 fake exercise/context 方法；free 在 side effect 前/后失败 × 首次后继入口为 plan.query/batch.close | 针对 M1 借用与新 producer guard；正常 free 才扣 live bytes、unknown 强留 owner/runtime、siblings 清理、旧包不换，随后所有 query/close sticky且无重复 free。exercise 退出后 weakref/GC 核 ledger 留存，不能由闭包假保活 |
| M16 / 1 CUDA | M12 成功一次后，新 scratch 的单 buffer 先真实 free 再抛；F4 同类有限 seam | 不假零 live、不发布，plan query/close 与 batch.close 都受 unknown lease 保护，不二次 free/关 runtime。M1 plan 只借 runtime，不凭旧 COL2 独立 runtime 测试声称新保护通过 |
| M17 / 每后端 1 | 源资格后新增 wheel 公共消费者：M12 两 sparse worlds，sphere/capsule，3 真步+stale拒绝+正常新 query | 严格 installed module origins、完整 source/raw/IDs/counters；独立小型解析参考，不 import repo tests/examples，不重跑旧全套 |

M03 的驱动/外载只为产生真实运动，本片不新增接触力。D04 affine 的真实构造继续由 M01/M02/M06 覆盖，不为“转动”改写它的 prismatic 定义。M03 必须逐帧比较完整独立候选集合；若某帧本来无 candidate，不得跳过此帧或伪造 distance 行。

M05 原 mesh literals 在新 reference 首 commit 逐字登记：cube 为 (0/2)^3 八点及原 12 facets；凹体 outline=((0,0),(2,0),(2,1),(1,1),(1,2),(0,2))、z=0/1，原 caps=((0,1,3),(1,2,3),(0,3,5),(3,4,5)) 和 sides；vertex IDs=11+13*i。不改面表追求少候选，共享边保两个 facet 身份。

## 4. 同次证据与 oracle 顺序

每 query 保存完整 inputs/model/bindings/filters/revision/limits/margin；静态 FP64→FP32 上传字及 delta；真实 accepted q/v、owner/registration/topology/epoch/active/world ticks/layout/runtime identity；device 当前点/轴、anchor/pair origin、packed 五点/radius/L、leaf/query bounds、BVH raw bitsets/counts/end markers、完整 candidates；全 COL2 raw/meta/ties/status/witness/bary/t/distance/gap/normal；实际 H2D/D2H/launch/sync/owned-live/ledger。所有数组带 dtype/shape/完整 bytes；失败非有限原字不替换成有限数。aggregate 共享内容按引用保存，不平方重复，也不删失败行。

断言顺序是 **独立 pose/material重建 → 完整候选集合 → 距离/见证**。F3 的输入来自独立重建的实际几何，不能只用产品刚发布的 packed 点自证；另按原来源表示门核实际 packed 点。BVH expected coverage 用固定 literals 的解析范围或独立逐 facet AABB 判断，不能调用产品 BVH。M01/M02/M03/M06 保存全部三步；同次 query 数据不是额外重复求解获取的证据。

原门不变：distance≤1e-4 L、witness≤2e-4 L、normal≤2e-3；L 只取本 pair 原 local 边/轴/radius 的 max，不含 pair 间距或绝对位置。原方向带 u=2^-24、normal_floor=32uL；可靠距离可与 normal=None 同时存在。不称严格 IEEE 证书、闭体内外或接触响应。来源/容量/身份无几何容差宽免。原 positive 除明确方向带/未决输入外，不接受任意 UNKNOWN 替代通过。

## 5. 接口依赖、文件责任与实现顺序

| 片 | 可执行工作顺序 | 精确责任建议 |
| --- | --- | --- |
| S0 tests-first | 冻结本表 identity/literals、binding/packet ABI、buffer 尺寸/字节式，再提交 reference 与测试；全部 UNEXECUTED | NEW tests/reference_newton/rr/resident_boundary_source_reference.py；NEW tests/collision/test_resident_boundary_query.py、test_resident_boundary_query_protocol.py；原 helpers 只复用不修改 |
| S1 来源 | 注册/四树 accepted lease、gather、frame；优先 M01/M02/M06/M07/M10/M11 | NEW backends/_resident_boundary_source.py、_tyx_boundary_source.py；父独占 tyx_world_batch.py 的 prepare/loan/owner-close；原 storage FK只借，binding 变更逐文件申报 |
| S2 候选/距离 | device bounds→BVH device-buffer入口→全计数→device pack→原 COL2；M03–M09/M12/M14 | NEW backends/_resident_boundary_query.py；受限 MOD _tyx_collision_bvh.py、_device_triangle_distance.py。原 _tyx_triangle_distance.py 和凸体数学核保持已验字节 |
| S3 发布/释放 | M13/M15/M16；全部 immutable result 和 cleanup 成功后一次 swap；普通失败 retry 与 unknown 区分 | 同新 source/query文件与父 batch-close hook；只 import已验 mujoco_tyx/_resources.py，不重写 ledger |
| S4 public/wheel | 父审 affected 邻居清单；源资格后 M17 | 父独占 phymeo/__init__.py；NEW tests/installed/check_resident_boundary_query.py，消费者后编写，不抢源资格 |

公共建议仍为 batch.prepare_boundary_query(model, bindings, limits=..., memory_limit_bytes=...) 返回 plan；plan.query(margin=...)、require_current(packet)、last_result、last_attempt、close()。这是原合同的**待父冻结名字**，当前没有 M1 public API，不能为赶进度直传未审私有指针。M1 packet 不与 COL3 HOST motion packet互换。

硬依赖先后：真实 accepted loan +共享 owner未知释放保护 → 注册/gather → bounds/refit/traversal受控device入口 → 完整候选计数/容量 → 设备五点pack/原核 → immutable packet与一次发布 → public/wheel。不能先用 HOST 当前坐标 pack 垫通再标设备成功。首片允许整数 bitsets/计数 D2H、稳定ID编排和整表H2D，全部明示计费；没有全设备 compaction 或零往返承诺。

查询不推进 state_epoch，不借 standby，不接 step_contacts。若共享 runtime 的 unknown lease 现接缝无法阻止 batch.close，先报告最小具体改动给父，不自行忽略错误/关闭依赖。M1 的真实来源是刚体 FK；没有 deformable producer 时 M2 仍 unsupported。

## 6. 通过与停止

- S1 成功：四树三真步、双方运动、完整材料/frame/epoch链通过，产品没有 HOST 当前坐标重建再上传；静态或单布局成功不够。
- 源码接受：17组所有适用新节点通过，HOST/CUDA专属项单列，原门/status/caps保持；相邻旧检查仅按实际 shared delta选择，不重复旧全套。
- 可交付：新wheel源码外公共 M17完成，source与installed资格分开；无新增Newton速度/内存比较。
- 立即保留失败并停止扩域：漏facet/跨world、坏source被消费、晚失败半发布、unknown丢owner或关runtime、以HOST冒device、或需要改输入/门/caps才通过。
- 一次有真实同次证据的通用最小修复仍不缩小障碍，就记明确scope缺口；不按shape/axis/world ID特判，不开始第三套算法追绿。
- M2要另有真实非刚性物理推进的accepted vertex bank/提交协议。COL3两个HOST快照、手写vertex更新或任意临时device array均不满足；CCD、载荷与历史各另冻合同，M1通过不自动授予。
