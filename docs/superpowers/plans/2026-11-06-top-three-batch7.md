# 第七批：浮根场景、圆体逐面候选与步内装配复用

> 用户已授权继续。三原作者隔离编码；父唯一项目执行和既有共享文件 owner。基线57791f4。功能优先、多案例验收；不跑远程CI、不改依赖/pin、不重跑同源已绿组。

**调研结论：** 完整浮根质量矩阵/偏置/COM漂移已存在，缺场景集成；TYX面级BVH已用于mesh–hull，mesh–round尚枚举；前批W32F8额外分页仅0.155%，分页优化门关闭，HOST姿态几何和材料装配才是已测热点。以[Newton 1.7关节](https://newton-physics.github.io/newton/1.7.0/concepts/articulations.html)和[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)作模块参照，不把其当前姿态碰撞当连续首触真值。

## A3 高：真实浮根加入共同场景

新增公开 `ContactSceneBuilder.add_floating_urdf`，复用现有MultiRobotScene的同入态采样、各物体一次积分、一次共同发布。真实根6DOF与子关节在同一完整质量矩阵中，不拆成独立刚体，不用虚拟关节模拟底座。允许一个浮根+自由刚体、浮树与固定普通/mimic树、多个浮树。HOST单world，普通标量关节；浮树mimic、fixed关节折叠、pads、非对角根惯量首片具名拒绝。根惯量必须正且与body轴对齐；其他原shape限制不扩大。

作者NEW：src/phymeo/builders/floating_urdf.py；src/phymeo/runtime/_floating_robot_participant.py；tests/test_floating_robot_scene.py；examples/floating_robot_contact.py；docs/benchmarks/p65-floating-robot-scene.md。
父旧文件：contact_scene.py、multi_robot_scene.py、_robot_participant.py、_robot_scene_contacts.py、_robot_robot_contacts.py、contact_events.py（均runtime），必要robot_scene.py的公共说明。禁止另复制完整scene。

新FloatingRobotAsset保留真实根惯量和children；FloatingRobotCommand(joints,external=None)用原RobotCommand控制子关节，root-only joints=None。external为FloatingRobotInputs，世界force/torque关于各体COM；其中effort必须零，内部驱动仅原PD/限幅通道。完整q为7+n、v为6+n、body_count为1+n，三者不能混同。participant提供state复制、body/velocity计数、真实contact geometry/J。根collider使用真实body0，children顺移；事件使用随动材料frame，原fixed-root静止frame不变。观察保留完整入/出态、实际外载、接触广义力/COM wrench、已限幅驱动，外载/接触各一次。

- [ ] tests-only先提交；父首RED：根2kg+子1kg X滑轨，根球r1/16，4kg自由球距.124，k2000零阻尼摩擦，h.001。M_x=[[3,1],[1,1]]，根−2N→a_root−1、qdd+1、child世界a0、自由球a+.5，总动量0。
- [ ] root-only力/力矩、偏心COM漂移、关节驱动反作用、重力和真实速度阻尼分别独立验。
- [ ] 浮–固定/浮–浮直接接触、换序/协变/多个局部同ID、球–有限胶囊/OBB原范围、root/child材料地址不混淆。
- [ ] 40步连续保持/撤离或自由/受载窗口、原入步速度不重置；第二participant越限/几何/事件/外载失败全回滚，修复输入后重试与重建一致。
- [ ] HOST专项、受改旧接口有限回归、公开例程、源码外wheel。

## B3 中高：静态/平移 mesh–sphere/capsule 真设备逐面候选

沿现mesh_midphase='bvh'显式开关，复用CollisionTrees.facets和现TYX TreeBuffers/refit_boxes/bvh_candidates。默认all_facets保持。双旋转保持原完整枚举，本片不声称全设备窄相。

作者NEW：src/phymeo/geometry/_mesh_round_candidates.py；tests/collision/mesh_round_candidate_cases.py；tests/collision/test_mesh_round_candidates.py；tests/collision/test_tyx_mesh_round_candidates.py；docs/benchmarks/p65-mesh-round-candidates.md。
父仅geometry/collision_pipeline.py、geometry/_mesh_round.py。

MeshRoundFaceSelector(shape,prepared_mesh,query_boxes)，sample(start,end,radius,vertex_distance2)与translation(start,end,drift,radius)返回原facet表排序唯一索引。callback接绑定/向外FP32面盒与查询盒，原设备驱动真实回读，无HOST fallback。静态U²=min完整顶点到有限轴段的精确距离²，上界pad=max(r,sqrt_upper(U²))，包完整最近面、所有并列及半径内边界；旧精确HOST下界筛除仍保留。平移用四端点扫掠盒扩radius。ClosedMesh全拓扑/inside、全顶点furthest、round_contains_mesh全部最远支持始终全量，绝不把子集当闭体。sample空集或坏回读明确失败。面数/原精度/资源上限不变。

- [ ] tests-only先提交；父有效RED证明旧bvh没有mesh–round面级调用。
- [ ] 盒[-1,1]³、球(1+3/32,0,0),r1/8→gap−1/32；有限轴杆身/端帽/轴外留出、轴穿面、完整并列。
- [ ] 球(-2,0,0)→(2,0,0),r1/8首次7/32；两端分离内部命中。
- [ ] 小圆体完全inside、大球包含mesh、凹U留出、整体协变/换序/重排/跨world/同key坐标更新/FP32边界。
- [ ] 108面细分盒独立全脸oracle，实际剪枝且不漏；容量/坏回读/设备异常不污染旧发布。
- [ ] HOST/interpret/CUDA分别完整5字段对all_facets；记录真实BVH计数，不预承诺速度。

## C 中：只在一次evaluate内复用相同输入

作者NEW：src/phymeo/solvers/_cloth_link_evaluation.py；tests/test_cloth_link_evaluation.py；scripts/benchmark_cloth_link_evaluation.py；docs/benchmarks/p65-cloth-link-evaluation.md。
父旧：solvers/cloth_link_contact.py与compliant_cloth_contact.py。不碰B候选接口。
显式local_reuse=False，program的cloth_link_local_reuse=False。新ClothLinkEvaluation在一次evaluate中拥有各link端点/CollisionMotion/COM线角速度、各face惰性CollisionMotion和节点速度；sample仍原精确static_capsule_face_sample，站点相对速度保持原算式/求和顺序。原law、散回、全owner/材料地址/记录不变。对象返回或异常即失效，不跨entry/mid/end、world、rebuild复用。

- [ ] tests-only首RED为既有构造器缺显式开关，而非缺import。
- [ ] 2/8面、多link/共享祖先及布节点、非连续ID/重排、真实变形和站点改变、质量/协变/重建/外部数组修改、退化/坏law输出和事务。
- [ ] 复用原固定场景与完整序列化，W1F8/W32F8做未插桩AB/BA/AB（各1cold+1warm+3formal、每窗口4步）；两端完整输出同，真实TYX页数/bytes保持；沿原32MiB/180s档预算。
- [ ] 无收益如实记录，默认不开；即使局部收益也不凭一档自动切换。

## 执行与收尾

- [ ] 三原树clean ff基线，tests先行→父唯一serial valid RED→隔离实现；父所有old hooks。
- [ ] 每项目命令fresh entrypoint-aware guard，保存source/command/raw exit/首失败；不kill，不重测完整绿同源。
- [ ] 原作者交叉只读审阅、父共享审阅，发现驱动必要修改和重验。
- [ ] 新stage一次build及源码外wheel组合使用，更新实际范围/开放项/续作入口，小提交不push。
- [ ] 全部执行槽释放，无新自动任务。

两轮没有新增可运行能力或更小复现则缩小具体节点/换方法，不堆手推、放宽oracle或重复绿组。A/B功能和C局部性能分别验收，不将本批称为前三大模块全部追平。
