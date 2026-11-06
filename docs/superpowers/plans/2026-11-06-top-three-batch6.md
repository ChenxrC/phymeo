# 第六批：机器人直接接触、双转轴检测与材料分页成本

> 执行：三原作者在隔离工作树编码；父唯一串行项目执行与既有共享文件 owner。用户已授权继续实施，无需逐阶段再次确认。

**目标：** 从第五批 f4695e4 出发，新增两个可复用功能，并用实测决定第三项优化。多个独立案例是验收入口；递布或 Panda 单一示例不代表通用覆盖。功能优先，性能不强制，不修改物理预算/依赖/pin，不执行远程 CI。

**当前调研：** MultiRobotScene 一律拒绝启用 robot–robot pair；mesh–round 在双方转动时拒绝；TyxNormalLayer 每 128 行完整传入13列/读回5列，页尾也全传且每页同步。WorldContactEngine 已按相位合批，不虚构逐世界同步瓶颈。三处由本仓库源码确认。Newton 1.7 的 [碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[关节](https://newton-physics.github.io/newton/1.7.0/concepts/articulations.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html) 用于模块参照；其当前姿态接触不作为连续首触真值。

## A2 高：不同机器人之间的真实双侧受力

保留公开 ContactSceneBuilder.pair + step(actions=...)。首片 HOST 单世界普通/mimic 固定树，完整检查启用树对的 collider 组合，支持球–球、球–有限胶囊、球–OBB；其他组合 build 时具名拒绝。实体直接接触，pad 不在此通道重复采样。不合并为虚拟机器人，不注入替代接触力。

作者 NEW：runtime/_robot_robot_contacts.py、tests/test_multi_robot_direct_contact.py、examples/multi_robot_contact.py、docs/benchmarks/p65-multi-robot-direct-contact.md（源路径均在 src/phymeo 下）。父仅接 multi_robot_scene.py：注册校验、同入态采样、双方完整 COM wrench/J转置合入，每树及刚体一次推进、一次共同发布。现有 event owner/frame 复用，不另造记录格式。

RobotRobotContacts(robots,pairs,model).evaluate(terms,poses) 返回逐 owner 全坐标载荷与唯一 contacts/details。两侧在共同见证中点用真实刚体速度，反向力、各自真实 COM 力矩及完整关节转置；mimic 由现有 participant 投影/观测真实约束反力。

- [x] A测试先行，公共 build 缺口的有效 RED 保存后再实现。
- [x] A01 两2kg球形滑轨，r=1/16，中心距0.124，k=2000、阻尼摩擦0：±2N；1ms后 v=±0.001、q=±1e-6。无自由体或 dummy。
- [x] A02 2/4kg、登记换序与刚体协变；有限胶囊轴z±0.25的杆身/端帽/轴外留出；OBB面、棱、域外。
- [x] A03 偏心转铰Izz=1/8，球心(0,.25,0)，另一球心(.124,.25,0)：无摩擦力矩.5、角加速度4；4kg Y滑轨vy=.02、mu=.25时Ft=-.5、ay=-.125、首侧力矩.531、配对功-.01。
- [x] A04 mimic两质量2/1、P=(1,-2)、偏移.003、从动受+2：Mred6、独立加速度-2/3、完整反力(-4/3,-2/3)，P转置反力0。
- [x] A05 额定内±2N平衡32步h=1/8192，再向外8N共8步h=1/256：真实冲量/两owner/末四步零载荷与结束事件。
- [x] A06 独立球–有限台2N只计一次；None排除、重复局部ID、根接触、第二树越限/事件失败全回滚与重建重试一致。
- [x] 完整专项、受改共享兼容、公开例程和源码外wheel通过，报告限域。

胶囊–胶囊、非球多点patch、mesh/圆柱/椭球、自碰、浮根、CCD、CUDA及新持续摩擦不在此片，不宣称机器人模块全部完成。

## B2 高：闭网格与圆体双方独立旋转

新增 _dual_turn_mesh_round.py 中 DualTurnMeshRoundQuery(prepared_mesh,round_shape,mesh_path,round_path,mesh_first)，复用 MeshRoundQuery 五项结果和闭体材料语义。父只接 collision_pipeline.py 双转动分派。在网格初始材料坐标中包围两次真实旋转合成及相对移动枢轴；有限胶囊轴而非无限直线。HOST保守窄相，TYX实际宽相。角度/采样/误差/cell限制沿现有上限，未知不得当作分离。

作者 NEW：该模块、tests/collision/dual_turn_mesh_round_cases.py、test_dual_turn_mesh_round.py、test_tyx_dual_turn_mesh_round.py、docs/benchmarks/p65-dual-turn-mesh-round.md。

- [x] B冻结输入/oracle，公共路由有效 RED 后实现。
- [x] B01 垂直非交换转轴：闭盒[-2,2]²×[7/8,9/8]，胶囊(0,0,0)→(0,1,0)、r1/8。mesh绕X转-2A，round绕Z转+A，A为存储pi/2。局部tip=(-sin a,cos(2a)cos a,sin(2a)cos a)，首触2sin a(1-sin²a)=3/4，首次sin a=1/2，t=pi_real/(6A)。独立pi区间给括区；不能直接硬编码1/3。
- [x] B02 同一输入的偏心球中心(0,1,0)，真实两轴、端点分离内部命中；胶囊还核真实朝向改变与完整有限轴材料。
- [x] B03 不同移动枢轴/角度、凹槽留出、整体包含/开离、真分离/预算未知；独立有限距离与闭体oracle不调用新producer。
- [x] B04 换序/镜像/整体刚体变换/重复局部ID/跨world；双侧材料重建、最早区间及占用见证逐项核。
- [x] B05 实际interpret/CUDA宽相与公开结果，受影响旧路由有限回归；不称响应/摩擦流形已完成。
- [x] 作者完整边界成本说明、父审阅与源码外wheel。

## C2 中高：先测材料分页，再决定一次相位传输

只测新问题：原TyxNormalLayer固定128行分页，TyxMultifaceContactBatch各相位串行调用所有页。原W4×8下载0.60/162.36ms不足以证明优化价值，不重跑旧同集。作者先新增固定协议、profile脚本、协议测试及报告；父串行真实CUDA测 >128 边界及W1/W32×2/8面、W3异构任务。所有物理状态、材料地址、完整输出和失败事务保留。

- [x] C0 固定测量矩阵/输入/阶段计时/输出一致性协议、测试及有效RED（仅新协议功能）。
- [x] C0 实际基线：分别记冷/暖、页数、launch/sync/bytes及完整step成本，禁止把重叠插桩时间相加充墙钟。
- [x] C成本门：只计第二页以后传输/提交/同步/读回的排他墙钟，占完整真实任务>=15%才研究一次相位传输；这只是候选成本上界。W32×8实测0.15468%，门未过，默认保持。
- [ ] **本批未触发：** 若过门，另行精确new-file授权，先独立固定FP32行与原内核逐字段一致；预分配/尾页/空/异常重试/拥有旧帧/容量完整覆盖。
- [ ] **本批未触发：** 新实现的同输入完整输出配对AB/BA/AB，报告退化小档和实际受益档的p50/p95/全步成本；默认切换需重复端到端收益。
- [x] 公共必要观测复用原材料身份与phase schema，不为优化建立重复记录体系。

## 父集成与止损

- [x] 三原树clean正常快进本计划；仅作者新增文件可并行，旧shared父独占。先tests author，父RED，再code。
- [x] 全部import/pytest/数值/profile/static/build父串行fresh guard；其他项目执行自然结束，无kill。
- [x] 原A/B作者互审、B作者另审C、父审C与共享集成；只按发现修改和必要受影响验证，不重跑同源已绿组。
- [x] 新stage一次build和源码外wheel组合验收；本地小提交、更新实际功能差异/续作入口，无push。
- [x] 每线真实结果、初失败和开放项入新 artifacts/p65-top-three-batch6-20261106-01。

同一节点两轮无新增可运行能力或更小复现，就调整具体接口/算法/范围；不增加长篇手推、弱化测试或重复绿组。A/B的功能验收与C性能成本门分别签收，不能称前三大模块追平Newton。
