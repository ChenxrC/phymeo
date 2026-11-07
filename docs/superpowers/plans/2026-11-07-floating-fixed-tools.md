# 浮根固定工具装配：开发与独立测试

状态：ACTIVE / CODE_AUTHORING。用户已取消时限并授权持续推进。当前 C2 滑轨设备宽相与本节点的 NEW 文件可隔离编码；所有执行仍归父唯一串行槽。功能目标是复用现有浮根动力学与共同场景，使固定安装的工具真正贡献质量、COM、惯量、碰撞位置及双方载荷。

## 调研结论与实施范围

现有 floating_urdf 显式拒 fixed/mimic；普通 _urdf_fixed 将静态根组惯量检查后丢弃，不能直接包装成浮根。原 FloatingRobotModel/Participant 已有完整根–子树 M、bias、COM Jacobian、受限驱动、一次积分和共同发布。复用这些能力，不添加另一积分器或虚构关节。

首片完成 fixed 装配，mimic 后续使用 diag(I6,P) 接入完整系统。首片保持公共 root q 对原根坐标系的含义：合成根 COM 惯量须在原根轴中对角，不能为方便悄悄转 root quaternion。非对角根明确拒绝并列为后续主轴桥任务；移动子组可复用已有主轴转换。根可只有固定成员且没有活动关节；正质量为整组要求，合法无质量固定标记不能制造自由度。

依据实际源码：[浮根 loader](../../../src/phymeo/builders/floating_urdf.py)、[固定组装](../../../src/phymeo/builders/_urdf_fixed.py)、[浮根参与者](../../../src/phymeo/runtime/_floating_robot_participant.py)、[浮根几何与雅可比](../../../src/phymeo/dynamics/floating_robot.py)。参照锁定 Newton 1.7.0 的未折叠 fixed/free 模型；既有 Newton fixed-collapse 惯量反例继续保留，不把其有误输出当 oracle。

## API 与坐标合同

父公开入口 load_floating_urdf(path, *, gravity=..., collapse_fixed_joints=False)。参数只接受 bool/np.bool_；False 保持旧行为；True 调用 NEW 模块 folded_floating_robot(name, root, links, order, incoming, gravity, path)。旧 loader 仍负责统一 XML/tree 检查；mimic 在两路均明确拒绝。NEW 模块可复用旧低级 _mass_component、_principal、_origin_pose、_joint_data、assemble_robot，不修改其共享实现、不调用固定根 folded_robot 冒浮根。

NEW FoldedFloatingRobotAsset 继承 FloatingRobotAsset：
- 保留所有 collider 的 authored link/name、原 material IDs 与 local_vertices；collider.link_index 以旧资产约定 -1 为根组、i 为活动子组。同组 shape ordinal 唯一，固定工具 key 必须归真实组。
- sites 为不可变全 authored-link 表，每项 FloatingFixedSite(body_index, local_position_m, local_orientation_xyzw)，body_index 使用真实浮根体序号 0/root、i+1/child；所有值 owned，非 caller 数组别名。
- authored_to_model_points(name, points_m) 返回该 named link 的原始局部材料点到所属真实组原点坐标；root 工具也返回 root-local，绝不返回静态 world。
- authored_link_pose(name, q) 返回真实世界原始链接 pose；collision_motion(q) 从同一 FK 和 sites 构造完整原材料点及 curved orientation。q_end 继续明确不支持，不能用端点弦替代旋转路径。
- 根质量、COM、惯量采用所有 root-fixed 成员及平行轴项；子组同理。固定链后接活动关节时完整组合 joint origin/axis；主轴转换同时作用关节、COM、sites，不能只转惯量。
- 合成根对角判定沿原 principal_inertia 的明确 FP64 尺度误差处理；真实非对角量不得被裁剪。保留 SI 与原限制/驱动力帽。

父旧接缝仅 floating_urdf.py 和 _floating_robot_participant.py：显式 opt-in 分支；contact_view 从已冻结 asset 传递 authored_to_model_points 适配，旧普通资产为 identity。几何在真实 entry pose 采样，原 key 重映射、完整 J 转置、一次积分、事务保持。若正文实现发现必须改变更多共享文件，先给具体依赖，不散改。不改 C2 文件、默认摩擦、旧力律或浮根积分器。

## 作者文件与顺序

原 RD agent / p65-world-scale，start4392249 clean。恰6NEW：
1. src/phymeo/builders/_floating_fixed_asset.py
2. tests/test_floating_fixed_tools.py
3. tests/fixtures/urdf/floating_fixed_composite.urdf
4. tests/fixtures/urdf/floating_fixed_probe.urdf
5. examples/floating_fixed_tools.py
6. docs/benchmarks/p65-floating-fixed-tools.md

先写固定输入/独立预期与测试 checkpoint，再源码、例程和报告。作者只文本/手推/file IO/Git；不 import/collection/probe/pytest/lint/format/build，不 merge 父、不改旧文件、不新增 agent。交接 CODE_READY_TESTS_PENDING，父读全部新增与 scope 后补旧入口。测试 missing module 不算行为 RED；新模块存在而旧公开入口尚不接受 opt-in 时可记录入口缺口，另以未正确承接工具材料坐标的真实受载场景检验接缝。

## 四组独立案例与成功门

F01 全固定自由组合体：root m=2 kg，COM=0，I=(1/5,3/10,2/5) kg m²；tool m=1、COM=0、I=(1/10,1/10,1/10)，相对 root 平移(1/4,0,0)。聚合 M=3、COM=(1/12,0,0)、I=(3/10,53/120,13/24)。tool 原点施 F=(0,3,0) N，转换到组 COM 的力矩为(0,0,1/2) N m；零速、零重力时 a_y=1、alpha_z=12/13。仅7个q/6个v，无虚构scalar。核材料点速度/J虚功、真实首步反应与至少1/4秒非重置窗口；方向/整体刚体变换留出。

F02 非 Panda R+P 探头：另一树含 root 固定安装座、revolute 子组、prismatic 子组和各自固定有限工具。质量/尺寸/姿态/命令在作者 tests-first checkpoint 和报告中先写死，以部件平行轴聚合和独立串联变换给 oracle，不从产品生成期望。保至少一个非零旋转固定工具、偏心 COM 和固定成员上的原 shape identity。自由外载及根–关节耦合用完整 M*a+b=tau 和独立总动量/功率核；不能只检查有限值或运行不报错。

F03 真接触窗口：F02 工具与独立自由物体接触、受载、撤离，双方真实运动，无 weld/pin/位置回填。按原碰撞模型共同力点核作用反作用、关于共同原点的总矩与各 owner 实际 J 投影；累计冲量对应观察到的动量变化。h 与 h/2、反向登记和刚体变换留出；持续1/4秒以上且有出生与离开事件。若固定输入未产生预声明阶段，保留真实失败、查原因，不按输出偷偷改 oracle。显式加载新 loader 进入原 MultiRobotScene，不能只用私有 helper 单测替代。

F04 事务与拒绝：全部固定成员无正质量、非对角 root 合成惯量、非法 fixed joint 控制/mimic、错误 frame/shape ownership、源数组修改、第二参与者拒步。失败时全部 q/v、time、events 与 history 不发布；修复重试与独立实例一致。至少两个独立场景不可共享可写资产缓冲。合法无质量标记及root-only是正控。

几何/惯量解析门先固定，解析 FP64差允许1e-11级尺度舍入；物理窗口采用原同类场景门，作者报告逐项明确，不临时扩大。功率一致性不称能量守恒。运动精度以合理 h 收敛和实际载荷为准，不再用精确浮点相触阻止整个框架。

## 验收与后续

父唯一串行槽：有效 RED、必要诊断修正、全新专项 HOST、公开窗口、受影响旧入口/参与者必要节点；不重跑旧完整绿集。复用已有环境。最终原 agent 交叉审阅、同阶段新源码外 wheel；Newton CPU/CUDA 参考只按新需要且串行，未运行不冒比较通过。本片是 HOST 资产/动力学场景，不自动称 CUDA 浮根支持。

完成条件为真实固定工具进入通用场景、独立多案例与事务通过、报告边界清楚。两次修订无新增能力或缩小具体失败时拆最小复现换路线，不能继续扩证明。后续浮根 mimic、任意根惯量主轴桥、转动机器人设备驻留仍独立推进。

## F03 authoring前固定输入

F02 采用 root→fixed mount→Rz arm→Px carriage→fixed rotated tool，另带无质量 witness。F03 joint q=(0,1/8)，全树初始 root COM v=(0,1/4,0)、角速与joint rates=0；1 kg自由球位于 tool 世界+Y有限侧面外1/1024 m，r=1/16 m。原接触law k=2048 N/m、d=0、mu=0；零重力、外力和drive，T=1/4 s，h=1/2048与1/4096 s。验收物理入触、受载、反弹离开；反弹卸载满足本节点撤离，不额外添加主动轨迹。q/v不复位、不回填；初始解析构型、整个窗口的接触冲量、根/关节/freebody反作用、总动量/共同点矩与功率分别存证。若此预声明窗口失败，保首错并分析真实原因，禁止按输出重设阶段或oracle。F01另保独立持续外载正控。
