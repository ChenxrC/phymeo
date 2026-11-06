# 浮动根共享机器人场景：Batch 7 A3

本节点把已有真实自由根与标量关节树动力学接到同一 HOST 场景事务。根与子链使用原完整质量矩阵、解析惯性偏置和根 COM 漂移；没有虚拟六关节、独立替身刚体、状态注入或第二次漂移。原固定树、mimic 入口和历史抓持失败保持原语义。

## 公开入口与边界

`ph.load_floating_urdf(path, gravity=...)` 返回 `FloatingRobotAsset`。`ContactSceneBuilder.add_floating_urdf(body, path, robot_state=..., kp=..., kd=..., effort_limit=..., world=...)` 与原固定树/自由体注册共存。`scene.step(dt, actions={body: ph.FloatingRobotCommand(joints, external)}, external=...)` 返回同一接受步的全部观察；原 scene-level external 仍用于独立自由体。

`joints` 是原 `RobotCommand`，根-only 为 None；其前馈与 PD 合并后受同一个额定帽约束。wrapper 的 `external` 是 `FloatingRobotInputs`，body0 为根、i+1 为子连杆，force/torque 均在世界系，力矩关于各体真实 COM；其中 effort 必须为零，内部驱动不能通过另一通道叠加。未传外载为零 COM 载荷，不代表关闭重力。

完整 q 为根原点 XYZ[m]、body-to-world XYZW、子关节坐标[m/rad]；完整 v 为世界根 COM 速度[m/s]、世界角速度[rad/s]、子关节速度。`body_count=1+n`、速度长度6+n、配置长度7+n严格分开。URDF 根质量及惯量真实保留，偏心 COM 允许；惯量必须在 authored link 轴中对角且为合法正惯量，不能仅取特征值后丢掉朝向。root-only 无虚假子关节。首片明确拒绝浮树 mimic、固定关节折叠、pads、非对角根惯量；普通固定树的上述既有能力不受影响。

HOST FP64、单 world，真实普通 bounded scalar joint 树。检测继续使用当前同态柔顺接触与原形状支持域，浮–固定/浮–浮允许球对球、有限胶囊、OBB；其他组合由原编译检查拒绝。没有宣称 CCD、自动防穿透、自碰撞、浮树 CUDA 或切向历史适配。加载器可以保留旧几何语法，场景可受力域仍由实际 sampler 决定。

## 物理接入

参与者只做 pure preparation/candidate：以相同入步 q/v 读取全部 COM/角 Jacobian 与完整 M/b。子关节 PD 的自由预测消去真实六维根加速度，使用 Schur `Mjj−Mjr Mrr⁻¹ Mrj`；已声明外载进入该预测偏置，接触在其后按共同入态采样。缺省前馈补偿该有效偏置，显式前馈保原值。实际根和所有子链由原 `step_floating_robot` 一次推进，接触 COM wrench、外载、内部 effort 各一次。

接触 view 使用实际动态 body0 根、body i+1 子，原碰撞 key 另行全场唯一映射。浮根事件为 `robot_root / robot_model_link / index0`，材料点从入步根随动 frame 重建；原固定根仍是静态 frame。球皮与胶囊皮肤地址使用 local_point_m，不能只凭顶点权重重建。

观察 q/v 与 full_q/full_v 均为完整浮树状态，entry_q/entry_v 属载荷时刻；drive_effort为受限子关节力，contact_effort_full为6+n，link_wrenches_world为(1+n,6)接触 COM wrench，external为独立不可变载荷，applied_effort_full为接触+外载+驱动广义力，acceleration_full读实际推进增量。重力和惯性在 bias，不混入applied。根不固定，因此 fixed_root_contact_wrench 为零。场景所有候选、时钟、几何、事件统一接受或统一保留原值。

## 预先冻结用例

| 组 | 输入与独立期望 |
| --- | --- |
| A01 | 根2kg+子1kg X滑轨，根球r=1/16、4kg自由球距.124m、k=2000N/m、h=.001s。真实2N接触，Mx=[[3,1],[1,1]]，根a=-1、相对a=+1、子世界a=0、自由球a=+.5；净动量0。 |
| A02 | 根-only COM偏置1/8m，F=2N、Tz=.75Nm、Izz=.75，分别a=1、alpha=1；另入步omega_z=2且vCOM=0，原点绕静止COM移动。内部±3N以及2N帽的根反作用独立验。kp=100、target=.01m时Schur有效质量2/3kg，实际effort=20000/20003N。 |
| A03 | g=(0,0,-1)，全树自由落体相对q不变；外载根+2N/子−1N、内部+3N时根ax=-.5、子世界ax=2。两边相向1/16m/s、damping8N·s/m得到3N、rate=-1/8、功率−3/8W。 |
| A04 | 世界旋转/平移/共同boost、反序注册；浮根与固定普通或负倍率mimic；两浮树root/child接触；独立球–台面力只算一次。有限胶囊轴部/端帽/轴外近失与旋转偏置OBB面，真实材料重建。 |
| A05 | 根2kg+子1kg Z滑轨，g=-1，初始根球压入台面1.5mm，32步h=1/8192、内部+1N，真实3N支撑且全树静止；随后8步h=1/128，根/子COM外力16/8N，无重置、真实分离。每步双方冲量、一次漂移、离散动能等式；末4步无接触，子关节保持独立运动。 |
| A06 | 第二浮树越限、后置事件异常、非法外载/命令、重合无唯一法向均不得发布半步；修正输入重试与重建一致。loader原asset、两个build、已返回状态/外载/模型隔离；根惯量/浮mimic/pads/设备等具名拒绝。 |

单步状态/漂移/动量绝对门1e-12（相应 SI 单位），接触力门1e-10N；实际窗口同门。输入和阈值均在实现/执行前提交，不从测量挑容差。测试源码为 `tests/test_floating_robot_scene.py`，公共例程 `examples/floating_robot_contact.py --output <json>`，可加 `--reverse-order`。例程每帧保留实际状态、载荷、命令、动态材料点、事件与同锁几何；最终JSON先统一转换 NumPy 标量，物理验收不由序列化隐式放行。

## 当前验证记录

- tests-first首节点21a9289，由父 `02-a-valid-red` 真执行：旧URDF解析成功，随后公开builder缺少add_floating_urdf，RAW1。该首错是功能缺口，非导入失败。
- 完整专项冻结52c5e67；两个NEW source检查点8af32b6。作者未执行项目import/test/numeric/lint/build。功能、静态、公开例程及源码外wheel结果等待父串行收据，当前不写通过数。
- 作者只拥有本节点5个NEW文件；旧builder路由、单/多机器人共享采样计数、接触view、动态根event、状态复制/事务与公共导出由父独立维护和验证。
- 没有本轮Newton独立浮树动态运行，也没有把旧浮树CPU/CUDA对照当作新共享场景验收。原100g mimic失败窗口没有重跑或修改。
