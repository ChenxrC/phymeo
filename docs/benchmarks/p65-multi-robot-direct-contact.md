# 多机器人实体直接接触：HOST 首片

## 能力与输入边界

公开入口仍是 `ContactSceneBuilder.add_urdf(...)`、`pair(robot_a, robot_b, RobotContactMaterial(...))` 与 `scene.step(dt, actions=...)`。本片允许不同普通或 mimic 固定根机器人直接接触，并沿原多机器人事务与已有自由刚体共同推进。没有替代机器人、dummy 刚体、直接写入状态或预制接触力。

首片是单世界 HOST，至少两台真实活动树，启用树对的每个 collider 组合必须属于球–球、球–有限胶囊或球–OBB。注册时验证全部组合，不能因当前距离远而漏掉不支持的组合；异常包括双方物理 owner、link/shape 名称及形状种类。`pair(..., None)` 明确排除，不调用该树对的窄相。这里的 OBB 是导入资产实际正交盒，不将一般 hull 冒称盒。

直接机器人通道使用实体 collider。即使同一个盒还注册为指垫，也不在机器人–机器人通道另加一份指垫采样力。原机器人–自由体的指面/实体选择与全场一次刚体–刚体采样由父共享接缝保持。

暂不支持胶囊–胶囊、非球多点 patch、网格/圆柱/椭球、自碰、浮根、连续碰撞、CUDA 或新的持续切向历史。本片采用实际入步姿态的顺应力采样，不保证物理步内新碰撞的防穿透。球心与另一球心/胶囊轴重合、或盒内部无唯一分离法向时明确拒绝该步。

## 接缝和受力语义

新增 `_robot_robot_contacts.py` 仅装配载荷，不积分也不发布状态：

```python
sampler = RobotRobotContacts(participants, pairs, collision_model)
loads = sampler.evaluate(terms, collision_poses)
```

`participants[(world, body)]` 是原真实 `RobotParticipant`；`terms[owner]` 是当前共同入步的 `(full_state, geometry, com_linear_jacobians, angular_jacobians)`。结果 `RobotRobotLoads` 有 `by_robot`、`contacts`、`details`。每个 owner 的 `RobotContactContribution` 含只读的 `joint_effort_full`、`link_wrenches_world` 和 `fixed_root_contact_wrench`。全场 `contacts/details` 仅追加一次，不能把各 owner 的同一对载荷再次全场累计。

实际材料表面来自既有 `LinkSolid.witness`：有限胶囊在两球心之间夹取轴参数，OBB 用实际局部轴与最近面/棱/角。两侧皮肤点分别是 pa、pb，满足 `pb-pa=gap*normal`；共同作用点是 `(pa+pb)/2`。COM 速度加 `omega cross (point-COM)` 给双方在该作用点的实际速度。原 `sample_loads` 给弹簧/法向阻尼/滑速正则摩擦，双方等反力各自乘以真实 COM 力臂，再用完整 `J_com.T F + J_omega.T M` 进入各自树。

mimic 保持完整物理轴载荷，原 participant 在同一步使用 `P.T M P` 和 `P.T load`，保留原每行约束反力。新适配器不重建 mimic、不将从动坐标直接写成目标。根 collider 用真实根 frame 和零速度；`fixed_root_contact_wrench` 仅表示根 collider 的直接接触合力/根原点矩，**不是所有关节传到固定基座的总动力学反力**。

事件沿用 `RobotContactEventAdapter.for_robots` 的真实 owner 和入步 frame。球/胶囊顶点权重只定位中心/轴上地址，完整皮肤材料点必须同时使用 `ContactMaterialAddress.local_point_m` 与 owner frame 重建。施力点、两侧表面点和两侧 COM 均保存，不能混为同一个地址。`sample_time_s` 是载荷入步时刻，`commit_time_s` 是共同接受末态时刻。

父拥有 `multi_robot_scene.py` 薄接缝：把逐 owner 三类贡献加进原机器人–刚体载荷；所有树和刚体仍各推进一次，末态几何、拥有副本和 events 完成后共同发布。新模块不修改旧采样律、既有模型/惯量或原控制预算。

## 实现前冻结的独立案例

所有长度为 m、质量 kg、时间 s、力 N、力矩 N m。除单列摩擦/阻尼外，k=2000 N/m、阻尼和摩擦为零、重力为零、原力帽 128 N 或 N m。单步 h=0.001 s；力门 1e-10、状态/几何门 1e-12，以各被检查量的对应 SI 单位表达。用例先于新模块提交，不按运行结果调整。

| 组 | 固定构型与独立预期 |
| --- | --- |
| A01 | 两台 2 kg 滑轨，各实际球 r=1/16，球心距 .124，压入 .001；无自由体。双方 ±2 N，首步 v=±.001、q=±1e-6，势能 .001 J，双方皮肤 x=.0625/.0615，作用点 x=.062。 |
| A02 质量/变换 | 2/4 kg，仍 ±2 N，a=-1/+.5，质量加权动量零。交换登记次序；旋转到轴 (.6,0,.8)，平移 (3,-2,.75)，保留实际材料重建。 |
| A02 胶囊 | 胶囊轴 z∈[-.25,.25]、r=1/16；球心 (.124,0,0) 杆身与 (0,0,.374) 端帽均为 2 N；(.124,0,.375) 真分离，不能按无限轴误撞。胶囊通过已有 MJCF loader 导入实际 geom，未发明 URDF capsule 标签。 |
| A02 OBB | 半尺寸 (.1,.2,.3)，局部中心 (.125,-.25,.0625)、局部 yaw=pi/2；真实面、棱、域外控制。注册同盒 pad 与不注册的面控制载荷同为 2 N，不重复加力。 |
| A03 转动/功 | 2 kg 偏心转铰 Izz=1/8，球心 (0,.25,0)，对侧球心 (.124,.25,0)：无摩擦力矩 .5、角加速度4。4 kg Y 滑轨 vy=.02、mu=.25、slip_speed=.01 时，对侧力 (2,-.5,0)，a_y=-.125；首侧力矩 .531，第二侧 COM 矩 +.031，配对功 -.01 W。 |
| A03 阻尼 | 两侧相向速度 ±1/16、damping=8 N s/m，法向相对速度 -1/8，N=3，配对功 -3/8 W；弹簧能率 +1/4 W，阻尼损耗 1/8 W。 |
| A04 mimic | 两物理轴质量 2/1，P=(1,-2)，从动偏移 .003，实际从动球受 +2 N。Mred=6，独立加速度 -2/3，完整加速度 (-2/3,4/3)，反力 (-4/3,-2/3)，P.T 反力=0，约束功=0；另一真实 2 kg 普通滑轨仍受 -2 N。 |
| A05 连续窗 | 两 2 kg 实体直接压入，以额定内 +2/-2 N 平衡 32 个 h=1/8192 步；再以 -8/+8 N 外撤 8 个 h=1/256 步。逐步核真实冲量/一次漂移/离散动能公式，末四步无载荷、事件结束、两侧真实位移越过 ±1 mm。 |
| A06 事务与边界 | 另加 1 kg 球–有限台 .001 压入，独立 2 N 全场只计一次；固定根真实偏心 collider 的接触/力矩；第二树越限与晚期 event 异常全回滚，重试等同重建；重复局部 key、观测/外载拷贝和旧事件材料不别名；None 排除、远处不支持组合、胶囊–胶囊具名拒绝、无唯一法向拒步。 |

A05 检查半隐式更新的精确离散动能恒等式 `delta K = h F v_in + h² F²/(2m)`。它验证实际载荷和积分状态的一致性，不将有限步弹簧接触称为连续能量守恒，也不把预载保持称为从桌面获取或抓放。

## 公开例程及数据

`examples/multi_robot_contact.py` 使用同一 A05 输入，在两个本地真实 URDF、共同载荷与原驱动下连续执行。没有自由体、状态重置或力注入。可选 `--reverse-registration` 只改变登记顺序。JSON 保存原 URDF 字节、初态、命令/原力帽、各入步/出步时刻、两树 q/v 与完整 COM 扳手、原载荷和事件材料；初始/保持末/最终完整几何均保存。

```powershell
python -m pytest tests/test_multi_robot_direct_contact.py -q -p no:cacheprovider --basetemp <fresh-absolute-directory>
python examples/multi_robot_contact.py --output <fresh-absolute-json>
```

已有输出拒绝覆盖；实际拒步和任一原门未达返回非零。文件和 stdout 都先转换完整结果为原生 JSON 类型，包含 NumPy 布尔值；例程不隐藏执行异常或把负结果改成成功。

## 当前验证记录

- 测试首节点提交 `9af43c902483f6f5ead2c201ba8fe11be9c796df`。父 `02-a-public-red` 实际 **1 failed / 1.13 s / RAW_EXIT=1**：旧 build 明确拒绝直接 robot pair。入/出源码快照相同，失败来自真实公共能力缺口。
- 完整 A01–A06 tests-first 为 `34fb94bc8f756b746f47be5e5f16f13cc0c7c1eb`；新适配器首候选为 `75645996f5fc1876fab3085f7da0ad2e8ffdf765`。
- 父 `15-a-host-first` 首次 collection 为 **RAW_EXIT=2**，测试末端 capsule–capsule 参数列表少一个外层右括号，尚未运行物理节点。父仅补该括号后开始 `16` 首次完整 functional；原输入/断言不变，首错收据保留。
- 本报告落盘时 `16` 结果、公开 CLI、static、共享 hook 兼容与最终 wheel 均仍 **PENDING**；不把作者文本检查当作执行通过。后续实际结果单列追加，不覆盖首错。

原始记录由父保存在 `artifacts/p65-top-three-batch6-20261106-01/`。A 作者只进行了文本/文件 I/O/Git；所有 import、collection、pytest、数值、lint、format、build 由父唯一串行执行。本片没有新增 Newton 动力学/原生接触或 GPU 对照声明，既有 mimic 100 g 历史失持窗口及其参数均未改变。

## 父串行最终验收

以下结果更新候选阶段的 PENDING 状态，保留前述 `02` 有效 RED、`15` collection 首错和全部原始输出。最终产品源码为 `3e6dff494d898ad19a3a3695d1de0c368b511805`；本节仅追加报告，不修改产品、用例输入、材料、力帽或容差。

`16-a-host-functional-first` 首次功能执行为 **5 passed / 1 failed**。失败是 MJCF 胶囊正 z 端帽的标签 oracle：`fromto=(0,0,-.25)→(0,0,.25)` 经既有 `_fromto` 与 `make_collider` 后，实际保存的 `local_vertices` 顺序为 `[[0,0,.25],[0,0,-.25]]`，所以正 z 端是 `end0`。父仅把该标签及对应测试分支从 `end1` 修为 `end0`，并增加实际端点数组的精确断言；几何输入、2 N 预期和所有容差不变。`17-a-cap-endpoint-fix` 该节点 **1 passed / 0.67 s / RAW_EXIT=0**。该修正区分实际材料端点编号与世界正负方向，没有为通过而翻转资产或接触法向；首次失败仍在[父原始记录](../../artifacts/p65-top-three-batch6-20261106-01/16-a-host-functional-first/stdout.log)。

| 父作业 | 实际结果 | 含义 |
| --- | --- | --- |
| [18 完整专项](../../artifacts/p65-top-three-batch6-20261106-01/18-a-host-complete/stdout.log) | 25 passed / 1.05 s / RAW_EXIT=0 | A01–A06 的真实力学、材料、限域及事务留出全部通过 |
| 19–24 静态 | 19 Ruff=0；20 format=1，三个文件待格式化；21 repair=0，22 format check=0，23 Ruff=0；24 五份 raw AST 完全相同 | 格式处理前后语法树一致，未修改物理输入或断言 |
| [25 公开窗口](../../artifacts/p65-top-three-batch6-20261106-01/25-a-public-window/stdout.log) | 40 个实际步、11 项门全部 true、RAW_EXIT=0 | 原 32 步保持与 8 步真实外撤连续完成，`execution_failure=null` |
| [26 旧共享场景兼容](../../artifacts/p65-top-three-batch6-20261106-01/26-a-shared-scene-compat/stdout.log) | 21 passed / 0.87 s / RAW_EXIT=0 | 原多机器人场景及父薄接缝兼容；不据此宣称本片未支持的形状已可用 |
| 47 构建 | RAW_EXIT=0 | 从最终产品源码构建 wheel |
| [48 源码外 wheel](../../artifacts/p65-top-three-batch6-20261106-01/48-outside-wheel-integration/stdout.log) | RAW_EXIT=0 | 反序登记的真实 40 步/11 项门，以及 B01 双转动首触均通过；全部已加载 phymeo 模块来自 wheel |

公开窗口最大冲量误差 **2.7755575615628914e-17 N s**，一次漂移误差 **2.168404344971009e-19 m**，离散动能误差 **3.469446951953614e-18 J**，保持力误差 **1.7763568394002505e-15 N**。合接触力、总线动量和点力虚功的最大误差均为 **0**。这些是实际保存窗口的各项读数，不是重新拟合的容差；完整[同源轨迹与几何](../../artifacts/p65-top-three-batch6-20261106-01/25-a-public-window/result.json)保留。

验收分为作者新模块/专项/例程与父旧共享接缝两层。新模块负责双真实 owner 的有限几何接触、共同作用点速度、各自 COM 矩及完整关节载荷；父 `multi_robot_scene.py` 负责把该载荷与原机器人–刚体及一次刚体–刚体载荷合并，再按原候选/共同发布事务执行。事件继续使用原双侧 owner/material schema。B 原作者对 A 作独立只读审阅，未发现阻断；A 对 B 的双前向旋转、全路径速度界、有限轴/双侧材料、未知/容量与旧路由作只读审阅，也未发现阻断。两项审阅不记作额外物理执行。

最终 wheel 大小 **1,115,791 B**，SHA-256 为 `5cd405e970bd1ef5709769d7b439f143613be69c23c1b9cc58c87c11b775aa09`。源码外验收实际使用该 wheel 中的模块，没有用源码 PYTHONPATH 冒充打包通过。本片范围仍为单世界 HOST 的普通/mimic 固定树、球–球/有限胶囊/OBB，以及原自由体同事务；没有新增机器人 CUDA、浮根、自碰、CCD、多点非球直接接触或一般抓取成功声明。所有执行由父串行完成，作者仅读取既有收据并追加本文，没有重跑项目。
