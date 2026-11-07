# F3 浮根惯量主轴与 authored 坐标分离

状态：T0源码调研完成 / T1测试先行，2026-11-07；无截止。父负责共享代码，原robot继续C4、原RD继续持续摩擦，不新增助手；全部项目执行仍父单一串行槽。

## 调研结论与实现路线
本机Newton1.7 `.venv-newton-1.7.0/Lib/site-packages/newton/_src/sim/builder.py:5301` 的add_link接受关于COM的完整3×3惯量；`sim/model.py:968`保留完整tensor。这里不需迁移到Warp或改变接触律。
本仓`FloatingRobotModel`假定root body轴就是主轴；`floating_tree._inertial_geometry`把所有惯量直接沿geometry.rotation旋转，root-only `step_free_root`则委托既有RigidSemiImplicitSolver。`floating_urdf._floating_from_tree`及`_floating_fixed_asset._root_group`因此拒绝合法非对角根惯量。子树已用原静态主轴装配，不能把根的特征值保留下来却丢掉主轴朝向。

选用独立的root惯性frame，保持q的原authored root-origin/body姿态、root COM世界速度与世界角速度、全部child frames/J/材料几何语义。FloatingRobotModel新增keyword `root_inertia_frame_xyzw`，默认identity，表示principal→root-body旋转；owned不可变。principal moments继续在model.root中；实际body tensor为Ri diag(I) Ri.T。场景消费完整M/J，无新虚构关节/粒子、无重设速度。
共享M/b只在根惯性旋转上复合Ri；root-only沿原RigidSemiImplicitSolver使用world principal姿态，返回时转回authored姿态，COM/世界w不变。identity路径保持原行为。任意Quaternion符号/重根主轴不改变物理tensor，比较几何和惯量而非eig向量唯一性。

## 可独立交付节点
1. **T1 核心frame（中）**：NEW tests/test_floating_root_inertia_frame.py；修改dynamics/floating_robot.py、solvers/floating_tree.py、solvers/free_root.py。先有效正常构造RED，再实施。独立完整tensor/陀螺矩/COM漂移/耦合M-b-a和40步窗口核；错误frame、owned读数和拒步保原态。
2. **T2 URDF与固定工具（中）**：复用原_mass_component/_principal等严格静态装配，普通/折叠/mimic loader共用，不复制XML parser。原root body frame保持，折叠保原source/sites/材料。新增合法非对角root正控替代原“合法但未支持”的拒绝项；无效惯量仍拒绝。不靠新夹具flag隐藏能力。
3. **T3 场景与交付（中高）**：实际共同场景受载/卸载、mimic保full根行、坐标协变与失败重试；必要旧共享回归、Newton保原部件CPU/CUDA独立对照以及新源码外wheel。只有全部实际通过后关闭F3；T1不能冒loader或全场景完成。

## T1固定多例与成功门
- 主矩(1/4,1/2,1/2)，Ri=Rz(pi/4)，body I=[[3/8,-1/8,0],[-1/8,3/8,0],[0,0,1/2]]，root质量2，COM=(1/8,-1/16,1/32)。静止施世界Tx=1，原姿态identity时alpha=(3,1,0)。独立3×3 solve及world变换对照。
- 非零world omega=(1/4,-1/8,3/8)，force=(1/4,-1/8,1/16)，torque=(1/8,1/4,-1/16)，g=(0,0,-1/2)。identity和xyz循环姿态/共同旋转；40步h=1/1024，独立世界tensor Euler+Rodrigues，比较每步COM/姿态/v/w及动能；完整保存原读数，不称严格连续能量守恒。
- 加一真实Rz子体，offset=(1/2,0,0)、m1、COM=(1/4,0,0)、I=(1/16,1/8,3/16)，q=.37、qd=.5。直接解析两个COM Jacobian、qdd=0解析惯性加速度、完整world I构造7×7 M/b和真实COM外载；核全M/b/投影/a，不能从产品M做自己的oracle。
- shape错误/非单位/非有限frame拒绝，输入数组改动/解除writeable不得改变model；非有限物理候选拒步保原q/v。默认identity及等价负quat兼容。
FP64这些有界核门：M/b/a rtol2e-11/atol2e-12；40步q/v/姿态/COM atol2e-10。全部门在首次执行前固定，不调原质量或窗口求过门。T2/T3固定完整输入在各自测试checkpoint补足，禁止先跑probe选夹具。

每节点须新增正常可调用能力或明确缩小失败；两次修订仍无此结果则保最小复现并改路线，不追加无限精确接触证明。性能不作本片放行条件，原100g mimic持稳和软布两初态失败继续保留。
