# F2 浮根 mimic：共享坐标映射、完整动力学和共同场景

状态：TESTS_FIRST_AUTHORING。用户已取消08:00截止，持续推进；当前只授原RD四个NEW案例/测试文件，产品实施在父审阅冻结输入和接口后接续，无需用户重复逐阶段授权。C2/F1已正式限域接受，证据见[固定工具](../../benchmarks/p65-floating-fixed-tools.md)和[滑轨宽相](../../benchmarks/p65-resident-slider-collision.md)。

## 为什么是下一节点

现有固定根 mimic 支持仿射无环图、被动约束反力和独立电机命令；浮根共同场景已有 full M/bias、根COM漂移、完整接触雅可比和一次原子发布。缺口是组合：浮根登记和参与者仍把物理活动子体数与独立控制轴数视为相同。F1只处理fixed刚性组，不替代mimic机械约束。先用非Panda案例补这一公共组合，不改变摩擦模型，也不把原100g mimic持稳失败写成解决。

依据实际源码：[固定根mimic图及反力](../../../src/phymeo/dynamics/joint_mimic.py)、[mimic XML](../../../src/phymeo/builders/mimic_urdf.py)、[完整浮根动力学](../../../src/phymeo/solvers/floating_tree.py)、[浮根参与者](../../../src/phymeo/runtime/_floating_robot_participant.py)、[接触维数约定](../../../src/phymeo/runtime/_robot_robot_contacts.py)、[共同提交](../../../src/phymeo/runtime/multi_robot_scene.py)。

## 已选技术路线

令完整物理关节 qj=P u+c，S=diag(I6,P)。完整tree保所有真实root/child质量、COM、惯量、固定工具和材料owner；独立 q=(root xyz,root XYZW,u)，v=(root COM速度,世界角速,udot)。公开读回另保存 full_q/full_v，不能把独立joint_names伪装成原FloatingRobotAsset的完整body列表。

在同一入态计算完整 M、b、全部接触/外载的J转置，形成 Mr=SᵀMS、br=Sᵀb、tau_r=Sᵀtau_full。PD在投影后的矩阵里消去真实根6维，电机只放在独立原轴；从动的限位/速度按原仿射图逆映射求交。根6维与joint同一次速度更新，复用原 _drift_floating_state 的COM/四元数规则；不能先推进无约束完整树再覆盖从动坐标。常数P使关节加速度同为P udotdot，所有根–关节交叉项保留。

完整残差 M a+b−tau_full须恢复为Gᵀlambda，G=[0,C]包含全部原mimic行；根6维约束残差独立核零，Sᵀ反力及每行/总约束功可核。零驱动、外载、真实摩擦等来源分开，不用约束反力补贴外部能量。沿原工程误差和h细分验收，不要求有限精度端点精确相触。

资产拟提供 load_floating_mimic_urdf(path, *, gravity=..., collapse_fixed_joints=False)，内部完整tree为FloatingRobotAsset或F1派生类型；提供 matrix/offset_vector/constraint_rows/independent_indices、full_state/reduce_state及 step(state, *, effort, dt_s, external=None)。step返回 state/full_state/loads；full载荷/加速度数组均6+n_full，独立effort是k维。external为完整FloatingRobotInputs，effort必须零，COM力/矩按全部真实body输入。场景复用已有builder loader入口、FloatingRobotCommand与原事件事务；不另造scene或积分器。

坐标图只维护一份通用实现：拟NEW dynamics/_mimic_coordinates.py的纯 compile_mimic_coordinates(names,types,lower,upper,velocity,effort,relations_tuple)，返回不可变映射/拓扑顺序/联合限位。parent让旧MimicRobotAsset委托；旧MimicRelation类型及公开路径保留，helper接raw tuple避免循环和类型迁移。不得复制两套图或忽略cycle/重复从动/同轴型/非零倍率/溢出/空限位检查。

participant.velocity_count仍为完整6+n_full，供原RobotRobotContacts完整v/J检查；独立状态/命令数另存。材料pose/contact_view全部来自真实完整tree和F1坐标桥。优先抽取或组合真实物理几何适配，不虚构root或额外注册body，不重复应用载荷/积分/发布。父负责旧factory/登记共享接缝。

## 批次与所有权

1. **T1 案例与测试，当前进行。** 原RD独立树仅4NEW：docs/development/p65-floating-mimic-casebook.md；tests/test_floating_mimic.py；tests/fixtures/urdf/floating_mimic_jaws.urdf；tests/fixtures/urdf/floating_mimic_elbow.urdf。固定输入、独立oracle、失败门先写。全部UNEXECUTED，父审全文及新增范围后冻结。
2. **T2 坐标与独立动力学。** 预计NEW _mimic_coordinates.py、dynamics/floating_mimic.py、builders/floating_mimic_urdf.py；parent旧joint_mimic/XML委托。以解析反冲、独立完整刚体几何/质量与鞍点方程验证，不能让product M/b为自己的oracle。
3. **T3 共同场景。** 预计NEW runtime/_floating_mimic_participant.py、test_floating_mimic_scene.py及公开例程；parent修改注册/factory/必要导出。固定F03物理输入后才执行，全部真实参与者与晚期失败共同发布。
4. **T4 验收。** 父唯一串行槽做新专项、必要旧图/普通浮根兼容节点、公开h/h2窗口、原agent交叉审阅和新源码外wheel。Newton只为新有需要的受约束语义作独立参照，不重跑F1原生或原绿套。

所有agent仅text/manualmath/fileIO/Git；imports、collection、数值、pytest、lint/format、build/install均父明确单槽并fresh guard。不重开RR限额失败agent，不新助手。当前T2/T3仅规划范围，尚未授作者实现。

## 独立多案例与成功门

- **F01 非等质量双P夹具。** root2kg、child1/2kg，关节原点/COM共X轴，axes均+X，q2=−q1+0.004m；零重力/初速，独立驱动1N。根平动a_x=1/14、u加速度5/14m/s²，原行lambda=−4/7N，根角加速度及根约束载荷0；总线动量与行功核。共轴条件不能省略，否则根转动交叉项改变解析值。加入图链/扇出、负比/偏置、联合限位/速度、电机上限及不一致full_state留出。
- **F02 非平行转铰+偏心固定工具。** 非单位倍率/偏置、三姿态、非零速度、工具COM外载；原部件独立几何/M、解析或独立能量导数bias及鞍点方程比对真实根反应、世界变换、短窗h/h2。误差门/差分步（若采用）在执行前固定，不从产品生成期望，作者不得执行数值probe。
- **F03 多owner受载。** 浮mimic工具与普通固定树/自由体同场，真实入触/受载/卸载和原材料见证；full/reduced读回、共同力点力矩、冲量及事件一次发布。固定完整输入先见casebook，随后父裁定；不能用仅prescribed pose或只跑通不报错当通过。
- **F04 拒步/恢复。** 联合限位、非法图/零倍率/异类轴、错误完整载荷，第二参与者及late event失败；所有q/v、时间、事件不发布，独立实例重试一致。拥有数据不可被调用方改写。

每个批次必须新增可调用能力或缩小一个具体失败；两次修订均无此变化则记录最小复现、改路线/拆更小节点，禁止靠放宽固定输入或重复审计自转。原持续静摩擦和软布失败独立保留。首片仅HOST树内scalar affine mimic，仍不包含跨树约束、浮根pads、任意根惯量主轴桥、CCD响应或CUDA浮根求解。

## 并行独立后继

原robot作者只读调研C3真实设备转动关节，当前没有C3产品授权；C2的球/滑轨接受态不能冒成多关节机器人驻留。待源码与案例矩阵形成后再冻结C3。F2和C3新文件可并行，旧共享接缝与所有验证串行。
