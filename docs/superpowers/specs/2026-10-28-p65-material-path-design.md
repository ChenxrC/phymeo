# P6.5 XPBD实际材料投影路径设计

用户已授权持续推进P6.5；本阶段是同域材料/接触共同账本的必要输入，不改变XPBD算法或将自由节点接口冒充布接触。实施者选择先暴露真实native材料路径，再独立认证与接入outer；不重复向用户询问常规实现选择。

## 为什么只存最终乘子不够

现有ClothWorkspace和tyx _NodeWorkspace仅返回distance_multipliers。逐次投影方向可能变化，真实配置载荷是sum_k(delta_lambda_k * direction_k)，一般不等于lambda_final * direction_final。三个未固定节点、两边、h=.1的独立探针中，NumPy两者等效速度差约.075m/s，actual interpret/CUDA同样出现。另一个无接触旋转硬边例，x0=(+-1,0,0)、v0=(0,+-1,0)，h=.1/.05/.025，Lz1=2/sqrt(1+h²)，不是Lz0=2；它是现有离散算法的损失，不能伪装成接触反作用或tyx缺陷。

数学背景：[XPBD原论文](https://matthias-research.github.io/pages/publications/XPBD.pdf)的增量投影与速度重建；[Global Momentum Preservation论文](https://sealab.cs.utah.edu/Papers/Dahl-2019-GMP/Dahl-2019-GMP.pdf)讨论PBD刚性动量损失。这里的具体原式/探针由本项目独立实现，不复制其代码；本阶段不采用后处理速度纠正。

## 实施者选择与替代路线

保留原积分与材料求解，增加opt-in真实执行记录。直接用最终lambda/最终梯度代替真实路径已被反例否定；改用新的守恒材料积分器需要单独冻结机器人/布/接触全步时序，不能作为记录功能偷偷引入。只暴露逐sweep净位移也不足以核对各边投影方向/乘子变化，因此记录每一条实际执行边。

## 接口与数据

ClothProgram(model,constraints,domain='cloth',*,record_material_path=False)、ClothParticipant(model,constraints,*,record_material_path=False)、TyxClothParticipant(model,constraints,mode,*,record_material_path=False)。make_cloth_runtime也转发该可选参数。只允许bool/np.bool_，快照签名包含此开关。默认不产生新诊断、不改变旧数学或旧tyx调用路径。

启用后每个临时workspace拥有本子步的frames列表；predict初始化为空，每次position_sweep追加一个(num_edges,18) FP64存储数组。零边/None的sweep记录形状(0,18)，与调用次数相符。字段顺序冻结：before_i_xyz(0:3)、before_j_xyz(3:6)、after_i_xyz(6:9)、after_j_xyz(9:12)、lambda_before(12)、lambda_after(13)、direction_xyz(14:17)、delta_lambda(17)。包含denominator=0时的零delta与实际未变状态；此时direction三列为零哨兵，表示没有施加投影方向，不冒充未使用的中间计算值。非零denominator的coincident采用原X轴分支。

诊断键material_path：schema='xpbd-distance-path-v1'、columns字段名、frames元组、predicted_position、predicted_velocity、alpha、h。所有返回数组为owned只读快照，多次diagnostics或新的trial/substep不能污染前次结果。记录是临时候选数据，不是新增持久history；外层提交/回滚仍用原SceneStepper。

NumPy project_distances(...,*,trace=None)写入调用者提供的FP64可写(count,18)数组；非None先验证再修改状态。tyx TyxNodeMath.project_distances同签名：traced kernel直接写实际FP32操作前后数据和delta/n，结果升FP64用于存储，不能在host重算来假装设备路径。关闭时使用原kernel及outputs，不增加默认launch/transfer。为不改变旧kernel字节，新增独立traced kernel镜像同一有序数学；两条路径的bitwise状态/乘子一致性是必须回归，后续修改公式必须同步维护。此有意重复为隔离变更的实施者选择。

记录没有全量独立物理认证；不能仅因frame存在就接受未知材料求解器。输入FP64→FP32转换、每sweep舍入、额外position_load或velocity_impulse不在材料帧内；后续ledger必须用predicted与帧连续性、实际最终状态核对这些差项。当前仅输出证据，不扩大InitialRobotNodeContactProgram允许范围。

## 验收

- 开关关闭/开启的状态、最终lambda、迭代数bitwise一致；numpy、actual interpret/CUDA，有限C++短例；空约束/None、固定点、coincident/zero denominator、多边多轮与失败候选均覆盖。
- 单边解析dlambda/方向/前后坐标；共享两边路径累计与实际位移一致，最终lambda*最终梯度反例仍有约.075差。每次投影原式验证，不只检查终点。
- 独立旋转两点Lz解析三步长及原P/K/材料能量观测，明确损失来自旧算法，不把它写成守恒或已修复。
- 诊断快照所有权、后段失败全局恢复、原输入/状态不可变，重新试算=fresh，新增记录不承担history提交。
- 小切片TDD，一次fresh最终review；必要全量CPU/CUDA、源码外wheel/uv-pip质量，正式NumPy/interpret/CUDA串行记录开关观测；C++长期TYX-003无修复不再压力测试。

## 后续不可省略

真正material/contact outer仍未实现。后续需独立原式核对input、prediction、每条材料投影、速度重建、external/contact work与总机械能，单独报告/门控离散角动量损失与步长收敛；不能将轨迹相关误差直接命名为物理摩擦。保持原FreeParticleLedger拒绝材料域，直到新合同通过，再做几何/CCD、弯曲/自碰撞、单爪、双臂与Three.js完整验收。性能只观测，缺陷分层记录。

Builder check：material_path_design_check返回nothing important left。产品记录API已在0bf0619实现；上一准确f259e69的CI40662503851与Documentation40662503807均success。验收结果以材料路径阶段文档为准，不扩大为联合接触。

实施者补充：直接trace目标要求C-contiguous且不与position/weights/alpha/lambda/约束数组共享存储，验证在任何修改前完成，避免诊断写入改变物理结果或重叠行吞记录。

审阅修订：零分母固定边的未使用方向运算可能产生NaN。traced kernel仍保留原算术路径，但不将未使用中间量作为必需有限物理输出；以明确零哨兵表示跳过投影。NumPy同一语义，18列不变；future ledger须结合weights/alpha核对跳过条件，不能把零哨兵当合法单位法线。此为记录表示修正，不改变原无记录路径或全局非有限门禁。
