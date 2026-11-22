# M2 场景与帧开发计划

G1 已实现构造校验、米制 XYZW 位姿组合、静态引用和帧节点覆盖，见[模块测试](../tests/test_scene.py)。光照扩展及后端时序验证留至对应关口。

**目标：** 将静态资产／实例／方向光与逐帧节点位姿分开，建立可独立验证的米制 XYZW 位姿和来源合同。

**结构：** `VisualScene` 由 `open_scene` 消费一次；`SceneFrame` 只携带帧键、时间和节点世界位姿。纯数据校验不读取设备、不拥有物理状态、不维护跨调用隐含历史。

**依据：** [当前 scene.py](../src/phymeo_rendering/scene.py)、[框架说明](../README.md)、[R0–R4 路线](../../docs/rendering/README.md)。合并按 [集成计划](08-integration.md)，证据按 [统一验收](09-acceptance.md)。

## 文件与测试归属

| 归属 | 路径 | 责任 |
| --- | --- | --- |
| 产品独占 | `rendering/src/phymeo_rendering/scene.py` | 现有场景、位姿、帧检查；纯位姿组合；R2共享光照场景与环境光输入合同。 |
| 测试独占 | `rendering/tests/test_scene.py` | 标准库 `unittest`；静态引用、节点覆盖、位姿组合和来源反例。 |
| 文档独占 | `rendering/plans/02-scene.md` | 计划与接口交接。 |
| 他方所有 | M1、M3、M5、M7 文件 | 资产声明、曝光绑定、会话生命周期及来源适配由对应负责人维护。 |

共享方法调用由集成负责人接入；不要在本任务中直接修改后端或摄像机文件。

## 输入、输出与依赖

当前类型及语义：

- `Pose(position_m, orientation_xyzw)`：子坐标系在父坐标系中的刚体位姿；米制、单位 XYZW 四元数。默认恒等。
- `Instance(id, mesh_id, material_id, node_id, local_pose)`：稳定正整数分割身份，视觉实例通过局部位姿附着到命名节点。
- `DirectionalLight(direction, illuminance)`：世界系单位光传播方向与线性 RGB 照度。
- `VisualScene(key, meshes, materials, instances, lights)`：静态清单。资产、实例绑定或灯光改变均需新的 `key`。
- `FrameKey(scene_key, world_id, epoch, revision, tick, phase)`：完整来源键；`phase` 仅允许 `authored`、`step_input`、`accepted_endpoint`。
- `NodePose(node_id, world_pose)`；`SceneFrame(key, time_s, nodes)`：每帧节点世界位姿和原来源时间。

上游：M1 资产与 M7 只读适配。下游：M3 用 `Pose`、`FrameKey` 绑定曝光；M5 `open_scene` 和 `capture` 消费清单／帧；M6 结果保留同一键和时间。

G1 已提供以下公共纯函数：

| 签名 | 结果与边界 |
| --- | --- |
| `validate_scene_frame(scene: VisualScene, frame: SceneFrame) -> None` | 成功无返回值；检查静态引用、帧场景键及实例所需节点覆盖。失败给出字段／身份诊断，不修改输入。 |
| `compose_pose(parent: Pose, local: Pose) -> Pose` | 返回 `T_parent × T_local` 的组合；不访问场景、设备或来源历史。供同帧 visual 和相机安装变换使用。 |

函数名变更须与 M3/M5/M7 及集成负责人共同更新，不另造同义实现。

## 顺序开发步骤

### 1. 先固定局部数据合法性

- [ ] 先写并取得失败：`test_identity_pose_and_empty_authored_scene`、`test_pose_rejects_nonfinite_or_zero_rotation`、`test_light_rejects_invalid_direction`、`test_frame_key_rejects_invalid_identity`。
- [ ] 实现有限三分量位置、四分量四元数与不可变元组检查。拒绝布尔冒充坐标、错误长度、NaN、无穷、零四元数。
- [ ] 单位长度的首版计划判据为 `abs(norm - 1) <= 1e-6`，用于四元数与光方向；这是待实现的输入合同值，不是物理求解容差。不得悄悄归一化或改变四元数符号；适配器若执行单位化须另存转换来源。
- [ ] 照度必须是有限、非负的三元组；零照度允许。身份必须为非空白字符串；epoch/revision/tick 必须为非负整数且不是布尔；时间必须有限且非负。时间不由 tick 或帧率重算。
- [ ] `Instance.id` 限定在 `[1, 2**32-1]`，与 M6 的 `instance_id_u32` 输出一致；0 留作背景。所有集合字段拒绝可变列表，保持冻结对象没有可变别名。
- [ ] 局部检查区分错误类型 `TypeError` 与非法值 `ValueError`，运行单元用例并记录证据。

### 2. 建立静态引用与逐帧覆盖校验

- [ ] 先写 `test_shared_mesh_with_two_instances`、`test_missing_asset_reference_is_rejected`、`test_duplicate_instance_or_node_is_rejected`、`test_scene_key_mismatch_is_rejected`。
- [ ] `VisualScene` 中 mesh ID、material ID、instance ID 各自唯一；所有实例的 mesh/material 引用必须存在。多个实例共享同一网格或节点是合法输入。
- [ ] 实现 `validate_scene_frame`：`frame.key.scene_key == scene.key`；帧内 node_id 唯一；所有 `Instance.node_id` 在帧中都有且只有一个位姿。
- [ ] 允许额外的非可见节点，例如腕部相机安装节点；不得把“实例引用节点集合”误当作整个机器人节点目录。空场景配空节点帧合法。
- [ ] 只检查给定场景与给定帧。相同 tick 的重复渲染、离线跳帧、epoch 切换和会话历史失效规则由 M5 显式管理，不在函数内缓存前一帧。
- [ ] M5 负责在会话打开后防止同 key 换静态内容；单个纯函数无法证明过去上传的是哪份清单，不伪称已验证这项历史属性。

### 3. 明确同帧空间组合

- [ ] 先写 `test_compose_pose_rotates_local_translation` 与 `test_composition_does_not_mutate_sources`，用独立手算点坐标作期望值。
- [ ] 实现 `compose_pose`，顺序固定为父世界变换在左：位置为 `p_parent + R_parent × p_local`，旋转为 `q_parent × q_local`。该函数明确按单位旋转计算：在局部计算副本上除以已验证的非零范数，并将新输出旋转单位化；原输入字节不改，返回值是合成的显示位姿，不能冒称原始物理 q。四元数存储顺序仍是 XYZW。
- [ ] 在本测试中使用节点绕 Z 轴 +90°、平移 `(1, 2, 0)`，实例局部平移 `(1, 0, 0)`、恒等旋转；结果位置应为 `(1, 3, 0)`。
- [ ] 测试 `q` 与 `-q` 表示同一旋转；比较旋转时采用两种符号下较小的四元数差，不能要求源字段被改成某一符号。
- [ ] 将组合函数交给 M7，用已保存节点位姿解析 visual／腕部曝光；不得调用任何物理 step 求得另一个状态。

### 4. R2：共同光照场景与曝光归属

- [ ] 以当前 `DirectionalLight` 作为首个共同光照输入：方向始终是世界系光传播方向，不是朝向光源的向量；线性RGB照度在两后端使用同一声明。M5 若需变号、换单位或转换色域，必须在后端边界显式记录，不能改写公共场景。
- [ ] 在 G1 协调可选环境光及扩展光源的场景描述。保持 `VisualScene(..., lights=())` 和现有方向光构造兼容；新的字段采用默认关闭的可选输入，不将环境光伪装为另一盏方向光。
- [ ] 环境光最少共同字段为：源资源键／URI／内容SHA、图像投影方式、经纬基准轴与朝向、线性RGB辐亮度语义、非负线性强度倍率。环境朝向用明确旋转，不能混入无意义的无限远环境平移；HDR浮点值不再进行sRGB解码。
- [ ] 在消费前登记 HDR 来源：M7 保存原始来源URL、作者、许可文本或标识、允许的再分发范围及内容SHA；M2 把已登记资源键纳入静态光照描述；M5 负责实际文件解码和设备环境采样。无来源资格时明确拒绝此资产，不默认联网获取替代环境。
- [ ] 需要点光、面光或聚光时，先登记形状、尺寸、坐标和强度单位，再进入 G1。先提交方向光＋环境光的最小合同；没有共同表示和后端能力证据的光源不得用不同的近似模型冒充公平对照。
- [ ] 所有光源、HDR内容、环境朝向或强度改变都生成新 `VisualScene.key`，重新打开静态场景；动态 `SceneFrame` 仍只更新节点位姿。原生后端的累积历史、加速结构和会话清理归 M5。
- [ ] M2 交付两路线逐值相同的光照场景与资产摘要；M5 声明并记录真实能力：光源类型／数量、环境投影及旋转、阴影、所选渲染路径。无能力时给出明确拒绝，不静默跳过环境或灯光。
- [ ] 区分两种“曝光”：M3 的 Camera 是曝光时刻的标定和世界位姿；画面亮度的曝光EV、白平衡、显示变换、采样数、反弹、累积和降噪均由 M5 管理并写入捕获配置／结果记录。M2 不靠修改灯光能量补偿某一后端的曝光差异，也不在 SceneFrame 增加显示参数。
- [ ] 实际R2比较必须同时固定场景、Camera、采样配置和显示配置。若两后端不能同样处理某项设置，保存差异并另列比较，不能仅凭相同 light ID 宣称输入公平。

先写以下不依赖 SDK 的测试，再实现 G1 冻结后的类型／校验：`test_directional_light_sign_and_illuminance_preserved`、`test_environment_descriptor_preserves_linear_radiance`、`test_environment_rotation_requires_new_scene_key`、`test_environment_rejects_missing_source_or_invalid_strength`。真实资源变更与key重用的跨调用检查由 M5 会话测试完成，M2 不伪造历史。

| 计划验收 | 输入与可计算目标 | 反例 |
| --- | --- | --- |
| 同一方向光 | 两路线消费前的方向／照度描述逐值一致，最大绝对差为0 | 仅反转方向或将照度改一分量，对照检查必须检出。 |
| 自有线性环境夹具 | 常量环境 `(1,1,1)`、倍率2的共同描述目标为 `(2,2,2)`，线性计算误差≤`1e-12`；该单测不声称像素会直接等于2 | 把HDR当sRGB转换，或把倍率误作EV，参考检查必须失败。 |
| 环境方向 | 测试亮斑源方向 `(1,0,0)` 经世界Z轴+90°旋转后应为 `(0,1,0)`，方向分量误差≤`1e-6` | 漏旋转或重复旋转分别检出；输入辐亮度未因旋转改变。 |
| 静态来源 | 改HDR摘要／灯光强度后场景key与旧key不同；旧曝光frame_key不能用于新会话 | 相同key悄悄替换HDR由集成会话拒绝；无许可／来源登记不进入真实图像测试。 |
| R2实渲 | 固定场景的两种画质及10秒已存帧回放；记录相同输入摘要、实际能力和显示配置 | 未支持的环境或光源明确失败，不能用丢光源后的“成功图像”满足R2。 |

## 正负例与可计算验收

| 层次 | 输入或事件 | 计划判据 |
| --- | --- | --- |
| 单元正例 | 一个网格、两个实例、同一节点、额外一个腕部节点 | 网格条目 1、实例 2、缺引用 0、缺节点 0；额外节点合法，帧内容逐值不变。 |
| 单元正例 | 上述 +90° 的组合例 | 平移最大绝对误差 ≤ `1e-12 m`；相对手算四元数、允许整体变号后的最小分量误差 ≤ `1e-12`。这是 FP64 简单算例目标，不是实渲误差。 |
| 单元负例 | 缺 mesh/material/node、重复 instance/node、scene_key 错配 | 每类独立拒绝；错误定位到对应 ID，不能生成缺件的部分“成功”帧。 |
| 单元负例 | 零四元数、范数为 2、非有限值、非法 phase、负 tick、ID 为 0 或 `2**32` | 全部拒绝；`1e-6` 单位长度界内外另有边界用例。 |
| R1 集成 | 基座／腕／手指，两个已保存姿态 | 六个锚点的世界变换与独立 FK 比较，逐点报告误差；不得用两处调用同一组合函数充当独立参考。统一验收负责最终锚点界。 |
| R4 集成 | 八世界中只重置一个世界 | 其余七世界帧键、节点及输出来源串世界错误数为 0；历史清理和缓存隔离由 M5 实测。 |

每次只读捕获前后，原 q/v/时间的保存字节应完全相同，物理步数增量为 0。该性质由 M7/M5 集成收据证明，M2 单测不能代替。

## 执行与合并依赖

独立包安装后，从仓库根目录计划执行：`python -m unittest discover -s rendering/tests -p test_scene.py -v`。

- 依赖 M1 类型；可先以当前骨架写测试，最终联合合并前必须使用 M1 的真实合同，不复制影子类型。
- M3 依赖 `Pose`、完整 `FrameKey` 与组合顺序；M5 依赖纯场景／帧校验入口。
- M7 拥有源坐标转换、四元数顺序转换和真实轨迹绑定；M2 不导入 MuJoCo、PhyMeo 物理包或 Torch。
- 覆盖 R0 静态标定桌、R1 两姿态锚点、R2 重复同帧以及 R4 多世界来源隔离；R3 的投影和深度指标由 M3/M5/M6 联合验收。
- 公共校验函数、单位长度界与额外节点规则合并前须写入集成合同；任何变更同步到 M3/M5/M7，不在后端内另设矛盾门槛。
