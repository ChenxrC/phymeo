# 到08:00：机器人、环境与策略连接

用户授权：北京时间2026-11-10 08:00前持续实施，优先看到G1完整舞蹈不跌倒，以及当前实际实现架构。启动约02:05，基线 `5a84d38`。本计划先落盘，再启动执行；不建立周期任务。保留原K1三个未提交文件，不推main、不发包、不改TYX pin。

## 早上的交付

1. **真实G1舞蹈**：先核完整20秒窗口；通过后继续同一次物理运行到官方 `dance1_subject2` 全部6574帧（50Hz，131.48秒）。初始q/v、29关节、原胶囊/球、质量、执行器、策略观测协议和力矩限额保持；h=.0005、策略50Hz。根自由，禁止用参考根轨迹或状态修补、固定/焊接假装不跌倒。
2. **公共连接**：训练环境可显式选择真实普通步/受载步；每世界拒步不伪造transition；HOST机器人场景和resident batch可通过同一任务/策略流程消费。关节目标/力矩控制与物理子步调度分开，动作保持及失败语义明确。
3. **可运行多案例**：除了G1，还验独立刚体/机器人受载环境、复位、拒步和策略时钟。完整舞蹈必要但不充分，不能用单案例推断全部引擎能力。
4. **实际架构**：中文MD与离线HTML/SVG图，实线只画已实现/已验证连接；区分HOST、TYX与CPU推理，标出安装/验证证据与未完成边。可回放真实机器人和足地力，不播放参考动作冒物理结果。

## 批次、难度与时间窗

时间用于容量控制，按真实结果调整；08:00停止启动新长执行，报告已完成/失败/未完成，不能到点把未过项目标成完成。

| 批次 | 预计时段 | 难度 | 交付与成功标准 |
|---|---|---|---|
| N0 固定合同 | 02:05–02:20 | 低 | 计划/当前源码/独立作者范围/执行队列落盘；复用已绿2秒和动力学，不重跑同组 |
| N1 G1长窗 | 约02:20–05:30，优先占串行槽 | 高 | 从原初态一次连续跑131.48秒，20秒作为中间验收点；完整原输入、实际状态、控制、足地力与首失败留存 |
| N2 向量受载与拒步 | 与N1并行纯编码，N1结束后验证 | 中 | 逐世界acceptance必须完整且boolean；部分接受或后置错误显式poison/reset，episode/reward不默默推进；显式受载stepper调用正确owner/plan |
| N3 HOST场景与策略接口 | 与N1并行纯编码，约05:30–06:40验证 | 中高 | 场景适配不写物理状态；公共策略节拍器支持持有动作、真实入步速度、重置与失败；两个独立受载任务接入；解释器/CUDA最小受影响组串行 |
| N4 运行连接与复现 | 约06:00–07:15 | 中 | 统一环境/策略短留出与选定安装包调用；G1长窗保持原语义，不能为了重构重跑已绿131秒前缀 |
| N5 实际架构与晨报 | 约07:15–08:00 | 中 | 图中每条实线有源码/测试/运行证据；真实轨迹回放可打开；最新续作入口、失败与后续优先级准确 |

上次4000步耗时约162.5秒；按同速全动作约3小时，当前有约6小时。此为运行容量估算，不是性能对标承诺。若长窗提前失败，先保存最小失败/原请求，再研究通用数值或接触接口；不做fixture/质量/阈值搜索。

## 冻结的接口边界

- `ResidentVectorEnv` 保留原构造与默认行为，新增显式可调用stepper（默认`batch.step`）。受载stepper只借用原batch/plan，调用 `batch.step_contacts(plan,h)`，不关闭它们。必须验证逐world返回map，不丢弃False。
- 全部返回False、部分False和异常分别留证；部分world已接受时，不声称物理回滚，标明mask并要求恢复/全部复位。失败不能生成虚假reward/done或推进episode计数。原成功路径的final observation/选择自动复位保留。
- HOST场景适配提供与环境相同的`set_controls/step/snapshot/reset`边界。一个逻辑world对应一个原公共scene；返回原不可变观测，复位通过原factory重建初态，不覆盖q/v。重建失败时保留旧可用对象，选定world隔离。
- 公共策略runner只做观测→动作/command与固定整数子步节拍；不拥有动力学，不把参考轨迹写成状态。checkpoint记录最近动作/逻辑时钟，物理拒绝后的推进规则可验证。G1资产/154维协议属于实验适配，不能污染通用核心。
- 新功能首批保NumPy/HOST边界，不能把它命名为零拷贝GPU训练。PPO与W256/W1024只有余量且前置连接通过后实施；未实施则图中明示。

## 并行与所有权

只复用原三个agent及各自干净隔离树；允许纯编码并行，全部项目import/collection/数值/pytest/lint/formatter/compile/build/install/profile仍父唯一串行。父持共享runtime、G1长运行、源码整合、架构图与最终验收。

- 原 `h1_device_capacity`：`training/resident_env.py`、NEW `training/resident_stepper.py`、原`training/__init__.py`及新专项测试/报告。修接受map与受载stepper，不改scene核心。
- 原 `h1_foot_support`：NEW `training/scene_backend.py`、新HOST专项测试/报告；依赖上述已冻结duck-typed边界，不改resident_env或旧足地测试。
- 原 `h1_actuator_model`：NEW `control/policy_runner.py`、新专项测试/报告；不改正在运行的G1例程、物理核心或旧执行器文件。

作者不得执行项目、启动辅助测试/评审agent、改旧共享代码或原WIP。父读增量、必要适配后唯一执行。每个完成单元按 `feature-commit-checkpoint` 验证并提交。

## G1成功、失败和工程精度

- 必须完成请求的全部物理步；首失败后停止并保存，不能跳过坏帧继续算成功。
- 原工程门：根高>=.35m、倾角<=1rad、策略关节跟踪RMSE<=.5rad、参考XY误差<=1m、峰值足力<=8倍体重、稳定段平均竖向支撑.5–1.5倍体重、原力矩限幅与双边反作用。记录足滑、接触间隙和实际运行后端，允许既定工程精度。
- 全动作在脚部的普通离散摩擦模型上运行；不冒称滚动材料静摩擦已经完成。若后续动作包含跳跃，双脚持续承载不是该阶段的成功门；应检查腾空/落地与真实支撑变化。
- 若20秒通过但131.48秒失败，应分别报告，不把短窗叫全动作成功；若全部失败，不用独立MuJoCo回放替代。

## 防止空转

同一实质失败连续两轮没有缩小原因时，停止扩写证明/统计，回到接触离散或控制接口最小复现。不得为一个舞步引入对象名/坐标/时间特判、调质量或追加固定根。不能把“默认姿态PD站立失败”冒成所有反馈策略失败，也不能用两秒成功抵消后续跌倒。

只在新里程碑、实质失败、完成或需用户决定时通知；长运行期间保持必要进度可见。到08:00交付真实结果和当前实现图，不保证尚未执行的舞蹈必过。

## 执行记录

02:25增补：N2/N3纯作者检查点已提前交接，尚未执行；利用长舞蹈独占执行槽期间的作者余量，追加**可选N4训练器连接**。按RSL-RL官方VecEnv/OnPolicyRunner文档和5.5.2接口作者适配，不自行发明PPO。先支持明确HOST拷贝的TensorDict边界，保留final observation/timeout，原NumPy接口不变。固定两个简单真实机器人任务、固定训练预算与未训练/训练后同条件评估；checkpoint保存/载入验证。只有核心连接与G1验证完成、安装和时间余量足够时才串行执行；否则保留UNEXECUTED，不能挤占完整舞蹈与架构交付。Torch/RSL-RL可选隔离环境，不更改TYX pin或核心依赖。

调研入口：[官方环境接口](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py)、[官方训练/保存/载入流程](https://leggedrobotics.github.io/rsl_rl/guide/overview.html)。该适配不是零拷贝或大规模GPU训练完成。

启动前状态与后续任务收据保存在 `artifacts/p65-connections-20261110-0800-01`，每项实际结果回填本节。

04:48长窗结果：通用轴入盒缺口修复并通过17HOST相邻/独立例后，从原q/v一次完成全6574帧、131.48秒/262960步，没有跌倒/拒步、没有状态修补。最低根高.439279m，最大倾角.630023rad、最大关节RMSE.291702rad、稳定段平均支撑1.00000865倍体重。原工程门峰值14.9124倍体重与参考XY误差1.274907m未过，RAW_EXIT1和两个failedcriteria完整保留；这不是全部工程门通过。新run结果在`artifacts/p65-connections-20261110-0800-01/g1-full-outcome.json`。原27.0495s失败仍在01，不重跑该旧前缀。

04:52连接专项：HOST72通过；interpret/CUDA各12通过。包含新受载计划跨episode、真实逐世界拒步、公共策略runner、真实HOST场景与日志语义。原firstlint5项与format14files失败已保存并修正；16源qualifiedAST证明和最终静态通过。新G1公共入口20秒、真实回放、可选PPO、源码外wheel及最终图继续按唯一父队列执行。
04:57可选N6增补（核心已通过后）：利用原capacity作者余量，在隔离树只写W256通用连接留出与报告，不改核心/旧案例/资源上限、不执行。最多两例：已有free/rigid families的真实多world策略/复位，以及原site/candidate容量允许时的受载prismatic spring多episode递推。8步、独立力–运动期望、完整actualreceipt留存；先核源码容量，不支持就明确缺口，不扩cap。该留出不叫PPO、纯GPU窄相或G1 GPU舞蹈。父在公共G1/回放/PPO/安装包优先队列和08:00余量内串行验证；无余量则保留UNEXECUTED。

05:55连接状态：公共G1完整20秒全部原工程门通过；全131.48秒实际回放已生成/浏览验收。可选CPU PPO两任务各以原固定seed/预算达到20%成本下降门并完成原生checkpoint重载，实际下降84.71%/99.01%。W256混合运动interpret/CUDA各2048world-steps通过，受载512-shape声明按原256上限拒绝。新wheel在源码目录外两真实受载scene完成32world-steps与隔离reset。消费者首次漏传必需robot_state导致API错误，22保留，22b补显式初态后通过，无产品修改。

### 可选N7：解除真实受载连接阻塞，不追G1专用路径

源码调研定位ContactShapeFrame已支持group_index0..32，但ContactLoads/sample_tree_contacts仍限17body。父c1292647已统一到32joint+root=33body；新30body最小测试先真实RED，修复后完整HOST53通过。原MAX_SHAPES256、site/candidate预算不变。原capacity作者只新增29/32joint最后body受载测试；父检查独立全M/J、根加速度/角加速度、双侧反作用和真实发布，不能拿无载容量当受载通过。

随后在余量内做**现有速度摩擦律的混合受载桥**。这是已用RobotContactMaterial/原HOST patch law到现TYX fullJ/M的连接，不增加静摩擦历史、设备几何或GPU策略张量。父范围仅`backends/_resident_contact_bridge.py`、`_resident_contact_plan.py`的材料admission与准确描述，和原bridge测试中已经过时的“正μ必须拒绝”预期；原actuator作者仅NEW `tests/test_resident_friction_bridge.py`、`tests/scale/test_resident_friction_connections.py`。不改G1参数、原力律、物理门、pin、caps、owner或发布流程。

先保留正μ被拒的实际RED，再连通原law；HOST独立核Fn/Ft、切速分段、patch权重一次、双方共同作用点/力矩和实际功；interpret/CUDA各用有真实入步速度的floating偏置球及fixed rotor解析全M/J/运动，保存原请求与完整收据。保持分离无力和非法材料拒绝。只靠删guard不算通过，未执行项继续标明；若出现额外物理/审计矛盾先保首失败，不调质量/阈值追结果。源码变更后新wheel重新验证受影响能力，之前wheel收据保留原source binding。

N5架构/结果优先；到08:00不启动新的长窗。设备站点力核、公开Torch缓冲/stream/epoch接口及单次发布分页仅完成调研，见两research文档，不画作已实现。

N7源码核对补充：public `tyx_world_batch.prepare_contacts`仅改过时frictionless描述，无逻辑变更；旧`test_resident_tree_contact_transactions`中的“正μ必须拒绝”改为非canonical材料pair key负控，原wrong-law/owner/topology/cap不删。全功/双侧力矩/输入速度/finish检查均为原3D路径，保留原诊断而不是新增豁免。44实际RED已证首个正μHOST例先通过原patch/law字面向量，再被bridge旧admission拒绝。当前源变更仍需首静态、HOST完整新旧桥及两backend新例/受影响admission负控，不能凭源码检查标绿。

06:16 N7实际通过：HOST完整新旧桥63项；interpret/CUDA各12项（两物理case各8连续接受步，加10个admission负控）；实际site cap0拒步保态与错owner仍拒。原Fn/Ft、全M/J、偏心力矩、入步速度、实际材料/作用点功均核对。仍为HOST几何/原HOST力律→设备响应，未做设备站点力核、卸载重入或G1设备窗口。

### 可选N8：完整G1设备受载入口短留出

在高body和非零μ各自独立物理留出均过后，原capacity作者仅NEW `examples/g1_resident_policy.py`，先源码核适用域/输入协议，再纯编码。复用原资产、原ONNX/154维协议/初始qv/限额、原球胶囊与有限地面；把同一29joint模型、真实geometry/group绑定、每h更新PD和接触载荷接到公共TyxWorldBatch/Env/PolicyRunner。案例协议保持在examples层，不动核心物理、旧G1例子、质量/h/pin/caps/物理门，不把HOST窄相说成GPU。

父在图/报告/最终包优先完成后，若08:00前余量足够，只先单CUDA2秒、从原初态、h=.0005做真正受载策略短窗，首失败停止/保原请求、真实状态、力矩/力与mask；不降h/换姿态/关摩擦或改容差追结果。新后端是新运行，不重跑旧HOST已绿长窗。若作者发现任何模型/法向/控制语义不相容则先最小文本报告，不复制HOST解算替代设备。未执行则保UNEXECUTED；不影响已完成的HOST全动作交付，也不自动授完整设备20/131秒长窗。
