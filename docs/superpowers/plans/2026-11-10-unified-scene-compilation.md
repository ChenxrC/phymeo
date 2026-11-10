# 统一场景编译与可选 MuJoCo 前端

用户授权：根据已核实的 Newton ModelBuilder/MuJoCo 编译机制制定计划并实施。基线 3b7e72ec；已有三个 K1 未提交文件保留。执行仍为父唯一串行槽，原三个作者隔离编码。采用 feature-commit-checkpoint，小单元验证后提交。

## 目标与已核实基础

现有 RobotAsset/FloatingRobotModel/SceneModel、HOST scene 和 TYX world lowering 继续复用。补齐的是同源声明到运行时的装配与能力检查，以及可选 CPU MuJoCo 模型编译；原生加载和默认物理语义保留。MuJoCo 编译不意味着 TYX 获得 tendon/equality/复杂传动求解能力。

参照：[Newton 1.7 MuJoCo 转换](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/mujoco/solver_mujoco.py#L7500)、[MuJoCo 模型编辑](https://mujoco.readthedocs.io/en/stable/python.html#model-editing)。正式 Phymeo 原入口不调用 MuJoCo；已有 MuJoCo 用法是独立参考。

## 实施节点

| 节点 | 难度 | 工作与文件所有权 | 验收 |
|---|---|---|---|
| M1 统一声明/编译 | 高 | 原 device 作者：NEW builders/scene_compiler.py 与专项测试；冻结资产实例、初态、控制、静态环境、材料、身份、过滤；compile 到现有 HOST/TYX，输出能力报告 | 同一声明运行固定根、浮根、affine 可支持组合；稀疏world、多次编译、独立状态/reset、完整接触反作用、拒绝不支持项且无静默回退 |
| M2 CPU MuJoCo 前端 | 高 | 原 actuator 作者：NEW builders/mujoco_compiler.py 与专项测试/fixtures；可选延迟导入MuJoCo，编译后显式解码公共资产与来源映射 | defaults/include/推导惯量、非主轴COM、标量关节/自由根、几何和a/D；读入错误、复杂传动/tendon/equality等按能力报告明确处理；与原MuJoCo FK/M/J对照 |
| M3 可选求解适配 | 高 | 原 support 作者：NEW backends/mujoco_runtime.py 与专项测试；同一原MjSpec编译模型的CPU执行，GPU MuJoCo-Warp按环境与明确能力另验；不借用原TYX通过标签 | 控制/状态坐标、真实力与步时刻、reset/独立world和错误处理；CPU实际步、可用GPU实际步分别记录；原接触律与MuJoCo接触律明确选择 |
| M4 集成与训练 | 中高 | 父唯一共享入口/exports/optional extra、组合例程、报告与架构 | M2资产进入M1 HOST/TYX，独立原MuJoCo参照；现有Env/PolicyRunner消费；新wheel在源码外使用可选前端和统一入口 |

M1首域为每world一个独立关节系统加有限静态环境；多关节机器人同world共同设备求解仍需原专门求解层，不能通过多次独立step冒充。现有支持域若不能装配，应先给明确能力诊断。统一入口复用已有算法，不新增状态拟合。

## 案例与判据

先声明小模型矩阵：平移关节、偏心惯量转动树、自由刚体/浮根树、affine/mimic、有限球/胶囊/盒地面；再用Panda/G1资产作规模与身份留出。G1已过长动作不重跑。每种模型核全部body/joint/shape身份、q/v地址、单位、惯量主轴、COM与a/D，静态参数不依赖某个姿态M矩阵。编译成功与动力学支持分别验收。

同律的小模型比较短步动力学与实际载荷；不同求解器/接触律使用明确工程误差和行为门，不要求长窗逐位一致。多world状态与reset必须隔离，编译失败不得产生半可用runtime。新编译模式不覆盖旧导入路径的精度声明。

父先读取作者具体API再冻结接口补充；只有父可运行import/pytest/lint/format/build/install/profile，执行前fresh guard。每次首失败原样保留，仅诊断修复受影响项；已绿同源组不重跑。两轮无新增可调用能力、无关闭实质失败、无可测收益的支线停止重选，不扩大数学证明。

## 实际进度

- [x] 源码与官方机制调研、计划落盘。
- [ ] M1作者/父验证。
- [ ] M2作者/父验证。
- [ ] M3作者/父验证，CPU/GPU范围分别记录。
- [ ] M4组合、源码外安装、架构和报告。

GPU MuJoCo-Warp是独立可选执行后端，TYX仍为主线；缺依赖或功能时保持明确未验证，不能把CPU编译成功写成GPU求解通过。

## 本批接口与执行补充

- M1：`SceneDeclaration(tuple[ArticulationWorld])` 只冻结装配声明，复用原资产/模型。`ArticulationWorld(world, asset, initial_state, kp=..., kd=..., effort_limit=..., static_shapes=...)`；`FiniteStaticShape` 保留完整world坐标几何与显式接触材料。`scene_capabilities` 给出拒绝项，`compile_scene(..., backend="host"|"interpret"|"cuda")` 返回独立运行时。同一资产使用原 `RobotCommand` / `FloatingRobotCommand`，控制格式不随后端改变。首域不启用机器人自身互碰，不把内部全排除包装成已支持self-collision。
- M2：`compile_mujoco(path)` / `compile_mujoco_xml(xml, assets=...)` 返回编译记录，含 `.asset`、公共坐标的 `.initial_q/.initial_v`、原ID/地址/坐标映射、完整compiled-native bytes。`require_asset=False` 可以保留只能由原生MuJoCo运行的模型，lowering issues不得被消去。固定根原模型未实现非零armature/damping的情况明确拒绝lowering；世界静态几何单列，不把资产加载冒充完整环境导入。公开native对象可变，独立runtime通过 `fresh_native_model()` 拥有自己的模型与timestep。
- M3：`MuJoCoRuntime(compilation, world_keys=(...), mode="cpu"|"cuda")`；`MuJoCoControl` 明确使用原生actuator/dof/body顺序。其q/v为原生qpos/qvel，并非Phymeo浮根COM速度。所有后端共享 `snapshot/set_controls/step/reset/close` 环境协议，但坐标和接触律在能力/结果中显式标记。原求解步contact与端点forward读数分别标注，不把端点force乘h冒充整步冲量。
- M4：同一无接触XML小树经CPU编译后进入HOST/TYX，另由原生MuJoCo实际步对照；原Env/PolicyRunner使用这些runtime协议。完整 `SceneDeclaration` 反向导出为MjSpec不属于本批交付，原生MuJoCo分支消费原编译记录。

父已在新artifact隔离target安装 `mujoco-warp==0.0.2 --no-deps`，复用原MuJoCo3.3.7/Warp1.17.0。固定两world正负力、各三实际CUDA步与CPU对照通过（03-warp-compatibility），只证明这组依赖可运行，尚不是新adapter通过。M3新增GPU实现及专项测试，必须先检查原生contact/constraint容量再回读，禁止截断后声称成功。既有环境及TYX pins未修改，所有执行仍父唯一串行槽。

## 用户追加：MuJoCo 的默认 CUDA 后端迁移到 TYX

用户要求单独原 agent 实现 mujoco-tyx，尽量复用 MuJoCo-Warp 算法及代码，只有显式选择 Warp 时使用 MuJoCo-Warp。原 support 作者负责新 TYX 后端及新专项测试，父负责公共 dispatch、打包与唯一串行执行。

先审查上游许可证、算子/数据结构与 TYX 能力，再按可验证模型域移植 forward dynamics、积分及接触约束。不得把原 Phymeo 不同接触律、Warp wrapper 或 CPU mj_step 改名为 mujoco-tyx；必须有真实 TYX CUDA launch、无 Warp/CPU 步进负控及原 MuJoCo 多案例数值对照。尚未支持的模型特征在分配或步进前拒绝，保留明确 capability report。默认切换以新后端实际验证为前提；本段是进行中工作，不代表完整 MuJoCo 已移植。
