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
