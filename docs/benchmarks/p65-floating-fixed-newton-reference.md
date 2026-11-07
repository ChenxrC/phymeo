# 浮根固定工具：Newton 1.7.0 未折叠原生参考

状态：CODE_READY_TESTS_PENDING / UNEXECUTED。作者仅文本读取、编写与 Git；未执行 import、数值、测试、静态检查或构建。父唯一串行执行后另记真实结果。本脚本不读取 Phymeo 产品生成期望，不合并父、不修改 C2 四文件或旧参考。

对应父计划 31c95e7、F03 输入补充 8835b51；原 URDF 由 ccca6c0 提供，F02 三姿态以 RD 1c39eee 的独立五部件定义为准。新增恰两文件：`tests/reference_newton/robot/floating_fixed_tools_reference.py` 与本报告。脚本必须显式给两 URDF 路径，便于在未合入资产的作者树运行，不复制资产或改其内容。

## 冻结范围与预期

- **F01 实际动力学一步**：`floating_fixed_composite.urdf`，2 kg base 与 x=1/4 m 的 1 kg tool，保留原生 FIXED。独立 M=3 kg、COM=(1/12,0,0) m、COM 惯量 diag(3/10,53/120,13/24) kg m²。零速/零重力，在原 tool COM 仅施 F=(0,3,0) N，h=1/1024 s，`SolverFeatherstone.step` 一次。期望整体 a_y=1、alpha_z=12/13；原 base、tool COM 的入步 a_y 分别12/13、15/13。组 COM 矩1/2 Nm由实际力臂形成，绝不再作为 tool 的 COM 外矩重复施加。另核首步漂移、实际线/角冲量和输入未被改写。
- **F02 三组真实 FK/J/M 读回**：(qR,qP)=(0,1/8)、(.37,-1/16)、(-.51,3/16)。第二组 root 平移(.5,-.25,.75)、XYZ 轴循环 XYZW=(.5,.5,.5,.5)；第三组平移(-.25,.5,0)。原生保留6个 body（五个正质量部件＋无质量 witness），FREE＋所有 FIXED＋R/P；9 q、8 v。五原部件质量、局部 COM/惯量、世界 pose/COM/惯量逐项比较，独立串联几何与平行轴/完整 Jacobian 求和作为 oracle。
- F02 速度固定为合成根组 COM 线速/世界角速/两标量速率 `(.1,-.2,.3,-.1,.2,.1,.4,-.3)`。原根组含 base+mount，COM 偏移原 base 的 x=1/20 m。令 d=C_base−C_group，显式 `v_base=v_group+ω×d`；记录完整 8×8 变换 B，比较 `J_native B`、`Bᵀ M_native B`，不直接把两套 COM 坐标的矩阵相等当真值。完整每体速度、动能及工具局部点(1/32,-1/64,0)的速度/J/虚功独立检查。
- F02 原资产只有两碰撞 shape：base sphere、tool box。保存真实 native shape ID/type/body/local pose/scale，逐一核球轴向表面采样或盒8角的世界材料位置；无碰撞声明的 mount/arm/carriage 不被补成代理几何。原 URDF collision name 与原生 label 分开保存，按本固定夹具的唯一 owner/type 映射，不宣称 Newton 解析器保留 collision name。

F02 仅 native FK/J/M 与虚拟功读数，没有施加工具力或调用动力学 step；F01 没有持续窗口。**未做 F03 原生碰撞、布、摩擦或整窗推进，也不把 CUDA FK/M 读回写成 Phymeo 浮根 CUDA 能力。** 旧 Newton fixed-collapse 惯量失败保持历史，本脚本完全不调用该路径。

## 原生约定与来源

锁定本机 `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0`，Newton==1.7.0、Warp==1.17.0；不安装或改 pin。脚本写入实际解释器、Python/NumPy/包版本、实际 device alias/name、每组 model/state device、原始 dtype。

以下均已静读该安装包源码，运行时保存完整 UTF-8 文本及 SHA256：

- `newton/_src/sim/state.py:123–145`：`body_qd=(v_COM_world,omega_world)`，`body_f=(F_world,tau_COM_world)`，都是 linear-first/angular-last；pose 原点和该 COM 不同。
- `newton/_src/utils/import_urdf.py:65–89,770–786,961–964`：显式 `floating=True, collapse_fixed_joints=False, collapse_massless_fixed_root=False`。真实 free root；保留全部固定关节及有质量体。
- `newton/_src/solvers/featherstone/solver_featherstone.py:451–469,635–651,1129–1138`：公开 COM/world 速度与内部 Featherstone 参考点转换；最后真实 FK 回公开状态。
- `newton/_src/solvers/featherstone/kernels.py:1498–1503`：输入 COM wrench 只在原生求解器内按实际 COM 转矩；`646–684`：FREE 的归一化 Euler 四元数更新。报告区分连续解析加速度与原生离散姿态，不把指数映射要求施加给另一积分器。
- `newton/_src/sim/articulation.py:1266–1288`：公开 COM/world Jacobian。`builder.py` 与 `model.py` 同存，绑定真实形状/关节/惯量接口。

零重力、无接触、无驱动是本参考的明确输入。导入后显式把 armature、被动/目标/限位增益及 joint friction 置0，原 parser 值和最终值都保存；位置/effort/velocity 限位元数据不变。固定坐标不是虚拟6个标量关节。

## 执行前固定误差门与证据

输入最大长度约1 m、部件质量0.25–2 kg，原生 FP32 ε约1.19e−7。预声明绝对门：位置/材料点5e−6 m；旋转元素5e−6；质量2e−6 kg；惯量5e−6 kg m²；速度3e−6 m/s或rad/s；完整 J 每元素5e−6（各列物理单位对应平移/旋转）；完整 M 每元素8e−6（平移/角/关节混合单位）；动能5e−6 J、虚功率5e−6 W。它们覆盖有限次 FP32 串联/矩阵求和舍入，仍远小于1/32 m材料偏移和实际关节响应，不动态随观察值扩门。

F01 的微小首步单列更紧门：a/alpha2e−5 m/s²或rad/s²，实际 v2e−8 m/s，漂移3e−8 m、quaternion2e−8，线/角冲量6e−8 Ns/Nms。h=1/1024，速度约1e−3、漂移约1e−6；这些门可区分未推进、丢固定工具质量或重复施矩。有限差分 acceleration 来自真实末速度；Newton 离散 COM 与归一化 Euler 姿态与连续精确解的高阶差保留。

单个 JSON 包含：完整原始 `joint_q/joint_qd/body_q/body_qd/body_f`；所有原体质量/COM/惯量与本地/世界量；joint type/parent/child/frames/axes/starts/limits/驱动元数据；原生 shape ID/type/material samples；F01 原力、原控制、入态/末态/入态重读；F02 原 J/M、坐标变换与独立 oracle；各固定门的实际误差、原生警告/异常；runner、两份 URDF 和上述7份 native source 的全文+字节哈希。原始非有限值用显式标记保存，不能冒通过。数值门失败也写全读回并返回1；路径错误/写入失败仍由父守卫保原异常。输出路径存在即拒绝覆盖。

父待执行两个独立命令（均用新输出路径与 fresh guard；作者未运行）：

```powershell
& '<Newton环境>/Scripts/python.exe' '<作者树>/tests/reference_newton/robot/floating_fixed_tools_reference.py' --device cpu --composite-urdf '<父树>/tests/fixtures/urdf/floating_fixed_composite.urdf' --probe-urdf '<父树>/tests/fixtures/urdf/floating_fixed_probe.urdf' --out '<新证据>/floating-fixed-cpu.json'
& '<Newton环境>/Scripts/python.exe' '<作者树>/tests/reference_newton/robot/floating_fixed_tools_reference.py' --device cuda:0 --composite-urdf '<父树>/tests/fixtures/urdf/floating_fixed_composite.urdf' --probe-urdf '<父树>/tests/fixtures/urdf/floating_fixed_probe.urdf' --out '<新证据>/floating-fixed-cuda.json'
```

预期每端为 F01＋F02三姿态共4记录，全部门且无异常才 RAW0。这里的4是执行设计，不是已运行/已通过数量。静态/CPU/CUDA与任何父集成接受均待真实执行补记。
