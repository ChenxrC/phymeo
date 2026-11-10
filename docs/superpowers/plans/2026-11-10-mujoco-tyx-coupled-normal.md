# MuJoCo-TYX T3a：有限形状与共同法向求解

从已验T2 `d3f2dbba` 继续。本批交付实际多接触能力，保留[扩展路线](../../development/mujoco-tyx-expansion-plan.md)的T3有限几何、T4摩擦、T5 G1组合顺序；T3a不冒称完成全部T3/T4。

## 固定语义与域

固定MuJoCo3.3.7与MuJoCo-Warp0.0.2，读取其constraint、collision primitive和forward源码，并对照[官方计算文档](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html)。复用T1/T2已验FK/J/M/bias，CPU只编译及明确端点观察，不在设备步内调CPU/Warp解算。

本批形状矩阵为plane–sphere/capsule/box，以及sphere–sphere/capsule/box；固定根1–8scalar或单free-root加0–29scalar。隐式碰撞mask按native OR，保same-weld/parent过滤。仅混合结果condim=1，明确拒绝其它可能配对、显式pair/exclude、非默认flags及未迁移模型语义。硬界128geom、16possible pairs、32预分配候选槽；plane-box保8角身份但按CPU原corner顺序/ldist≤0/最多4真实contact。容量按实际detected与active分别计，恰满可接受，溢出共同拒绝。

每row在原contact中点用完整两体当前J，正normal指geom1→geom2。`a0=M^-1 rhs`，`A=J M^-1 J^T+diag(R)`，`b=J a0-aref`，解 `min(f>=0) .5 f^T A f+b^T f`。R使用compiled body_invweight0两体之和及native impedance；不把实际A对角当diagApprox，不加额外Jdotv。Euler随后解`(M+hD) a_int=rhs+J^Tf`。

数值求解明确称`normal-dual-active-set-v1`，只声明同一frictionless QP；不复现或伪称native Newton/PGS迭代与warmstart。source iterations（1..512）映射为有限active-set线性解预算；scaled KKT门为max(source tolerance,2e-5)，公开actual iteration、绝对/归一残差和预算。未收敛、非有限或坏pivot全批拒绝，不退为各行独立clamp。有限FP32解不称精确解。

## 所有权与接口

原source/test/audit三个隔离worktree继续，保旧分支/WIP。source作者只NEW `_normal_geometry.py`、`_normal_solver.py`、`_normal_device.py`及source报告；测试作者只NEW专项和预登记案例；audit作者只NEW固定语义研究/审阅。父独占`_normal_model.py`、旧compile准入/shared runtime/schema/NOTICE及整合文档。全部项目执行仍父单一串行槽；作者只文本/编码/Git。

新static plan包裹base kind2/3，上传geom local pose/size、pair参数及stable slot身份。设备五阶段：旧geometry→旧mechanics→新geometry contacts→新共同normal solve→旧Euler solve。最后一段使用独立total_applied与disposable constraint scratch，保原applied/a0和实际Jᵀf观察。所有candidate state/force/receipt在原同一次publication发布。

公开contact改为通用双方geom/body、contact frame、J、normal force与双方COM world wrench；保detected inactive、active但零force。receipt区分smooth qacc、含constraint qacc和integration acceleration，raw rows/dual matrix/solver diagnostics可审阅。旧T0公开记录保持兼容。

## 开发与验收

- [ ] 原生固定来源研究、预登记六配对/多接触/反例/工程容差。
- [ ] 新几何/共同求解、父准入/公共receipt和真实设备接线，独立源码审阅。
- [ ] 父首次静态与interpret、CUDA专项；首次失败保留，仅诊断修复。
- [ ] 同slider双接触独立解析反例、偏心四角、双脚树、双活动兄弟body、有限边缘撤离/重入、29scalar末body负载。
- [ ] W4稀疏world、detected/active恰满与溢出、选reset/clone下一步、晚故障重试和禁CPU/Warp/HOST负控。
- [ ] 少量受影响T0/T1/T2相邻、公共ENV与源码外新wheel双backend。
- [ ] 同次保存/源码/包绑定、正式结果及续作入口；本批释放后进入T3余下形状或T4摩擦。

案例与容差由独立作者在首次执行前登记，禁止观测后改模型/容差掩盖漏项。一次新可调用能力作为结束物，不用一个人形动作代替通用矩阵，不重跑旧完整已绿组。无远程CI/push/main合并/pin变更。
