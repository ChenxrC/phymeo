# 统一场景编译首批验证

本批执行父为唯一串行 owner；原三个作者隔离编码，所有原始命令、首失败、源码快照与原始退出码保存于 `artifacts/p65-unified-scene-20261110-01`。这是一批新的入口与适配器验证，不替代旧物理模块验收。

## 已完成的功能与实测

| 组件 | 新增能力 | 本批实际验证 |
|---|---|---|
| 统一场景 | 同源冻结声明进入 HOST / TYX interpret / CUDA；固定根、浮根、affine、多world、显式有限静态环境 | HOST 13；interpret / CUDA 各4；原报告time与实际device h分开 |
| CPU MuJoCo编译 | MjSpec.compile、MJB拥有型快照、ID/q/v/惯量/COM映射和显式资产转换 | 原27案例通过；新增3个collision override/disabled actuator拒绝案例通过；URDF独立质量1例通过 |
| 原生 CPU 适配 | 独立 MjData、多world控制/reset/clone、真实mj_step和原生contact | 专项21 + 原Env/PolicyRunner集成1通过 |
| 显式 MuJoCo-Warp | 真实CUDA step、双buffer发布、真实contact/wrench、容量失败保持旧状态 | 初套12通过；新增容量恰满2和FP32控制溢出3案例通过，CPU步进禁用相邻案例1通过 |
| 跨层组合 | 同一CPU编译资产进入 M1，独立原MuJoCo短步参照 | HOST/interpret/CUDA各1通过 |
| 原机器人规模留出 | Panda 9joint/19collider、G1 29joint/33collider，初态和完整identity映射，selected reset隔离 | HOST/interpret/CUDA各1脚本通过；只验证装配，不重跑G1动作 |
| 可选依赖边界 | 新模块导入不加载MuJoCo/Warp/TYX | 新进程阻断可选依赖后通过 |

native CPU 的新 stored_control 字段经原motor节点再次针对性核对；显式Warp选择调整后有真实CUDA相邻验证。各计数有少量针对新变化的重复node，不相加冒充唯一测试总数。设备专项中的 deselected 是显式选取HOST与device标记；实际选择内无skip。

## 保留的首失败和修复

- M2首套23通过后，一个multiple-articulation负控因继承limited=true却无range，在MuJoCo编译阶段先失败；修正新负控的未限位声明后完整27通过，并补真正fixed-child拒绝例。
- 10个新Python文件及后续组合文件的初次Ruff/format检查发现import和排版问题，原输出保留；仅对精确新增文件修正。
- 只读交叉审阅发现M2显式pair/exclude与禁用actuator组不能静默转换；现有明确LoweringIssue与3个负控。
- Warp0.0.2的connect/weld内核在恰满约束容量时可能省略最后一组，适配器保守要求nefc严格小于capacity；FP64有限控制转FP32溢出在整批控制替换前拒绝；receipt同时保请求控制与实际存储控制。

## 支持边界

M2.asset是机器人资产，不含完整环境，world geom ID单列。M1 first-domain是每world一个关节系统和有限静态环境；无新增self-collision/CCD/多机器人同world共同设备求解。HOST与TYX各自保持原力律与精度。

原生CPU/Warp保原MuJoCo坐标和接触律；环境Task需要显式适配，不能仅因方法名相同就假设观测与控制schema相同。Warp的actual last_step与detached CPU endpoint-forward观察分开。任意SceneDeclaration反向导出MjSpec尚未实现。

## 用户追加的原交接状态（历史）

新mujoco-tyx由专门原agent实现，CUDA默认选择将使用TYX，Warp只显式选择。本节尚未赋予新移植任何物理通过结论。第一批为受限平移树/自由球体/单法向接触；完整旋转树、多约束/摩擦/G1需要独立补齐。公共选择逻辑已有10个分派/拒绝测试通过，但这不是TYX求解测试。

以下是本轮完成后的实际交付记录。

## 用户追加的 MuJoCo-TYX 已完成首域验证

新后端源码真正执行TYX算子，不依赖Warp运行时或步内CPU mj_step/mj_forward。默认 `MuJoCoRuntime(..., mode="cuda")` 进入TYX；显式 `implementation="warp"` 保留原Warp，`mode="cpu"` 保留native CPU。无支持能力或设备时明确报错。

- 最终解释组53：27 passed（含环境集成）；真实CUDA组50：20 passed。
- affine actuator/spring reference/disabled group两模式和CUDA环境集成组54：3 passed。
- 相同source 0.05kg/4kg平移体、0.5kg自由球持续承载、卸载重入，逐步核q/v/normalforce；斜面非正交轴、8DOF分支完整耦合M、真实world torque、priority/directref/下降或零width impedance、精确margin零力site、容量/FP32/限位/晚故障均覆盖。TYX CUDA不是从Warp或CPU步包装而来。
- 采用同MuJoCo单法向软约束的闭式QP，能力字段区分source solver和actual numerical solver；不是完整Newton/CG/PGS算法移植。旋转tree、多contact/friction、一般浮根树与G1明确未支持。
- 首执行失败49/51/52分别为异常重导出、NumPy标量coercion和TYX不支持augmented assignment；已修复，完整原失败保留，没有放宽物理误差或改变原fixture。

新增[首域实现说明](../development/mujoco-tyx-port.md)、[通用机器人扩域计划](../development/mujoco-tyx-expansion-plan.md)和[接手说明](../development/unified-scene-handoff.md)。

## 新wheel与源码外消费者

58 build / 59 no-deps install全部成功。60首消费者因既有venv的editable路径仍在sys.path被预检拒绝，尚未执行物理；v2在导入前移除该路径/对应editable finder，并验证运行后所有phymeo模块都位于新安装target。67–73的7路径各2world×20实际步，解析末速度±0.04m/s、时刻0.02s和原运动方向门全部通过，无skip/fallback：

`host / interpret / cuda / mujoco-cpu / mujoco-cuda(default TYX) / mujoco-warp(explicit) / mujoco-tyx-interpret`。

源码 `fd8b5198`，wheel SHA256 `6e72c312afaf99f37ea40542a7e64b0c2fce6c14288e21723b80316ec98b5c5b`。12个新增产品/许可文件源码—wheel—installed bytes全同，optional extras和许可成员存在，收据 `wheel-parity.json`。实际新wheel为 `artifacts/p65-unified-scene-20261110-01/wheel/phymeo-0.1.0a0-py3-none-any.whl`。

这完成本批统一编译、明确后端选择、训练消费与首域TYX移植的交付；任意SceneDeclaration→MjSpec导出、完整MuJoCo/G1 TYX求解及整个P6.5仍开放。未运行远程CI、未push/发包、未改TYX pin，原K1三个WIP保持。
