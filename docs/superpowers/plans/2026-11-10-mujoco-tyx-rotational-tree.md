# MuJoCo-TYX T1：固定根旋转树

用户已授权按[扩展路线](../../development/mujoco-tyx-expansion-plan.md)继续。当前基线 `af3a7854` 的 T0 首域已验收。本批交付一个新的可调用模型域，验收后继续 T2；不重开已经通过的完整 T0 组。

## 开发与所有权

- 原 support agent：固定根 1–8 个 hinge/slide 的每步 FK、COM Jacobian、完整质量矩阵、速度/重力偏置、外载映射与 Euler 积分。保留静态 welded body、非主轴惯量、joint anchor/ref、armature、damping、spring 和原无状态驱动语义。只在独立 worktree 编码。
- 原 actuator agent：独立 CPU MuJoCo 3.3.7 多案例测试和预先登记的 FP32 工程误差门，不读取产品结果来构造 oracle。
- 原 capacity agent：对固定上游和新增源码作有针对性的语义审阅，重点为坐标、速度偏置、驱动力与阻尼的计入时刻。
- 父：共享入口、代码集成、唯一串行项目执行、安装包验证与验收。三个作者不运行项目代码、检查器或构建。

原默认 CUDA→TYX、显式 Warp 选择保持。T0 持续使用已验收实现；T1 在同一入口明确分派。所有物理更新必须来自真实 TYX 算子，不能在每步调用 CPU/Warp 求解，也不能冻结初态质量矩阵。

## 可验收结果

1. 原生坐标下的入步 q/v、每 body 位置/旋转/COM、世界线/角 COM Jacobian、M、bias、passive、applied、smooth acceleration 从同次设备计算发布；端点 CPU 观察仍明确单列。
2. 偏心单摆（解析重力与 M）、有非零速度的双摆、R/P 混合、共祖分支、7 关节非主轴惯量、局部 anchor/ref、welded 子体和偏心外载，用独立 MuJoCo 比较同一入步和短轨迹。
3. W1/W4、稀疏 world ID、控制差异、selected reset/clone 后下一步一致；晚回读失败不发布，输入与持有控制保持。限位继续采用明确的 strict-interior guard，不声称已有停止约束求解。
4. 分开执行解释与真实 CUDA；保存首失败、修复理由及最终同源结果。针对改动的 T0 相邻 smoke 即可，不重跑完整旧组。
5. 新 wheel 从源码外加载运行旋转树与同次动力学字段，许可证/新增模块随包；现有 Env/PolicyRunner 消费新模型的短窗验证。

当前无接触：只接纳不存在可能碰撞对的模型。自由根关节树、多接触、摩擦和 G1 属于后续批次，不通过删除源参数或静默回退来扩大支持声明。

## 执行次序与停止无效支线

先独立案例与接口冻结，再源码检查、精确新增/改动文件静态检查、解释专项、CUDA 专项、相邻与环境连接、wheel 源码外调用。所有项目执行前做 fresh guard，原始日志放独立 evidence 目录；每个完成单元小提交。

如果 M/J/b 不一致，定位首个不同物理量与 body/joint 地址，不调质量、初态、阻尼或容差来掩盖。已满足工程门的误差不追加精确证明。两轮没有新增可调用能力或实质失败闭环时停止当前支线并重新定位最小反例。

## 本批完成（2026-11-10）

已按本计划完成源码集成、独立解释21/CUDA17、相邻8、Env/PolicyRunner及新wheel源码外验证；T0驱动clamp语义另修复并通过6项。实际命令、误差门、来源与限制见[验收报告](../../benchmarks/p65-mujoco-tyx-rotational-tree.md)。产品/测试提交 `71ea1dc0`；不重复这些已绿组。下一项是原扩展路线T2自由根树及容量，保持多案例先于单一G1留出的顺序。
