# P6.5 静态凸特征距离证书 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development for the user-authorized implementation delegation; TDD. One fresh final code review for this independent stage, followed by necessary fixes. The completed builder check is not a code review.

**Goal:** 提供不依赖递布编排、可复用的点/三角和边/边静态距离上下界，为后续连续碰撞认证建立严格数学输入。
**Architecture:** 单个纯函数模块从原FP64几何和可行权重计算精确有理凸KKT界，按有限binary64有向舍入返回不可变标量区间；不修改旧closest或接触程序。
**Tech Stack:** 现有Python3.11+/NumPy、标准库Fraction/struct；无新依赖，host FP64输入输出。
**Spec:** [距离证书设计](../specs/2026-10-29-p65-distance-certificate-design.md)。

## Global Constraints

- 只支持(N,M)=(1,3)、(3,1)、(2,2)，显式FP64有限存储；权重非负、精确总和正。
- 精确归一只构造距离见证，不修改材料附件；退化凸包有效，不改变旧closest退化拒绝合同。
- 区间是设计公式的最紧FP64向外包围；有效但松的[0,U]不代表碰撞/安全，不采用经验epsilon。
- 无history、时间推进、网格枚举、碰撞侧、厚度、法向、冲量、CCD或案例硬编码；不借用求解器私有数学模块。
- 当前材料接触准确push/SHA/CI闭环后再启动产品实现；保持旧求解器/积分器/容差/tyx pin不变。
- 用户要求至少两个非递布独立场景、状态回滚及实际后端边界；性能只观测，不扩P7/P8，不重复TYX-003长期压力。

## Review Focus

- 权重浮点总和溢出或不精确为1，但精确有理总和有效；按spec接受，不用FP64归一化悄悄改变见证。
- 次正规根、极端共同坐标或候选见证距离超过maxfloat；正确区分保守零下界与不可表示有限上界。
- 非最优见证与退化凸包不能产生过大的下界，也不能用永远返回零满足表面不等式。
- 任意手工构造的结果记录不是证明；调用方必须针对当前几何调用认证函数，避免缓存旧构型“证书”。
- 最后子步证书异常与资源退出失败必须保持跨域history/state全恢复，纯函数无可变缓存。

### Task 1: 公共证书、独立几何与薄事务接入

**Files:** Create `src/phymeo/geometry/feature_distance_bound.py`, `tests/test_feature_distance_bound.py`, `tests/test_feature_distance_bound_transaction.py`; modify `src/phymeo/__init__.py` and public installed smoke if needed. Do not edit current material-contact algorithms, surface_features math or SceneStepper behavior.
**Interfaces:** Consume FP64 finite arrays via core._arrays.float_array and core.capabilities.SolverFailure. Produce `feature_distance_bounds(first, second, *, first_weights, second_weights) -> FeatureDistanceBounds`, frozen scalar fields `lower_m`, `upper_m`, `method='convex-kkt-distance-v1'`. Trusted outputs arise from the function on actual inputs, not arbitrary result construction.

- [ ] 写公开API缺失RED：三角板(0,0,0),(2,0,0),(0,2,0)，探头(.5,.5,3)，权重板(.5,.25,.25)、点(1,)应返回[3,3]；按spec校验数据和标量结果。运行`.venv\Scripts\python.exe -m pytest tests/test_feature_distance_bound.py -q`，确认因缺API失败而非collection/helper错误。
- [ ] 写非最优见证RED：同一探头、板权重(1,0,0)，精确Q=7.5/S=9.5；几何真实距离为3，返回端点按Fraction平方及相邻FP64证实上下方向与紧性。然后实现精确见证/KKT/有界位序二分，不复制生产函数作为测试oracle。
- [ ] 增加杆件解析正常例、零距离、边/顶点最近、平行/近平行/重叠、退化点/线凸包。用独立解析或测试侧精确凸优化参照证明界包住真实距离；两个非递布场景不只替换名称。
- [ ] 增加Review Focus的舍入/范围RED：FP64权重和溢出、极小正权重、最小次正规距离、平方开方跨ULP、大共同平移、上界不可表示；根据spec实现接受或明确异常，禁止跳过几何对。
- [ ] 增加错误dtype/shape/组合/NaN/Inf/负与零权重、不可变结果/输入不变/失败后再调用、顺序重排及可精确表示变换。一般浮点变换对实际变换后输入独立认证，不要求不可实现的位级旋转不变。
- [ ] 在`test_feature_distance_bound_transaction.py`用测试侧SceneVerifier调用公共证书；真实多域候选几何+显式history，最后子步认证SolverFailure与验证后资源关闭故障分别RED；检查原state/history不变及retry=fresh。不改变运行时来让纯函数“拥有事务”。
- [ ] 运行新几何/事务与原surface_features/scene_stepper相关回归；thin actualtyx case检查设备状态确由tyx产生、几何证书明确在host运行，记录不伪装CUDA几何。无新tyx kernel时不为凑数重复此前材料接触完整CUDA长轨迹。
- [ ] 完成public export和源码外wheel安装smoke。交回实现diff、实际RED/GREEN结果、文件范围、后端支持/限制与剩余问题；主agent验收后commit。该项只完成静态距离基础，不能勾选动态CCD或H1完成。

### Task 2: 阶段审阅、工程证据与后继交接

**Files:** 新阶段acceptance/engineering/review与性能记录、可选小型serial benchmark；更新progress/continuation及框架双线状态。原current material stage结果不得重写成此阶段结果。
**Interfaces:** 仅消费Task1公共API；输出独立能力证据和后续动态接触设计输入，不新增碰撞实现。

- [ ] 主agent核对Task1 spec覆盖，派一次fresh最终code review，Critical/Important按独立RED修复，不再派第二轮。保留原始失败、适用范围与CPU/host定位。
- [ ] 必要core311/nativepip回归、Ubuntu新核心、ruff/docs/边界/compile、锁文件导出一致、源码外wheel；实际tyx薄集成单独记录。选测与未执行必须明示，不复用旧全量计数冒充新结果。
- [ ] 数值和review探针退出后串行计量独立证书；若含closest见证生成，分开报告。样本/批次/精度/源码与脚本hash固定，观察极端输入成本，不做优化/不归因tyx/不声明Warp差距。
- [ ] 更新能力“静态距离证书已验收”与案例“动态接触/CCD尚未接入”两栏，commit/push并核对精确SHA/CI；下一任务必须继续接触生命周期/厚度侧/特征/history和连续路径，不把总目标改成只交数学工具。

计划自查：spec接口/支持组合/错误与最紧包围/退化/所有权、两独立场景、精度攻击、薄事务和后端声明分别落入Task1；工程/性能/交接落入Task2。builder已返回nothing important left。产品Task1/2均尚未开始。
