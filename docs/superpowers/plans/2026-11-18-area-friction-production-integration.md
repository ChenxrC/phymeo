# 面积摩擦生产接入：下一阶段可执行计划

依据：截至2026-11-18的[已连接架构](../../development/area-pressure-connected-architecture-20261118.md)、[源码接缝图](../../development/area-pressure-friction-integration-map-20261118.md)、[I3q HOST](../../development/area-pressure-coupled-friction-report-20261118.md)与[I3r CUDA](../../development/area-pressure-tyx-coupled-friction-report-20261118.md)。这是当前法向链路之后的开发计划，不是已交付功能或完整Newton对齐声明。

## 目标与边界

让同一实际场景中的法向与切向载荷共同决定下一步运动，并在现有状态、观察、恢复机制中一致发布。保留最终法向平滑律：Fn=max(0,-k(phi+h vn))；Ft=-mu Fn vt/sqrt(vt²+vs²)。它有微滑，不称精确静摩擦。不能先解法向再额外施一次摩擦，也不重新运行I3q/r去追求更小误差。

第一产品范围为现有完整压力体对有限刚面、现有无activation/BALL的构型，只含新面积约束的世界。任何剩余可能native contact、limit或dry-friction row在同一世界显式拒绝；原native-only与friction=none世界继续原路径，可以在异构批次中并存。一般混合约束之后单独做联合方程，不能串联两个solver冒共同求解。

首版容量应从实际完整面目录确定。I3r当前6DOF/8点不是生产容量；需要在实现前选择有预算证明和测试的公开边界。至少覆盖已有完整有限面产生的积分点，不允许丢点/缩面/重归一来迁就设备。若首版保留6DOF限制，必须公开，不能称29关节机器人已接通新律。

## 开发节点

| 节点 | 难度 | 开发内容 | 独立验收与进入下阶段条件 |
| --- | --- | --- | --- |
| F1 显式law与容量 | 中 | 定义owned mu/vs、单位、law版本、每pair身份；选择并记录样本/DOF/世界容量及完整buffer预算；旧配置逐字节保持 | 非有限/负值/FP32正数下溢/容量/错pair预拒；旧bundle仍friction=none；无执行后择basis/自动降级 |
| F2 设备共同求解 | 高 | 从同阶段M、D、bias/passive/applied与完整两侧point J直接装配；将已验证Newton算术迁入backend，复用现runtime/ledger，完整清padding | 无HOST数值polish/二次workspace；保留nonsymmetric Jacobian和pivot；真实全部样本、非对角M、双方可动、mu0、坏末world/数值失败；资源计入manifest |
| F3 积分和原子发布 | 高 | 由共同Qcontact沿原native坐标积分，核最终实际v与求解v一致；新版本记录Fn/Ft/Ftotal，状态与载荷同次提交 | 不重复施力；物理qacc与integration_acc区别保留；末积分/下载/receipt失败不部分提交；所有世界都通过才换bank |
| F4 观察与恢复 | 中高 | 显式新面积记录版本、统一frame、力矩reader、asset digest、checkpoint/clone/selected reset | 旧v1不悄改语义；重载/恢复后下一步同输出；late观察失败不返回旧帧；源law/参数/目录/状态完整绑定 |
| F5 真实多案例 | 高 | 自动当前面→共同解→原生积分→公共观察，覆盖下表 | 每个案例都有独立物理区分和完整保存轨迹；不以supplied-point实验代替；完成才称生产摩擦接通 |
| F6 规模与一般约束 | 高 | 扩DOF/样本、mixed native/area方程、完整步成本，再连接训练特征 | 先固定同模型同输出和误差；未支持组合明确失败；无完整收益不扩大移植 |

F1–F4可分小提交，但对外不能将尚未完成的内部接口开放为已通过生产路径。只在F5完成后更新用户能力表。父任务拥有既有共享接缝；独立数值kernel、测试oracle和文档可由原隔离agent并行author，项目执行始终单一队列。

## 必须保持的三条动力学连接

1. 令b=applied+passive-bias，现passive已含-Dv0。A=M+hD，vfree=v0+h A^-1 b。如果沿I3r输入形式，f0=b+Dv0；不得直接再以b作为非阻尼外力而重复阻尼。
2. G=JA-JB来自同一入步两侧完整点Jacobian。切向不能由Jn反推。保留dFn/dv对Ft的交叉导数，不把Newton矩阵对称化后套T4的SPD因子。
3. 如果保留现末段积分器，将同一Qcontact加到total_applied恰一次，并检查实际发布v与共同解一致。FREE局部角速度/世界线速度与四元数积分继续用现native实现，不另写HOST状态更新。

## F5固定案例类型与判别

具体质量/材料/步长/窗口在实现前从I3l/o/s已验证资产选择一次并冻结；下面是必须覆盖的物理区分，不能执行后删除难例。

| 案例 | 允许的真实运动 | 不能被掩盖的错误 |
| --- | --- | --- |
| 平面滑动与低载启动 | 自由切向与法向DOF，同步承载 | 事后补摩擦、遗漏微滑、错误方向或正摩擦功 |
| 压缩后卸载分离 | 原速度/外载推进接触启闭 | 最终Fn=0而保留入步Ft、旧帧摩擦 |
| 偏心转动与有限边缘 | 真实hinge或FREE角速度，当前接触面随q变 | 只用线性J、错误力臂/世界局部变换、面消失后残留载荷 |
| 两侧可动／多区 | 全部真实质量与速度共享响应 | JB=0、逐pair独立解后相加、反力矩不闭合 |
| 旧路径与事务混合批 | 新律world、旧none world、native-only world同批 | 错world/错law/错目录、晚world失败部分提交、旧胶囊伪升级 |

比较两层误差：同一步方程是否被正确求解；模型是否满足预期功能。工程精度门沿I3r已公开FP32资格再对真实积分给独立界；不要求解决Newton/XPBD也不保证的问题。温和数值穿透可明确量化，不能把漏接触、重复力或错身份称精度损失。

## 成功与停止

成功：五类场景由自动几何产生，载荷和运动来自同一解；完整保存M/J/实际输入与输出、原始设备记录、残差、力矩、reset/restore/故障结果。生产旧路径兼容，容量有界可解释。

遇到一次具体失败，先保原证据并定位到几何、law、线性求解、积分或生命周期；修复该层后仅验证受影响范围。若模型本身不适用，公开缩小范围或更换明确law，不能调案例/门槛追绿。没有新功能或整步成本收益，不追加相似案例或重复旧几何证明。

后续优先级：F1–F5完整单路径 > 一般约束/更大机器人 > 全设备几何及端到端规模 > 面积观察的策略编码与训练 > 变形体压力散射。FFT非局部材料、Kamino独立参考和通用自动选择保持分离，不作为本次接线的额外前置。
