# 最终功能概览：独立文本审阅建议

状态：**只读审阅建议，非新增验收**。依据父当前报告及概览草稿，截点为
`ec0e2997`；本作者未运行项目、读取大轨迹、重做哈希或保存审计。
最终结果以父随后实际包外收据为准，不由本页把待验提升为通过。

## 图中保持三条运行架构

| 路径 | 已连接的公共主链 | 必须直接写在图上的边界 |
| --- | --- | --- |
| Resident机器人 | `ParticipantSceneDeclaration` → 原资产/控制/安装来源 → `ResidentSceneBatch`/`TyxWorldBatch` → 接触载荷、各机械块求解、world共同发布 | `host_patch`与设备查询不是同一响应算法；有限表面查询本身不施力，不是跨树隐式KKT |
| 持久节点与有限刚柔 | D1持久节点bank → D2真实双源查询 → D3入步NormalLayer/散射 → 节点及刚体候选 → 一次共同发布；D5负责公共声明装配 | 每world仍一个节点模型＋一个ordinary单body运动球/胶囊；不是任意机器人/多刚体刚柔；有HOST载荷/诊断往返 |
| MuJoCo-TYX机器人 | CPU MuJoCo编译/原模型来源 → 显式准入 → TYX完整M/J/接触/约束/积分 → 公共runtime、训练或异构分桶 | CPU编译、明确的观察/参考不等于TYX步内CPU物理fallback；异构是同模型桶的跨桶事务，不是单异构kernel或跨桶碰撞 |

interpret/CUDA是执行后端轴，不能画成上述三条功能路径。异构和训练是第三条
路径的消费者，不是第四套物理求解器；不要画D3自动连MuJoCo机器人或任意
SceneDeclaration转换成MjSpec的实线。历史完整HOST观察仍有成本；已有GPU
任务/动作及PPO连接不能简写为全部闭环零读回。

D6在第二条路径中增加明确可选节点：`原elastic reconstruct → 一次有序轴向
implicit velocity pass → 完整候选验证 → 原共同发布`。原接触仍在入步采样，
不在耗散后重算本步力；None/全零走原路径。图旁标注“轴向、一阶分裂、固定
几何；平膜法向零模仍存在”，不写成全布阻尼或全网络隐式解。

## 最终状态卡建议

| 节点 | 当前可写事实 | 必须保留 |
| --- | --- | --- |
| D5-A | 首包113629b3两后端各38 calls/57 worldsteps/6显式query有限安装接受 | 仅上述单body＋节点域；不是D4稳态支持，也不继承关节机器人资格 |
| R3-N | 源码55＋兼容7；首包两后端各56 calls/112 worldsteps接受 | FREE+BALL/NONE JOINT/primitive normal有限组合 |
| R3-P | 源码69＋原路由兼容4通过；当前新包资格仍待父实际结果 | condim3完整连续参考已验；mixed1/3仅编译路由资格。activation、limits/dry、condim4/6等仍未接 |
| G2a | 完整原容量内源码矩阵；首包W16/B3与W16/B8两后端各27 calls/432 worldsteps接受 | W1/W8有源码资格，未新增对应安装运行；cap仍8模型/16world，live/peak内存UNKNOWN，无Newton性能比较 |
| D6-A | 核、D1、D3及D5已获有限源码功能资格；安装待父闭环 | run104动量strict与112原elastic速度strict均保留FAIL；新耗散、算术、连续工程及事务门仍有效，不称精确守恒 |
| D4原N1 | interpret/CUDA各完整3072步，入层/卸载/离层/重入链路可观察 | 两预定稳态窗的支撑力与速度RMS失败，另有原short精度失败；总体FAIL/DIAGNOSTIC_ONLY，原另三格未验 |
| D6-B新应用 | 独立新材料/新case协议，预登记c=4096 N·s/m | 实际结果未到即PENDING；即便以后通过也不覆盖原D4失败、不外推所有膜振型 |

数值损失必须分层：显式generated-FP32-subnormal交付记录、elastic舍入余项、
新damping requested/actual impulse与有符号能量差分别保存。它们解释离散
算术，不是用补偿力修状态，也不能解释掉D4真实稳态功能失败。

## 两份草稿的必要修订位置

- `p65-feature-acceptance-matrix-20261113.md:19–22`：旧PENDING/运行中截点已过期；
  按上表替换，另加R3-P/D6-A/D6-B行。第35/97行的pyramid缺口改为“源码已验、
  安装待闭环”，不要与仍缺的FREE+BALL activation混写。
- `p65-feature-architecture-draft-20261113.md:145`及200：原R3-N限定仍对，但当前
  总能力不能继续写pyramid全拒；第181/182/185行更新首包有限接受。D6插在
  原NodeCandidate与Commit之间，D5节点明确single-body，不借ParticipantArticulation名称扩大能力。
- 两稿页首绑定最终tested/product/wheel版本；保113629b3首包事实与后继新包的
  独立资格，不把旧模块来源/步骤数倒灌到新产品。状态表统一分“功能接通、
  指定源码/后端、安装消费者、完整应用门、性能”五层，不用PASS数作完成比例。

依据：父[D5报告](p65-deformable-scene-report.md)、[R3-N报告](p65-free-ball-normal-report.md)、
[R3-P报告](p65-free-ball-pyramidal-report.md)、[G2a协议末节](p65-heterogeneous-scale-protocol.md)、
[D4协议末节](p65-deformable-normal-load-protocol.md)、[D6源码报告](p65-distance-dissipation-report.md)
及[D6-B新应用协议](p65-damped-normal-load-application-20261113.md)。
后继优先新增可独立验收的组合及明确材料应用；不能因本轮收尾自动关闭
G1长舞、J/R学习、任意刚柔、CCD/摩擦或整个Newton功能对齐。
