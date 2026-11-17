# 面积摩擦生产接入：后续行动计划

更新：2026-11-18。当前 codex/exp 的[已连接架构](../../development/area-pressure-connected-architecture-20261118.md)以实际证据为准。用户优先完成可复用功能；本计划不把独立算术、测试数量或局部几何当成生产接通。

## 已完成与真正缺口

法向主链已接通：当前模型状态→自动有限面→真实面积积分→MuJoCo-TYX共同法向解→统一状态/载荷→设备观察与任务Env。几何仍为HOST或设备拓扑＋HOST精确复核。

摩擦已选 final_normal_smooth_v1：Fn=max(0,-k(phi+h*vn))，Ft=-mu*Fn*vt/sqrt(vt²+vs²)。其微滑是模型性质，不称精确静摩擦。

| 已有交付 | 能复用什么 | 不能据此宣称什么 |
| --- | --- | --- |
| [I3u](../../development/area-pressure-backend-friction-report-20261118.md) | backend共同求解，6DOF、128完整点、32world、完整非对称导数 | 未积分q，未接Scene |
| [I3v](../../development/area-pressure-entry-friction-report-20261118.md) | 真实历史入步M/J/载荷与源身份适配，区分重复阻尼 | 是历史counterfactual，不再施回已推进状态 |
| [I3w](../../development/area-pressure-resident-friction-pack-report-20261118.md) | 实际resident布局→七输入→共同kernel，中间无HOST数值读回 | 用原设备字节重放，不是live runtime或生产加速 |
| [安装包](../../development/area-pressure-friction-wheel-report-20261118.md) | 源码外完整128点输出、字节/包来源核验 | 没有替代整套未来场景验收 |

下一批的交付目标是**一条可选的真实场景摩擦路径**，同时连接材料声明、当前设备数据、原生积分与新的载荷记录。不要再增加相似的独立reference作为前置。精确切口见[live集成交接](../../development/area-friction-live-integration-handoff-20261118.md)。

## 首个产品范围

每桶1..32world、1..6DOF、完整样本≤128（现三点三角积分最多126）、机体≤128、pair≤16。全量拒绝超限，不删点或分次独立解后相加。新law世界只含面积约束；明确拒绝BALL/activation、equality/tendon、limit/dry及任何仍可能生效的native pair。旧native-only与friction=none可在别的桶继续原路径。首版不意味着29关节机器人或同世界任意混合约束已接通。

压力体对有限刚面，逐片g=grad(p)·n>0；保持真实面积、全样本与双方J。普通cube滑出可能出现p>0而g=0，必须在物理推进前整包拒绝；不得加epsilon、删片、缩板或称精度损失。偏心转动的已知g>0路径与任意滑出分别验收。

## 下一批 B1：最小生产闭环（高难度，最高优先）

这一批跨越F1/F2/F3及F4的最小发布部分；拆成小提交，联合验收后才开放场景选项。

1. **明确声明与身份**：新增owned law envelope，mu无量纲、vs单位m/s，绑定有序pair、材料、源、精度和layout版本。同MJB不同law/参数不能同桶或互换checkpoint。旧asset v1的friction=none含义不改。
2. **接真实设备入步**：在现mechanics/actuator后、旧constraint solve前调用已验pack和共同solver。复用owner runtime，scratch/status/双accepted bank全部进入allocation manifest和统一ledger，不新建每步workspace，不下载再上传M/J。
3. **一次原生积分**：[I3x](../../development/area-pressure-native-friction-bridge-report-20261118.md)已用四个固定历史入口验证这一接缝，下一批优先将同一Q加到原applied一次，再调用旧native solve。所有发布必须在actual vnext上核最终Fn/Ft、共同方程和有限性；不能仅检查v*。若该路线不相容，提取共享native q-only尾段直接消费v*，保留WXYZ/local omega/Kahan clock与全部原生地址，禁止另写HOST运动或先法向后摩擦kick。
4. **新载荷记录与共同提交**：保存Fn/Ft/总力/相对速度/双方矩、全样本身份、law、入出步相位；不能塞进旧F=nFn/type7记录。所有候选检查通过才翻状态和载荷bank。尚未实现的旧consumer/恢复操作必须在动作前明确拒绝新law，不能少算切向力。

B1固定短例复用I3v FREE偏心COM/六分量速度、mu=.5与mu=0、旧normal-only对照及空D40滑块，同批三步；再固定同资产的上离初速，验证入步有面但最终Fn=Ft=0，随后有效空面。原材料/几何/h不因结果调整。独立CPU只mj_forward/fullM/jac和mj_integratePos，不用CPUstep代替TYX运动。

成功门：公开scene入口由当前q自动生成面，实际q/v和新载荷同次发布；原始M/J/输入/输出完整保存；正确与重复阻尼、一次与二次积分、最终与滞后Fn均可区分。内核通过但无public实际步，B1仍未完成。

### B1接线清单（截止前源码核查，待开发）

避免再造一套同名材料类型：将现有 AreaFrictionPairLaw 移入不依赖 runtime/ModelSpec 的 backends/area_friction_law.py，原 mujoco_area_friction.py 重导出同一个类对象。保留原导入路径和构造签名；用新旧导入次序、类对象一致性、不可变参数和原历史adapter的针对性兼容检查验证。这是未来变更，当前代码尚未移动。

| 现有接入位置（均相对src/phymeo/backends） | B1必须连接的内容 |
| --- | --- |
| mujoco_heterogeneous.py：ModelSpec、_signature、_model_preflight | 末尾可选law tuple；完整有序pair；四名body/material身份、law、mu/vs、版本和精度进入分桶签名与预检 |
| 同文件：Batch.__init__ | leaf创建传递声明；新law条件性进入bucket绑定，旧normal-only绑定不增加空字段 |
| _mujoco_tyx_runtime.py：__init__ → mujoco_area_pressure.py：compile_area_plan | 捕获owned声明，统一能力、资源和准入；复用MuJoCoRuntime已有options工厂 |
| _mujoco_tyx_runtime.py：_clone_selected | reset/restore重建不得漏传law而退回旧法向路径；未支持操作在重建前拒绝 |
| mujoco_checkpoint.py：_model_binding；mujoco_area_pressure_state.py：area_binding；mujoco_entry_checkpoint.py：_entry_binding | 分别绑定物理配置与新accepted-record版本；旧schema不补造切向历史 |
| mujoco_heterogeneous_area_scene.py：Scene.__init__ | 从当前只核双方body名，补到完整body/material身份；几何摘要和law摘要分别绑定 |
| area_pressure_assets.py：_models/_match、Bundle.from_assets/assets_for | v1的friction=none封套不能直接匹配新law模型；明确拒绝而不静默改律 |

BodyAreaPressureAsset本身只含几何与材料身份，可用于显式新建摩擦模型；旧v1 bundle仍按原无摩擦含义解码。允许用户显式提取几何并重新声明新law，但这不是旧物理checkpoint的restore。将来保存新摩擦配置使用新版本封套，分别绑定几何与law；新磁盘格式不作为B1内存Scene接通的额外前置。

上述接线与同批实际三步、最终分离、真实载荷发布联合验收；仅提交law数据类不算B1完成。尤其保留不同mu/vs同MJB分桶、相同body但material不同拒绝、clone/reset不丢law三个负控。

## B2：真实双方、观察、恢复与晚失败（高难度）

复用原异构五world场景，保h=.0005和完整资产；hinge与moving plate检查真JB/双方反作用与虚功，native-only world对原leaf保持原路结果。导轨案例不冒自由双体总角动量定理。

新law-aware reader/codec绑定实际最终速度、Fn/Ft/总力、完整f64面积、材料与源。完成step→checkpoint→多步→selected reset→restore→同mode重演；同bucket只reset所选world，控制相位保真。改变mu/vs/law/源/资产需拒绝。

故障按相位分别测：pack/共同solve、最后桶下载或提交前失败，保全部accepted q/v/clock/tick/revision/force/receipt/geometry，清故障后可重试step；scratch诊断允许变化。物理已经提交后的观察失败，只保旧观察帧和游标，不回退已接受物理，也不能重复step，只重试观察；Env若已poison则按成对restore协议处理。restore末replacement失败保持原owner和已接受状态，清故障后重试restore。旧v1消费者不能默默读新数据。

成功门：真实双侧反力、正常路径混桶、完整恢复和晚失败均有实际结果；不以I3w的局部status测试替代Scene原子性。

## B3：连续任务与完整成本（中高难度）

在B1/B2通过后冻结三类连续轨迹：有法向支撑的切向滑动与低载微滑；冲击/压缩/卸载/再接触；g>0域内偏心转动和两处接触耦合。自动当前面、同次运动与载荷、真实可视化全部保留。先完成可观察功能，不要求超越Newton/XPBD精度。

比较1/小批/32world的相同模型与输出，分列初始化与稳态，覆盖FK、候选、完整面、装配、求解、积分、验证和必要传输。旧5/20/32的9.7–11.3%收益只属于其固定法向案例；不能倒灌到新摩擦路径。只有定位到真实成本才扩设备几何。

## 独立边缘准入节点 E0（中高难度，不阻塞B1）

先比较固定正梯度、零梯度侧片、负梯度姿态三类输入。现g>0拒绝是基线；若选择新压力律，先声明其力—间隙、功与求解域，再比较完整面积/合力/力矩/单步运动。论文的忽略polygon策略不自动成为本项目默认合同。具体手推反例与原始资料见交接第8节。

一次诊断与一次明确修正仍不适用，就公开限制或换模型；不循环调几何和阈值。尚未选定直接压力、滤面或其他新律，不能在B1实现中暗中处理g=0。

## 后续大节点与并行分工

| 优先级 | 节点 | 难度 | 进入条件 |
| --- | --- | --- | --- |
| 1 | B1真实单路径摩擦闭环 | 高 | 已有I3u/v/w直接复用 |
| 2 | B2 reader/codec/restore/真双侧与事务 | 高 | B1原始证据完整；可并行author独立codec和测试 |
| 3 | B3连续多案例与端到端成本 | 中高 | B1/B2 public路径通过 |
| 4 | 扩DOF与同世界mixed native/area方程 | 高 | 统一联合残差与资源上限先声明，不能串联两个solver |
| 5 | 全设备几何/大规模与训练特征 | 高 | 同模型同输出、完整成本有依据；策略确实消费面积观察 |
| 6 | 变形体面积压力散射 | 高 | 当前变形面/材料坐标/fullJ/双向响应单独验收 |

父拥有旧model/preflight、allocation、advance、runtime事务、checkpoint/reader等共享接缝。原scene agent可隔离author新增law声明/独立验证模块；原training agent可隔离author上述固定测试与oracle。先约定互不交叉文件范围，再并行编码；项目import/测试/数值/静态/build/install始终只有一个执行owner。当前不自动重派已经结束的任务。

FFT非局部半空间、Kamino独立参考与通用自动算法选择保持独立，不作为B1额外前置。成功看新增的可运行能力与保存证据；模型不适用、积分不相容、身份错误、漏接触和重复力必须分层处理，不能统称精度问题。没有新功能或整步收益，不继续扩大相似案例。
