# Phymeo 当前续作入口

> 2026-11-17 I2a真实接线完成：源码a898e3d1，31个新HOST/interpret/CUDA节点及4个原生T4接缝回归通过。当前面数据经真实机体Jacobian进入共同求解，面积压力与双方运动/载荷共同发布；真实关节阻尼、偏心力矩、能量、零压分离、容量及晚失败原子性已验。限显式supplied-surface/elastic-normal：D=0可混原生约束，D>0仅area约束；自动裁剪更新、摩擦、设备area观察及checkpoint/clone仍待接。下一优先I2b当前状态→新接触面→连续入层/承载/卸载，详[报告与下一步](area-pressure-tyx-step-report-20261117.md)。exp本地提交，dev不变；原作者/执行槽释放，无新期限/周期任务。下文为历史。


> 2026-11-17 双峰新几何完成：源码8fcfa2b8，73新HOST通过；12次外部CPU半空间参考/KKT及五组256→512一致性全部通过。100μm独立峰相加比共同解总力高8.06%、CoP偏0.1255mm；固定force-only PCHIP在40μm误差9.16%超过5%，实际初始化REFUSED，未benchmark未追加点。见[实验图与报告](multiasperity-contact-reference-report-20261117.md)和[检测＋动力学目标架构](contact-model-architecture-20261117.md)。pair身份不等于独立材料域，需保留跨接触区非局部耦合；总力表不能直接接I2。下一产品节点仍为MuJoCo-TYX单路径I2实际响应/阻尼/共同发布，再设备生命周期与resident，非本批已集成。exp继续，dev不变；原作者和执行槽已释放，无新期限/周期任务。下文为历史。

> 2026-11-16 有界自动选择原型完成：源码1c83de97，98新HOST通过；75/150μm两个新FFT512均KKT/原5%门通过。单点标定＋五点资格、相同输出成本筛选已实际处理244请求（65选择179拒绝），可解释域/能力/精度/预算拒绝。3/2先验标定F/k/U最坏误差0.51044%，两合格公式计时范围重叠，选择开销数十μs高于公式本身，未证明整体加速。见[报告与交互选择器](bounded-contact-autoselection-report-20261116.md)。下一优先初始化选择/复用/失效，以及真正需要数值参考的非轴对称或粗糙半空间；非Kamino/FFT在线混合或TYX接入。原旧FFT面积失败和dev保持；项目执行、原作者已释放，无新期限/周期任务。下文为历史。


> 2026-11-16 Kamino/FFT选择实验：codex/exp外部参考四格CUDA/3600步全部功能与h比较通过；FFT九格CPU离散约束通过，但原5%面积目标全部FAIL_RECORDED，最细载荷误差0.20%–0.73%、面积6.47%–15.01%。新47项HOST及一次序列化修复的新1项通过；没有改变数值参数追绿。已完成[报告、两类交互实验与24秒中文视频](kamino-fft-selection-report-20261116.md)，下一优先FFT离线标定→低成本接触律的有界验证；Kamino双指/关节为后续独立扩展，不自动继续Hydroelastic I2/在线FFT移植。dev76a240d8不变，原作者释放，数值执行已结束，无新期限/周期任务。下文为历史。


> 2026-11-16 I1：codex/exp测试源码9640a1de，真实三维tetra压力网格—有限刚面裁剪首次45新HOST/2.67s通过。七个同次保存解析案例覆盖斜面、有限边缘、共享面去重、跨cell、完整边界零压力体和多法向；面积/压力力矩及负控/容量均通过。见[报告与三维查看器](area-pressure-tetra-intersection-report-20261116.md)。仅小型HOST静态生成，full/local和g限制保留；下一I2共同实际运动/关节响应，不重跑旧几何。dev76a240d8、V03外部FAIL及独立T-smoke待接不变；父/原两作者全部释放，无期限/周期任务。下文为历史。



> 2026-11-16 R4/I0：codex/exp测试源码777b7f12，三维面积压力数据与完整双方Jacobian/HOST力映射首次59新HOST通过。真实dA、变法向、线性压力力矩、纯力偶、12DOF及共享关节坐标/虚功已验；原日志与13项绑定已核。见[结果与下一步](area-pressure-3d-port-report-20261116.md)、[接线裁定](area-pressure-3d-integration-decision-20261116.md)。仅接收已裁三角，未接碰撞生成/动态/GPU；下一接触批I1真实tetra–triangle裁剪，再I2共同解。V03 FAIL、dev76a240d8、独立T-smoke待接保持。父/两作者已释放，无新期限/周期任务。下文为历史。


> 2026-11-16 V05动态有限支撑：codex/exp数值源码052efc4b，首次20新HOST与两条0.6s真实轨迹/1800步全部通过。0.5kg体实际稳定承载4.905N，4N推动后俯仰越边，约0.382s离开有限支撑并卸载；两h峰值差0.01168%，均出现侧面接触，完整保存148文件/12事件已核。见[报告、离线查看器与24.2秒视频](area-pressure-planar-dynamics-report-20261116.md)。仅通用凸截面2D挤出NumPy参考、单运动体/固定支撑，未接完整3D/TYX/产品。下一接触节点R4选型及首个3D表示/I0–I2接口裁定；V03外部FAIL和T-smoke独立主线保留，不重跑既过组。dev仍76a240d8，无新截止/周期任务。下文为历史。


> 2026-11-16 V06双自由体动力学：codex/exp数值源码423e872b，首次17新HOST与四条0.5 s真实轨迹/3000步全部通过。实际压力面积积分驱动双方有限质量；零耗散细步长能量误差0.007981%，有耗散末态相对分离速度0.048256 m/s；峰值步长差异均<0.25%。完整入层/压缩/卸载/分离、16压力事件及151文件绑定完成，见[报告、交互查看器与真实运动视频](area-pressure-impact-report-20261116.md)。限同轴居中零转矩NumPy参考，未接TYX/产品或一般6DOF。下一接触批V05真实支撑/水平推动/俯仰与边缘退出，再R4及I0–I2；V03外部FAIL保留，T-smoke独立。dev仍76a240d8，父和原作者已释放，无新期限/周期任务。下文为历史。


> 2026-11-15 V05有限支撑边缘：codex/exp数值源码fdf05a42，首次14新HOST与102固定案例全部通过，34组三档/末两档比较通过。通用凸投影/仿射gap裁剪、真实dA及压力中心/COM力矩已验证；相同压入下移到40 mm时承载从4.905降为2.4525 N，到60 mm归零。仅规定姿态的有限顶面参考，不是自由滑落或runtime集成。见[报告、查看器与30秒说明视频](area-pressure-finite-support-report-20261115.md)。下一接触批转入V05/V06力与运动共同更新，保持原1/0.5 ms动力学工程门，不重做已绿静态几何。V03外部FAIL与T-smoke训练待接状态保持；dev仍76a240d8，所有执行和原作者已释放，无期限/周期任务。下文为历史。


> 2026-11-15 V04分布摩擦/扭转：codex/exp最终数值源码e13a7216；25新HOST与26固定格全通过，保留近单位法向反例及一次投影修复。最细扭矩误差0.17%–0.24%，三档最后变化<0.74%。单/双盘共同轴、反转、曲面法向和作用反作用已验；仅规定压力/速度参考，不是动力学/静摩擦锁止。见[报告、查看器与30秒说明视频](area-pressure-torsion-report-20261115.md)。下一接触节点V05有限边缘，再V05/V06共同运动响应；V03外部面积/收敛FAIL仍保留，T-smoke训练主线独立。dev保持76a240d8；本批执行与作者已释放，下文为历史。


> 2026-11-15 V03外部参考更新：codex/exp已执行独立双软球曲面与Newton1.7.0 CUDA七格，43新HOST通过；原计数读取首错已修复并保留。最细力误差约3.4%、面积约6.46%，最后两档收敛也超过原5%门，明确FAIL_RECORDED；不能称交叉验证通过或runtime集成。见[本批报告与28秒说明视频](area-pressure-newton-reference-report-20261115.md)。仅静态碰撞前端，零solver.step；不追加网格追绿。下一接触独立节点V04分布扭转，现有raw误差归因与T-smoke主线分开推进；dev未改。所有数值执行和两作者已释放，下方为历史。


> 2026-11-15 分支与实验更新：稳定基线及 GitHub 默认分支为 `codex/dev`（76a240d8），当前父工作树为 `codex/exp`。已清理131个已合入本地分支/6个远程分支；12个有独有补丁的本地分支和远程rigid WIP保留，见[整理清单](branch-consolidation-20261115.md)。exp首批V00/V01/V02独立NumPy参考完成：源码48c4a14b，14 HOST通过，30固定格通过，交互压力图和15秒静态参数扫描视频已生成，见[报告](area-pressure-static-reference-report-20261115.md)。仅静态积分/声明压力场，尚无Drake/Newton参考执行、共同动力学或新训练结果。下一步仍是T-smoke共享语义PPO接线，以及独立V02/V03外部参考；已绿本批不重跑。所有项目执行已结束，无新截止或周期任务。下方旧批次均为历史。


> 2026-11-15 用户调整：先跑通真实训练，暂不要求操作成功率。当前已验证产品截点 d7c8195e；合入现有远程 codex/dev，不纳入三份 untracked device-convex-patch 实验。下一开发批是8原CUDA环境的真实状态/目标编码、共享masked策略/PPO更新及恢复，T1b接触切换和抓放效果不再前置。独立接触预研首批为面积缩放、偏心压力、软球压平面；见[小案例与训练计划](../superpowers/plans/2026-11-15-area-pressure-contact-visual-cases.md)、[总计划第21节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。本轮只合并与规划，无新的训练、压力接触数值结果或安装。下方第十二批及以前是既有验收历史。


> 2026-11-15 第十二批：目标位姿评分、有限换点/撤离/绕行与进度monitor已接入同一连续CUDA episode，产品32aae3ab。96个新HOST节点（91+3+2）通过；三窗25700步/2573保存帧审计通过。RR/RRR完成各6段，12秒内归一代价下降30.065%/16.201%；PPPR首段下降42.341%，随后撤离路径碰到转动T横梁，1700步明确停止。原完整PushT目标仍0/3，不能把综合代价下降当朝向或IoU成功。见[本批报告](p65-manipulation-push-feedback-report-20261115.md)、[计划第20节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)、[公共用法](p65-manipulation-env-usage-20261114.md)。下一步T1b-2d先做当前几何退出/保持接触与有界模型修正；不是延长固定动作或放宽深度。全部执行与作者槽释放；语义编码/共享训练、完整放置仍未完成，无新截止/视频/安装/push/旧窗重跑。下方旧批次仅保留历史。


> 2026-11-15 第十一批：PushT完整接近与真实CUDA短推已接通，产品580b9d3a/5cfb0afc。54新排序＋20 consumer（新6/兼容14）HOST通过；RR/RRR/PPPR各2000步，6000 CURRENT entry及603端点帧审计通过。沿推向实际位移5.926/12.338/17.231mm，原完整PushT目标0/3。RR转向违背目标、PPPR出现偏航；下一步是T1b-2c有界换点与目标位姿闭环，不能只延长固定法向。见[本批报告](p65-manipulation-push-route-report-20261115.md)、[计划第19节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。原两作者及root执行槽释放，未派下一任务；语义编码/共享训练仍未开始，无新视频/安装/push/期限。下方旧批次只保留历史。

> 2026-11-14 第十批：公共contact force/COM moment测量、保存读取器与有限PushT外边界已集成（64a6b206/16f6c9ea/b1a521f6），新增82 HOST通过。world1/3共4798条旧entry消费通过；14个native同状态比较中world1第1900步0.0181N差保留未归因。原RR/RRR/PPPR各24候选、22几何usable，两端点单seed IK收敛10/10/14，尚非整机路径/真实推行成功。见[本批报告](p65-manipulation-wrench-push-report-20261114.md)、[计划第18节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。下一步优先PushT完整接近路径与真实短推，同时以实际pad覆盖/相对姿态证据设计6R持握改进；语义编码和共享训练仍未开始。无新CUDA窗口/视频/安装/push；父和原两作者释放。下方第九批及以前保留历史。


> 2026-11-14 第九批：完整姿态IK与C1采样路径检查已集成，产品316df74f，新46个HOST首次通过。world1/3四条参考路径各2402样本合格，各一次2400步CUDA及全部4800步/482帧保存核对通过。盒最高离台26.98mm但原grasp未过；6R胶囊实际抬升38.66mm，仍转动/下滑且原place未过，不称完整放置。见[报告](p65-manipulation-route-report-20261114.md)、[计划第17节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。下一批先用已存轨迹补接触位置/COM力矩/相对姿态支撑诊断，再登记有依据的持握改进；原PushT边界路线可隔离纯编码并行。root/两作者执行完成后释放；无新截止/安装/push/训练，不重跑这两窗。下文第八批及以前为历史。



> 2026-11-14 第八批：通用姿态审计、公共接触点滑速测量与有界双指反馈已集成，产品02f44795，新增88个HOST通过。固定11候选中4个镜像seed的TCP虽到位却pad穿台，明确需要夹爪方向与全路径约束。world0一次CUDA反馈对照在第2038步通过原短窗抓球Goal（entry约2.037s、离台48.38mm、连续低速保持.2s）；完整保存因果审计通过，原模型/物理/目标保持。仍有缓慢滑移，其他7任务本批未新增操作成功，尚无语言/共享策略训练。见[报告](p65-manipulation-pose-feedback-report-20261114.md)与[计划第16节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。下一批为方向受限接触姿态/全路径、边界PushT、完整放置与迁移。root和两作者已释放，无新截止/安装/push；下文旧0/8是历史基线。

> 2026-11-14 第七批：T1a通用PD/有界IK、真实CUDA诊断采集及八画面视频完成，产品5a281455、渲染3bed01fd。12个新HOST通过；原8世界2200步/2.2秒，17600 world-step和221保存帧已核，**任务成功0/8**。world0夹起后滑落，world7推约17mm未到目标，其余6条点IK路线未收敛并保持原参考。见[报告及视频](p65-manipulation-control-report-20261114.md)。没有共享策略、语言模型或optimizer，视频明确标诊断控制。下一批按[计划第15节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)补接触姿态/有限seed、持握与可达推行；当前root和两作者均释放，无新截止/安装/push。下文第六批及以前保留历史。

> 2026-11-14 第六批：B3异构操作Env已完成源码资格，产品 `135752e4`。24项checkpoint边界与10项CUDA生命周期节点通过；原八场景W8/W1、32步支撑、选择性reset/换目标、final observation和配对恢复已验。见[本批报告](p65-manipulation-env-report-20261114.md)与[公共用法](p65-manipulation-env-usage-20261114.md)。下一批按[计划第14节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)从T1a真实抓/放/PushT控制可达性开始，随后扩八构型并接真实语义编码与共享策略。当前尚无操作成功率或训练结果，success评价仍有HOST读取。当前无执行owner，无新截止/安装/push；下文第五批及更早结果保留为历史。
> 2026-11-14 第五批：M2-B2c/d 公共GPU接触观察与同入步SuccessFrame已完成源码资格，产品 `3ceb77cb`。新增19项（interpret9/CUDA10）和受影响回归52项全部通过；世界力/真实力偶、双侧body-origin力矩、Torch副本、同字节tick0重置与配对恢复已验。见[本批报告](p65-contact-observation-report-20261114.md)和[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。`include_contacts=True` 现在可用，默认仍关闭。下一批按[计划第13节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)完成B3异构Env的动作掩码、目标/episode/history、终止帧和组合checkpoint，再推进T1真实抓放/PushT及语义共享策略。当前无执行owner，无新截止、安装、push或训练结果。下文第四批及更早结果保留为历史。

> 2026-11-14 第四批：M2-B2 持久入步记录、全接触路由归一和 reset/clone/checkpoint 存储已完成源码资格，产品 `994b0842`。19个新节点（interpret7/CUDA7/HOST codec5）与95项受影响回归通过；原首力偶fixture失败及基于接触点运动学的修正均保留。见[本批报告](p65-entry-record-report-20261114.md)。下一批按[计划第12节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)完成公共GPU entry/body/world-wrench观察与SuccessFrame，再接B3 Env；公共contact_tokens仍关闭，不能混用B1端点与入步载荷。当前无执行owner，无新截止、push、安装或训练结果。下文第三批及更早结果保留为历史。


> 2026-11-14 第三批：M2-B1 公共 GPU 端点 body 观察已完成源码资格（`f1c3d9a0`）。`prepare_device_workspace(include_bodies=True)` 从 accepted q/v 导出 body origin 世界位置/线角速度及 body→world 矩阵，与 native 状态共同发布。新增解释6/CUDA8及受影响兼容58项通过，见[本批报告](p65-body-observation-report-20261114.md)和[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。下一批按[计划第11节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)实现 B2 accepted-entry 接触持久双 bank、同相位状态/载荷与 reset/checkpoint，再接 B3 Env 生命周期。接触 token、真实操作可控性、语言编码及共享策略训练仍未完成；无新截止、当前执行槽已释放。

> 2026-11-14 第二批历史：M2-A公共异构GPU状态/动作/reset接口已完成源码资格（2de488b9）；八份真实操作场景静态24项通过，并在同GPU各32步/256环境步与独立CUDA W1逐字节一致、全部出现物体支撑接触。见[本批报告](p65-heterogeneous-manipulation-io-report-20261114.md)及[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。合计143个不同验证节点通过。下一批M2-B是端点对象/接触特征与Env目标生命周期；M0可达性、实际抓放/PushT、语义编码及共享策略训练仍未完成。该批执行已结束。

> 2026-11-14 新应用主线：单GPU上8个异构机器人环境、抓取/放置/PushT共用一个自然语言条件策略。参见[研究与实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)。首批M1共享多树动力学已完成源码资格：产品0c217310，新增28项多树测试和95项受影响兼容节点通过，见[M1报告](p65-mujoco-tyx-forest-report-20261114.md)。固定臂与独立完整FREE物体已可共同进行CUDA接触/运动；下一批是公共异构GPU训练接口及8任务场景。预训练语义编码、共享策略训练、完整抓放/PushT成功率与新安装交付仍待完成。下文原验收保留。


更新：2026-11-13；**G2b-A共享CUDA数组预算与32构型展示已完成限定源码/安装验收**。产品e1401fbf，源码64项通过、新包536成员同字节；包外interpret12步、CUDA256步（8,192worldsteps）均STRICT_PASS。257帧实际CUDA轨迹的视频已生成，见[报告与证据](p65-g2ba-resource-gallery-report-20261113.md)和[公共用法](p65-g2ba-resource-gallery-usage-20261113.md)。原两作者及父执行槽均释放。BALL-S、M-B2/M-B1、G2-R0与R3-A原有限验收复用。
D5-A、R3-N/P、G2a、D6-A 已获各自有限源码/安装资格；D4 原稳态功能仍 FAIL。
D6-A 的 FUNCTIONAL_PASS 与旧弹性 STRICT_FAIL_RECORDED 同时保留，不能合并成严格通过。
整个 P6.5 / Newton 功能对齐未完成。旧07:30计划、M-B2、BALL-S与G2b-A均已收口。下一资源模块是G2b-B完整HOST/native/context/JIT等观察与外部owner计量，再另冻更大规模及公平性能格；不继承旧截止或session，不重跑已绿组。

## 从这里继续

0. [G2b-A验收报告](p65-g2ba-resource-gallery-report-20261113.md)、[公共用法](p65-g2ba-resource-gallery-usage-20261113.md)：六solver-bank数组manifest/预算及32无接触异构构型；不是总显存预算、融合kernel或跨world接触。[BALL-S](p65-ball-scalar-constraints-report-20261113.md)原混合约束资格复用，BALL角限位仍未实现。
1. [M-B2验收报告](p65-articulated-deformable-report-20261113.md)及[公共用法](p65-articulated-deformable-usage-20261113.md)：非根连杆、完整J/M投影与共同发布已验；[实施合同](p65-articulated-deformable-implementation-20261113.md)已关闭。[上一批计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)仅作历史。
2. [功能验收矩阵](p65-feature-acceptance-matrix-20261113.md)：源码、后端、安装、应用和性能分列；不估整体完成百分比。
3. [当前连接架构](p65-feature-architecture-draft-20261113.md)、[统一编译架构](../architecture/unified-scene-compilation.md)：声明、设备来源、查询、载荷与共同发布。
   [离线架构总览图](p65-functional-architecture-20261113.html)保留G2a时点的8/16容量标记，当前32及预算以本页/新版文字架构为准；图中未连接的域仍明确分开。
4. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)与[新版离线全景图](phymeo-newton-panorama-2026-11-13.html)：下一通用模块与未完成范围；本轮详细事实以验收矩阵及原收据为准。
5. [D6用法](p65-distance-dissipation-usage.md)、[D6设计](../superpowers/specs/2026-11-13-distance-dissipation-design.md)、[D6-B应用协议](p65-damped-normal-load-application-20261113.md)。
6. [Batch C](p65-three-track-batch-c-20261112.md)、[Batch B](p65-three-track-batch-b-20261112.md)、[旧历史入口](p65-continuation-history-through-20261112-0035.md)仅按需引用，不恢复旧周期任务或重做旧审计。

当前G2b-A证据：`artifacts/p65-g2ba-resource-gallery-20261113-01`，产品e1401fbf、536成员同源，22正式CUDA及25/29视频已经结束；完整压缩轨迹/状态/视频保留，不重跑或解压补副本。原BALL-S证据 `artifacts/p65-ball-scalar-constraints-20261113-01`、d7105a9d、534成员资格复用，不重启原已结束组。
M-B2原证据 `artifacts/p65-articulated-deformable-20261113-01`、快照 `6b4a639c` 与验收 `cdccbe44` 复用；旧D6-B仍以143结束收据覆盖历史pending状态。
本页是续作导航，不替代原完整输入、输出、原始失败、source snapshots 与保存收据。

## 已验公共功能与边界

| 功能 | 当前资格 | 仍然不能声称 |
| --- | --- | --- |
| G2b-A共享数组预算与异构32 | 六solver-bank实际manifest、跨owner/边界峰值、UNKNOWN计费；64源码项；包外W32/B32解释12步与CUDA256步STRICT_PASS，真实视频 | 完整进程资源峰值、融合kernel、跨world接触、更大W、全部约束组合或Newton性能胜出 |
| BALL-S 球树混合约束 | fixed/FREE BALL/R/P中scalar软限位、DOF dry与NONE/FILTEREXACT、无接触/原condim1/3共同求解；源码132＋恢复兼容8、包外各448worldsteps通过 | BALL角限位、G1长舞、BALL刚柔、更大异构规模；下方R3各行限制是其原首片历史范围 |
| D5-A 公共刚柔声明 | `ParticipantSceneDeclaration(contact_response="deformable_normal")`＋`ParticipantDeformable` 自动装配唯一 D3 owner，资产球/程序胶囊短窗及生命周期源码、安装通过 | 任意关节机器人或持续稳态支撑；多独立单体见M-B1 |
| R3-N 浮根球树法向接触 | FREE＋BALL/R/P、NONE JOINT motor、primitive condim1；完整同入步与连续参考及安装通过 | 人形长舞、limits/dry 等全部组合；activation见R3-A |
| R3-P 浮根球树摩擦接触 | 同一约束/CSR路径的 primitive condim3 正控、安装通过；mixed1/3仅编译路由资格 | condim4/6、elliptic、FREE＋BALL limits/dry 已放行 |
| R3-A 浮根球树有状态执行器 | FREE＋BALL/R/P的NONE/FILTEREXACT JOINT，no-contact/normal/pyramidal；源码117＋兼容6及安装两后端各352worldsteps通过 | limits/dry、多FREE根、任意机器人/刚柔组合 |
| G2a 原容量内异构规模 | W1、W8/B3、W16/B3、W16/B8 源码格及晚失败/retry通过；W16两个格各有两后端安装资格 | W1/W8额外包外资格、单异构kernel、跨桶碰撞或W32以上 |
| D6-A 可选轴向耗散 | 保存 elastic reconstructed velocity 后一次 ordered 速度pass；D1/D3/D5有限源码通过，D1/D3安装 FUNCTIONAL_PASS | 精确守恒、旧elastic短门通过、全网络BE、全局drag、所有膜振型衰减 |

原 S1–S3 resident 多参与者查询、M1/M1-A 资产来源、D1设备节点、D2有限球/胶囊–triangle 与 D3共同发布资格继续复用。
D3/D5经M-B1扩展为每world一个节点模型＋多个ordinary单body动态球/胶囊；M-B2另支持每world一个ordinary R/P浮根关节树，子连杆球/胶囊真实受载、完整B×6与J/M共同发布。两类world可混batch。总participants≤8、worlds≤8、每world bodies≤33；入步采样、逐facet NormalLayer、零摩擦，存在HOST载荷/诊断往返。
原几何 CONDITIONING 不得丢弃后继续施力；查询不自动授予响应、CCD、实体内外或面积压力。
Resident 刚柔与 MuJoCo-TYX 机器人是分别准入的路径，不能把支持表相乘成任意联合求解。

详细范围：[D5](p65-deformable-scene-report.md)、[R3-N](p65-free-ball-normal-report.md)、[R3-P](p65-free-ball-pyramidal-report.md)、[G2a](p65-heterogeneous-scale-protocol.md)、[D6-A](p65-distance-dissipation-report.md)。

## 安装产品与保留失败

- 首包 `113629b3`：530成员同源；R3-N 75/76、G2a W16/B3 77/78 与 W16/B8 79/80已验，按原证据复用。
- 第二包产品 `ec0e2997`：run127核532成员 source/wheel/installed 同字节；不是全部历史套件重跑。
- 第二包 D6 128/129各44 worldsteps，均 FUNCTIONAL_PASS，同时每后端20项旧弹性精度记录、`strict_elastic_precision=FAIL_RECORDED`。
- 第二包 R3-P 130/131各112 worldsteps，D5 132/133各57 worldsteps，均有限安装通过。
- D6专用安装消费D1/D3；带耗散D5声明有源码资格，不能借普通D5安装消费者声称另一个带耗散D5包外矩阵。

**原D4无阻尼N1：** run50 interpret、run69 CUDA各完整3072步/3秒，入层、释放、重入等事件发生，但两预定稳态窗支撑力均值与速度RMS不合格，均保留 FAIL / DIAGNOSTIC_ONLY。
这不是仅精度问题；另有run46第301步旧短速度门失败，原其余三格未执行。生成载荷subnormal窄修前后的两轨迹不冒称同一产品快照。
**D6-A：** run104原总动量严格失败、run112旧elastic重构速度严格失败及安装strict记录保留；完整算术余项解释不修改状态、请求冲量或原阈值。
固定反力包含外/接触、elastic及damping冲量；requested与实际速度冲量、理想功与有符号舍入差均分别保存，不使用能量补贴。

## D6-B 实际结果与下一步

新案例统一四径向边 `c=4096 N·s/m`，来自事先静力/线性化量纲选择；原四格几何、质量、初态、3秒时序、h、物理门全部保持。
这是独立的新材料应用，不覆盖原D4 FAIL；接触阻尼单位仍s/m，不能与材料系数混淆。
新短合同 run140 interpret14通过，run141 CUDA单返回故障1通过；**这些不是长窗资格**。
包外首launch142因辅助目录不在search path而0step失败；原记录保留，未发生物理推进。
run143仅补明确的源码外helper路径，已完成3072步/3秒，returned=checked=3072、control3、close1，453.315秒；状态 **FAIL / DIAGNOSTIC_ONLY**。
两窗支撑均值21.46127079/19.36648514 N、刚体RMS .30439415/.21362712、中心RMS .16520125/.11754240，均未达到原稳态门。

1. 143入层/卸载/再接触、逐步phase/力/算术/连续长参考完成；peak约51.95 N、gap约−.03037 m、原侧约+.09463 m通过原门。7项旧elastic短精度失败仍记录。
2. 按预登记停止这一系数的应用扩展：不启动该格CUDA或其余固定格，不扫描c，不修改质量/h/时间窗/验收门。
3. 新D6-B失败与原D4失败分别保存；完整运行和可解释耗散不是稳态成功，不能把未运行格或新模型外推成已验。
4. R3-A已完成117源码＋6兼容、第三包532成员parity和164/165两后端各352worldsteps；详[正式报告](p65-free-ball-activation-report.md)。M-B1已完成多独立单body与同片的地址/反作用/共同发布；新增28源码节点及S8/M3容量4节点通过，两后端安装各68worldsteps/132participantsteps，见[正式报告](p65-multibody-deformable-report-20261113.md)与[用法](p65-multibody-deformable-usage-20261113.md)。
5. 异构扩容先补完整生命周期内存预算；live/peak无真实observer则UNKNOWN，Newton公平性能比较另立阶段。

## 执行与保护边界

父集成：`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`，当前实验分支 `codex/exp`，稳定与默认分支 `codex/dev`。
父/root独占旧共享接缝、合入及唯一串行项目执行；原三作者仅在既有隔离worktree按allowlist并行作者工作。
继续前先核实时agents/Git/WIP和当前grant；已释放任务不重派，同源已绿不重跑，同hash证据不重审。
所有import/collection/数值/probe/pytest/lint/formatter/compile/build/install/profile必须由父fresh entrypoint-aware guard后串行执行。
旧06:15冻结/07:30收口已经结束。新一批按用户继续指令与M-B2合同执行，没有新的时间截止；仍不授权伪造通过或自动扩大模型域。
不建周期任务，不poll/rerun/dispatch远程CI；按用户授权推送dev/exp，不做mainmerge或发包，不改pin或盲跑寿命stress。
三份无关未跟踪稿、证据Junction及其canonical归档、用户Torch和当前venv/pins全部保留，不跟随Junction删除或全解压补副本。
观察器在物理返回后失败不能称物理回滚；UNKNOWN_RELEASE保owner/lease，不能重复free或伪报零live。

旧G1长舞46.29s XY门/131.48s完整动作、J/R学习、R03两项净加速度、B-point touch-SE仍开放。
通用摩擦/换面历史、CCD、更多刚柔机器人组合、传感、可微物理、密集布/体软体、杆缆与MPM/流体仍是后继；本轮不因安装绿而关闭。

第三包R3-A产品08d7a1b6、全部164/165返回和压缩trace已保留canonical installed-closed-archive-01/wheel-03；不重跑同源已绿矩阵。M-B2现有独立实施合同，BALL limits/dry与G2b仍为后继提案。

## 本轮后段接受与下一交付

- 第四包产品23757ae6：534成员同字节，191/192 M-B1各44worldsteps/88participantsteps；201/202 S8/M3各24/44。S8=4world/8participants，不外推8world满配。
- G2-R0：178预期RED，185源码12GREEN、193包外12GREEN；实际恢复晚失败保原发布、原异常与cleanup，确切候选强owner及真实关闭。UNKNOWN未被重试，资源完整预算仍未实现。
- 新canonical `artifacts/p65-feature-until-0730-20261113-01/installed-closed-archive-04/wheel-04`：33文件，原件保留，禁止重复归档/全解压。没有新项目执行owner；已绿同源不重跑。
- M-B2已验后，下一两条独立功能线：BALL树标量limits/DOF dry与完整mixed checker；G2b真实allocation manifest/生命周期预算。两者仍为提案，先冻结各自多案例与接口，再由原作者隔离编码、父共享接缝及唯一串行验证。
- D4/D6-B只按[可否证的短试样后继](p65-sustained-support-next-experiments-20261113.md)另冻实验；不选第二个c盲跑。本轮沒有新增该实验或BALL limits产品执行。

## M-B2 本次交付

- 产品/测试快照6b4a639c：新解释15＋CUDA10，受影响旧路径兼容39项通过；没有功能测试失败或数值修复。
- 新包534产品成员与源码/安装同字节，包外解释/CUDA各40worldsteps、147完整事件，三个owner均一次关闭成功。完整原始结果、格式首错与source snapshots保留，新安装归档9文件/3,959,181 bytes；原件保留，不重归档或重跑。
- 实际覆盖R–P串联与非对称R/P分支、两world、真实非零入速/控制变化、完整body来源与力学、晚失败/retry、restore/partial reset。多关节机器人同world、MuJoCo BALL刚柔、摩擦/CCD/持续支撑及新关节D6组合仍未获本批资格。
- 现无项目执行owner；下一批从上述两条提案选独立节点，不恢复已关闭M-B2任务。
