# 8 环境 GPU 异构机器人与语言条件共享策略

日期：2026-11-14。状态：M1 / M2-A SOURCE_VALIDATED；M0已具静态准入与8模型CUDA短窗资格，完整操作/语义策略仍未验收。
本计划接用户“8 个并行、不同构型、不同物体、抓取/放置/PushT、一个网络优化”及“目标使用语义编码”的要求。
当前为单 GPU 上 8 个独立环境，不是 8 张 GPU。当前无新的项目执行 owner；下一项由父串行分配。

## 1. 目标与验收口径

同一组策略参数接收每个环境自己的真实状态、机器人结构、自然语言指令和可见目标，输出对应有效执行器动作。
共享策略包含可训练状态/结构编码、融合主干、actor/value 输出；冻结预训练语言编码器是共享条件输入组件。
world/task 编号只用于路由、样本分组和日志，禁止作为数值特征、one-hot、查表 embedding 或关键词分支替代语义。
文本 tokenizer 内部的词元编号属于正常文本编码过程，不等同于离散任务标签。
关节类型、拓扑地址与连续几何量有各自物理意义，仍然保留；不把精确坐标序列化成文字来控制。

CPU MuJoCo 可以编译 MJCF/URDF、提供独立对照；正式环境推进使用 MuJoCo-TYX CUDA。
8 环境必须同时拥有独立设备状态、时间、控制、目标和 episode；策略及优化器在 GPU 上运行。
现有 facade 按模型分桶提交，不能把这种实现称为一个异构融合 kernel，或承诺 8 个 kernel 同时运行。
先完成正确的 GPU 批处理闭环，随后独立测量吞吐/读回/启动开销；不以 CPU 步进冒充 GPU。
接口通过、真实接触通过和训练达到成功率是三种不同结果，分别报告。

## 2. 当前代码核验：真正缺失的公共能力

本次仅文本核验，父产品 HEAD c41d63f5；此前 TYX 升级证据独立保留。

| 接缝 | 当前事实 | 必须补的通用功能 |
| --- | --- | --- |
| [floating model](../../../src/phymeo/backends/mujoco_tyx/_floating_model.py) | 唯一 leading FREE；所有其余关节必须为该根后代 | 同一 world 的固定臂与独立 FREE 物体、多树地址/惯量/运动学及真实耦合接触 |
| [fixed tree](../../../src/phymeo/backends/mujoco_tyx/_tree_model.py) | fixed tree nv<=8 | 7R+两指已有9个执行自由度，另有物体6 DOF；覆盖完整新布局，不能裁掉关节 |
| [heterogeneous batch](../../../src/phymeo/backends/mujoco_heterogeneous.py) | 32模型/世界准入、逐桶准备共同发布；只有HOST控制mapping | 公共设备观察、按world有效动作映射、选择性状态重置；不能访问私有桶拼装训练 |
| [task workspace](../../../src/phymeo/backends/mujoco_task_workspace.py) | 固定q/v宽度、二次目标奖励 | 设备机器人/物体/接触/目标特征与独立任务评估接口 |
| [CUDA VecEnv](../../../src/phymeo/training/mujoco_tyx_env.py) | 同构native布局、已有done/timeout/final observation | 异构token布局、语义目标生命周期、目标随reset/checkpoint绑定 |
| [RSL adapter](../../../src/phymeo/training/rsl_mujoco_tyx.py) | 只接当前同构VecEnv | 多模态观察与共享masked action分布；复用PPO，不另写一套优化器 |

原 gallery 禁用了碰撞；原 Panda 抓放和部分 Resident 载荷路径不能替代本案例的 CUDA 操作验收。
共享接触表已有 plane/sphere/capsule/box；首批采用 scalar R/P、pyramidal condim3。
不得通过移除单树检查、固定物体旋转、给物体直接施加策略力、焊接物体、改坐标轨迹来制造通过。

## 3. 八个场景的首版矩阵

以下为待构建的程序机器人，不冒称已导入商业机器人资产。每个物体是独立 FREE 刚体。
R/P 分别是转动/平移关节；双指为两个独立 P 关节。所有 reachability、初态无穿插、力矩裕量需在训练前核验。

| 环境 | 机器人构型 | 操作物体 | 自然语言指令示例 |
| --- | --- | --- | --- |
| grasp-sphere | SCARA RRPR + 双指 | 红球 | 抓起红球，让它离开桌面并保持稳定。 |
| grasp-box | 空间4R + 双指 | 蓝盒 | 把蓝色盒子夹起并稳住。 |
| place-block | 空间5R + 双指 | 绿长盒 | 把绿色长盒放到左侧托盘，然后松开夹爪。 |
| place-capsule | 空间6R + 双指 | 黄胶囊 | 将黄色胶囊状物体放到右侧区域并松手。 |
| place-compound | 空间7R + 双指 | 橙色阶梯复合盒 | 把橙色阶梯块搬到中间托盘，放稳后松开。 |
| pusht-planar | 平面2R + 圆推杆 | 标准T块 | 将T形块推到轮廓内，并对齐它的朝向。 |
| pusht-redundant | 平面3R + 圆推杆 | 宽T块 | 推动宽T形块，让它与旋转后的目标轮廓重合。 |
| pusht-cartesian | PPPR + 圆推杆 | 窄T块 | 将窄T形块推到指定位置和方向。 |

T/阶梯形状由同一物体内多个box geom组成；圆推杆用capsule。保持完整物体六自由度。
每条指令有中文/英文及独立同义改写集。左右、目标颜色、目标区域和任务动作必须变化。
表内映射仅是首个同时运行的快照；不能永久一机器人只对应一条指令。
抓取/放置在可达夹爪构型间交叉分配；每个PushT构型轮换位置/角度/同义表达。
后续引入同场景多个候选物体，以验证语言的目标选择；不预先把正确物体高亮给策略。
每个训练/测试拆分按原始场景与目标组合分组，不能把同一轨迹的改写文本泄漏到测试集。

## 4. 目标与状态如何进入同一个网络

推荐数据流：

自然语言原文 → 共享冻结预训练文本编码器 → 有顺序的语言token + attention mask
真实q/v/接触/物体位姿 → 共享数值编码器 → joint/object/contact tokens + 有效mask
机器人轴/父子关系/尺寸/执行器约束 → 结构编码器 → morphology tokens/拓扑特征
目标区域/轮廓的实际位姿与描述 → target tokens
上述tokens → 共享attention融合主干 → 每个有效执行器的共享动作头 + 共享value头

语言在目标变化或reset时批量编码并缓存，不在每个物理子步调用大模型。
缓存绑定原文、模型/权重revision、tokenizer、最大长度和pooling配置；超长指令明确报错或预定义处理，不静默丢掉后半句。
首版候选为支持中文/英文的预训练文本编码器，例如 multilingual MiniLM；这是工程候选，尚未下载/测试或锁定版本。
优先保留token序列，通过cross-attention将“红色方块”“左侧托盘”“松开”与场景实体及关系关联。
单个句向量可作为基线，但不得把高余弦相似度当作左右、否定或先后关系已经正确理解。
词元特征/池化接口必须与实际预训练模型匹配，不能把未经相应训练的任意hidden state冒称合格语义表征。
先冻结文本编码器，训练状态编码、融合层及actor/value；行为评估有证据后才考虑微调文本层。

第一版状态使用仿真可观测真值，不声称视觉识别：
- joint：归一化q/v、轴、类型、限位、执行器能力和父子拓扑；quaternion与scalar分开处理。
- object：位姿、线/角速度、几何尺寸、颜色/名称描述；实体顺序可置换，并保持地址映射一致。
- target：所有可见候选区域/轮廓的位姿、尺寸、描述。语言决定选择哪个，不能只把答案目标送入输入。
- contact：观测到的接触/夹爪状态；禁止暴露未来成功、教师动作或隐藏阶段标签作为普通actor输入。
- action：公共共享每执行器解码及物理比例映射；只允许控制机器人，不能直接推动目标物体。
- mask：同时作用于动作执行、log_prob、entropy、KL及PPO采样；只把padding动作归零是不够的。

结构化目标另供环境评价器保存 object/target关联及成功阈值。这些是作者给定的真值，不由语言模型运行时自报。
同义表达必须绑定同一个物理谓词；矛盾或指向不存在物体的指令按明确的拒绝/澄清协议处理。
连续目标尺寸、位置和角度仍是数值量；自然语言负责行为与实体/关系，不能替代精密几何。

## 5. 物理成功与语义成功

每任务数值阈值、评估窗口和训练预算须在正式执行前预登记。当前只确定谓词，不臆造已通过数值门。

- 抓取：物体实际离台、手指真实受载、桌面支撑消失，并在规定高度/速度范围保持；手指闭合不算成功。
- 放置：从桌面开手初态出发，历史上发生抬升和搬运；末态位于目标、夹爪释放、桌面承托且稳定；起始已在目标不算成功。
- PushT：物体实际轮廓重合率、平面位置/yaw及低速保持同时合格；推杆位置不是物体成功依据。
- 原始穿插/冲量/能量/越界诊断保留。工程容差明示，不以无限加精度或更改测试来拖延功能交付。

语义验收包含：
1. 同场景同状态，指令换“左/右”“红/蓝”“抓起/放下”，最终行为与对应物理谓词一起变化。
2. 未见同义表达、中英改写，保持相同任务行为；训练句不能仅改编号。
3. 保留“先…再…”顺序和否定目标的独立测试；不能默认为一个embedding自然解决。
4. 置换对象行序、环境编号和padding内容后，语义关联和有效动作不改变。
5. 去掉/打乱语言的对照应显著损害跨目标表现；否则仍可能仅凭构型、初态或目标泄漏猜答案。
6. 留出机器人×任务组合、物体属性/目标位置组合，按组报告；不把简单8项训练记忆称为通用泛化。

## 6. 实施节点、依赖与停止条件

| 节点 | 难度 | 交付与验收 | 依赖 |
| --- | --- | --- | --- |
| M0 场景与语义数据 | 中 | 8份MJCF/初态/指令/目标谓词；native静态编译、可达性和正式TYX准入报告；失败真实保留 | 本计划 |
| M1 共享多树动力学 | 高 | 完整固定树+独立FREE物体的FK/J/M/接触耦合及CUDA结果；不同拓扑、物体/关节顺序、反作用、独立参考与失败事务 | M0最小两例 |
| M2 GPU异构训练接口 | 高 | 公共设备token导出、动作分发、独立reset及checkpoint；W1/W8、padding/排列、晚失败和恢复 | M1 |
| L1 语言/状态编码 | 中 | 冻结真实预训练编码器、token/mask/缓存、结构/对象编码；实际模型来源与语言最小对测试 | 可与M1设计/作者并行 |
| L2 共享策略/PPO接线 | 中高 | 一个主干/actor/value与一个优化器；8行样本对同一参数更新，masked概率，目标/episode恢复一致 | M2、L1 |
| T1 三任务可控性 | 高 | 独立诊断控制器验证真实抓/放/推的可达性和接触；控制器只作物理对照，不冒学习策略 | M1 |
| T2 多任务训练 | 高、结果不预保证 | 固定预算、多seed、按任务均衡采样/损失尺度；逐构型逐任务成功率与最差组、语言反事实/留出组 | L2、T1 |
| V1 展示与安装交付 | 中 | 同一真实GPU轨迹的8宫格视频、指令/目标/成功标注、共享checkpoint、包外复现 | T2实际结果 |

M1与L1可隔离worktree并行编码；所有import/test/数值/编译/训练严格单一串行owner。
M2必须先确定共享public协议，不由示例私自读取桶/复制状态再拼接。
第一次训练用准确状态降低感知变量；后续视觉编码是独立扩展，不阻止本批物理/语义连接。
八个环境是功能最小格，吞吐与样本效率不能按大规模训练承诺；后续同一模型族增加重复world另测。

避免循环：
- M0若正式准入失败，停在记录明确缺失的多树能力，直接进入M1；不对同一不支持场景反复跑训练。
- M1先以最小固定臂+FREE物体验证真实耦合；一个机制至少两种构型和一个未参与设计的holdout，之后才扩8例。
- L1真实编码完成仅算输入资格；不因余弦相似度或随机网络动作不同宣布语义理解。
- T1控制器无法完成时检查接触/可达性；T1通过而T2失败时定位探索、目标编码或信用分配，不能无预算地加训练。
- 每轮训练事先登记步数/种子/预期改进；连续两轮同类失败且无新证据，停止重复，记录对照后只更改一个有依据的因素。
- 不以loss下降、优化器参数变化、总平均reward掩盖某一任务完全失败。

## 7. 调研依据与适用边界

- [Octo 官方项目](https://octo-models.github.io/)：语言/目标图像条件与灵活观测、动作接口；本方案参考其token融合思路，不声称直接继承其训练能力。
- [Octo tokenizers 源码](https://github.com/octo-models/octo/blob/main/octo/model/components/tokenizers.py)：语言与低维观测编码的独立职责。
- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)：跨机器人共享策略与跨本体数据依据，不代表仅8例即可学会泛化。
- [Sentence Transformers 官方预训练模型](https://sbert.net/docs/sentence_transformer/pretrained_models.html)：多语文本编码候选，中文/英文支持不等于机器人空间/时序指令已验。
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)：PushT需操作真实T形物体达到目标轮廓/方向，而非推杆自身到点。

2026-11-14 本次完成的是源码审查、研究与实施计划；未执行新的物理、GPU训练、模型下载或成功率实验。

## 8. 第一批落地（2026-11-14）

M1 已完成源码层有限验收，产品 `0c217310`；[详细报告](../../development/p65-mujoco-tyx-forest-report-20261114.md)。
统一原生多根地址、FK/J/M和FREE/BALL积分，复用接触求解及共同发布。
新增28节点及受影响兼容95节点通过（interpret/CUDA分列于报告），未执行全仓库或新安装包验证。
两项旧组件准入断言经批次前源码对照纠正，最初偏心力矩fixture错误也保留；无产品数值门放宽。

第一批结束时M0只有最小两类机器人与独立物体的通用fixtures；第二批更新见第9节。
下一批先实施M2公共异构GPU观察、有效动作映射/掩码、状态重置与checkpoint；
同步完成M0八个场景的静态检查与物理谓词，再推进T1真实抓/放/推可控性。
L1语言编码、L2共享策略与T2真实学习依然未实现，不从当前接触短窗推导训练成功。

## 9. 第二批落地：M2-A 与 M0 静态/短窗（2026-11-14）

M2-A产品 `2de488b9`，8场景集成最终快照 `7fed0cc4`；
[本批报告](../../development/p65-heterogeneous-manipulation-io-report-20261114.md)和
[公共用法](../../development/p65-heterogeneous-manipulation-io-usage-20261114.md)为当前续作依据。

- 公共异构GPU accepted q/v/act/clock、有效掩码、模型常量、Torch独立D2D输出已实现。
- 全桶设备动作先准备/审计再共同发布，partial state reset和已有checkpoint共同边界已连接。
- M2-A新增18interpret+20CUDA，原兼容80；M0静态/谓词24通过。
- 原八份模型同GPU各32步（256环境步），独立CUDA W1逐字节相同；全部对象出现真实支撑接触。
- 仅源码资格，未跑新安装包；M0可达性/控制任务与M2整体尚未验收。没有实际语义编码或策略训练结果。

## 10. 下一批执行顺序与停止条件

| 节点 | 难度 | 具体产物与验收 |
| --- | --- | --- |
| M2-B1 端点对象/结构特征 | 高 | 从accepted q/v复用现有lowering/FK，在独立工作区导出body世界位姿/速度、FREE local→world角速度和mask。至少多树/非首FREE/BALL/固定体/W1-W8/排列；禁止观察改写physics bank；同端点独立原生对照；第三批源码结果见第11节。 |
| M2-B2 接触特征与相位绑定 | 高 | 复用原设备接触结果，保留真实geom/body、点/法向/双侧力矩、input tick/time、有效mask及overflow。以输入位姿/速度/接触力共同形成SuccessFrame；不能把旧scratch或旧epoch当端点。覆盖无接触/落地/双侧载荷/最后桶失败及reset后的旧记录拒绝。 |
| M2-B3 Env生命周期 | 中高 | 在既有VecEnv/RSL接缝扩展异构观察、masked动作、目标/episode绑定、选择性reset、final observation/terminated/truncated与checkpoint；验证目标与物理状态一起恢复，padding不进入概率/损失。 |
| T1 可控性首批 | 高 | 在八模型矩阵中先选不同构型的抓/放/推各一例，使用真实执行器和同相位载荷证明目标可达与接触有效，再覆盖其他构型；控制器只作物理诊断，失败保留，不当学习结果。 |
| L1 真实语言编码 | 中 | 选定实际中英预训练模型及revision后冻结token/缓存协议，接自然语言与全部可见目标；任务ID不进网络。可与M2-B纯代码/文档独立推进，项目执行仍串行。 |

前三项依赖顺序B1→B2→B3；先形成一条可检查的公开协议，不由应用访问私有bucket。
本批已通过的M2-A用例与M0静态检查按原证据复用，仅改动影响的门和新能力执行。
若B1/B2发现相位或native地址不一致，先用最小双模型反例修正共享机制，不能通过改数据行顺序/复制HOST状态掩盖。
32步落地不足以承诺长窗支撑；T1须预先登记任务窗口和物理判据，不能以单步力或平均reward替代完整操作。

## 11. 第三批 B1 与下一批 B2（2026-11-14）

M2-B1 公共 `include_bodies=True` 已完成14项新源码验证（解释6、CUDA8）；
[详细报告](../../development/p65-body-observation-report-20261114.md)记录受影响兼容结果与证据。
端点 body origin 世界位姿/线角速度及 body→world 矩阵与 accepted q/v/act/clock 共同发布；
默认关闭。跨构型、排列、W1/W8、失败、partial reset/restore、Torch及八场景两步已覆盖。
B1 不提供接触 token，也不授予抓放/PushT 成功或训练资格。

B2 按以下依赖执行，不直接读取旧 scratch，也不将 HOST receipt 再上传伪装为设备观察：

1. **B2a 叶子 accepted-entry bank（高）**：在 step prepare 内 D2D 保存本次真实入步 body 状态与接触结果，
   output/input tick、补偿时钟、有效性与溢出随全桶物理提交一起接受。单独的 `entry_valid` 区分 reset 后无记录和成功空接触。
2. **B2b 各接触路由归一（高）**：覆盖 T0/T1 固定 rows、T3 normal_rows、T4 contact_map/contact_wrench；
   保留原 geom/body/feature/condim、frame/点/距离/实际 wrench。T4 contact index 不等于 slot index，
   limit/dry 约束行不能当接触。世界力、双侧反作用与力矩参考点须明示。
3. **B2c 公共观察与完整边界（高）**：workspace 只 scatter 已接受记录；失败保留上一 entry，
   selected reset 只清所选行，checkpoint/restore 保存/恢复完整 entry 及相位，而非从端点重建过去载荷。
   controls revision 变化不伪造新的物理 input tick。
4. **B2d 接成功判据（中）**：SuccessFrame 的位姿、速度、力来自同一 entry，按 input tick 消费一次。
   reset 后不生成虚构零接触帧；首个成功 step 提供 tick0 入步记录。B3 再连接 goal/episode/history 与物理 checkpoint。

测试矩阵须包含：无记录/有效空集合/非激活检测/实际受载、非共线摩擦与多个接触点、limits/dry混合、
非首FREE/变构型/行序与padding、W1/W8、最后桶失败/重试、reset/restore/下一步重放、
Torch副本、分配预算/关闭、禁止HOST状态或receipt重建，以及端点与入步故意不同的相位反例。
先验两种路由再扩完整既定矩阵；若无法保留原始双侧载荷或边界一致性，停在具体缺失的共享机制，不绕开公共协议做私有训练脚本。


## 12. 第四批：持久记录与路由归一完成（2026-11-14）

产品 `994b0842`；[报告](../../development/p65-entry-record-report-20261114.md)。
本批完成第11节B2a、B2b及B2c中的reset/clone/checkpoint存储部分：
真实入步native q/v/act、补偿时钟、身份/接触点/frame/完整六维载荷在设备D2D生成，
随同原物理bank共同接受；失败保留旧记录，空接触与无历史记录分开。
body入步观察尚需从保存q/v复用FK，不借用B1端点数据。
19个新节点（7 interpret/7 CUDA/5 HOST codec）及95项受影响兼容通过；
原首个力偶fixture失败和诊断修正保留，未降物理门。仅源码资格。

下一批按下列顺序完成剩余连接，不回头重写已验存储：

1. **B2c公共GPU entry观察（高）**：workspace输出保存的input tick/clock/valid/count/mask及contacts；
   从保存q/v计算同相位body pose/velocity；frame转world wrench，明确双侧符号和body-origin力矩。
   覆盖W1/W8、native入步oracle、非首FREE、condim4/6、inactive/empty、controls-only revision和相位故意不同反例。
2. **B2c观察边界（中高）**：最后桶/异常数据拒绝后旧frame保持、reset/restore重新绑定owner、Torch独立D2D副本与alias/close。
   不从HOST receipt造GPU特征；原记录和物理状态查询前后字节保持。
3. **B2d成功帧（中）**：SuccessFrame只消费valid的同一input tick一次，无记录不冒空接触；
   位置/速度/载荷共同来自entry，端点另列。再接B3目标/episode/history。
4. **B3→T1→L1/L2**：先Env生命周期及真实操作可控性，再接真实语义编码和共享策略，维持第6节停止条件。

本批执行已结束；未启动训练、模型下载、安装或新的持续任务。原所有预算/精度门保持。


## 13. 第五批：公共接触观察与成功帧（2026-11-14）

本批完成第12节B2c公共观察、B2d同入步SuccessFrame与读取边界；
实际节点/后端/首失败/源码版本以[本批报告](../../development/p65-contact-observation-report-20261114.md)为准。
公共入口为 `include_contacts=True`，原默认端点接口保持；不把接触点力偶丢掉或混成COM力矩。
reset的无记录、成功空接触与inactive检测分别表示，controls-only revision不产生新样本。
成功帧读取器消费input tick一次，public per-world boundary区分同字节tick0重置与重复读取。
cursor恢复需核对实际已恢复公共物理frame；不允许靠改episode或旧空cursor绕过reset。

### 下一批 B3 的实施与验收节点

| 节点 | 难度 | 具体工作与可判定结果 |
| --- | --- | --- |
| B3a 异构Env适配层 | 中高 | 复用已有VecEnv transition/error语义和公共heterogeneous workspace，按每world模型常量映射动作，保留native地址/有效mask。现有MuJoCoTyxVectorEnv依赖单模型task workspace与统一nu/nv，不能直接冒充异构支持。应用不访问私有bucket。 |
| B3b 目标与终止生命周期 | 中高 | 每world绑定Scene、Goal、episode、history与horizon；物理step后消费同一entry，完整保存input tick/action与endpoint关系。先产final observation/旧episode终止事件，再按选中world reset。terminated/truncated分开，inactive padding不影响动作或评价。 |
| B3c 组合恢复与失败 | 高 | 在明确update/Env边界同时保存physics、reader cursor、goal/history、episode长度/回报及reset计数；先完整校验来源与形状，再提交。物理已接受而后续观察失败须标needs_recovery，不伪称回滚或生成成功learner transition。 |
| B3d 多案例源码验收 | 中高 | 原8场景GPU、W1/W8/排列、不同horizon和目标、只reset部分世界、同字节tick0、错目标/错episode、晚world失败、恢复后下一step相同、Torch独立副本；至少两个构型实际接触。所有执行仍串行。 |

B3完成接口测试不等于任务可控性或学习成功。其后T1先以不同构型抓/放/PushT三个独立诊断控制器验证实际操作，
再覆盖8场景与未参与开发的组合。L1另冻真实预训练语义模型/来源与最小对；不能拿task整数替代语言输入。
L2/T2只在前置物理和语义门通过后启动固定预算训练，继续遵循第6节的停止条件。

## 14. 第六批：B3 Env 生命周期完成（2026-11-14）

产品 `135752e4`；[本批报告](../../development/p65-manipulation-env-report-20261114.md)、[公共用法](../../development/p65-manipulation-env-usage-20261114.md)。
B3a-d已取得本批源码范围资格：每world masked native动作、目标/episode/history、独立horizon/reset、final observation、terminated/truncated与组合checkpoint。
24个checkpoint边界节点与10个CUDA节点通过，首次CUDA测试输入错误及修复保留。原八场景W8/W1及排列逐字节对照，32步真实支撑成立；不称完整操作或学习成功。
旧求解器、物理门与原场景未修改；没有新wheel/训练/语义模型下载。root串行槽和两个作者均已释放。

### 后续顺序：先验证动作能完成任务，再训练

| 节点 | 难度 | 具体下一步 | 成功与停止条件 |
| --- | --- | --- | --- |
| T1a 控制可达性与通用适配 | 高 | 从原8场景选不同构型的抓、放、PushT各一例；读实际关节/执行器/限位/接触模型，确定控制输入、可达路线、增益、窗口及同相位物理记录。复用公开Env，不访问私有bucket。 | 先登记路线与窗口，再执行。找出不支持的执行器、不可达目标或接触机制时先修通用缺口，不用修改物体轨迹、焊接或只改reward绕过。 |
| T1b 首批真实操作 | 高 | 诊断控制器只通过真实机器人动作接触物体；保存同一连续episode的入层/抓持/搬运/释放或推行过程、载荷和失败。 | 原Scene/Goal成功判据成立，力与运动相位一致，目标切换与错误目标负控有效。不是端点摆拍，也不是学习结果。 |
| T1c 八构型与holdout | 中高 | 首批机制通过后扩原8例及未参与设计的组合，保持统一接口与逐任务结果。 | 逐例报告，不能以平均reward掩盖某类全部失败。若同类失败两轮无新证据，停止重复并定位一个具体缺口。 |
| L1 语言输入 | 中 | 独立确定实际中英预训练模型及revision、token/缓存协议；自然语言与全部可见目标进入策略。 | 正确来源与最小语义对照通过；无task整数或选中目标真值泄漏。不以embedding相似度代替任务成功。 |
| L2/T2 共享策略与预算训练 | 高 | 在T1与L1门通过后接一个共享网络、形态/状态编码、masked动作分布和本批Env；固定种子、步数及评估窗口。 | 完整8任务逐例成功率/holdout/语言对照；同时记录失败与吞吐。loss下降或权重变化不单独算成功。 |

下一批从T1a开始。调研、文本和隔离编码可以并行；所有项目执行仍由一个owner串行。
新Env只支持一动作一步，success评价当前是HOST消费者；若训练需要decimation/GPU评价，另以明确需求和对照扩展，不能默认为已支持。
本批既过节点按原证据复用，不重跑来堆测试数；不继承旧截止或启动周期任务。
## 15. 第七批：T1a控制诊断及视频完成，操作成功仍未取得（2026-11-14）

[本批报告](../../development/p65-manipulation-control-report-20261114.md)记录产品5a281455/最终渲染3bed01fd。
12项新HOST通过；原8世界2200步CUDA正常返回，保存审计与最终视频完成。
结果0/8：world0实际夹起后滑落、world7推行约17mm未到目标；其余6条单seed点IK路线未收敛并保持原参考。
这不是完整T1通过，更不是共享网络训练结果；L1/L2/T2尚未开始。

### 下一批先解决可证伪的通用控制缺口

| 节点 | 难度 | 实施与验证 | 停止条件 |
| --- | --- | --- | --- |
| T1a-2 接触姿态候选 | 高 | 由物体表面、pad/pusher几何与机器人FK生成有界接触姿态候选；检查pad角点离台、双侧重叠、原限位和机器人碰撞。保留中心点失败对照，不能仅追object中心。先登记world1后侧候选与world3原高度侧夹，作为有限候选验证，不声称已可行。 | 几何不合法候选不进入物理；有限集失败只标未找到，不证明全局无解。不得移动物体或改原Goal。 |
| T1a-3 破除单seed分支限制 | 中高 | 对5R/6R/7R同一合法目标，预登记zero与两条镜像腕部seed，原每次IK100步预算不变；记录误差、姿态、pad净距和碰撞。用独立FK重投影及不变性负控验证通用接口。 | 原失败源中腕部保持0只是局部分支证据；不无限随机重启或扩大预算。 |
| T1b-1 持握诊断与闭环 | 高 | 首先消费本次world0保存轨迹，核同入步切向力/法向力、负载、相对滑移与pad下缘；必要时建立同模型同输入的native对照再定位机制。之后一次预登记反馈控制对照，保留本次滑落基线。 | 原稳定抓持判据成立才通过；双指有力或曾抬升均不足。没有新定位证据不扫描增益/摩擦/solver参数。 |
| T1b-2 可达推行路线 | 高 | RR/RRR以真实T边界外偏pusher半径选可达接触feature，验证短推进的入步力及关于COM的力矩/yaw符号；PPPR也核偏心接触。先固定有限候选，再验证原目标的位姿/IoU。 | 不用COM误当body origin；短推正确只过传力诊断，不自动过完整PushT。 |
| T1c/L1/L2 | 后续 | 首抓/放/推机制通过后扩原8场景/holdout，再接真实文本编码与共享网络优化。 | 逐任务成功率和负控；不以视频播放、reward均值或loss下降替代。 |

world1原指定低位中心TCP受几何长度限制、RR/RRR原最终TCP点超圆形工作域的文本分析，只约束那些点，不否定可抓姿态/可推边界。
5R–7R原零seed的Z/X轴腕部列与平面残差可能形成不变分支；候选测试负责验证，不能把猜测写成故障结论。
root保持唯一串行执行；原作者可按互不重叠文件纯编码。原模型、目标与全部成功阈值保留。


## 16. 第八批：姿态筛查与首个短窗抓持成功（2026-11-14）

产品02f44795；[本批报告](../../development/p65-manipulation-pose-feedback-report-20261114.md)、[固定试验登记](../../development/p65-manipulation-pose-plan-20261114.md)。
新增pose/grip/contact共88个HOST通过；原7快照native比较最大接触世界力差6.79e-7N。
固定11姿态中2个端点候选可继续设计，4个镜像seed虽点IK收敛却有pad穿台，均未投入物理。
world0唯一新CUDA对照2038步后通过原抓球Goal，成功entry2.037s、连续低速保持.2s、离台48.38mm；2038步因果反馈/实际指令保存审计通过。
原模型/solver/摩擦/质量/PD与Goal保持。该结果仍有缓慢滑移，限短窗单例，不覆盖长期持握、其他7任务、语义或网络训练。

| 已取得的通用能力 | 边界 |
| --- | --- |
| audit_native_contact_poses | 1..3 seed、BOX pad角点/PLANE距离、sphere/capsule/box投影及输入绑定；仅静态筛查，不是全SAT或路径规划 |
| measure_finger_slip | 公共同入步body-origin速度、接触点与载荷；双方顺序/多接触/无记录/无接触/有效mask；HOST消费者 |
| BoundedSlipClosure | 双指载荷门、滑移死区、限速与原关节界内增量；不修改物理状态、无目标物体额外力 |
| 可选CUDA诊断反馈 | 原motor真实驱动、一次连续episode和原Goal评价；最终Env checkpoint不包含诊断控制器恢复状态 |

### 后续三个具体节点

1. **T1a-4，难：方向受限的接触姿态与全路径检查。** 在原机器人/物体/Goal下登记夹爪闭合方向、完整pad几何与碰撞禁区；使用解析关节Jacobian和固定预算有界候选。加入本批TCP到位但pad穿台负控，保留5R有限集未解。先验证world1后侧和world3高位候选的接近/闭合/抬升完整路径，再进入新物理窗；不靠增加点seed掩盖方向约束缺失。
2. **T1b-2，难：原PushT目标的可达边界推行。** 以真实T边缘和pusher半径生成有限接触点/路线；短推先验同入步力矩和yaw符号，再按原Goal验证位置/角度/IoU。原PPPR约17mm失败保留，不更改目标或success门。
3. **T1b-3/T1c，中高：完整放置与多构型迁移。** 原目标区内实际释放、重新支撑和低速保持；抓/放/推机制通过后扩8场景和holdout，并评估有限更长持握窗与控制器checkpoint需要。之后才接L1真实文本编码、L2共享网络优化。

本批root与原两作者完成后释放，不自动启动以上新增运行；后续按用户继续指令实施。
没有新截止、周期任务、安装、push或CI；已绿同源组与旧保存审计继续复用。


## 17. 第九批：完整姿态与采样路径已接入实际 CUDA 操作（2026-11-14）

产品 `316df74f`，见[报告](../../development/p65-manipulation-route-report-20261114.md)和[执行前登记](../../development/p65-manipulation-route-plan-20261114.md)。
新增通用pose IK / C1采样路径两模块；46个HOST首次通过。world1/3两姿态与nominal/upper参考共四条路径各2402样本通过；两世界顺序各2400CUDA步/2.4s，全部4800步反馈/PD与482保存帧同次审计通过。
world1盒最高离台26.98mm，原grasp失败；world3胶囊最高38.66mm、真实lifted，但后续转动/下滑且原place失败。该短窗没有执行目标区释放，不把它称为完整放置。原world0短窗成功复用，本批未增加完整操作成功任务数。

下一批按以下顺序推进：

| 节点 | 难度 | 开发/验证及停止条件 |
| --- | --- | --- |
| T1b-4 载荷与姿态支撑诊断 | 中高 | 消费这两条保存轨迹的接触点、世界载荷、COM力矩、pad覆盖和相对转动，区分几何/控制/求解误差；必要native比较预登记同状态同输出，禁整窗无依据重跑。形成通用姿态筛选或闭环接口，不能只用两指法向力大小作为抓稳门。 |
| T1b-5 一次有依据的持握改进 | 高 | 依据诊断固定有限候选/控制变化，保持原物理和Goal；实际抬升与低线角速度保持均成立才通过。无新定位证据不扫描增益/摩擦或继续同路重试。 |
| T1b-2 PushT可达边界 | 高 | 与抓持不交叉的纯编码可并行；先原T边界/pusher几何与真实传力/COM力矩，再目标位姿/IoU。全部执行仍单owner串行。 |
| T1b-3/T1c，L1/L2 | 后续 | 稳定持握后接完整放置和八构型/holdout，再实际文本编码及共享策略。采样clear不是CCD或动态物体路径证书。 |

本批两作者及父执行结束后释放；不继承旧deadline，不开始新训练/安装/push或旧同源套。

## 18. 第十批：载荷/COM矩与有限 PushT 边界已接入（2026-11-14）

产品64a6b206、保存坏输入补足16f6c9ea、PushT最终b1a521f6；[本批报告](../../development/p65-manipulation-wrench-push-report-20261114.md)、[执行前登记](../../development/p65-grasp-wrench-plan-20261114.md)。

新增82个HOST通过；原world1/3共4798条public entry载荷/COM矩消费通过（各末条缺后续完整entry，明确未补造）；14个固定同状态native比较中13个最大pair力差≤2.655e-6N，world1第1900步最大0.0181N单独保留未归因。测试中临时MjData接触视图生命周期引发的访问异常已修复，只复制独立oracle数值，54项PushT完整绿，原失败保留。

原RR/RRR/PPPR每场景24个边界候选、22个几何usable；原单seed预算下两端点都收敛分别10/10/14。查询保完整真实BOX并集外边、pusher半径、整段进入距离、全部拒绝尝试及native COM名义力矩；不是整个机器人路径、真实载荷或PushT成功。本批无新CUDA物理/视频/训练，原抓球一例短窗成功复用。

### 下一批开发与验证

| 顺序 | 节点 | 开发/验证闭环 | 有界停止条件 |
| --- | --- | --- | --- |
| 1 | T1b-2b PushT接近与传力 | 消费这72个已保存候选，按原Goal方向和几何力臂登记有限排序；接完整关节路径检查，再用真实CUDA motor执行一次短推，保存同entry所有接触/COM矩、物体位移/yaw；确认传力后延长同轨迹推进原目标。 | 端点可达不能跳过整机路径；全候选失败保留具体碰撞/限位原因。不得修改物体坐标/原Goal或扫参数。 |
| 2 | T1b-5 有限pad覆盖和姿态 | 原6R胶囊优先，以本批边缘退化/相对转动/COM力矩记录约束候选，再一次固定控制变化。4R盒的自由度/覆盖约束保留，不只重复TCP点IK。 | 无新覆盖或姿态依据不重跑旧路；原低线角速度保持才算抓稳，双指有力或曾抬升不算。 |
| 3 | T1b-3/T1c与L1/L2 | 真实搬运、目标区释放和再支撑；八构型/holdout后接文本语义编码、单共享网络训练。 | 原各任务指标/负控分别报告；视频、候选数、loss不能冒操作成功。 |

native 0.0181N偏差保留作独立有界同状态诊断，不以未证明归因修改solver，也不阻塞独立PushT功能连接。所有执行继续root单队列；原两作者已释放，无新deadline、自动化、安装、push或CI。


## 19. 第十一批：PushT 接近与实际传力完成（2026-11-15）

[报告](../../development/p65-manipulation-push-route-report-20261115.md)、[预登记](../../development/p65-push-route-plan-20261114.md)。
580b9d3a 接完整关节参考和CURRENT entry采集；5cfb0afc接兼容保存consumer。54新排序HOST＋20保存consumer（新6、兼容14）通过。
原RR/RRR/PPPR各一次2000步CUDA，全部6000入步/603保存帧审计通过；沿推向实际位移5.926/12.338/17.231mm。原完整PushT目标0/3，语义与共享网络未开始。

下一节点为 **T1b-2c（高）：目标位姿闭环、有限换接触点与同轨迹继续**。
RR的负yaw违背正yaw目标，PPPR纯平移目标也偏转，故不简单延长固定法向；以实际位姿、剩余目标和同入步接触载荷决定继续推、退出或有界绕行再接触。
先纯作者通用状态/输入/候选/阶段记录及多构型负控，再登记完整原Goal的有限运行。每段保持物理连续，禁止重置物体、调摩擦/质量或放宽原Goal。
原两作者可按互不交叉文件纯编码；共享seam及所有验证仍root单队列。未找到路线或目标恶化要输出具体原因，不无限重试。
T1b-5的pad覆盖持握、T1b-3完整放置、T1c/holdout及L1/L2语义共享训练继续保留。
本批槽已释放，无新期限/安装/push/CI；不重跑本批三窗。
