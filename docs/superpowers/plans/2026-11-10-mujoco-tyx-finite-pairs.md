# MuJoCo-TYX T3b：有限 primitive 配对与卸载重入

基线 `275872be` / T3a产品 `a9a39dcb`，2026-11-10。延续[扩展路线](../../development/mujoco-tyx-expansion-plan.md)，本批新增实际可调用的 capsule–capsule、capsule–box、box–box，并完成同一真实轨迹上的承载→卸载分离→重入。摩擦仍是T4；不扩大到mesh/CCD或G1已验。

## 调研后冻结的合同

固定CPU MuJoCo3.3.7及已安装MuJoCo-Warp0.0.2源码。来源为 [collision primitive](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c)、[collision box](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c) 和 driver。保CPU实际manifold与距离语义，不以换用通用最近点算法代替原模型接触规则。

- capsule–capsule：非平行1点；平行按原顺序生成最多2点。Warp0.0.2的parallel TODO/单点不能直接继承。
- capsule–box：最近segment/box特征及原第二点，最多2点。
- box–box：保原SAT、face/edge选择和候选顺序；完整raw候选最多24（16边交+4面角+4对方角），并保driver扩展盒1.01外点剔除、精确位置去重保后项和稳定压缩。CPU face距离的半深度语义照源保留，不套Warp完整depth。
- 新pair slots为 **2/2/24**，box预留全部保守枚举界，不静默截8。研究初期8来自Warp的暂值，在编码/任何执行前已由CPU原源修正。原总界G128/P16/C32保持；两box–box pair超布局明确拒绝。
- 新三pair feature_index为native最终紧凑输出ordinal，保原重复点规则；不宣称它是跨步材料地址。原六pair身份和几何保持。
- condim1、混合margin非负、原source options/可能pair过滤、完整当前J/R/aref、共享M共同QP及原发布事务不变。不同接触不能分成独立物理步。

## 分工和设备接口

仅沿用三个原作者，分别在现有worktree从基线新建分支，旧分支/WIP保留。作者仅源码/测试/文本与Git，所有import、collection、pytest、数值probe、lint、formatter、build/install均父唯一串行owner执行；每次执行fresh guard。无CI/push/pin变更。

源码作者修改所属 `_normal_geometry.py` 并新增 `_normal_extended_geometry.py`，独立测试作者只新增专项和案例文档，研究作者只新增固定源研究/审阅。父独占静态layout、device/runtime、NOTICE、公共ENV及整合文档。

新kernel ABI：`extended_contacts(count,ngeom,npair,nslot,geom_ids,geom_data,pair_ids,pair_law,motion,rotation,geom_world,raw_rows,extended_scratch,status)`。raw为 `[W,C,11]`：present、distance、point3、normal3、tangentseed3。scratch为 `[W,64,3]`，0:32候选，32:64向量/矩阵临时区；具体分区及边界由源码报告和独立审阅确认。原contacts在scratch之后新增raw参数，共用原完整J/接触律。

源码细化（执行前）：box最多24个临时候选直接写该pair预留raw槽，过滤后在同buffer内稳定前向压缩，present只在最终通过时置位；scratch64留给矩阵/向量/顶点/线段。没有扩大资源或丢弃候选，完整索引分区见最终source报告。

有扩展pair时路径为 FK→mechanics→extended manifold→原contacts/common rows→原joint QP→Euler，共6launch；仅原六pair保持5launch。raw预分配，不调用CPU/Warp几何，无分支失败fallback。预算或非有限不许只保部分候选。

## 多案例与验收

- [x] 独立固定源研究/先登记案例和工程门；原cap-caps两点、box driver规则源码审阅。
- [x] 通用新三pair编码、static admission、原device及共同发布接线。
- [x] 平行/斜交/端点cap-caps，cap-box面/端/边/角，box-box面/边/偏心转动；同tree双活动body互易反作用。
- [x] 有限box上的动capsule/box，真实初态连续0.512s：0..0.096s承载、0.096..0.160s世界向上3mg外载、撤载后重力落回。m=.25/2kg，h=.001/.002s；不得reset状态制造重入。竖直slider检查力与分离，独立自由根案例检查转动/力矩。
- [x] 同入步native完整manifold/J/R/aref/force/QP/qacc；独立连续CPU轨迹。保T3a工程门，不按输出改变fixture/阈值。
- [x] 容量恰满/溢出、零力或inactive完整保留、W4、reset/clone真实下一步、晚故障重试、步内禁CPU/Warp/HOST求解。
- [x] 父首次interpret与CUDA、少量受影响旧六pair、公共ENV和新wheel源码外双backend。
- [x] 保存证据/源码/安装绑定、实际报告和续作入口更新；执行槽释放。

候选近退化处可有FP32工程误差，但必须诚实保诊断和原门；不能截断native contacts、偷偷改margin或用fixture tag选择算法。测试发现来源/坐标/漏项先修具体问题，避免无期限精确证明。结束物是三类新配对及真实连续重入能力，整个T3/T4/G1另按各自门验收。

## 09 后的跨精度局部编号裁定

09 首次 yaw-face 对照保留 RAW1：双方完整8点和对应力一致，TYX ordinal到CPU ordinal为[5,1,2,0,4,6,3,7]。原保存的CPU Rzz=1.0000000000000004，TYX纯yaw公式Rzz=1；独立源码给 c2-c1=(s-1)(.05-z-.06)，使CPU选择另一等深参考面。这是数学平局的舍入破局，不是漏点或窄相算法错误。产品几何不改、不加epsilon或fixture分支。

按用户接受工程精度、以通用流程为优先的要求，撤销“跨精度相同ordinal”这一过约束；feature_index明确为各后端各pair本步紧凑局部序号。测试先独立核双方各pair数量和完整不重复0..n-1，再用原位置/法向容差构造确定性增广路完整双射。每个重复点是独立节点，不能去重、截断、按力或J择配。保存原native编号；仅旧helper的局部拷贝适配编号，所有physical字段、rawslot、J/R/aref/force、H双轴、KKT及连续q/v检查原样保留。整个原参数网格不变。

补一个纯合成负控节点，覆盖置换、重复数、坏ordinal/位置/法向、贪心陷阱、Hall失败、力字段不参与选择。现共68节点（解释35、CUDA33）。由于公共对应规则发生变化，最终完整解释重验所有35节点有新依据，不是重跑同源已绿集；真实CUDA33仍首次。原08/09失败作为历史，不改为通过。两个原作者独立纯文本复核已完成，原产品几何保持。

## 连续窗口追加观察（原 .512 s 门不改写为通过）

16 第一条 capsule/.25kg/h.001 从原入步速度完整跑至.512s，每步原native几何/力/动力学/q/v和时间门全部通过，入层、离地、重入均成立；末.02s平均支撑1.61804796N，小于原.7mg=1.71675N，故原末态静支撑门RAW1保留。物理仍处回弹恢复，不能修改力律让它人为更早站稳。

后续所有8条原case保持原初态、pulse、mass、h、参数和逐步工程门，延长同一连续状态至1s（无中途reset）。.512s原支撑门继续计算并保存true/false，不把它改写为通过；1s新增恢复检查沿用同一[.7,1.3]mg/末速度/高度标准。原reentry必须在.160..48s与至少.02s完全离地仍必需。原始前缀有失败，增加真实后续状态后才复验；不是无理由重复已绿全套。

14时间首次失败已保留；15确认当前TYX无f64类型，随后用两FP32 high/comp缓冲在同一设备积分kernel内Kahan累加。发布high-comp和原始前后pair一起核对，q/v/M/J/接触精度不变，不是HOST修补运动。16完整.512s时间门已通过，原clock型别与reset/clone/rollback仍需最终CUDA/相邻验证。

## 最终完成记录

产品/测试 `de623915` 的[实际验收](../../benchmarks/p65-mujoco-tyx-finite-contacts.md)已完成：解释36、CUDA34、相邻33及新wheel两路径通过，28任务尝试/296绑定/67保存JSONL/471包成员核对0差异。最终共有70节点，前文68节点是增加正margin回归前的历史阶段。

上述连续窗口项以“原.512s行为如实记录 + 同轨迹1s恢复”结项：原.512s每backend四条胶囊末态支撑门失败、四条盒子通过，不能改写原门全过；所有八条1s恢复门通过。全部原参数和逐步误差门保持。时钟和受影响旧路径已实际验证；父执行槽及三个原作者均释放。后续T4摩擦/限制、T5 G1另验，不把此批当整体完成。