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

有扩展pair时路径为 FK→mechanics→extended manifold→原contacts/common rows→原joint QP→Euler，共6launch；仅原六pair保持5launch。raw预分配，不调用CPU/Warp几何，无分支失败fallback。预算或非有限不许只保部分候选。

## 多案例与验收

- [ ] 独立固定源研究/先登记案例和工程门；原cap-caps两点、box driver规则源码审阅。
- [ ] 通用新三pair编码、static admission、原device及共同发布接线。
- [ ] 平行/斜交/端点cap-caps，cap-box面/端/边/角，box-box面/边/偏心转动；同tree双活动body互易反作用。
- [ ] 有限box上的动capsule/box，真实初态连续0.512s：0..0.096s承载、0.096..0.160s世界向上3mg外载、撤载后重力落回。m=.25/2kg，h=.001/.002s；不得reset状态制造重入。竖直slider检查力与分离，独立自由根案例检查转动/力矩。
- [ ] 同入步native完整manifold/J/R/aref/force/QP/qacc；独立连续CPU轨迹。保T3a工程门，不按输出改变fixture/阈值。
- [ ] 容量恰满/溢出、零力或inactive完整保留、W4、reset/clone真实下一步、晚故障重试、步内禁CPU/Warp/HOST求解。
- [ ] 父首次interpret与CUDA、少量受影响旧六pair、公共ENV和新wheel源码外双backend。
- [ ] 保存证据/源码/安装绑定、实际报告和续作入口更新；执行槽释放。

候选近退化处可有FP32工程误差，但必须诚实保诊断和原门；不能截断native contacts、偷偷改margin或用fixture tag选择算法。测试发现来源/坐标/漏项先修具体问题，避免无期限精确证明。结束物是三类新配对及真实连续重入能力，整个T3/T4/G1另按各自门验收。
