# MuJoCo-TYX T3b：有限配对、连续卸载与重入

2026-11-10，产品/测试提交 `de623915da4815232e0578f0d00face6d75bf4ff`，本机限域验收完成。基线是[T3a共同法向接触](p65-mujoco-tyx-coupled-normal.md)。本批补齐 capsule–capsule、capsule–box、box–box，进入同一套当前姿态动力学、共同法向求解和状态发布。解释36个节点联合覆盖、真实CUDA34、相邻33项，以及新wheel源码外两后端均通过。

原0.512秒“末态已恢复静支撑”的假设没有全部成立：每个后端四条胶囊轨迹均未达原支撑下界，四条盒子轨迹通过。原失败保留。所有八条轨迹保持原初态、质量、步长、外载脉冲及误差门，不reset连续推进至1秒后，全部达到原同一支撑/速度/高度标准。因此接受的是**三类配对及连续分离、重入、1秒恢复能力**，不是原0.512秒静支撑门全部通过。

## 新能力与范围

固定CPU MuJoCo3.3.7的primitive、box、driver规则；MuJoCo-Warp0.0.2用于源码交叉研究，未作为缺失规则的替代。详[固定源研究](../development/mujoco-tyx-t3b-research.md)、[实施计划](../superpowers/plans/2026-11-10-mujoco-tyx-finite-pairs.md)和[案例库](../development/mujoco-tyx-t3b-cases.md)。

| 新配对 | 实际覆盖 | 每对静态预留 |
|---|---|---:|
| capsule–capsule | 平行两端、斜交、端点重复、双活动支路与近共轴转动 | 2 |
| capsule–box | 侧面、端、边、角、偏心自由根及双点 | 2 |
| box–box | 面、转动面、边、偏心转动、正margin分离 | 24 |

box的24槽是完整保守候选界：16边交、4参考面角、4对方面角；按固定原生driver过滤和精确重复规则稳定压缩，不静默截8。总上限仍为geom128 / possible-pair16 / static-feature32，两box pair需要48槽时明确拒绝。运行检测/约束容量分别验证恰满与溢出。

与原六类合计九类配对。支持域仍为condim1、混合margin非负；固定根1–8 scalar DOF，或一个free root加0–29 scalar joints，nq≤36/nv≤35。不支持同世界多个独立free root、摩擦、joint limits、mesh/CCD、equality/tendon或完整G1原生支撑。

有新配对时设备阶段为FK → mechanics → extended manifold → common contact rows → coupled normal QP → Euler，共6个launch；仅原六类保持5个。所有行共享当前完整M和双方J，在一个QP中求力，再积分一次；没有逐接触独立施力、CPU/Warp/HOST步内几何或动力学回退。原 `normal-dual-active-set-v1` 名称、预算和KKT门不变，不声称复现原生Newton/CG迭代。

`feature_index` 是各pair当前manifold的紧凑局部序号，`capabilities.extended_feature_identity` 明示 `current-manifold-local-ordinal`。它不是跨步材料地址，也不是FP32与FP64之间的固定同号身份。native比较仅按原点/法向容差确定完整双射，保留重复点数量，随后核原J/R/aref、力、共同矩阵双轴和KKT；不按力选对应。

## 实际验证

主环境：Python3.12.14、MuJoCo3.3.7、TYX0.1.0a0、NumPy2.5.4、pytest9.1.1。产品执行全部由父单一串行owner进行，每条命令fresh entrypoint guard；三个原作者仅隔离编码/文本复核并已释放。

| 任务 | 原始结果 | 覆盖 |
|---|---|---|
| 16-interpret-compensated-clock | 16 passed、1 failed、34 deselected，26.94s | 对应规则、12个静态几何例、3条动态例通过；首条原.512s静支撑门失败 |
| 17-interpret-continuation | 20 passed、50 deselected，576.57s | 原八条轨迹继续至1s；其余容量、失败事务、公共ENV |
| 19 / 20 / 21 | exit0 | 精确11个Python文件Ruff、format-check，以及格式前后原始AST全同 |
| 22-cuda | 34 passed、36 deselected，87.34s | 最终源码完整真实CUDA专项及ENV |
| 23-adjacent-clock | 33 passed，7.08s | 原六配对24项及T0/T1/T2受影响积分路径9项 |
| 24-build / 28-install | exit0 | 新wheel与源码外无依赖安装 |
| 26 / 27 wheel consumers | 各exit0 | 安装包解释/CUDA公共有限接触环境、reset/clone下一步 |

interpret最终36节点按16中的16项和17中的20项联合覆盖，格式后的AST等价另由21绑定，不重跑同源已绿项。CUDA34节点为最终完整组。没有skip；deselection来自后端分组及17明确复用的已绿16节点。70总节点包含两个无后端合成/准入检查。

每后端八条连续窗口为两种形状 × 0.25/2kg × h=0.001/0.002s，共6000个窗口步。初始竖直速度-0.02m/s保留，0.096–0.160s施加向上3mg外载，然后撤载。每条验证持续承载、至少0.02s完全分离、0.160–0.48s内重入，以及1s恢复。另有24/24/32步的偏心胶囊、偏心盒子、双活动支路。

每步在同一TYX入步状态做独立native manifold/J/M/R/aref/force/qacc核对；另有从相同初态独立推进的CPU轨迹，检查完整q/v和时间，不回灌CPU状态。原模型、质量、步长、脉冲、逐步误差门未调宽。完整状态/控制逐步保存；详细receipt按预定首/中/末、脉冲边界、首次分离/重入保存，不冒称每步完整receipt全量归档。

公共ENV使用真实free root加两关节、有限box接触、稀疏worlds4/31。控制每3步刷新并读取已接受接触力；连续12步后selected reset和clone下一步核对。每个wheel消费者完成主runtime26个world-step、clone1个world-step，并断言所有Phymeo模块均来自本次安装目录。W4、容量拒绝、inactive行、实际晚回读失败重试及步内禁fallback也通过。

## 首次失败与实际修复

1. **08端点预期错误。** 固定native源码的有序尝试产生两个同位置接触，原测试预期一个。按独立原源推导改为两个，保留重复点和原输入；08 RAW1保留。
2. **09跨精度局部编号过约束。** yaw face两侧均有完整8点、对应力相同，FP64/FP32对等深参考面有不同破局。改用完整几何双射，分别核本地0..n-1序号；加入置换、重复数量、贪心陷阱、Hall失败、坏点/法向/序号与力不能选配的负控。原几何参数不改。
3. **12冗余SAT轴错误替换。** 倾斜box实际出现4点对native2点、深度/力翻倍。精确共轴谓词证明cross轴等价于已有face时，在非负penetration域禁止其替换face，仍核全部15轴分离。负penetration保原偏置语义；无epsilon或fixture分支，并加正margin分离回归。
4. **13近共轴Gram相减消去。** 双活动胶囊第1步真实det大于原mjMINVAL，但FP32差平方相减归零。改用等价的cross积平方及对应三重积解，保持阈值、顺序夹紧与真平行规则。
5. **14/15设备累计时间。** 普通FP32累计在431步超过原2e-6s门；TYX没有f64类型。三条积分kernel改为设备内两FP32的Kahan high/comp累加，原始前后pair、入步绑定和递推均验证；发布high-comp。运动、M/J/力精度不变，无HOST运动修补。reset、clone、rollback和旧积分路径均已通过。仍要求高位每步严格增加，不声称任意长时间均支持。
6. **16恢复时间假设。** 首条胶囊.512s末.02s均力1.61804796N，低于原0.7mg=1.71675N，且native同样处于恢复期。保留原RAW1及所有八条.512s原支撑门；延长同一轨迹到1s，未改物理参数或阈值。
7. **静态与安装脚本。** 02/03首次import排序/格式失败修复后通过；25安装配置引用不存在的解释器，pip未启动。其RAW_EXIT=0是旧LASTEXITCODE，已明确标记INVALID_LAUNCH，不计成功；28使用已存在环境完成安装。原日志和纠正说明均保留。

## 证据、交付与下一步

[原始证据](../../artifacts/p65-mujoco-tyx-t3b-20261110-01)和[最终文件核对](../../artifacts/p65-mujoco-tyx-t3b-20261110-01/final-audit.json)记录28次任务尝试（含25无效启动）、296项当前绑定、67个保存JSONL及30个固定native案例身份。16个最终窗口身份为每后端八条；原.512s支撑门8条false、8条true均保留。文件核对只读取既有证据，没有重新求解。

新wheel SHA256：`3fbe521f8e0c3e7d57a5f81aaa4cad852df18a14b9ecdc0e958adb1c9fced89c`。471个产品/资源成员与源码、安装目录逐字节一致，Apache LICENSE/NOTICE/AUTHORS随包；0 mismatch，fresh执行匹配0。审计SHA256：`911d2ca2cf3ecbe3dee0ac35112e9a1c380304e145c39682f4b99d5eefad4967`。产品已提交，证据为ignored本地产物；没有push/远程CI/发包/依赖pin改动，三个原K1 WIP未动。

下一批是[扩展路线T4](../development/mujoco-tyx-expansion-plan.md)：先冻结native摩擦cone/约束求解语义与案例矩阵，补condim3、joint limits及G1资产实际需要的frictionloss，再做T5真实G1持续支撑与策略组合。当前condim1完成不等于摩擦、人形支撑或全MuJoCo移植；HOST调度、NumPy观察和有界dense solve也不构成全设备训练或性能超越Newton的结论。
