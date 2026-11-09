# 机器人—环境—策略连接：11月10日到08:00结果

本批优先交付真实G1完整舞蹈和可核实的公共连接。**官方全131.48秒动作在Phymeo HOST上完成，没有跌倒或拒步；原两项工程门仍失败。** 新公共入口另完成20秒，全部原工程门通过。两者分别记录，不能互换验收标签。

[已实现架构](../architecture/robot-environment-connections-2026-11-10.md) · [离线连接图](../architecture/robot-environment-connections-2026-11-10.html) · [原批次计划](../superpowers/plans/2026-11-10-0800-robot-environment-connections.md)

## G1真实运动与工程边界

原29关节、30body、33个原球/胶囊、自由根、质量/惯量、执行器a/D和官方ONNX反馈策略保持。策略50Hz、物理h=.0005s，控制目标生成限幅力矩，没有将参考姿态或根轨迹写入物理状态。

| 实际窗口 | 公共工具入口20秒 | 原物理入口全动作131.48秒 |
|---|---:|---:|
| 接受物理步 | 40000 | 262960 |
| 最低根高/m | .6665476371 | .4392791326 |
| 最大倾角/rad | .3814782653 | .6300226791 |
| 最大关节RMSE/rad | .1965080147 | .2917020632 |
| 最大参考XY误差/m | .3589852642 | **1.2749072320，原门1m未过** |
| 峰值足力/体重 | 6.5153636616 | **14.9124216782，原门8未过** |
| 稳定段平均竖向支撑/体重 | 1.0004113175 | 1.0000086550 |
| 跌倒/物理拒步 | 无 | 无 |
| 原工程门/原始退出码 | 全部通过 / 0 | 两门失败 / 1 |

全动作独立入口物理源为`0cd6dec`；新工具入口使用已验证的`SceneBatchBackend → ResidentVectorEnv → PolicyRunner`。全动作不是通过新env重跑131秒，也不是设备G1舞蹈；新env的20秒标签不覆盖后111.48秒。

首次长窗在27.0495秒因胶囊轴端进入有限盒体内部而停止。保留原请求与失败；通用修复采用有限盒减线段的凸体特征、严格歧义拒绝和真实材料见证，没有G1/时间特判。17项HOST相邻/独立留出及原保存请求一步通过后，再从原初态完成全动作。原首失败不删除，也不重新解释为跌倒。

## 本批新增连接与实际验证

| 能力 | 实际证据 | 限域 |
|---|---|---|
| 显式受载env stepper、完整world接受map、部分/未知发布错误 | HOST公共四文件72项；interpret/CUDA公共三文件各12项，原始退出0 | 包含真实受载与拒步；部分protocol/parser测试不是物理运行 |
| 同一受载plan跨episode、final observation、选择reset | 上述真实设备留出和HOST原场景留出 | 不复制solver，不伪称已接受物理可回滚 |
| 无状态公共PolicyRunner | 动作保持、逐世界整数时钟、真实原命令、独立reset | 不是RNN/完整物理checkpoint协议 |
| 完整tree body载荷包 | 30body实际RED后修复旧17body门；完整HOST桥53项通过 | 既有32joint+root对应33body，原shape/site/candidate上限不变 |
| 29/32joint最后body设备接触 | interpret/CUDA各2项，通过完整M/J、仅最后body施加2N和偏心-.5Nm、根运动与双侧反作用 | 解析同轴树，不是设备G1舞蹈；μ=0 |
| 速度摩擦混合桥 | 新旧HOST63项，interpret/CUDA各12项；两个真实运动case各8步 | HOST几何/力律→设备fullM/J；不声称静摩擦历史或G1设备窗口 |
| W256公共连接 | interpret/CUDA各256world×8步=2048world-steps | 128平动球+128真实转动关节；无接触几何，NumPy策略边界，无吞吐结论 |
| W256受载注册边界 | 512个有效shape在原256单模型门被拒，0物理步 | 是准确缺口负控，不是受载W256通过 |
| 可选RSL-RL | 10项可选适配测试真实执行/通过，未skip | HOST/TensorDict复制；不包含GPU互操作 |
| 两真实HOST物理任务PPO | 固定seed/各100迭代；原20%学习门均通过 | 两个单关节任务，不是G1训练或受载抓取训练 |
| 源码外wheel | 两真实受载scene16步/32world-steps，独立力—运动递推与选择reset通过 | 安装到独立target、cwd在仓库外Temp，模块来源断言指向wheel安装 |
| 实际长动作回放 | 262960条原接受记录→6575可视帧，离线HTML浏览/首末/播放/受力显示检查通过 | 保存的实际q/v与载荷，非参考NPZ；姿态与入步力时刻分别注明 |

旧17body限制同时存在于采样器和载荷包构造，修复父`c1292647`统一为33。新高body测试先以独立几何/质量/J解析期望核载荷，再观察原设备响应；没有用源码常数作为通过oracle。

## 学习、依赖与回放

实际可选环境为Torch2.15.0 CPU、TensorDict0.15.5、RSL-RL5.5.2、NumPy2.4.6。`pip freeze --all`及`pip check`已保存，后者无破损依赖。核心依赖仍只有NumPy；Torch是独立可选extra，TYX pin及原运行环境未变。

| 固定训练任务 | 未训练cost | 训练后cost | 下降 | 原生checkpoint重载 |
|---|---:|---:|---:|---|
| prismatic-reach | .4059548378 | .0620736182 | 84.7092% | 新建原世界后192步观测、动作、cost、实际q/v及终态mask复现 |
| revolute-velocity | .7493789196 | .0074049751 | 99.0119% | 同上 |

PPO使用原RSL-RL rollout/update/save/load，两任务均为16个HOST环境、seed20261110、100×24步。没有重挑seed、加训练预算或改学习门。G1仍使用原公开预训练ONNX；这里没有训练G1、物理反传或零拷贝GPU训练的结果。

实际回放位于`artifacts/p65-connections-20261110-0800-01/g1-full-actual-replay.html`；PNG同名。全动作工程门未全过在viewer中明确显示。每40个物理步抽取已保存接受姿态并保留最后一帧；原完整JSONL以逐字节可还原gzip保留，未经删字段或重序列化。

## 原始证据与失败保留

本批根目录：`artifacts/p65-connections-20261110-0800-01`。

- `01-g1-full-dance`保留原27.0495秒失败；`07-g1-full-overlap`保留完整长窗、源前后绑定和两个原工程门失败；`g1-full-outcome.json`归纳真实结果。
- `08*–10*`为公共连接首静态失败、修复、qualified AST、HOST及两设备结果；`11-g1-common-connection`为独立公共20秒；`12-g1-full-export`与`replay-browser-qa.json`为真实回放。
- `14–19`为隔离依赖、适配测试及两任务PPO；`ppo-outcome.json`、`41-training-freeze`、`42-training-pip-check`记录实际版本与学习结果。
- `23–29`为W256作者格式修复、原shape边界与两backend物理结果；`30–40`为旧body门RED、HOST53、纯格式AST同一和两backend高body正控。
- `20b/21b/22b`为加入33body修复后的wheel构建/安装/源码外消费者。`22`首次消费者漏必需robot_state而失败；`22b`仅补显式原初态后通过，无产品修复。后续若再改产品，需新wheel绑定，不沿用旧包通过标签。
- `44–53`为速度摩擦真实RED、原静态/AST及新旧桥63、两backend各12；`54–59`为产品37aa1c40的新wheel、448成员0mismatch、源码外HOST32world-steps及interpret/CUDA各8摩擦受载接受步。后者为当前产品包，不借此前旧包标签。

未重跑既有同源全套、旧审计或旧长窗。未动原三个K1未提交文件、未推main或发布包、未改TYX pin、未启动远程CI。

## 剩余连接与后续顺序

1. **G1设备组合与站点力核**：高bodyμ0与非零μ两运动机制已分别验证；[N7混合桥](p65-resident-friction-connections.md)不借HOST G1标签。N8先检查完整G1设备短窗，再迁移站点力核；固定材料点静摩擦/滚动历史仍为另一能力。
2. **受载W256**：先做完整world分页、FP64合载、每storage一次solve和一次共同发布；不能循环调用旧plan两次或只把256改大。
3. **公开策略张量数据面**：基于固定TYX的公开Torch互操作，明确stream/owner/epoch/reset/拒步及终态副本，再减少HOST往返；CPU PPO不是该结果。
4. **全动作工程质量**：14.91倍峰值力和1.275m参考误差保持开放。在通用离散、支撑/滑移诊断及控制协议上改进，不改原质量/门槛、不给G1专用姿态修补。
5. **整体引擎**：跨独立participant设备共同求解、密集柔性接触及各物理族仍沿全景分别补齐。这一批接通机器人工具链，不代表整个P6.5或Newton功能对齐完成。

后续依据：[设备策略/摩擦研究](../development/p65-device-policy-contact-research.md)、[驻留接触规模研究](../development/p65-resident-contact-scale-research.md)。两篇是源码与官方资料调研，不是新增实现或冻结合同。
