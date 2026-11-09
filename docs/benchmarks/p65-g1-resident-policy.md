# G1真实设备受载反馈：CUDA 2秒留出

**已通过，2026-11-10；新示例源`5bc3c380`，共享物理源`37aa1c40`。** 原29joint/30body、33个原球/胶囊、自由根及执行器a/D，通过`TyxWorldBatch → ResidentContactStepper → ResidentVectorEnv → PolicyRunner`完成4000个接受步、2秒，原九项工程门全部通过，RAW_EXIT0。

这证明完整G1、高body身份、非零摩擦和公共控制接口可以组合。计算边界是**TYX CUDA动力学/FK/宽相，HOST有限几何/速度摩擦，HOST NumPy观测，CPU ONNX策略**。不是全设备数据面、GPU推理、GPU PPO或完整GPU舞蹈。

| 原门/观察 | 实际值 |
|---|---:|
| 物理h / 策略节拍 | .0005s / 50Hz，每40接受子步更新raw action |
| 接受步 / 拒绝步 | 4000 / 0 |
| 最低根高 | .7640795112m |
| 最大倾角 | .1006874068rad |
| 最大关节RMSE | .0805607640rad |
| 最大参考XY误差 | .0557616281m |
| 峰值足力 | 2.7280680429倍体重 |
| 稳定段平均竖向支撑 | .9996313565倍体重 |
| 稳定段双脚承载比例 | 1.0 |
| 非足地面载荷 | 0 |
| 原力矩上限 / 双侧反作用 | 通过 |

## 同源模型与一次真实步

原MJCF、deploy.yaml、dance1_subject2.npz、policy.onnx及外部data均存原SHA。每h从真实checkpoint重算原capped PD；设备KP/KD为0，保留原模型物理armature/damping。原raw action按50Hz保持，不将参考q/v写成状态。地面仍为有限盒、原k50000/d100/μ1；机器人内部排除与原HOST例程一致，不声称自碰撞窗口已测。

33个collider经`ResidentFloatingTreeBinding.from_asset`绑定原局部材料、principal frame、model token与逐body身份；每次接触力/力矩经完整J和M到原自由根/关节候选。载荷时刻t和接受姿态t+h分开，完整接触receipt、M/J、实际驱动力、反力和全部请求保留。只有全部接受后才推进策略/episode时钟。

首次66在注册阶段把应用robot ID误当全部连杆body ID，原无歧义guard拒绝，0物理步。修正仅保留资产逐body/shape身份、给地面不冲突ID；没有删guard或增加cap。失败现场仍在66。69修正后从原初态连续完成原2秒，不重放/跳过失败物理前缀。

## 环境、证据与成本

新隔离Python3.12子venv只读借用原TYX环境的site目录，核实pin99f6df7…、NumPy2.5.4；ONNX Runtime1.22.0与PyYAML6.0.3安装在子环境中，与原HOST策略版本一致。两个原环境及TYX pin均未改。实际模块路径和direct_url保存在65。

`artifacts/p65-connections-20261110-0800-01/69-g1-resident-cuda`保留完整轨迹。原1447436860字节JSONL经独立全解压SHA/bytecount及原件再次未变核对，保存为455342215字节gzip；收据`g1-resident.storage.completed.json`证明same bytes后才去除冗余plain。`g1-resident-outcome.json`记录11个源before/after/current无漂移。未因无plain而重生成证据。

带全诊断/逐条fsync日志的本次wall为06:36:37→07:02:22，约25分45秒。该数包含完整调试数据面，**没有分解内核、HOST几何、同步或写盘成本，不作为吞吐比较**。它明确尚未达到大规模训练所需工作流。

77仅读取原生产者统计：4000次query、100次策略、4000次控制；batch记录40005 submitted kernels、28007同步、12003次浮根state read，contact下载168000000字节。plan和batch展示的contact计数重叠，不能相加。78对一个保存的末步输入做0物理步HOST profile，wrench逐值相同；该次14对采样调用168次segment_minimum，剖析成本集中于静态胶囊–凸体的逐三角形/有理数路径，而非仅Hull构造。一个插桩输入不代表整段wall占比，下一批应扩大代表输入后再选算法/缓存和设备化路线。

## 尚未验收

CUDA20秒/131.48秒长舞蹈、GPU PPO/无HOST往返张量、材料静摩擦/滚动历史、更广地形/自碰撞、受载多世界和吞吐仍开放。原HOST完整131.48秒及公共HOST20秒分别沿各自报告复用。设备短窗日志目前未接到HOST长动作播放器，不能借用它的回放标签。
