# 2026-11-11 公共连接与分页设备接触验收

状态：本批公共接口与安装包已验；G1完整动作未通过。产品/测试冻结提交 `e3d15707c16fe346f856bd26961be856c1d7ef06`。本页记录实际完成范围，不把接口、查询或保存审计等同完整Newton功能。

## 本批连通的功能

- MuJoCo-TYX公开物理checkpoint/restore：固定根与单自由根、接受状态、控制、时钟和CAS；恢复后实际下一步一致。
- Resident world-complete受载分页：每个世界完整归一页，各页载荷共同装配，每storage一次动力学、全批共同发布；支持默认HOST patch与显式device patch。
- MuJoCo-TYX设备task/action：接受bank生成观测/奖励/终止；三类控制通道、producer stream、完整字节审计与CAS；29执行器动作宽度和生命周期均已验证。
- 限定设备S/C/OBB/H凸体patch进入原材料点、完整J/M、双方力和COM力矩。设备几何之后仍有HOST转换、装配和诊断；不是全GPU训练。
- G1当前IMU-site观测语义已修正并通过短窗；全131.48秒跟踪仍失败，见[诊断](../development/g1-drift-diagnosis-20261111.md)。

两种运行模式是TYX interpret与真实CUDA；两条物理实现是resident与MuJoCo-TYX。它们的形状、约束与规模资格分别报告，不互相外推。

## B3分页设备接触的实际验证

| 项目 | interpret | CUDA |
|---|---:|---:|
| 完整新B3矩阵 | run100：29 passed / 263.85s | run101：29 passed / 213.07s |
| 受影响HOST兼容 | run102：2 passed | run103：2 passed |
| 同次固定保存文件 | run104：28完整payload，PASS | run105：28完整payload，PASS |
| 新wheel公共分页接口 | run121：5次成功物理调用 | run122：5次成功物理调用 |

各功能命令RAW_EXIT0，无跳过；29个节点含5个有意不保存的非法选项检查。保存审计重核原输入、完整query/raw/source、转换差分、连续步、世界/页身份和实际计数，没有重算producer或物理。对象进程身份与11种故障瞬间的发布不变性依赖实际pytest断言；保存文件包含成功前缀和重试结果，不冒称有独立故障瞬间快照。

W129/W256主例连续三步；W256为256个隔离世界，每世界一个2kg slider和一个有限静态support，两页各128world/256shape，各512个BB站点。非零入步速度、驱动与外载保留。另有四类树混排、单页B2与HOST同输入等价、世界/页/划分顺序变化、partial reset、每world容量拒步、末页错误和同输入重试。

这不是256台G1训练。硬界仍为8页、每页原shape/vertex门、每非空页设备站点1..4096；原每world受力站点上限独立。最后一页转换失败不能留下前页物理成功。无页/空页与有shape但零allowed-pair页分别覆盖；后者分配并计费32字节物理metadata，清空站点后发布逻辑零行。原未初始化尾项缺陷已修复，毒化缓存负控实际通过。

调用见[公开设备接触接口](../development/resident-device-contact.md)。所有query属于入步epoch/bank历史，不把它当作下一步的当前查询。

## 安装包资格

run110构建、111安装到源码外新目录、112核对**483个产品文件在源码、wheel、安装目录逐字节相同**，均RAW0。wheel SHA256：

`3b238492a3ca1f3f8b9d76f1f3639e0bb1cebdd93ac11036c7b33b6ff490890a`

安装后只走公共API和现有依赖，不将父/src加入搜索路径：

| 公共能力 | interpret / CUDA实际命令 | 验证内容 |
|---|---|---|
| A1恢复+C1分页 | 125 / 114 | fixed/free各两世界恢复并继续；W129两页三步 |
| 设备task | 115 / 116 | 实际物理后task消费、重置/恢复失效、来源和资源 |
| 设备action | 117 / 118 | 真TYX消费者→三控制通道→四次物理推进；独立HOST控制路线同输入 |
| B2单页device patch | 119 / 120 | 三个连续受载步、完整来源/力学收据 |
| B3分页device patch | 121 / 122 | 三世界两页，连续三步、selected reset、末页容量错误及同输入重试，共5成功调用/15world-steps |

十组全部RAW0并保存成功记录。没有重跑旧全套或再次构建同源wheel。

113预检误把解释器venv根当源码，物理0case；v2仅放行精确绑定的解释器根。124随后正确拒绝旧venv editable .pth自动加入的主/src，物理仍0case。最终统一源码外stdlib bootstrap在任何phymeo导入前只移除这一已知entry，保存before/removed/after；所有实际phymeo模块来源仍须在新安装目录。未改venv、.pth、产品或物理门。两次首错保留。

父第一次汇总把task/action的JSONL误作单JSON，非终止PowerShell错误导致无效状态抽取；原命令/输出不变。正式汇总为 `final-wheel-public-api-acceptance-v2.json`，分别读取真实终记录并保RAW0、script/source before-after绑定。旧汇总明确被取代，不作为证据。

## 其它已验增量及性能边界

A1/C1/C2/C2b/B1/B2的原focused执行按[本夜进度](../development/p65-parity-2026-11-11-progress.md)复用。B2单页interpret/CUDA各67节点及每后端47个保存文件已完成；B3没有把这些全部重新运行。公开安装检查覆盖最终组合。

action-pack复用活跃物理Runtime编译缓存，固定W1/8/32小案例的稳态提交从约13–15ms降至0.65/0.80/0.98ms；复制、同步与HOST审计保留。这是局部接口测量，不是整体训练吞吐或Newton速度优势。原C1 W256 benchmark使用HOST patch，不能引用为B3设备patch性能。

## 证据与后继

完整本机原始证据根：`artifacts/p65-parity-20261111-01`。B3为100–105、静态107–108；安装为110–122及125，113/124首错保留；483成员核对复用112，不再重哈希全库。正式wheel汇总SHA256 `296c6e3bb1d0363da60253b6220bc814a81dc70c781920534e699ffddf8e1fd4`。

当前全景见[八方向表](../development/phymeo-newton-panorama-2026-11-11.md)。下一开发主线为[GPU训练N1→N2→N3](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md)：独立CUDA Torch资格/真实策略消费、两任务实际PPO更新、更新边界组合恢复。目前三项仅为已调研计划。G1保留失败，采取有限保存诊断；不以第三次全窗试参替代通用模块补齐。
