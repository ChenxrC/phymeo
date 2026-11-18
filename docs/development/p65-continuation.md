# Phymeo 当前续作入口

更新：2026-11-18。当前 codex/exp；B1/B2核心提交2d2ac75a，B3测试提交28fef171，查看器与最终报告接续其后。dev仍76a240d8，本轮不push。先查实时Git/WIP和agents，再按本页前进；不恢复旧heartbeat、07:10已结束任务或旧运行会话。

## 本轮进展

两个滑动关节已在同一 CUDA 世界联合求解面积接触与一条原生干摩擦行。三个工况只让近端接触平板，从设备直接观测到远端世界加速度随导轨位置和摩擦上限改变。首步、连续第二步、恢复重演与错误约束行拒绝均已核验。[两节滑块的接触载荷传递与实测表](area-friction-coupled-guide-report-20261118.md)

同一 CUDA 世界已联合求解面积摩擦与一条原生滑动关节干摩擦约束。1 kg 滑块在 2 N、0.25 N 两种导轨摩擦上限下的步末速度、面积推力和导轨反力均符合预先固定的解析基准。分离、容量、恢复与原有路径回归已验证。[首步实测值与支持范围](area-friction-guide-report-20261118.md)

CUDA 面积摩擦支持每个模型 1 至 8 个速度自由度，新增可动平板与 6/7/8 自由度混合批次已通过验证。原中心跌落案例的全部 150 个步末已按有限板覆盖范围测量，最大压入量为 1.96249 mm（32 ms）。[新增测试、深度数据与下一项混合约束的解析基准](area-friction-wide-report-20261118.md)。

## 此前完成的三批

用户要求的三批B1 live摩擦、B2双侧/观察/恢复/故障、B3连续案例与整步成本已完成。父为唯一项目执行owner；两个原agent已完成独立author/read并交接，没有新的agent任务。

主报告：[面积摩擦B1/B2/B3](area-friction-live-report-20261118.md)。计划：[三批计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。[真实轨迹查看器](assets/area-friction-live-20261118.html)默认当前最终冲击轨迹；蓝法向/橙摩擦/绿色实际裁剪面，历史失败单列。旧[07:10架构图](assets/area-pressure-architecture-20261118.html)是历史快照，当前增量见[架构说明](area-pressure-connected-architecture-20261118.md)。

## 最终验证

- run23：29 passed /25.80s。当前Scene自动几何→resident M/J→共同Fn/Ft→原生q/clock尾段→状态/力原子发布；v2真实混合观察、双方响应、分桶/材料身份、检查点/selected reset/重演及晚失败。
- run16：双接触区40步，1 passed /6.20s。之后只排版，AST等价，不重跑。
- run25：4 passed /62.22s。冲击与偏心支撑各150步、同桶滑动/释放各20步、1/4/32world各3步完整数值输出一致。事件按实际记录，未观察到的明确保留NOT_OBSERVED。
- run26：原接口兼容五文件20 passed /16.34s。这54项只代表相关新增和兼容范围，不是全仓库通过。
- run30/31：36个Python文件Ruff/format通过。run21证明34个raw AST同、2个仅导入排序；run38补scale/viewer最后排版，scale仅导入排序、viewer raw AST同。后续viewer schema修复以run40和浏览器验证为准。
- run40：读取10个完整payload，生成14个world/尝试视图、979286B HTML，无物理重跑。1280×1000/390×844无横向溢出，播放/暂停/拖动/放大、失败前缀与成功切换已验；临时检查服务已关闭。

原始evidence：artifacts/area-friction-b1-live-20261118-01。所有raw失败、源证明、完整gzip、成本和查看器截图保留。run11/12速度不一致及run14矩不一致导致采用共享原生尾段直接消费v*；原残差/实际速度/力/矩误差门未放宽。run39为查看器schema读取错误，run40修正；guard拦截没有RAW_EXIT，不能算测试失败或通过。

## 成本与边界

1/4/32world后两步的完整观察吞吐约6.85/9.03/9.76 world-step/s；第一步和初始化另列。仅两后续步、保留已有JIT缓存，不能叫稳定benchmark/训练效率或快于Newton。主要时间位于Scene.step，尚未分解到几何、求解或生产核验，下一轮先定位。所有成本比较保同模型同输出。

可选final_normal_smooth_v1，CUDA 单独面积接触，每桶1..32world、每模型1..8DOF、完整点≤128（三点三角积分≤126）、pair≤16、body/geom≤128。保完整dA/双方J/物理M与A=M+hD；Fn/Ft共同解，原生坐标和时钟只推进一次。

旧normal-only/native-only可在别桶同场景存在。混合面积摩擦现支持一至两个无黏性阻尼的 SLIDE，一条干摩擦行可位于任一关节。多条原生约束、BALL/activation、限位、equality/tendon 仍待接入。旧资产v1的friction=none保持原义，旧reader拒绝新v2。g>0仍是模型域，p>0/g=0不加epsilon、不删面。

几何仍为HOST或设备拓扑加HOST精确复核。不是全设备碰撞/面积CCD/29关节摩擦机器人/任意有限边缘/精确静摩擦。FFT/Kamino和通用自动算法选择保持独立；新面积摩擦策略特征/PPO、一般变形体散射尚未接通。

## 后续优先级

查看器在本批 run04 和 run05 通过 Ruff 与格式复检，历史生成与浏览器记录仍见上一批 run40。

1. 在双滑块上接入单侧行程限位，验证未激活、受载和卸载时的面积力、限位反力及另一关节响应，再扩至多接触机器人。
2. 对Scene.step做分段成本定位，再选择几何/装配/求解/核验优化；同模型同输出才评价收益。
3. 新v2面积摩擦观察进入真实训练输入；新磁盘law封套、一般有限边缘和变形体扩展各自独立验收。

新 wheel 和 dev 合并仍待单独验证。

## 操作约束

保留三个原未跟踪凸patch文件及原agents WIP。所有项目import/pytest/数值/static/build严格串行；明确唯一owner和fresh entrypoint-aware guard，外来实际执行自然结束，不kill。每条命令显式workdir，父写/Git/执行require_escalated。不以sandbox不可见推断仓库坏。

不重跑同源已绿组/同hash审计，不重开旧会话；失败候选不发布，物理提交后观察失败只重试观察。原Temp证据可能为Junction，只用canonical archive，不遍历删除或全解压。本轮不改dev/main、push、发包、pin、周期任务或清理环境。

长期历史按需读[历史入口](p65-continuation-history-through-20261118-i3o.md)。不以模块数/测试数给P6.5完成比例。
