# Phymeo 当前续作入口

更新：2026-11-18。当前codex/exp；本轮未push，dev仍76a240d8。先核实时Git/WIP与agents，再读本页和最新报告，不恢复旧历史中的运行任务。

## 当前任务与状态

用户要求本批截止北京时间07:10。本轮I3f–I3x已按各自有限范围完成，06:50结束执行与预览收尾；fresh guard无项目执行进程，原两个agent均已交接释放。当前无执行owner、无运行会话、无已派下一批任务。后续继续时重新取得唯一串行槽；原执行全过程保持import、测试、数值、静态、build/install串行。

最后有界[I3x原生积分检查](area-pressure-native-friction-bridge-report-20261118.md)首次1pass/1.73s：4个固定历史入口、两次原native核，actualv上原Fn/Ft与共同方程通过原工程门。没有重跑接触解或旧轨迹，也未改产品源码。下一批优先复用原积分器并保留逐步actualv核验，不能把该诊断叫live Scene。执行收据见[本轮计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)及各报告。

## 最短阅读顺序

1. [当前连接架构](area-pressure-connected-architecture-20261118.md)及[离线架构图](assets/area-pressure-architecture-20261118.html)。
2. [I3s FREE真实连续窗口与视频](area-pressure-free-body-support-report-20261118.md)，了解已经能运行的法向链。
3. [I3u 128点摩擦backend](area-pressure-backend-friction-report-20261118.md)、[I3v实际历史入步](area-pressure-entry-friction-report-20261118.md)、[I3w设备装配](area-pressure-resident-friction-pack-report-20261118.md)。
4. [下一批B1–B3计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)与[源码live交接](area-friction-live-integration-handoff-20261118.md)。后者包含实际文件、native坐标备选与普通边缘g=0反例。

## 能力边界

当前模型q/v→自动有限面→真实dA/p/g→MuJoCo-TYX共同法向解→统一状态/载荷→持久观察与任务Env已接通。生产面积律仍无摩擦；几何为HOST或设备拓扑+HOST精确复核，不称全GPU精确碰撞。

连续导轨承载/卸载、三可动物体双区释放已验；另有三种FREE初态各150步，实际旋转、独立M/J/原生坐标审计和同轨迹视频。I3s原oracle平面绑定首错保留，修正只审原450world-step，无物理重跑，不把该pytest初次失败改写为全绿。

摩擦backend已6DOF/128完整点/32world资格；I3v历史适配和I3w设备pack均已验。源码外新wheel也通过对应组。尚未将新Fn/Ft接入真实Scene积分、law-aware载荷发布与恢复；不能把旧type7、v1的F=nFn或friction=none悄改含义。普通滑出有g=0准入边界，不能删面追绿。

固定5/20/32world同输出后暖步成本减少约9.7–11.3%，仅属于该法向案例。没有普遍快于Newton、任意复杂机器人或摩擦生产提速结论。面积策略特征/PPO、一般软软/薄布折叠、面积CCD、通用自动选型仍未接通；FFT与Kamino保持独立参考用途。

## 下一步与复用规则

下一批B1直接完成一条可选公开Scene摩擦路径：声明/身份+真实resident同入步+一次原生积分+新载荷记录同时交付。B2补真双侧、多桶、reader/codec/restore/故障；B3连续多案例与完整步成本，再扩DOF/一般约束、GPU规模和训练。

物理提交前失败重试step；提交后观察失败只重试观察或依协议restore，不能重复推进。各报告原raw、完整输入/输出、失败与source绑定直接复用；不重跑同hash已绿组、不重新开已关闭session，不以更多相似证明替代新功能。

原44,556字节入口保存在[历史入口](p65-continuation-history-through-20261118-i3o.md)，只作历史索引。旧全景和当前模块报告共同判断范围，不给P6.5总体完成百分比。

## 操作约束

父树拥有所有旧共享接缝。保留三个原未跟踪文件及原agents WIP；命令显式workdir，必要require_escalated。每次项目执行fresh entrypoint-aware guard；外来执行自然结束，不kill。仅沿用原agent，不重复派已完成任务。

本轮不改dev/main、不push、不发包、不改pin、不建周期任务。旧Temp证据入口可能是Junction；使用canonical archive，不跟随删除、不补副本或全解压。工具开源整理仍是后续优化。
