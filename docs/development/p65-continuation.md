# Phymeo 当前续作入口

更新：2026-11-18。当前分支 codex/exp；本轮改动尚未 push，dev 保持 76a240d8。先核实时 Git/WIP 与 agents，再以本页和最新报告继续，不从旧历史中的“正在运行”恢复任务。

## 当前任务

用户要求本批持续到北京时间 07:10。执行入口：[本轮计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)。

I3f–I3r 已按各自范围完成。I3s 正在隔离 author：一个实际 FREE 模型、三个初态、150 步自动有限面/共同法向响应，独立原生 M/J/bias 和积分核验；尚无执行结论。原 scene/training 两 agent 只纯编码，root 是唯一项目执行 owner。全部 import、测试、数值、静态和构建串行。

## 先读这些

1. [当前连接架构与差距](area-pressure-connected-architecture-20261118.md)及[离线架构图](assets/area-pressure-architecture-20261118.html)。
2. [450 步真实支撑窗口与视频](area-pressure-heterogeneous-support-report-20261118.md)、[三可动物体双区释放](area-pressure-three-body-release-report-20261118.md)。
3. [独立 CUDA 摩擦参考](area-pressure-tyx-coupled-friction-report-20261118.md)、[生产接缝图](area-pressure-friction-integration-map-20261118.md)。
4. [后续生产摩擦 F1–F6 计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。

## 当前能力及边界

- 当前模型 q/v → FK → 按模型选择 body-local 资产 → 自动有限面 → 真实 dA/p/g → MuJoCo-TYX 共同法向解 → 整批状态/载荷提交 → 持久设备观察已接通。
- 默认几何 HOST；可选设备拓扑＋HOST 精确复核。批量筛选/一次网格准备在固定5/20/32世界、相同输出下取得约9.7–11.3%整步减少，不泛化为全GPU或快于Newton。
- 任务 Env 已能驱动此链；native+area多body力/力矩一次读回、资产保存/加载、选择reset/成对恢复均有各自证据。新面积特征尚未完成共享策略/PPO接线。
- 生产面积律仍为无摩擦、冻结入步几何的弹性BE法向响应。摩擦HOST/CUDA只是独立共享DOF参考，未改type7、v1观察或friction=none资产。
- 已跑连续承载/卸载和三体释放使用明确导轨；FREE长窗仍待I3s。没有宣称真实自由抓持、一般软软/薄布折叠、面积CCD、任意机器人或通用算法自动选择完成。
- FFT非局部半空间与Kamino动力学各有独立参考用途。双峰实验已显示独立峰相加和force-only表的局限，不能以格点独立弹簧替代材料耦合。

## 复用证据，不重复执行

| 连接 | 报告 |
| --- | --- |
| 批量候选 / 当前网格一次准备 | [I3f](area-pressure-batched-candidates-report-20261118.md)、[I3g](area-pressure-single-preparation-report-20261118.md) |
| 固定多规模比较 | [I3h](area-pressure-batch-scale-report-20261118.md) |
| 显式area角色 / 任务Env | [I3i](area-pressure-success-reader-report-20261118.md)、[I3j](area-pressure-manipulation-env-report-20261118.md) |
| 多body wrench / 一次读回 | [I3k](area-pressure-body-wrench-report-20261118.md)、[I3m](area-pressure-wrench-batch-report-20261118.md) |
| 完整资产文件 | [I3n](area-pressure-assets-report-20261118.md) |
| 摩擦选型 / HOST共同解 / CUDA参考 | [I3p](area-pressure-friction-model-report-20261118.md)、[I3q](area-pressure-coupled-friction-report-20261118.md)、[I3r](area-pressure-tyx-coupled-friction-report-20261118.md) |

各报告指向原完整raw日志、实际输入/输出、失败和保存清单。原生旧路径只在确实受影响时回归；不重跑同hash绿组、重建已结束session或重审同档案。原44,556字节入口已原样移至[历史入口](p65-continuation-history-through-20261118-i3o.md)；其中“当前/运行中”均为历史，不是新指令。

## 操作约束

父树是旧共享接缝唯一owner。所有命令显式workdir；需要时require_escalated，sandbox不可见不等于仓库坏。保留三个既有未跟踪文件及原agent WIP，不reset覆盖，不新建周期任务。执行前fresh entrypoint-aware guard，外来项目执行等待自然结束，不kill。

本轮不改dev/main、不push、不发包、不改依赖pin。旧证据Temp入口可能是Junction；只用canonical archive，不跟随删除或补回副本，不把压缩JSON叫缺失。测试/工具的开源整理仍为后续优化。

## 下一步判断

先完成I3s一次固定窗口及同轨迹图；06:35收尾、07:10报告。之后按F1–F5接入真实场景摩擦，再扩混合约束与规模。每一步须增加可观察能力，失败定位到对应层；不用追加相似测试、调案例参数或重复几何证明消耗时间。局部接触进展不代表整个P6.5或Newton功能对齐完成。
