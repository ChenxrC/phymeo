# I3i：显式面积接触载荷读取（2026-11-18）

实现提交 `7ca002ff`。新增 **8 个 interpret/CUDA 节点和4个原生短兼容节点全部通过**；没有数值失败。当前完成的是HOST任务评估reader，尚未接Env或声称抓取/训练成功。

## 接口与语义

`ManipulationSuccessReader(workspace, scenes, area_roles=...)` 接收每个面积模型的完整有序 `AreaPressureRoleBinding(identity, object_side, role)`。角色为finger0/finger1/support/other；对象A/B侧与真实Scene body匹配，命名手指或支撑不能标other而漏载。身份/角色复制后持有，拒绝可变数组枚举或歧义body。没有面积声明的原调用保留原binding和fingerprint路径。

面积法向力已经是N，读取时**不再乘dA**。世界力定义作用于A，对象在B时反向。完整面积目录、材料、样本/来源顺序、历史输入tick/time/configuration、声明mask和数值数组绑定同一帧；原生contact又含已声明area body pair时拒绝，避免重复施力。read与restore完整验证后才提交cursor。controls-only保留旧输入历史，reset要求新episode，restore必须配对原cursor。

full FREE对象、XML/timestep、原native接触语义和checkpoint六列结构保持。面积模式有独立版本binding；旧cursor不静默迁移。

## 真实短步结果

新固定协议Scene使用undamped/unlimited的3个真实P关节和motor、独立FREE球、独立支撑；h=.0005、最多3步。A/B交换两侧、同压力面四分、native共4world。面积表面在本批显式提供，用于独立检验reader；自动几何将在I3j接Env，不能混为本批已完成。

CUDA首步同次保存的载荷：

| 模型 | 左 / 右指法向力 N | 对象向上支撑 N |
| --- | --- | --- |
| area-a | .199750364 / .199750356 | .200365335 |
| area-b | .199750379 / .199750349 | .200365335 |
| area-split | .199750341 / .199750334 | .200365283 |
| native | .100465335 / .100465320 | .945560455 |

每pair实际面积1e-4m²；四分后每片面积变为1/4，载荷总量按原FP32量级门一致。每个值与真实typed step receipt独立汇总对照。native使用另一接触律，未要求和面积模型数值相同。支撑数字也不是稳态承重或抓持成功证明。

负控覆盖缺角色/缺area workspace、跨model重复数字ID的材料错配、末world坏目录/计数/面积/非有限载荷、同tick修改、精确时间错位、native重复bodypair；失败后原cursor不动。生命周期覆盖controls-only、三次真实步的成对恢复重放、选择性reset。

## 验证与范围

证据：`artifacts/area-pressure-success-reader-20261118-01`。
- 06原native短兼容：4passed/18.90s，2个未变长轨迹节点未重跑。
- 10新interpret：4passed/25.83s；11新CUDA：4passed/26.98s，均RAW_EXIT0。
- 13/14：3文件最终Ruff/format-check均0。
- 01 source与07 test首次Ruff均只有import排序；对应02/08修复、03/09排版。04/05/12是外来执行guard，未运行产品命令。
- 原作者SHA `9d267affab6cfb1f0d4837f7fe757601a7293d8b485ffed938e0220ac336d2b1`；父增加同次accepted-loads.json和排版，未改物理参数/门槛。3个最终source/test执行前后SHA相同。
- 源码审阅在执行前修正可变enum别名、命名角色漏载和历史时间近似比较；不是测试失败或物理调参。

下一按[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)做I3j：动作→当前状态自动接触面→共同求解→任务观察→独立episode/reset/checkpoint。面积摩擦、actor新特征、PPO、柔性资产和一般CCD均不在本批通过范围。
