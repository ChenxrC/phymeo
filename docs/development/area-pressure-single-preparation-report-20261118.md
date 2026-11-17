# I3g：每步当前网格一次完整验证

2026-11-18，codex/exp；实现 df6289815313331b9dfc1d1358ffed38657241b6。属于[07:10执行计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)。

## 变化

scene 从当前 owner 的实际 FK 捕获原始世界坐标、拓扑、压力和身份，内部声明拥有不可写数组副本。统一 preflight 中各构造一次完整 TetraPressureMesh/RigidPressureMesh，然后进入同一 HOST 或混合求交核心。不跨步缓存世界几何，不因“刚体运动应保拓扑”跳过 stored FP64 结果检查。public intersect/intersect_many 仍从公开字段重构验证，不接受 caller private cache 或 trusted 标志。

raw 声明仅做有界字段、形状和数据复制，不是几何证书。build 前再次检查固定字段；完整正体积、交叠、incidence、full-pressure 边界等都在本步构造执行。HOST 默认路径同步去重。checkpoint/reset/clone、求解方程、物理预算和默认选路均未改变。

## 验证结果

9 新＋63 受影响兼容，共72 passed，0 skip；6文件最终Ruff/format通过。

- 新owned输入HOST＋interpret：5 /16.00s；新CUDA：4 /16.57s。
- 原HOST几何核心＋scene协议：35 /4.35s。
- I3f批量、混合scene/cost、异构scene：28 /87.83s；五个未改的exact边界证书节点未重复执行。

真实 validator spy 调用原函数，逐步统计四个面积请求各1次 pressure/rigid 完整验证。反序world、同桶不同材料、第二步当前几何变化、选择reset换owner、同帧设备观察全部对照HOST。末world有限大平移使实际FP64坐标塌缩时，完整检查在设备launch前拒绝；state/geometry/checkpoint/观察和上次query发布保持。raw原输入修改/clear不影响捕获值，数组不可改；非法重叠/退化/full边界压力以及额外mapping字段均拒绝。

首Ruff有2个import格式问题。02修复命令被外来执行guard阻止，未运行；04的五项在排版前通过，05/06随后只做import/layout，最终CUDA及全部接缝组通过。原始日志保留，未出现功能测试失败。

## 同批整步成本

保持原5world/3bucket/12步；HOST与mixed完整几何、物理和观察逐字节相同。CUDA首步HOST1.426464s/mixed1.405469s；后11步HOST1.367434s/mixed1.203290s。interpret后11步7.037030s/6.822355s。短窗口不足认定稳定加速，不能把不同批次的绝对耗时直接作因果比较，默认仍HOST。

mixed工作量保持：192原候选，156严格证空、36设备复核、0HOST裁剪回退，12launch；上传4608B/下载22272B，workspace9408B。CUDA查询prepare0.615112s/device0.036519s/refine0.029389s/total0.723143s。prepare现包括唯一一次完整网格构造；之前scene外围那次重复构造已去掉，因此不能只按此分项跨运行作减法。

## 下一步

固定5/20/32世界、25步CUDA矩阵，容量64；32是当前公共world上限。单列模型声明、初建和首步/三段暖窗，比较同输入同输出。随后补显式area body-pair载荷读取和同帧历史绑定；reader不是Env，不能将该接通称为训练已完成。

证据：artifacts/area-pressure-single-preparation-20261118-01，包括validation-summary.json、guard/原始命令/RAW_EXIT、batch-scene-basetemp的完整成本JSON。旧已通过组仅因本次改变共享入口而选取相关兼容检查，无旧物理长窗重跑。
