# Phymeo 当前续作入口

更新：2026-11-12 三主线后续计划已调整。**直接球电机接触（BALL-MC）和原资产设备边界来源（M1-A）已完成源码与安装资格**；M2只完成调研设计。没有新截止，整个P6.5/Newton功能对齐仍未完成。旧阶段见[历史入口](p65-continuation-history-through-20261112-0035.md)，历史pending/deadline不覆盖本页。

## 当前入口

1. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：当前已有、限域、缺口和优先级。
2. [本批实际结果](p65-parity-motor-asset-20261112.md)：两项实现、源码/安装证据、首错、边界及下一片。
3. [后续三主线计划](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)：当前行动顺序、节点依赖和验收；[上一批计划](../superpowers/plans/2026-11-12-parity-motor-contact-asset.md)已关闭。
4. [M2生产者调研与设计](p65-m2-producer-bank-design.md)：先持久变形状态M2-P，再查询M2-Q；尚未实现。
5. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo编译、CPU/TYX/显式Warp与resident路径。
6. [07:20架构快照](../architecture/phymeo-newton-parity-2026-11-12.html)：历史图，尚未绘入后续直接BALL、资产装配、M1/M1-A；当前以本页和全景为准。
7. [上一批实际结果](p65-parity-m1-assembly-20261112.md)、[07:20记录](p65-parity-until-0720-20261112.md)：可复用历史，不重做已完成开发。

不先通读历史heartbeat或重复旧hash审计；需要具体范围时定位相应报告。

## 最新已交付

- **BALL-MC**：直接BALL JOINT电机与原normal/pyramidal接触共同求解，完整力/运动、双方COM力矩和事务；51新源码节点、6兼容节点；解释/CUDA源码外各112worldsteps。限fixed树、stateless直接球电机、na0；不自动开放FREE+BALL、activation或limits/dry。
- **M1-A**：ordinary/affine/floating/floating-affine四类原资产来源进入真实accepted设备FK→gather/BVH/有限边界查询；保完整原shape/facet/model/graph身份。2HOST+37interpret+37CUDA，旧direct/static每后端8兼容；安装各6worldsteps/3queries/24rows/2stale拒绝。
- **当前统一包**：产品d362799f，516成员source/wheel/installed字节相同；四个新源码外消费者通过。两份源测试参考/API期望修正和一个运行目录错误完整保留，没有放宽产品门。
- **M2仅设计**：确认现有节点/复合布/coupled临时设备数组不能借作长期accepted状态；已提出M2-P双bank owner、提交/lease和M2-Q接入，以及真实形变正负案例。不是新增功能通过。

## 既有资格复用

- BALL-M1的完整native JOINT moment映射、ParticipantScene显式多参与者资产/控制/材料对装配、M1四树direct设备来源已在上一批完成。该批产品2f5f19d6的515成员与六个安装消费者按原证据复用；不倒灌为新包全量回归。
- ROB2b、BALL1/A/C、COL1a/2/3及旧resident载荷范围保持。COL3 HOST当前快照、M1/M1-A真实设备来源、MuJoCo-TYX求解器是不同接口/资格，不能任意相乘。
- N5状态随机reset、N6 D=1..8/GRU、N7 J8/D2/GRU与S128/S256/D4更新/共同状态通过；不等于新学习成功。紧凑HOST镜像移除普通步native构造/copy，不代表零HOST或Newton性能胜出。
- N8 J/R/S共768真实转移、0新Adam更新、297条同次保存审计保持；未证明弱学习原因，不能自动改clip/lr。

## 后续从这里开始

后续统一按[三主线计划](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)推进，优先级和共享集成顺序如下；本轮仅更新计划，尚未启动这些产品节点。

1. **统一场景到设备碰撞（S1–S3）**：ParticipantScene逻辑world/participant到真实多bank的身份、同入步状态及lease映射 → 设备查询桥接 → 公共场景入口和源码外多案例。覆盖多机器人、工具、物体，M1-A单资产资格不代替该组合。
2. **真实变形状态到连续载荷（D1–D5）**：M2-P持久设备状态 → M2-Q真实顶点查询 → 合法接触表示/双向力与运动共同发布 → 持续支撑、摩擦/换面、卸载 → 统一场景和安装验收。M2-P、M2-Q只算各自子节点，不能提前称刚柔响应完成。
3. **机器人组合与GPU规模（R1–R2、G1–G4）**：更新能力矩阵，分别补direct BALL activation、FREE+BALL等组合 → 异构world功能/容量 → 同模型、同输出、同质量门的Newton性能基线 → 根据热点优化。功能优先，超越性能为附加目标。

下一开发批：原三个agent分别负责场景adapter、M2-P、机器人下一组合，可隔离纯编码；父独占共享协议与集成，全部项目执行串行。M2-Q依赖真实producer，响应依赖合法接触表示/物理端口；性能测量等待相关功能格通过。本轮没有发产品编码或执行任务。

训练N8诊断保留为第三线需要时的支撑项，沿原saved数据分析，不成为三条主线前置；其它八方向缺口仍列全景。允许事先声明的工程精度损失，以多案例稳定流程、载荷与运动一致为验收，不恢复精确相触/无限精度的旧障碍。

按多案例→差异模块→受影响验证→源码外消费者推进。同源已绿范围复用，两次针对尝试不缩小问题就保存最小反例、调整假设。

## 保留的失败和缺口

J/R原学习门、G1原46.29秒参考XY门及完整131.48秒动作、R03两项小净加速度门仍未通过。B-point touch-SE法向与曲面见证不相容，候选隔离未合入。M2、通用CCD/持续载荷/摩擦历史、更广机器人组合、批量传感、跨域、密集布/体软体、杆缆与MPM/流体见全景。局部资格不替代这些验收。

## 所有权与证据

父目录C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo，分支codex/p65-parallel-modules，本批本地未push。原三名作者与审阅均已释放；本批项目执行已结束。下一片仍用隔离worktree作者，父是共享接缝唯一owner；所有import/collection/数值/pytest/离线分析/lint/formatter/build/install/profile由唯一串行owner在fresh entrypoint-aware guard后执行。

新证据artifacts/p65-parity-motor-asset-20261112-01，installed-wheel-01共61文件，installed-close.json绑定516产品/四消费者。上一批证据artifacts/p65-parity-m1-assembly-20261112-01与旧07:20证据保留，不重审、不恢复旧session、不全解压。

三个原K1未跟踪文件、用户Torch、当前venv/pins、旧证据Junction和全部首失败均保留。无remote CI、push/mainmerge、发包或P7/P8扩域。本批关闭不等于整个阶段complete。
