# 三主线 Batch B：公共场景、变形查询、浮根球关节

状态：AUTHORING / TESTS_PENDING。基线 `c45a0174`（产品 `d2a18bee`），2026-11-12，用户授权继续下一批，无新截止。沿[三主线路线](2026-11-12-scene-deformable-scale-roadmap.md)，复用[Batch A资格](../../development/p65-three-track-batch-a-20261112.md)，不重跑同源已绿组。

## 并行交付与所有权

| 节点 | 原作者及隔离树 | 本批开发范围 |
| --- | --- | --- |
| S3 公共场景接线 | mtyx_t2_tests / p65-h1-actuator | 原 participant_scene_compiler 的公开 query 准备/生命周期，必要新增纯装配 helper；独立源码测试、第二类工具/自由物体安装场景及报告。底层 S1/S2 复用 |
| D2/M2-Q 真变形查询 | mtyx_t2_source / p65-h1-support | 新 deformable source adapter、必要纯设备 gather 核、源码/事务测试、真实变形安装消费者。复用原有限距离/BVH数学，不新写求解器 |
| R2 FREE+BALL 第一格 | mtyx_t2_audit / p65-h1-capacity | 先核实际浮根/球关节ABI、完整耦合动力学及独立native参考，登记固定多案例/容量/代码边界后继续实现；不靠删guard开放任意组合 |
| 父共享接缝 | root / p65-parallel-integration | D1 owner/borrow/runtime到通用query的协议、shared source/query、公共导出；MuJoCo model/runtime/receipt/checkpoint按明确单owner协调。全部项目执行由父串行 |

三作者从本批同一基线建分支，保留原稿和未跟踪文件；先简短S0源码/接口/资源核对，再范围内tests-first编码。共享文件不交叉写，新增需要接口时用具体签名和字段交父协调。不得以文档代替可调用实现。

## 多案例与完成门

S3沿同一声明完成 compile、prepare query、真实step、读取、reset/restore后继与关闭；自动保原participant/body/shape/facet identity和过滤、不同anchor、稀疏world。新增与Batch A两arm混合family不同的工具/自由物体场景；查询失败不倒滚既成物理步。用户无需私接bank；没有新接触力资格。

D2借用D1当次accepted x bank：真实gather→变化的bounds/BVH→原球/胶囊有限边界查询。固定原三节点/共享边弹性片、多world固定点、多帧近面/远面/共享边、反序与过滤；证明面和边长实际变化，禁止HOST当前顶点回填。核原点面身份、旧revision和loan失效、完整候选/资源cap及late failure、reset/restore后的实际步与query。静态球/胶囊归同plan/runtime，尚无运动刚体双owner借用、CCD/闭实体或接触响应。

R2先冻结一组通用FREE根+BALL子关节案例：完整root twist/四元数、qpos/DOF/祖先和M/J/bias，非零初速、不同轴/惯量/驱动、正反卸载、恢复后继；每格独立同入步和连续CPU MuJoCo参考、实际TYX力与运动。接触/activation/limits/dry等组合必须逐项按实际实现声明，原工程门不为追绿改动。新组合须复用现有框架，不能针对某模型/坐标/时间特判。

父最终按片审阅、fresh guard、环境/精确静态、解释/CUDA新增及受影响兼容，再一次统一wheel和源码外消费者。保存所有首次失败、输入、实际返回及独立参考；同源完整绿组不重跑。两次针对同一假设仍不缩小问题，则保存最小反例，调整假设或转独立节点。

## 后续与操作边界

S3通过才关闭主线一当前公共连接目标；D2只关闭查询，D3双向接触力与共同发布、D4连续载荷和D5组合安装仍待补。R2第一格不替代异构功能/容量或性能比较。下一阶段按已验能力推进，功能优先、性能超越非强制。

沿用三位原agent，无新助手。所有import/collection/数值/离线probe/lint/formatter/compile/build/install/profile和参考执行严格root单槽，开始前fresh entrypoint-aware guard。保原K1三未跟踪文件、作者旧稿、环境/pins、首失败和证据Junction；无CI、push/mainmerge、发包或环境清理。证据根 `artifacts/p65-three-track-b-20261112-01`；本计划不是执行通过或整个阶段完成声明。
