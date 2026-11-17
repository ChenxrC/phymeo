# I3x：共同载荷经过原生积分后的实际速度检查

2026-11-18，codex/exp，测试与固定夹具41efaf43。首次唯一新CUDA节点 **1 passed in 1.73s / RAW_EXIT=0**，在I3w已验证的源码外安装包上执行。没有修改产品源码、重新构建wheel或重跑已接受的摩擦求解。

## 本批回答的问题

共同摩擦solver给出速度v*和全样本力F。若把Q=Σ(JA-JB)^T F加到原applied，再调用既有native solve一次，FP32重新求得的vnext是否还能与原F对应？

固定四个已接受历史入口回答为“在原工程误差门内可以”。这支持下一批优先复用现有积分器，**不能证明所有输入都相容**；生产必须保留actual vnext上的逐步力律和共同方程检查。共享q-only积分入口仍是明确的结构性备选，未在本批改造旧路径。

## 输入和实际执行

直接复用I3w完整pipeline保存结果，SHA256 883dc7603c69d5789ea1e697ec8d64be33cb85c998bd244e4e05aba1a6b235a0。仅取实际FREE前三lane：mu=.5、mu=0、空面反事实；排除synthetic非零JB第四lane。另取实际空D40滑块。

原q/v/M/bias/passive/applied、h、阻尼和完整双方J均绑定原I3v实际入步与相同XML。HOST仅作本次诊断的存储F聚合，RN32(applied+Q)后调用：
- 原floating.solve一次，FREE三lane；
- 原tree.solve一次，scalar单lane。

合计2个原生设备kernel、0次新摩擦solver/pack、0次Scene step或CPU mj_step。CPU仅静态模型编译及mj_integratePos坐标参考。测试禁止workspace.solve_many和原共同solver入口。没有把该HOST聚合称为live resident零读回接线。

原记录没有raw两字clock。仅在original initial tick/time与receipt input_time均严格为0的绑定成立后，按原fresh初始化重建(0,0)，保存clock_basis；实际native核仍执行原Kahan更新。不得把重建说成读到了原clock字节。

## 实际结果

| 固定入口 | actual vnext与原v*最大分量差 | actual速度的力律方程残差 | 实际存储力的方程残差 |
| --- | ---: | ---: | ---: |
| FREE mu=.5 | 8.37446e-6 | 2.05408e-8 | 1.46506e-10 |
| FREE mu=0 | 4.47035e-8 | 3.16601e-10 | 3.12225e-10 |
| FREE空面 | 5.21541e-8 | 8.57136e-10 | 8.57136e-10 |
| D40滑块空面 | 1.86265e-9 | 4.12881e-10 | 4.12881e-10 |

速度分量含m/s与rad/s，最大值仅为该向量数值诊断，不称单一物理单位误差。残差为原M+hD逆度量；全部通过原2e-5*scale门。实际存储Fn/Ft/总力在actual vnext上通过原.002绝对+.002相对门；原点矩、切向非正功与Coulomb容量同时检查。严格HOST 1e-10门四例均false，未隐藏。

物理qacc按M求解、integration_acc按M+hD求解，分别独立核验；原D40正确与重复扣阻尼公式仍可区分。q仅按actual vnext积分一次，FREE世界平移与WXYZ/local角速度对独立mj_integratePos；两字clock、原输入字节不变均通过。不会用v*坐标参考掩盖实际积分后的速度差。

## 证据、限制及决定

[测试](../../tests/test_area_friction_native_integration_bridge.py)333行，SHA256 328a2c9546ed69b810f370fa604f8fc131153f219ac217f7eaab5ce78718c04d。首次Ruff通过；首次format-check未过，纯格式化后完整AST相同，最终静态通过。没有数值失败、修复或重跑。

artifacts/area-pressure-native-friction-bridge-20261118-01保存全部raw命令、隔离来源、执行前后源绑定、完整输入/原输出/实际native输出及坐标与力律证书。payload为native-integration-bridge.json.gz，20,224B，SHA256 08a29bd2ff3c5505a5d1d472a7449837ca65b8d33f3d6481d6e179eafdebb21d。复用原I3w wheel SHA95bcf0e92709977e5a09059f5e2e5811bc58f48b847b49f3536dbf01b262afb1；native模块与父源码相同，已加载phymeo来源均受隔离检查，未重复全包审计或安装。

结论是保留最小原生积分复用路线，随后直接实施[B1真实场景摩擦闭环](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。本批不是live runtime接线、持续轨迹、状态/新载荷共同发布、reader或restore验收；不能据此宣布F3完成。旧type7、v1与normal-only路径保持原含义。
