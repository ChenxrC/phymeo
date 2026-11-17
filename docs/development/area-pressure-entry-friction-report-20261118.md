# I3v：真实入步数据到面积摩擦问题

2026-11-18，codex/exp，源与测试825902af。首次完整组 **3 passed in 4.39s / RAW_EXIT=0**，无数值修复或重跑。

[新适配器](../../src/phymeo/backends/mujoco_area_friction.py)将受信实际入步receipt转成owned问题。绑定源三hash、layout、world/tick/time/q/v、材料和全部source quadrature（含空面）；由入步COM与COM Jacobian生成完整双方点J。没有重跑FK、接触求解或积分。D=RN32源阻尼，非阻尼项f=applied+passive-bias+Dv0，避免重复扣阻尼。几何为原FP64，动力学为设备数据的FP64解码，再由I3u统一打包。

scope明确为historical-same-entry-counterfactual-no-integration。输出不能再次施加给已经推进的场景。原Scene/type7/v1与friction=none没有改动；生产仍须在积分前使用同阶段数据。

## 一次实际验证

[测试](../../tests/test_mujoco_area_friction.py)仅一次双模型CUDA场景step，随后一个三问题CUDA solve_many：
- FREE模型有偏心COM、六个非零入步速度、18个真实当前面样本；同入口分别mu=.5与mu=0。
- 水平滑块m=.5、D40、v0=.02，无接触。正确自由速度.018518518m/s，重复阻尼错误值.017037037m/s；实际明确区分。
- CPU仅mj_forward/mj_fullM/mj_jac独立核M、完整JA/JB、bias/passive/applied，不mj_step。mu0另由独立全active线性解核且先证Fn均正。
- 三组通过原FP32残差、最终Fn容量、Ft、双方力矩、非对称Jacobian及实际存储力代回门；严格HOST 1e-10门三组均false，未隐藏。
- 28个实际绑定/容量负控；另有源hash、非法law、原生候选、limit、dry friction、BALL、activation、inactive equality与fixed tendon负控。
- counterfactual前后snapshot/revision不变，不计为额外物理轨迹。

问题维度(6,18)、(6,18)、(1,0)；一个共同kernel，上传71364B/下载94632B/workspace165996B。kernel约.0632s，solve约.0864s，仅诊断，不作速度比较。

## 证据和剩余边界

artifacts/area-pressure-entry-friction-20261118-01保原日志、source-before、author、同次完整输入/raw结果/trace、saved-summary.json。实际payload为06-payloads/test_actual_entry_native_oracl0/entry-counterfactual.json.gz，72010B，SHA256 50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6。

首Ruff通过；首format-check未过，格式化前后两完整AST相同，最终format-check通过；执行前后source/test SHA相同。父文本审阅补neq/ntendon拒绝在首次执行前，不是数值失败。

仅1..6DOF/128完整样本（现三点三角积分最多126）、新law、无剩余原生约束。历史receipt受信，不提供密码学来源证明；output_time不是新积分证明。真正生产F2/F3、新载荷观察和摩擦窗口待做，见[后续计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)。
