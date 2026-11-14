# T1b-4：实际抓持载荷与质心力矩诊断

起点61840202；2026-11-14。root唯一串行执行；两原作者只做隔离纯编码/文本研究。保留全部旧结果与原场景、Goal、物理参数。

## 在运行前固定的新分析

仅消费上一批world1/3各2400步原保存轨迹。固定步骤1100/1300/1500/1700/1900/2100/2400，每世界7点，共14点；同一trace[k-1]端点q/v与trace[k]实际native ctrl/qfrc/xfrc恢复native scratch。CPU只mj_forward求同状态输出，不做mj_step，不新增CPU轨迹或GPU重跑。

每点记录：全部物体接触与分组pad/其他（包括台面）的世界合力、关于真实COM的合力矩；contact frame的force/couple及body1/body2符号；pad-local接触点、COM位置/边缘余量；对象相对tool朝向与角速度；各pair native输出合力/COM矩差与contact数。不以某个最接近的点丢弃其他点或合并不同物体所有者。完整保留raw contact与native对照。q/v/time前后不变。
原MuJoCo3.3.7编译XML/惯量/重力/摩擦/软接触；只对照当前实际状态，不能推出整条CPU/GPU轨迹一致。误差如大于先前经验必须如实定位，不调整物理或容差使通过。

## 本批可复用接口

新增measure_object_contact_wrench：只消费公共accepted-entry NumPy记录、显式object body/同相位COM、0..8个group geom IDs。保留所有物体接触，包括未分组台面；从完整contact轴变换force/couple，再把moment移到COM。完整校验同相位meta/mask/frame/condim/ownership/有限性，冻结逐contact、group/other/total输出。该量是测量，不是力闭合或动力学平衡的证书。新增独立手算、反号、非零质心、协变、力偶与坏输入测试；父一次完整串行HOST。

另一原作者只先研究原PushT两BOX外边界/pusher有限候选，不依赖抓持模块；在接口冻结前不执行/不改源。

## 后续分支规则

先读这14点新证据，再决定是否有依据登记一次控制变更。若主要是接触几何/力矩支撑不足，补通用接触位置/相对姿态测量或筛选；若是模型规定的软接触creep，保留真实误差并设计有依据的闭环；若native差异实质明显，先定位实现。不得直接扫描摩擦/增益，也不得把双指有法向力当任务成功。不强行为本批凑新GPU成功；产生可复用诊断和已定位缺口本身可验收，整个操作目标仍保留。

依据：[MuJoCo接触模型](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#contact)与[contactForce接口](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce)，以及已实现public entry轴/载荷schema。无新安装/push/CI/周期任务。
## 公共保存记录消费验收（执行前追加）

新增 CLI scripts/measure_manipulation_wrenches.py；先原world1、后world3顺序读取每条trace中后一记录的feedback_source，与前一记录的contact IDs/data及SuccessFrame逐项绑定。每世界预期2399个entry，共4798；末步2400因缺少后一条完整public entry而明确未覆盖，不补造数据。使用同entry对象pose和原编译model.body_ipos求显式COM，逐contact独立对照原已发布world force/couple与双侧body-origin矩平移，转换差门2e-6；不是native求解精度或物理成功门。全部entry和摘要保存gzip JSONL，原输入SHA保持。
只编译原XML/读取model常量，不调用mj_forward/mj_step或新GPU。首失败原输出保留，修复后新目录执行；测试不重复已有物理轨迹。
