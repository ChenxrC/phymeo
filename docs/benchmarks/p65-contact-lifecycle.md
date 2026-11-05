# B1：已接受物理步的接触生命周期

本批为普通 `ContactScene` 增加 `contact_history`，记录同一次实际入步的正法向载荷、材料见证与代次。它复用原有 `NormalLayer`、摩擦和积分，不新增静摩擦记忆、恢复系数或持续约束求解；原计划 B 的其余部分仍未完成。

源基线为 `104e49a`，父维护的场景事务接缝 `dbfa815` 在作者树对应 `ab9a736`。本报告同提交的新增源为 [contact_history.py](../../src/phymeo/runtime/contact_history.py)，专项为 [test_contact_history.py](../../tests/test_contact_history.py)。本批执行由父串行完成，作者没有并发运行项目。

## 公开语义

`scene.contact_history` 是不可变快照，初始 `step_index=0`、两时刻为 0。成功物理步后才递增；`sample_time_s` 是力和见证所在的入步时刻，`commit_time_s` 是该步接受后的状态时刻。重复 `sample()`、`geometry()`、`observe_scene()` 不消费代次或事件。

身份为编译场景内的 `(first CollisionKey, second CollisionKey, generation)`。两键均含 world/body/shape；浮点坐标不进入身份。只有实际 `PairLoad.force_n > 0` 的形状对进入 `active`。出生、持续分别为 `begin/persist`；已载形状对不再受载时为 `end`，其当前 `record=None`，原因区分本步没有 PairLoad 的 `absent` 和仍在材料层但法向力为零的 `zero_force`。重入增加该对的 generation。零力不是几何分离的别名，原始零力 PairLoad 仍可从本步 `last_frame.loads` 取得。

每条记录保留当前 PairLoad 的力、法向、间隙、功率、层储能、实际表面点和施力点。材料地址单列原节点权重；刚体另存入步刚体坐标系下的表面点，粒子面保留原权重。摩擦的共同施力点与两侧表面点不混用。`material_changed` 比较精确材料数值，末位变化也可能触发；`support_changed` 仅比较非零节点支持集。这两项都不是额外的稳定曲面特征 ID，也不运输摩擦历史。

快照只保存本步 active/events，以及每个曾受载注册形状对的最后代次；不在场景中无限串接完整历史。父接缝在外层 workspace 事务中、物理结果 owned copy 成功后纯计算新快照；成功退出事务后与状态和时间一起发布。容量或发布拷贝异常时，旧状态、时间、物理帧和生命周期保持不变。

## 实际验证

| 组 | 实际运行 | 结果与范围 |
| --- | --- | --- |
| 正常入口首 RED | HOST，1 项 | 原 ContactScene 缺少 contact_history，1 failed，0.87 s；原因明确保留 |
| 新专项 HOST | 9 项 | 9 passed，1.04 s：有限台离边、材料支持变化、整体轴循环和平移、原摩擦、自由节点、世界隔离及拒步 |
| TYX interpret | 1 项 | 1 passed，1.18 s：真实三步受载→零力→重入及同样物理数值 |
| TYX CUDA | 1 项 | 1 passed，2.20 s：同一三步在实际 CUDA 路径执行，要求 TYX 存在 |
| 源码检查 | 仅新增两文件 | Ruff 通过；格式化前后 2 exact AST matches |

上述用时是 pytest 报告的专项耗时，不是物理吞吐。九项 HOST 留出没有因此全部被标为 CUDA 已验。设备两端仍使用现有 TYX 法向层和刚体积分；几何复核、材料映射与新生命周期记录在 HOST，未新增设备核或宣称整链驻留。

关键独立判据：

- 两个 1 kg、半径 1/8 m 球，间隙 1/32 m，层宽 1/16 m、刚度 128 N/m，零重力。首步实际力为 4 N；h=1/256 s 后速度为 ±1/64 m/s，净线动量不变。入步记录的 4 N 与随后新状态 sample 的力不同，防止混标时刻。
- 同几何用原阻尼 4 s/m 与预先固定外力：第一步外力 ±60 N，与接触合为 ±64 N，速度达 ±0.25 m/s；下一入步间隙 17/512 m，原法则裁剪法向力为 0。第二步外力反向 ±128 N，回到初始间隙并以相对速度 −0.5 m/s 接近；第三步原法则实际力为 12 N，身份进入第 2 代。没有改质量或材料来匹配身份结果。
- 一个球从有限箱顶沿 +x 运动 16 步；真实材料支持从面内移向边，法向出现 x 分量，随后退出层。pair/generation 在连续受载期保持，退出只发一次且不携带旧力。
- 双球原动摩擦 μ=0.5、初始相对切速 0.5 m/s：原施力为 (−2,0,4) N、功率 −1 W；共作用点 z=9/64 m，而两球各自局部表面点 z=±1/8 m。双侧力矩连同轨道项相消。
- 球与三正质量自由节点有限片的权重为 (1/4,1/4,1/2)，h=1/512 s 后节点 z 速度为 (−1,−1,−2)/512 m/s，球与片总线动量守恒，节点真实位置差异非零。
- worlds 7/42 复用局部 body 4/9，身份各自保留；容量拒步与 owned-copy 注入失败后，重试和直接成功得到相同快照及状态。

## 复现与收据

父证据根为 `artifacts/p65-postnight-batch1-20261105-01/`。任务 `b1-red`、`b1-host-first`、`b1-interpret`、`b1-cuda`、`b1-lint-first`、`b1-format-ast` 保留 raw exit、参数、源 SHA 和输出。CUDA 收据绑定的新模块 SHA256 为 `7609833F502BACE14E0B1077417CAAE5F679C1BBCEADDC65FAFA72138B212344`，测试为 `3B677D1171F407E89D5666340BA86E3F151FE8C0D7FDCF9B47B27149EF7994B2`；当前两文件与上述 SHA 一致，格式化前后也已证明 AST 不变。

专项选择为：

```powershell
python -m pytest tests/test_contact_history.py -m "not tyx" -q
python -m pytest tests/test_contact_history.py -m tyx --tyx-mode interpret --require-tyx -q
python -m pytest tests/test_contact_history.py -m tyx --tyx-mode cuda --require-tyx -q
```

本切片不自动覆盖 A1 的另一种机器人场景返回类型，不改变普通 sample/last_frame 原义。源码外 wheel 与跨组组合由父后续验收；这里仅报告已完成的作者树专项。

