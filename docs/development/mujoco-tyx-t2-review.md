# MuJoCo–TYX T2：自由根语义与源码审阅

2026-11-10，基线 `4ab18bd5`，独立分支 `codex/mujoco-tyx-t2-audit`。本工作只读固定源码、手推和审阅；没有项目import、probe、测试、静态工具或构建。旧T1审阅稿保留未改。T2新代码的实际通过范围由父串行执行记录决定。

## 固定来源与坐标结论

本地读取父 `artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src` 的0.0.2源码，另查官方MuJoCo **3.3.7**；不能把Warp main或现有Phymeo COM坐标语义代入此版本。

| 必须保持的语义 | 固定原生来源 |
|---|---|
| free qpos是根body原点的世界位置与WXYZ；FK只归一化局部副本，不修改入步qpos | [kinematics 50–58](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L50-L58)；本地 `smooth.py:87–95` |
| 前三列是世界平移；后三列绕当前body局部基，世界轴为R的列，转动支点是根原点 | [motion DOFs 209–226](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L209-L226) |
| 速度偏置不能仅保重力；free translation先加入，再处理三维转动，保全身体传播与RNE | [comVel 1792–1841](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_smooth.c#L1792-L1841)；本地 `smooth.py:1669–1731` |
| native四元数积分归一化原quat，再右乘局部角速度增量；Warp另归一化结果。这是FP32实现选择，非native精确舍入等式 | [quatIntegrate 215–224](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_util_spatial.c#L215-L224)；本地 `math.py:160–171` |
| Euler先更新velocity，再用新velocity更新位置；有阻尼时物理解M与步解M+hD分开 | [Euler 768–815](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L768-L815)；本地 `forward.py:227–264,326–352` |
| free spring含平移差和归一化四元数差的角向量；不支持就明确拒绝。dof damping按全部nv原坐标计 | [passive 60–111](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_passive.c#L60-L111)；本地 `passive.py:84–148` |

## 独立推导及已发送的检查点

令根原点o、根旋转R、局部角速度w、质心局部偏移r。世界角速度为Rw，COM速度为 `v_o + (Rw)×(Rr)`。在native qacc=0的偏置递推中，根原点加速度为0，角加速度也是0，因为 `Rdot w = R(w×w)=0`；但COM加速度仍有 `omega×(omega×Rr)`。子体的关节运输与Coriolis项继续按真实父速度传播。

根三个转动列必须为 `Jw_i=R e_i`、`Jv_i=(R e_i)×(COM-o)`；这同时确定原body COM世界wrench的广义力投影。M含全部动态子树及welded惯量，静态世界marker的J为0；不得按body数量猜root ID或从DOF数量截断身体。

输入receipt绑定存储FP32 qpos；FK使用归一化副本。不要为了让receipt quaternion变成单位长度而在HOST偷偷改写原状态。源root不一定是body1，nq与nv也不能共用宽度或切片，29scalar对应nq36/nv35。

保留T0已验纯自由球路径是显式结构分派，不应采用“先T0失败再fallback T2”。T2有潜在接触时应清晰拒绝；单个无接触时刻不能作为整个模型接纳依据。

## 新T2源码审阅

产品作者最终交接 `98ecb8aea97695534e3101ca525a34ba7db18478`（静态表 `ec1e9458`、kernel/device `9594c1e3`）。本次已逐段读取三个新模块；作者确认最后仅追加交接报告，产品字节没有在审阅后变化。源码审阅未发现阻断项；这不是解释/CUDA运行通过结论。

| 审阅点 | 实现与结论 |
|---|---|
| native身份及准入 | `_floating_model.py`单独保存root body ID，保全部身体和welded惯量，scalar按原qpos/DOF地址建表；root以外的独立活动树、free spring、复杂驱动和潜在contact pair明确拒绝 |
| 根坐标与偏置传播 | `_floating_kernels.py:geometry`直接覆盖root origin/R，world线速度与R乘local角速度分开；root零偏置a/alpha后，子体和COM仍计算运输、离心与Coriolis项。root编号不是常量1 |
| 完整耦合 | `mechanics`逐body/逐DOF构造world COM Jacobian及M，保根6列、后代列和所有交叉项；原非对角惯量先转world，bias同时含线性和转动项；COM wrench由同一完整J回传 |
| 被动力与积分 | 根六维a/D全保；scalar spring使用qpos_spring，FK使用qpos0，affine actuator使用raw q。M物理解和M+hD步解分开；右乘local角增量，输入bank不被归一化覆盖 |
| 状态与失败发布 | `_floating_device.py`动态nv向量/矩阵和父传入nq状态分开；download绑定old bank输入和new bank输出；复用父publish前的整批检查。非零status不被后续pass重置为0，失败pivot保护值不会成为已接受物理解 |

另对父未提交的 `_model.py` / `_mujoco_tyx_runtime.py` 变更做了针对性文本检查：显式旧T0结构在自由根检查中保留，复杂free树先分派T2，T1仍处理旧fixed tree；没有捕获失败后换后端。q宽度为max(8,nq)，v及applied宽度为max(8,nv)；kind3限位取原scalar joint的qpos/DOF地址；dynamics input_q使用nq，M/J/qacc使用nv。当前共享草稿未发现阻断项，仍需父按最终源快照执行验证。

独立新测试稿已覆盖0/1/7/16/29 scalar、非恒定根姿态/三轴角速度、偏置COM、非对角惯量、root编号大于1、massive welded、root和child力的互相响应、独立CPU同入步动力学及连续轨迹、稀疏多world/reset/clone、晚回读故障整批回滚、禁CPU/Warp/HOST求解fallback。父新ENV稿还保非identity根姿态、offset root ID2、两个scalar以及独立连续CPU轨迹。这些是测试设计覆盖，不是已收集数量或通过声明。

## 本轮结论与后续验收边界

本lane只完成T2语义和源代码审阅，没有改产品、测试或共享接缝，也没有领取执行槽。父后续需记录最终源对应的interpret/CUDA结果、独立动力学和连续轨迹误差、事务结果及NOTICE映射。结果归属最终tested commit；不能倒灌给本未执行作者版本。

T2仍是no-contact自由根树。29个scalar的容量、无接触自由运动或成功ENV连接都不能表述为G1足地持续支撑/完整舞蹈通过；接触、复杂传动/约束和更大规模性能仍按扩展计划另验。旧T1审阅稿保持未改且未纳入本提交。
