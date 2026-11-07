# 持续抓持材料声明：T1 短验收库

本片只交测试与记录，不实现声明工具、不运行新物理窗口。父计划 `3e70cfc` 授权，作者基线 `4de207a`；公开首 RED checkpoint 为 `4220ee16f1f0c55cbeb776e157c00c0b137f03e9`。父在 `c68f132` 导入后执行 `238` 收据：**1 failed / 0.83 s / RAW1**。原 `make_case("mimic-box")`、2→3 mimic 投影、真实 q/v、100 g 与原 dt、两有限指面和八个独立材料锚点、初态无目标正 N 全部先通过，只在缺少 `phymeo.material_stations_from_pads` 时失败。这个有效 RED 不重跑。

其余新测试均 **UNEXECUTED**，作者未运行 import、collection、数值、pytest、静态检查或格式工具。父原 F01–F06 的已验结果不作为本工具或 F07 抓持通过证据。

## 冻结接口和原输入

```python
phymeo.material_stations_from_pads(
    scene, *, robot, payload_key, pads,
    station_keys, counter_features, chart_ids,
) -> MaterialStationRegistry
```

`robot=(world, physical_body)`，本例是 `(0, 10)`；普通 single `scene.shape_key` 查询仍用整数 body `10`。三种 mapping 均以 authored `(link_name, collision_name)` 为键：分别是与声明 offsets 同序的 `StationKey` 元组、物体完整面 material IDs、`(pad_chart_id, payload_chart_id)`。八个非连续 token 固定为左 `(101,107,131,149)`、右 `(211,239,263,307)`；chart 固定左 `(1001,2001)`、右 `(1002,2002)`。重排须同时重排 offsets 与 token；不从采样行、最近点或世界坐标生成身份。

测试使用原 `examples/robot_force_acquisition.py:35 make_case` 与原控制器。原例程尚未返回 `info.pads`，本测试只用公开 `PadFace.named` 重申原声明：四 offsets 为 `(±.005,±.005) m`，没有新几何。M 左/右是 axis=0、side=+1/−1；P 左/右是 axis=1、side=−1/+1。M 对面的 payload IDs 分别 `(0,1,2,3)` / `(4,5,6,7)`，P sphere 为 `(0,)`；不能扩大为 box 八点 chart。

M 继续原非等质量 mimic jaws，初态 q=(0,0)、full q=(0,0,.004)，payload 0.1 kg、h=.001 s。P 继续原九坐标 Panda、原固定折叠资产、sphere 0.04096 kg、h=1/1600 s。原质量、法向/动摩擦 law、控制目标 2 N、原力帽、时间、容差均不变；只预声明可选 kt=200 N/m 的八站连接。声明工具不得采样力、启用模式或推进历史。

## 短矩阵与独立判据

唯一测试路径 `tests/test_persistent_grasp_stations.py`，文本预期 16 个实例，实际 collection 以父收据为准：

| 节点 | 实例 | 冻结判据 |
|---|---:|---|
| `test_named_pad_registry_predeclares_real_open_mimic_sites` | 1 | 已完成有效 RED；原几何/初态前提先成立，八登记公开入口缺口 |
| `test_pad_declarations_survive_mimic_projection_and_panda_fixed_frames` | 2 | 原 M/P authored 点经真实 model frame/FK 重建 world 点；完整 face、counter、chart、切轴、原 1e−8 m 匹配容差；禁止声明时 sample/step/enable |
| `test_declaration_reorder_and_source_mutation_do_not_change_owned_registry` | 2 | 两资产声明顺序与 offsets/token 同步反序不改登记；修改输入 mapping 不污染两份 owned registry；输出不可重新赋值 |
| `test_invalid_material_declaration_rejects_without_scene_publication` | 9 | 错 robot world/body、station world、payload world、重复 token、缺 chart、未知顶点、整 box 伪面、未知 authored pad 均拒绝，所有公共状态不变 |
| `test_first_loaded_pad_sites_and_retry_publish_once` | 2 | 各原资产只推进至首次真实正 N；完整双侧材料/owner/application、原法向/未选力、独立广义载荷与自由体冲量；同一步失败→同命令重试一次发布 |

模型局部锚点用资产既有 `authored_to_model_points` 转换，另由 authored URDF 固定尺寸/偏移及真实 FK 重建世界位置核对。M 原面 x=.218/.282 m；P 原 pad authored center y=±.00758 m、half y=.0076 m、center z=.04525 m。材料 owner 来自真实资产 link_index/link_name，不从 robot body ID 推连杆编号，不将 Panda authored 点直接当 model-local。

两个短动态实例各只有一条原 acquire 前缀：原 dt/原 posture/原 controller，最多原 acquisition timeout `ceil(1.5/h)+1` 次尝试。未到首次正载即 timeout/empty/lost 时直接失败；不改初态预压、不重置、不调目标或把首碰推迟到“好看”的帧。首次正 N 不要求恰为 2 N 或恰有八行，所有实际正载 site 必须一一绑定，使用原 `site.force_n`，不再次乘 fraction。

在首次正载实际 finish 成功后注入一次 MemoryError，检查 state、reading、geometry、events、history、receipt、last_frame 全不变；原控制器同 observation 再 propose 必须完全相同。恢复真实 finish，用同命令重试；selection、候选 history/receipt 与未发布候选完全相同，step/history 只推进一次，出生 generation=1。M 同时把两个合法 payload 对面声明互换，用**同一个真实入步样本**验证 feature 拒绝；两资产都检验正确 owner key 搭配错误 link frame 必须拒绝，不允许观察消耗 episode。

正载的独立账包括：所有未选 PairLoad 原样保留；所选 N、gap、normal、权重及共同作用点不变；按每个实际 site 对自由体累加 force 和 COM torque，对照公开完整载荷。自由体保持原 `m Δv = h(F+mg)`、误差 ≤3e−12 N·s；通过原公开点 Jacobian 复算完整关节载荷，mimic 再核 `P.T @ full`，力帽余量 ≤1e−9 N。材料/作用点运动门仍为 1e−6 m，能量残差门仍为 1e−8 J，未放宽。

每例 `tmp_path/<case>-first-load.json.gz` 完整保存原输入/声明、首次正载入步 reading/geometry/proposal、真实 events/bindings/owner frames/法向/weights/两侧 material 与 application、未发布候选、原完整载荷、成功 reading/geometry/history/receipt/events，或实际失败类型与最后公共状态。复用既有 `examples/general_scene_friction.py:641 public_payload` 无损字段序列化；不只保存 hash，不把归档 IO 放在物理回滚合同内。

## 父执行入口与边界

首 RED 已有收据，下一阶段须先由父授权并实现纯声明工具；本提交没有产品 source、root export、原例程钩子或长窗 driver。父以后先声明类短节点、再上述两个首次正载节点，所有项目执行继续仅父串行。可用 `-k 'not first_loaded and not named_pad_registry_predeclares'` 选择尚未执行的纯声明矩阵；产品有实现后的首公开 GREEN 与两个动态节点由父按收据排队，不重跑 238 RED。

G01 原 100 g mimic、G02 零摩擦、G03 原超载、G04 Panda 的四个完整候选窗仍全部 **UNEXECUTED / 本片未实现**。本片不宣布 acquired、held、released，不替换原八门，也没有新增跨面迁移、任意转动或自动材料匹配。首次 sampled-pad 绑定如暴露真实 face 漂移或运动门错误，应保存原始反例交父裁定；不能修改材料 ID、扩 chart、加匹配容差或偷偷退回旧切力。

## T2 作者交付：纯声明工具和原协议消费者

父 `d319d6e` 授权后，新增 `src/phymeo/runtime/pad_material_stations.py`，checkpoint `3052ee6` / `72f9385`。它同锁读取已编译注册、复用原 `_pad_points`，核原采样面及 offsets 集合，返回 owned registry；不采样、求 FK、启用切向或推进。普通场景调用 `shape_key(body,...)`；Multi 按确切 `(world,body)` 参与方的 named keys 定位并复核 world 与 body_keys，不能用全局首个同 body 名称。原 fixed-root/affine-mimic cuboid pads + 动态 COM-local sphere/cuboid 是明确范围；静态物体的旧 sampler 不消费 pad，故拒绝；floating/其他几何未自动扩域。合法但方向相反的完整 payload 面可声明，真实绑定时仍严格检验，不靠当前最近点替调用方挑面。

父已报告的新收据按发生次序保留：

- `249`：纯声明 14 passed / 2 deselected / RAW0。
- `250`：mimic 已到真实 birth/finish 与注入失败，但测试误用 `scene.tangent_receipt is receipt`。既有属性通过 `replace` 返回 owned 副本；父仅改为完整 public 值相等，保留 history/events identity、全快照和所有物理门。不是产品/物理修复。
- `251`：mimic 首次正载、失败回滚与同命令重试通过。Panda 首个真实正载绑定拒绝 `sample weights leave first_feature`：原左 pad y− face 应是 `(0,1,4,5)`，从 world 点逆算 trilinear 权重引入对面 2/3/6/7，约 1e−18…1.47e−17。父将从原 face/offset 保留 bilinear 元数据并沿既有显式权重桥传递；本作者不改旧 source/T1，也不把这次真实失败覆盖成成功。后续 Panda 修复结果待父收据。

NEW `examples/persistent_grasp_holdouts.py` 与 `tests/test_persistent_grasp_holdouts.py` checkpoint `4167322`；后一作者纯文本修正将 elapsed 取值放在末次 source hashing 前，以与说明一致，没有执行/物理变化。Driver 全部仍 **UNEXECUTED**。它只做一次原 `make_case`、一次公开声明/kt200 配置、一次原 `run_case`，不复制控制器、IK、阶段机或八门。依赖父 `20dac2b` 的原 `info.pads`、`on_accepted/on_rejected` 钩子与 root export。

回调严格消费 `grasp-protocol-observation-v1`：完整 entry 与 accepted/retained 七字段、实际最后 proposal、原失败类型/消息/accepted step/time。回调没有 live scene、没有控制返回值。每接受步保存一条完整记录，另核连续 step 序号和最终原 accepted_steps；每条 history/receipt/events/material/load 都留存。附加审计只读真实切力、原 μN、kt 弹性力、储能、实际作用点功与原 motion/energy 误差门；原八门完整保留，新增审计另列，不放宽或重标持稳结论。

`observations.jsonl.gz` 是可顺序解压的完整 gzip members/JSONL：输入与真实 registry、每个 accepted/rejected record、完整原 result（含其 trace/events），以及必要 incomplete 收据。总压缩上限仍 256 MiB，其中常规预算 192 MiB，64 MiB 留给最后一个尚未归档的完整记录与 64 KiB 停止收据；单记录上限为该保留量减停止收据。若记录本身无法容纳，明确记录未完整保存，不声称可恢复。部分写失败仅将本次新生成文件退回此前完整 member 边界；退回也失败则明确 prefix 不完整，不覆盖/删除既有证据。JSON summary 只做小索引，不代替原完整 result。

Driver 将原例程异常结果标为 `physics_rejected`，而被动回调/IO错误标为 `observation_incomplete`；后者不会重试物理、撤销 accepted state 或伪装成原 physical rejection。状态机按原失败分支继续卸载的行为仍由原 `run_case` 决定。源码身份绑定实际导入的 Phymeo checkout、HEAD、整包 `.py`、三份相关 examples、原 mimic/Panda URDF SHA；结束复核同一组字节，改变则不得给 passed。记录的应用 elapsed 含配置、原协议、被动审计、JSON/gzip/IO，不含 source hashing；不是 solver 核时间或新旧摩擦性能比。首个正载后再接受 64 步只发一条实际进度收据，不重启前缀。

短协议测试文本预计 13 实例，**待父 collection/执行**：原 open mimic 仅一真实步后 observer IO 错仍保持 accepted；原 step/propose 拒绝的 request/None 与完整 retained；rejected observer 错不重分类/重试；完整字段 gzip round-trip/拒覆盖；压缩预算保留最后记录；partial-write 回退新 member；替身协议恰调用原 run 一次且八门不改；accepted observer 写错不再推进；新绝对目录门及 schema 错原样保存。替身只测消费协议/IO，不用其数据当物理验收。没有新增四窗 pytest，也不重复旧原始窗口。

父下一次真正执行的 CLI（每条 fresh 绝对目录，原固定顺序）：

```text
python examples/persistent_grasp_holdouts.py --case mimic-box --out-dir <fresh-absolute-G01>
python examples/persistent_grasp_holdouts.py --case mimic-zero-friction --out-dir <fresh-absolute-G02>
python examples/persistent_grasp_holdouts.py --case mimic-overload --out-dir <fresh-absolute-G03>
python examples/persistent_grasp_holdouts.py --case panda-sphere --out-dir <fresh-absolute-G04>
```

这四窗仍 UNEXECUTED；唯一执行 owner 仍是父。作者只交上述两个 NEW 产品/工具路径、一个 NEW 短协议测试及本报告更新，T1 原测试与全部旧 shared 不再改动。原 100 g 失败尚未被新的持稳窗口关闭。
