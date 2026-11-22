# M7 外部数据适配开发计划

G2 已实现有限程序数据源 `AuthoredSceneSource`。[模块测试](../tests/test_adapters.py)覆盖独立遍历、帧身份和多世界时序。封存轨迹和具名模型的公开物理快照留待后续接入。

**目标：** 把程序创建的场景、保存轨迹和 Phymeo 已发布快照转换为同一 VisualScene／SceneFrame 输入，保持坐标、时间、身份与世界边界，渲染不推进或改写物理。

**架构：** 保留 SceneSource 的单向协议；静态视觉清单先建立，动态帧只携带节点世界位姿。首先用标准库的程序／保存帧适配器证明合同，再在可选物理依赖边界消费已发布快照；适配器不持有推进物理的职责。

**技术栈：** 核心 Python 3.11+ 标准库，测试使用 pytest。具体 Phymeo／MuJoCo 读取依赖放在可选适配器内部，不使单独安装 phymeo-rendering 时导入物理、CUDA 或图形运行时。

**依据：** [独立项目说明](../README.md)、[原 R0–R4 路线](../../docs/rendering/README.md)、[SceneSource](../src/phymeo_rendering/adapters/__init__.py)、[帧合同](../src/phymeo_rendering/scene.py)、[相机合同](../src/phymeo_rendering/cameras.py)。总调度见 [计划索引](README.md)，跨模块与最终证据分别归 [08-integration](08-integration.md)、[09-acceptance](09-acceptance.md)。

## 范围与文件所有权

| 所属 | 文件／职责 |
| --- | --- |
| M7 独占产品入口 | rendering/src/phymeo_rendering/adapters/__init__.py：保留 SceneSource 公共协议，显式导出具体实现 |
| M7 可新增产品文件 | adapters/authored.py、adapters/replay.py、adapters/phymeo.py：分别容纳程序帧、保存帧及可选已发布物理快照转换；不必为未支持来源创建空壳 |
| M7 独占未来单测 | rendering/tests/test_adapters.py |
| M7 计划 | rendering/plans/07-adapters.md |
| 上游协作，不直接修改 | assets.py、scene.py、cameras.py 的类型；物理公开快照／轨迹 schema 及其拥有者 |
| 下游 | RenderBackend.open_scene(scene)、RenderSession.capture(frame, request)，以及同帧腕部相机装配 |

当前 `adapters/` 公开 `SceneSource` 协议和程序场景实现。下文的轨迹解码与物理快照适配是后续工作。视觉资产缺失应列出缺件；碰撞代理只在调用方显式选择的诊断模式中显示。

## 现有接口与输入边界

当前公共协议只有：

~~~python
class SceneSource(Protocol):
    def scene(self) -> VisualScene: ...
    def frames(self) -> Iterator[SceneFrame]: ...
~~~

一个 source 的所有帧都引用 scene().key。实例、网格、材质、灯光或 node 绑定改变时创建新 VisualScene.key 和新会话，不能在同一场景键下偷偷换清单。世界可以共享静态资产，但帧、动态状态和分割身份以 world_id 隔离。

第一步具体实现采用显式有限数据：AuthoredSceneSource(scene, frames) 与 ReplaySceneSource(scene, frames) 接收 VisualScene 和已解码的 SceneFrame iterable；在对外产出前保存不可变副本。前者限定 phase=authored；后者保留保存帧的原 phase 和时钟，frames() 每次返回从首帧开始的独立迭代器。文件 schema 解码另放 replay.py 的明确入口，不做任意 pickle 或凭扩展名猜格式。

Phymeo 适配器先确定所支持的一个真实公开快照／保存 schema，再实施转换。没有 world、epoch、revision、tick、phase 和来源模型绑定的输入应拒绝，而不是从视频帧号、文件顺序或节点数量猜补。具体 API 与 source owner 对齐后写入该适配器文档和测试，不把私有 active bank／scratch 指针作为长期公共输入。

R1 另定义一种离线归档输入。对具名的 `panda-box-control-v1`，以源归档摘要建立独立世界，epoch 为 0，revision 是零基原始记录索引，tick 是对应的一基物理步号，时间须与记录的步长一致。模型摘要、关节顺序、原始行摘要和记录相位共同绑定。该身份只在归档内有效，实际运行时的发布身份保留为空。源端点的 q、方块位姿用于画面；同一记录中的力和指垫输入见证属于该步入步几何。[两帧摘录与提取方法](../tests/fixtures/panda_replay/README.md)

静态视觉清单负责 body/link 名称到 node_id、网格／材质及 visual 局部位姿映射。适配器输出 body/link 的世界位姿；实例世界变换由下游按 node_world × instance.local_pose 合成。避免在适配器预乘 visual 偏移后，下游再次乘一次。

## 坐标、时序与只读规则

| 项目 | 规则／拒绝条件 |
| --- | --- |
| 长度与缩放 | Pose 与 Mesh 局部坐标按米；来源毫米等需显式单位声明并只换算一次。未知尺度拒绝；同一清单不得把网格和节点各缩一次 |
| 姿态 | Pose.orientation_xyzw 为单位 XYZW；来源 WXYZ 显式重排。轴系变换应用于完整刚体变换，禁止只交换位置轴或混用左右手系 |
| 相机 | 输出相机约定 +X 右、+Y 上、−Z 前，像素左上原点、首中心 (0.5, 0.5)；腕部相机由同帧腕位姿 × 安装外参求出，M3 负责标定与投影 |
| 静态身份 | 帧内 NodePose.node_id 唯一，所有实例和相机安装点引用的节点均可解析；允许有位姿但没有视觉实例的腕部节点。源映射中未声明的引用、缺失的必需节点和重复位姿均须诊断。实例正 ID 在世界内稳定，跨世界以 (world_id, instance_id) 解释 |
| 动态身份 | FrameKey 保留 scene_key、world_id、epoch、revision、tick、phase 全部字段；不得只按 tick 或 batch 的行号缓存 |
| 物理末态 | accepted_endpoint 只能来自已经发布的成功积分终点；被拒绝候选或未提交 scratch 不得标成已接受 |
| 入步数据 | step_input 的位姿和时间须来自同一入步记录。入步接触／力不能贴到该步末态，phase 不得因为画面“最新”而重标 |
| 回放 | time_s 是源物理时间，不是视频索引／fps；允许跳帧、重复曝光同帧及同 tick 的合法不同发布修订，不要求相邻帧恰好相差一个固定步长 |
| 重置／回退 | epoch 改变允许 tick/time 重新开始；同世界同 epoch 内静默时间倒退须拒绝，显式 seek 应创建独立回放遍历／会话，不能伪装连续发布 |
| 异步多世界 | 各世界自己的时间线独立验证，不能按到达顺序对所有世界施加一个全局 tick／time 单调条件 |
| 所有权 | 输出不可变自有值；源缓冲复用、随后 step/reset 或释放源，不改变已产出帧；源的一次转换前后发布版本必须一致 |

读取不能调用 step、reset、控制提交或改变原时钟；也不得借渲染触发一次“预热物理步”。若必须从 q 重建 FK，应在独立拷贝的运动学数据中进行，明确来源和数值路径，不能改变实时物理对象。只读不等于来源可信：快照的已发布身份、所有权、完整性和几何映射仍要验证。

如果公开快照没有能直接对应的 epoch／revision，先与物理 owner 明确映射和 reset/restore 语义；不得自增一个本地计数冒充物理发布修订。当前源注释已明确原生 FREE q 使用 WXYZ，且 accepted endpoint 与 entry contact scratch 不同；实现时须再次核实选定公开 schema，而非把这些注释当所有来源的通用假设。

## 可以本地独立完成的步骤

### 1. 有限程序帧与保存帧合同

- [x] 在 test_adapters.py 用标准库构造一个网格、一个材质、两个实例、两个 node 和两个世界，写 scene()/frames() 正控；先取得缺具体实现的 RED。
- [ ] 实现 AuthoredSceneSource／ReplaySceneSource，显式保存静态清单与有限不可变帧。两次 frames() 的迭代进度彼此独立，禁止共享一个已消耗生成器。
- [ ] 写源 key 不符、节点重复／缺失、非有限时间／位姿、未知 phase 的负控；验证继承现有数据类型之后仍有明确运行时拒绝。
- [x] 同一个完整 FrameKey 若再次出现，时间及节点位姿必须一致；允许同帧被多个相机重复消费。发生矛盾时报告世界、键与字段，不覆盖先前记录。
- [ ] 跑本模块单测。保存文件入口只读显式受支持 schema，校验模型／视觉清单身份；未知版本、截断文件、长度不符和来源不全必须失败。

计划中的最小独立迭代测试：

~~~python
def test_replay_iterators_are_independent(scene, frame_a, frame_b):
    from phymeo_rendering.adapters.replay import ReplaySceneSource

    source = ReplaySceneSource(scene, (frame_a, frame_b))
    left, right = source.frames(), source.frames()
    assert next(left) == frame_a
    assert next(left) == frame_b
    assert next(right) == frame_a
    assert source.scene().key == frame_a.key.scene_key
~~~

这里 scene/frame 夹具须在 test_adapters.py 明确定义；scene 至少有一个真实 node 绑定，不能用空场景绕过映射测试。

### 2. 坐标和静态映射的手算验证

- [ ] 建立 identity、非零平移、绕 Z 轴 90°、固定 link、非零 visual 局部偏移的正控。源位置若为 1000 mm，转换后精确为 1 m；WXYZ 的 (√0.5, 0, 0, √0.5) 必须成为 XYZW 的 (0, 0, √0.5, √0.5)。
- [ ] 对节点位置 (1, 2, 3)、绕 Z 轴 90°、visual 偏移 (0.1, 0, 0)，独立计算网格原点应为 (1, 2.1, 3)，抓重复乘局部位姿错误。纯 FP64 小例坐标绝对误差 ≤1e−12 m、旋转矩阵分量误差 ≤1e−12；这只约束解析夹具，不把 FP32 实测 FK 强行套用此门。
- [ ] 非单位四元数、零四元数、未知轴系／单位、左右手转换错误的输入拒绝。合法 q 与 −q 按相同旋转比较，不因四元数双覆盖误判，也不静默“修好”非法源姿态。
- [ ] 构造腕节点移动而基座固定的两帧。完整旧曝光与新帧配对时应在拍摄前拒绝。另用独立变换重算新帧腕位姿，检出上游已经写上新键、却仍使用旧位姿的错误；单凭键相等无法检出这一类错位。
- [ ] 静态清单变化必须使用新 scene key；更改材质却沿用旧 key 的已绑定来源，在其清单校验入口拒绝。缺视觉资产保留明确缺件诊断。

### 3. 只读物理快照与时序

- [ ] 与 source owner 选定一个公开已发布快照入口，给适配器传快照值或只读提供器；明确真实 schema、模型身份、世界键、reset/restore 修订和节点世界位姿来源。
- [ ] 写禁止操作的测试双：step/reset/control 写入一旦被调用立即抛 AssertionError；转换一个快照应只读所需字段，前后 q/v/time/epoch/revision 逐值一致。
- [ ] 读取源版本、复制所有字段、再次核版本；若读取期间版本改变则拒绝该次输出，不拼接半个旧帧和半个新帧。若上游已提供原子自有快照则直接使用其完整性保证，不另读实时数组。
- [ ] 明确覆盖 accepted_endpoint、step_input、拒绝候选三个输入：前两者按其真时相输出；第三者拒绝成为 SceneFrame 的 accepted_endpoint。
- [ ] 修改／复用源可变数组后重读已发布帧，要求字节／数值不变；销毁适配器或源也不使该帧失效。
- [ ] 验证无可选物理包、无 GPU 的进程仍可 import phymeo_rendering 及 authored/replay 模块。具体物理适配器缺依赖时给显式诊断，不影响其他来源。

建议节点：test_adapter_never_advances_source、test_snapshot_copy_survives_source_reuse、test_source_revision_change_during_copy_is_rejected、test_rejected_candidate_is_not_endpoint、test_input_and_endpoint_phases_remain_distinct。

### 4. 多世界独立流

- [x] 构造 A 在 tick=10、B 在 tick=3 的交错帧，验证合法异步到达不会被全局单调门拒绝。
- [ ] A、B 都使用 instance_id=1 和相同 node_id，但位姿与 world_id 不同；逐帧交接时不得按节点名或 ID 复用另一世界位姿。
- [x] 只 reset A：A 的 epoch 增加且 tick/time 重新开始，B 的原帧与后续连续帧保持逐值不变。
- [ ] 同世界同 epoch 的陈旧发布、错模型绑定、错 scene key、缺 world 身份、重复完整键但内容不同分别失败。跨世界相同 tick 不构成冲突。
- [ ] 将这些数据交给 08 的真实双世界拍摄，保留重复 ID 的独立掩码和相机绑定；适配器单测通过不等于 GPU 多世界已隔离。

未来聚焦执行命令（仓库根目录，当前不执行）：

~~~text
python -m pip install ./rendering
python -m pytest rendering/tests/test_adapters.py -q
~~~

## 验收与合并时机

1. **可独立合并的 M7 核心：** 两个有限数据源及单元测试通过；无 GPU／物理依赖；静态清单引用、完整 FrameKey、坐标转换、源所有权、异步双世界负控均有证据。仅声明程序与保存帧能力。
2. **可选物理来源合并：** 选定真实公开 schema 的一个受支持模型通过前后状态不变、零步进、输入／末态相位和复制所有权测试；模型不支持时明确拒绝。不能以测试双通过就声称真实接入。
3. **R0：** 程序标定桌经 SceneSource→后端→Capture 形成首图；物理调用次数为零。M7 不等待 Panda 外观资产就可先交此来源。
4. **R1：** 接入真实轨迹和可追溯视觉资产；两个姿态验证基座、腕、手指锚点，含固定链接、visual 偏移和单位。资源许可、hash 与模型映射交 M1／M2，适配器不得自行找近似资产替换。
5. **R3：** 同一已发布帧的外部／腕部相机与像素标定一致；保持原 640×480、0.25–2 m、投影 ≤1 px、平面内区深度 ≤2 mm、有效区 ID 全一致门槛。真实 FP32/FP64 位姿差异记录来源，不能用放宽像素或深度门掩盖坐标错误。
6. **R4：** 1→2→8 世界、每世界 2 台 320×240 真实采集；单世界基准对比及单世界 reset 隔离通过。回放保源时间；预热后 300 帧的成本统计区分快照复制、FK、渲染、同步、读回和编码。

R2 画质与 R4 性能不是适配器独立合并的前置，但不可变来源和同帧相机是它们的前置。没有固定的完整身份／坐标／时序合同时，不合并“先画出来再补绑定”的物理适配器；涉及 scene.py 或 cameras.py 的公共变化先交 owner 协调，不在 M7 私自加另一套帧类型。
