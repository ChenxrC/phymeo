# C4 固定凸包的精确表示预构造复用

本批起点为 clean `9e1693110880f23245d8d2e780d37316f1d7d21b`。依据 C3 同次实际 loaded W32 每步 96 次 HullShape/边界、576 次 MaterialChart 构造的读数，尝试复用不变材料几何。它不是接触结果缓存；本批开始时不宣称性能收益，深拷贝与绑定检查本身有成本。

## 范围与所有权

作者新增 `_validated_hull_cache.py`、`test_validated_hull_cache.py`、本文及预先声明的小脚本 `profile_validated_hull_cache.py`。父负责 `_convex_support.py`、`_convex_support_path.py` 和 `collision_pipeline.py` 三个旧文件的窄接缝；不改变几何算法、原 C1/C3 驱动、原独立物理 oracle、内核、依赖或容量。

`ValidatedHullCache(shapes).get(shape, path, builder=None)` 属于工作区私有实现。登记表最多 256 个形状、总顶点不超过既有 8192 上限；只为真实登记的 hull 对象存条目。单 hull 仍由原 HullShape 强制最多 64 点。每个 hull 最多保留一个成功姿态，没有无限历史、全局缓存或跨工作区引用。

命中必须同时满足：

- 同一实际注册对象身份，以及所有 dataclass 注册字段的完整表示，包括 world/body/shape、材料 ID 次序、radius 的浮点字节、group/mask/revision 等。相同 key 的另一个对象不获得该条目；破坏 frozen 登记的修改明确拒绝。
- 原始 CollisionMotion 的实际 FP64 start/end 数组形状、dtype、strides 和逐字节内容一致；None 与显式相同端点不同，零角 turn 的原始 pivot/axis/angle 与 body orientation 也绑定。没有用局部坐标、平移等价、旋转等价或近似相等替代实际存储世界坐标。
- 仅静止路径复用。移动路径走原构造器且移除该 hull 的旧条目；窗口内每个真实采样姿态由原路径代码生成后，才可按该次存储点相等命中。连续区间证明不减少。
- 当前 builder 对象身份及类的实际 `__init__` 身份一致。默认在每次调用时读取真实 HullShape，不用定义时捕获的默认参数；父 hook 显式传当前几何模块中的 HullShape，原 monkeypatch/计数可观察。

失败不存条目，非法替换每次再次通过真实构造器。成功 miss 先深拷贝 factory 返回原件为私有存储，再深拷贝给消费者；hit 每次也返回新深拷贝。因此即使 consumer 或自定义 factory 保留原返回值并修改 points/face normal/chart，未来命中不会受污染。只靠 ndarray 的 writeable=False 不足以提供这一保证。工作区关闭清空条目，stats 返回独立字典；计数记录真实 attempts/builds/hits，并与接触状态分开。

## 保留的逐次工作

父 hook 只替换外部球–凸包投影中的 `HullShape(hs,hp)`。当前球心到真实面或边的搜索、精确距离界、法向、两侧材料权重与 offset、世界坐标分辨率门、完整时间窗验证都继续执行。初态包含球的通用 `_Body/_Query` fallback 保持原构造，不承诺减少其次数。plane/hull、hull/hull、mesh/hull 和 hull/triangle 原 prepared_hull 路径不在本次改动范围。

原 envelopes、实际 backend classify、policy/filter/capacity、code 0 与 code 5、所有合法性和失败发布规则不变；先前已发布 tuple 不被重写。非法几何可能按照原约定发布 invalid_geometry，这与设备/容量失败保持原 publication 的规则不同。实际 CUDA 仍执行原设备宽相；这一精确窄相复用发生在 HOST，不能称为 GPU narrow phase。

## 冻结验收

首测试提交 `ce4c5b4`。父在唯一串行槽执行 `test_public_repeated_fixed_hull_reuses_validation_after_real_witness`，真实首 RED 为 `05-cache-first-red`：1 failed / 0.81 s / RAW1。旧 public detect 先通过独立 1/32 m 净距、正向法线、双侧实际材料重建；改变球的横向位置后真实 HullShape.__init__ 多调用一次，正是机制缺失。没有以新 API 不存在作为这条 RED。

专项包括完整 public dataclass 的每一字段/数组字节/Fraction/浮点位比较；面、真实棱、角、触碰、内部 fallback；换序与刚体变换；一个 ULP 的实际点改变、重排、None/end/zero-turn/姿态表示、非零运动、身份/修订/容量、重复非法几何、factory 与 consumer 别名、构造器替换；多世界重复局部 ID、容量失败/恢复、真实设备故障、世界精度未决与原 affine 首触。设备专项禁止 HOST broad fallback，同时检查真实完成 launch。

作者全程仅文本、文件和 Git。源代码就绪时本专项 GREEN、格式与真实设备结果均待父执行；不得将未执行的测试或候选脚本写成通过。

## 小范围同次成本比较

唯一新脚本每次只接受 loaded W1 或 W32，HOST/interpret/CUDA 显式选择。沿用原 C1 同一公开 ContactScene、4 N 真实支撑滑动、`H=1/128 s`、原质量/重力/容差/材料及独立 scalar oracle。A 为本次同一源码的原 HullShape 逐次构造，B 为新缓存；不拿旧二进制或不同历史时段作为速度基线。

每边独立场景，物理第 1 步作为保留的 warm 调用，第 2、3 步为两个正式观察，正式次序 BA、AB。只在串行脚本范围替换 getter，两个变体均有相同的轻量实际 getter 计数；上下文 finally 恢复，嵌套或外部替换拒绝。没有系统性 Python profiler 或线级 tracer，不能把这一小窗口称为稳定吞吐排名。

计时从 `scene.step(H)` 入口到紧接 `scene.geometry()` 读回结束，拆 step/geometry；history/events 读取、oracle、编码和归档在此边界外。公开完整 frame、geometry、contact_history、contact_events 一个字段不删，逐帧 canonical bytes 必须相同；每条新无损 gzip 流在同次解码一次。旧大帧不再读取或哈希。

只允许四个原时钟不同：`geometry.envelope_ns`、`geometry.broad_ns`、`geometry.midphase_ns`、`geometry.narrow_ns`。新增三个机制计数独立保留且按原固定几何明确核验：每边实际 getter 调用为 3W；A 的缓存 stats 为 0（走原构造器）；B warm 的 attempts/builds/hits 为 W/W/2W，之后为 0/0/3W。除此所有原 counters 和实际 native kernel/传输/分配增量必须相同，不能把未知字段当时钟删除。纯 GPU device time 记 null，API 墙钟不冒充设备核时。

来源前后绑定原 benchmark/profile/cases 三工具、此次脚本/测试/cache 文件以及实际已加载产品模块与 clean HEAD，全部不变才能接受。每次保存 `pair.json`、`manifest.json`、两条完整 `*.json.gz`；首调用、两样本 p50/p95、实际字节及全部失败保留。2 个正式样本仅用于发现粗大收益或回归，无统计稳定性承诺。

父在 fresh guard、明确实际 source、clean HEAD 后可运行：

```powershell
python -m pytest tests/collision/test_validated_hull_cache.py -m "not tyx" -q -p no:cacheprovider --basetemp <全新绝对目录>
python -m pytest tests/collision/test_validated_hull_cache.py -m tyx -q -p no:cacheprovider --basetemp <全新绝对目录>
python scripts/profile_validated_hull_cache.py --backend host --worlds 1 --expected-head <完整HEAD> --out <全新父artifact目录>
python scripts/profile_validated_hull_cache.py --backend host --worlds 32 --expected-head <完整HEAD> --out <全新父artifact目录>
```

真实 CUDA 可用同脚本 `--backend cuda`，父按队列选择执行。每调用 6 个真实公开步，不跑旧九档或 C3 大矩阵。HOST 排程目标每档 20 s，超目标只如实标 `host_budget_exceeded` 并保留完整自然结束结果，不截取成功片段或悄悄减量；若它暴露成本问题不追加规模。存储预留约 64 MiB/档是估计，实际逻辑和压缩字节由收据决定。是否保留默认优化，以完整输出与有限同次实测裁定，不因构造次数减少预先许诺更快。

## 父最终验收（2026-11-06）

18-cache-first-functional：23 HOST passed / 1.00 s，另一个设备节点分别在 21-cache-interpret 和 22-cache-cuda 实际通过（1 / 0.68 s、1 / 1.20 s）。最终 Ruff/format 与源码外 wheel 通过。父 hook 限外部 sphere–hull 原 code 0 分支，原其它几何分流不变。

30–33 四个预声明小实验均 RAW0，24 个完整公开帧一致，完整 payload 同次核验，getter 均恢复。两个正式样本的 step+geometry p50：HOST W1 6.421→3.390 ms、HOST W32 191.985→86.879 ms、CUDA W1 8.159→5.163 ms、CUDA W32 207.183→89.495 ms。W32 暖态为 96 hits / 0 builds。保留该窄范围优化；两样本不是稳定吞吐排名，也没有比较 Newton 性能。实际 CUDA 宽相仍执行，HOST 几何复用不能称为设备窄相。

产品提交 8c4a7db，原始收据与限制见 [第四批综合验收](p65-scene-tangent-batch4.md)。父执行槽已释放；不追加旧矩阵或重复本同源已绿组。
