# D：原始 incoming 的离线时域可否证实验

本目录不是 GPU 预览档，也不是已显示 RGB 的平滑器。不修改任何 renderer、
shader、直接光、镜面或公共传感。它只收集**原 SparseBackend 全率 guide**中真实
traced anchor 的 incoming，再做独立 CPU FP64 评审。Compact 的 guide 是低分辨率，
本入口明确拒绝；GPU compact_history 是另一个实现，不共享本目录的验收结果。

## 最短执行链（由唯一执行 owner 运行）

当前 author workdir：`C:/Users/38347/.codex/worktrees/render-motion-preview/phymeo`。
使用已有 NumPy/Pillow 环境的 Python，不安装依赖。两个输出目录必须全新。

```powershell
python -m unittest discover -s rendering/examples/preview_temporal -p test_*.py -v
python rendering/examples/preview_temporal/collect.py --runtime C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples/preview_sparse/runtime.py --prepared C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/vr2 --output C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-raw-01 --device-index 1 --start 55 --count 12 --camera detail
python rendering/examples/preview_temporal/replay.py C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-raw-01 --output C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-review-01
```

采集为同原 Panda 模型、材质、灯光、720p detail 相机、seed0、录制状态55至66，
10次预热后12次真实公共四通道 capture；每次读取实际 incoming guide，不额外capture。
它会重新采这12个状态，因为既有电影只有RGB/传感，无法从RGB恢复incoming。
`--count`仅允许2至32个连续状态。原图不重采以外推，不伪装完整300态性能。

原始输出：

- `summary.json`、`inputs.json`：真实源文件/包/DLL哈希、物理帧/相机/刚体变换、
  光/材质/信号token、采集次数和关闭状态。
- `frame-0055.npz`至`frame-0066.npz`：实际全率RGBA32F guide及原depth/ID/valid。
- `public-0055.png`至`public-0066.png`：未改色的公共RGB8原件。
- capture墙钟、guide额外同步读取、文件编码成本分别记录；不把它们称为纯GPU耗时。

离线输出 `summary.json`逐帧、逐策略保留状态计数、拒绝原因、复用观察年龄、
修改incoming的RMS与signed RGB/luminance变化。它们是**当前样本的变化量**，不是
对真值误差或降噪比例。没有独立incoming参考时，`quality_claim=false`。
默认另存姿态59、60的精确`accepted`/`traced`掩码（0/255）与`age`计数PNG；
这些不是渲染图，也不覆盖原始RGB。

## 冻结的门与边界

- `reset-any`：任何对象刚体变换、场景/世界/epoch/phase、光/材质/信号token、
  相机身份/内参/尺寸、断登记帧或超过100ms间隔均清历史。相机外参移动可真实重投影。
  真实机械臂连续运动可能每帧全重置，零复用必须如实报告。
- `primary-only`：允许依当前/上一刚体重投影，但无法证明被移动物体改变的二次
  遮挡/入射光可见性；每帧明确标记该限制，不能推荐为可靠GPU时域方案。
- 同正ID、有效有限正深度、两相traced、相对深度0.003、Ns dot≥0.995、往返≤0.75像素；
  公共miss必须是`+inf/ID0/valid0`，不作为背景历史。
- 当前同ID/深度/法向邻域clamp，半径在看结果前固定为源tile间距（tile4→radius4），
  至少3个实际traced支持；保留观察数上限8。年龄不是独立样本数。
- SH fallback和全部拒绝像素逐元素保持current；显露区不从别的ID继承。
- 每帧同时给一个故意错误的同屏旧图半混合负控，记录它会影响的ID/深度变化区，
  明确不是有效对应、不是产品结果；不为了展示差异挑像素。

此版本仍需父执行HOST与真实采集才能报告通过。纯HOST包括相机平移、物体移动、
新显露身份、深度/法向拒绝、光/epoch/镜头重置、所有权/失败原子性、原包逐字绑定。
没有独立的真实“只改灯/只移遮板”采集；不能把这些合成控制用例称为现场光路验证。
真实原档以1600Hz物理tick下采样为电影帧，55→56实际tick2987→3040。
评审保原key不动；在完整pose序列与逐字source绑定后，另加presentation index、
previous exact source key、collection SHA作为邻接证明。仍拒重复物理tick、丢帧、
错前驱、换collection或超时。旧首轮报告因误用物理tick+1而全reset，须保留；
修复只重放已有NPZ到新review目录，不重采这12帧。

## 真实静态场景 / 小相机运动正例

另建光学控制：冻结同一录制场景的pose59全部节点，原材质/灯/相机内参不变，
三次曝光相机沿自身+X移动0、0.1、0.2mm（在看结果前固定）。公开SceneFrame标为
新的`authored`世界/时钟，保原pose59完整来源；不冒连续物理。10次暖身加3次实际capture。
评审严格核所有原节点仍相同、相机确实按声明移动，要求两策略各至少16个历史复用像素；
记录实际instance身份，不根据画面调整步长或支持门。

```powershell
python rendering/examples/preview_temporal/collect.py --runtime C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples/preview_sparse/runtime.py --prepared C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/vr2 --output C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-camera-01 --device-index 1 --mode camera-control --control-pose 59 --count 3 --camera detail
python rendering/examples/preview_temporal/replay.py C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-camera-01 --output C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/temporal-camera-review-01 --mask-poses 0 1 2
```