# 独立 GPU 低支撑补样原型

这是 cp4 派生的独立生成包。现有默认、基准 DLL/SPV 和旧诊断不改。作者未运行构建或设备；结果以父串行执行收据为准。

同一次公开 capture 的路径：原全率主光/镜面及传感 → 原紧凑 incoming → 原 K25 支撑计算及通用选择 → **同一真实场景全率、原深度 EQUAL 的额外光栅 pass，对选中像素真实追16个一跳提案** → incoming 过滤与合并。不是第二个 owner 的图像合成。

选择固定为 rho>0 且实际 K25 接受支撑数≤2，不使用 fixture、对象ID常量、亮度或69像素ROI。原16提案从绑定包 `runtime/.../resident_indirect.glsl` 原文件复制；有 HDR 时为8+8 MIS，无HDR为16分层余弦，保实际 Ng 半球和原 terminal 材质/灯光。它与 cp4 visibility-ratio 是不同估计器，统计不能混写成“全图16 spp”。

新增fragment只写 incoming；主画面的 direct/specular 与本像素 rho 原样保留。选中 incoming 的5×5二项核仅接收本次确有结果、同ID/valid、相对深度≤.003、Ns点积≥.98、当前direct+spec显示码差≤2的邻点。无 primitive-ID 门；不声称不跨三角形。均匀环境不为零时这一guide不等于旧完整环境guide。

有成本：额外全尺寸RGBA32F附件（1280×960为19,660,800字节，不含原资源），一次全率几何覆盖绘制、一次额外合并compute与原选择中的原子计数。缺少新硬件计时前不声称便宜。默认最多262144次合格fragment调用，绝对上限1048576；包含共面overdraw。超限标invalid并拒绝整个发布，不返回截断采样图。每次成功实际记录选择像素/合格调用/真实16提案函数调用/合并像素/滤波邻点；后者×16是提案数，不是hardware ray/query计数，Ng拒绝可不发射射线。首版无history，仅MSAA1。

## 执行入口（父唯一串行槽）

```powershell
$author='C:/Users/38347/.codex/worktrees/render-motion-preview/phymeo/rendering/examples/preview_edge_rescue'
$parent='C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo'
$adapters='C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples'
$py='C:/Users/38347/AppData/Local/Temp/phymeo-g5-5fdf4ec3-20261123-02/runtime/env/Scripts/python.exe'
$prepared="$parent/artifacts/rendering/nov24/edge-prepared-01"
& $py -m unittest discover -s $author -p test_build_source.py -v
& $py "$author/build_source.py" --base "$parent/artifacts/rendering/nov24/cp4" --output $prepared --adapter-root $adapters --max-invocations 262144
# 使用本机已验证的 SDK include/lib/glslang 路径配置，不安装工具：
cmake -S "$prepared/source/_native" -B "$prepared/build" <原SDK/CMake选项>
cmake --build "$prepared/build" --config Release
```

生成器对每个原包声明文件核SHA，所有变更锚点必须唯一；新目录独占，失败不能就地覆盖重试。新98/99标记、扩展compute描述符和独立edge ABI都是真门。`runtime.activate(prepared)`必须先于任何 renderer import。每进程只能一个prepared。编译输出及实际导入包均留独立路径。

```powershell
& $py "$author/run_smoke.py" --prepared $prepared --examples-root "$parent/rendering/examples" --full-library <已验原全率DLL> --device-index 1 --validation --output "$parent/artifacts/rendering/nov24/edge-smoke-01"
& $py "$author/capture.py" --prepared $prepared --fixture <原guided_motion_study.py> --reference <原thin3-ref目录> --assets <原Panda资产目录> --device-index 1 --output "$parent/artifacts/rendering/nov24/edge-rescue-01"
```

smoke按odd封闭室、无紧凑锚点薄片、零光三组做真实新包与原全率一跳对照；全三传感逐字，实际额外attachment计数、薄片真实补样、sensor-only resize后同态重拍逐字、close/validation均硬核。画质误差只记录，不据此宣称通过。

`capture.py`保原14→15真实运动、原69px掩码与两份独立1024spp显示参考；全幅传感/帧身份须与既有参考逐字一致。另保存每个69px的真实额外incoming与有效位、计数、PNG、scene-linear总量、signed bias/RMSE、成本和关闭状态。注册像素少一个未真补样即拒绝；不补填或换ROI。旧 `diagnose.py` 的8HOST及实际near/cp4/vr2证据原样保留。
## 已验证全距离补样与可选12 cm组合

父串行首版验证：6 HOST、build、odd/thin/zero以及原14→15均通过；登记69像素全部有真实新incoming。两份独立1024spp显示参考下RMSE为0.00751/0.00767（原cp4 0.0411/0.0408，旧全率16+guided 0.00756/0.00771）。这是固定薄板案例结果，不是300帧成本或全场景质量结论。

后续可选新包以已封存 `near120-sealed` 为base，只有原低分辨率guide走120 mm真实tmax及平滑hit衰减，超半径按miss处理。选择条件完全相同；额外16提案明确走非guide全距离分支，主直接光、近场镜面和传感保持原路径。因此是**近场稀疏照明+全距离低支撑补样的空间混合模型**，不把二者称同一个传播半径。本次只比较这一套新组合，不覆盖首版通过产物。

源门核0.1200只存在于guide宏分支、两处query的真实tmax和原fade；补样编译时若意外带guide宏直接拒绝。新增完整设备descriptor/resource限额核验；过滤权重先乘精确1/256，避免大HDR值的中间加权放大，仍保最终finite拒绝。新compute实际sampler数如实从8记为9。以上增量需父验证，不沿用旧包实测状态。

```powershell
$new="$parent/artifacts/rendering/nov24/edge-near-prepared-01"
& $py -m unittest discover -s $author -p 'test_*.py' -v
& $py "$author/build_source.py" --base "$parent/artifacts/rendering/nov24/near120-sealed" --output $new --adapter-root $adapters --max-invocations 262144
# 按上一段SDK命令构建新目录，再绑定全部实际DLL/SPV：
& $py "$author/build_source.py" --output $new --seal
# 薄板仍用 capture.py，prepared=$new，另开output。
```

`motion.py`只适配父原 `preview_sparse_motion/run.py`，不复制采集循环。每轮原10预热+300原状态，两个通道组、第二轮逆序；原detail/recorded、帧键/相机/传感/hash/P95/分相位/编码逻辑完整复用。新schema=`phymeo-preview-edge-motion-v1`、backend=`experimental-native-compact-edge-rescue-v1`、profile含`local120mm-rescue16-full`；旧文件夹通用槽仍叫`sparse`，不表示旧稀疏候选。每帧actual中额外保存`edge_rescue`与`total_primary_proposals`，分别保留原4路锚点和16路真实补样。公开request到返回墙钟包含真实计数读回；GPU profile单列。

```powershell
$helper="$parent/rendering/examples/preview_sparse_motion/run.py"
& $py "$author/motion.py" --motion-helper $helper capture --prepared $new --baseline-library <原同构建全率DLL> --baseline-shaders <其SPV目录> --output "$parent/artifacts/rendering/nov24/edge-near-motion-01" --device-index 1 --resolution 720p --camera detail --studio-look recorded --channels rgb all --rounds 2 --poses 59 60 149 --film sparse
# 诊断时加 --quick；不把其结果称300帧验收。复用已测基线时显式加 --candidate-only，报告不声称本次配对重测。
& $py "$author/motion.py" --motion-helper $helper encode "$parent/artifacts/rendering/nov24/edge-near-motion-01" --media-output "$parent/artifacts/rendering/nov24/edge-near-media-01"
& $py "$author/motion.py" --motion-helper $helper score "$parent/artifacts/rendering/nov24/edge-near-motion-01" --references "$parent/artifacts/rendering/nov24/ref1/summary.json" "$parent/artifacts/rendering/nov24/ref2/summary.json" --output "$parent/artifacts/rendering/nov24/edge-near-score-01.json"
```

评分仍是同一已登记全距离一跳1024spp参考；近场混合模型差异必须保留在解释中，不能称无偏全距离积分。`--candidate-only`没有本次baseline四通道组，原评分器会按原证据门拒绝正式配对评分，不静默借未绑定旧结果。