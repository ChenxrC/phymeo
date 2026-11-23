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