# 薄边显露：实际支撑取证（尚无 GPU 补样产品）

本入口只拍原 `guided_motion_study` 的14、15两态，从公开实验 capture/readback 取
实际几何、incoming、SH/Ns和重建总线性颜色。固定相机/灯/材质/曝光/seed0；参考复用
`thin3-ref` 中两份独立1024spp原图，逐字核验原69像素右指显露掩码。

诊断不改着色、不修改原实验目录，不执行物理；每个prepared独立进程。某69像素误差
较大不自动意味着过亮：报告保存 signed RGB/luminance、RMSE、两参考间差异，以及
每像素K25逐候选首拒绝原因、权重、incoming和支撑数。无样本的子组为null而非零误差。

## 原理和范围

- compact采用实际低分辨率hit的ID/depth/Ns，并二次核对同位置全率anchor；奇数尺寸
  超界anchor无效。旧sparse的最后tile会夹至末像素，单独处理，不混用两种规则。
- 支撑核验保持原同ID、深度相对0.003、Ns点积0.98和K25高斯权重。这里用CPU FP32
  重演已有源门，不把结果冒称GPU逐像素附加状态；靠近比较边界的样本明确标记。
- 当前薄板fixture没有HDR或uniform studio。`sparseFallback`返回0，无支撑会丢失
  间接光。若测得正亮偏，不能仅以这个回退解释；同ID/depth/Ns不证明二次可见性一致。
- near120实际有两层行为：guide的rayQuery tmax为0.12m，且命中贡献乘
  `1-smoothstep(0,R,t)`；超半径按miss处理。记录trace/estimator源SHA。
- near120与cp4是compact/4路径；旧vr2是全率guide/2路径。只有匹配的near/cp4对照
  才能较好隔离局部模型，不能把所有差异统称“同模型采样噪声”。有限参考也不是真值。
- 对照的旧首帧是全率16路径加guided显示；原参考长积累1024路径未滤波。
  显示误差为RGB8逆sRGB后的display-linear；实际scene-linear总颜色另存，不混称。
- 这是无暖身两capture诊断，墙钟含四通道，debug读回单列；不作30FPS或吞吐宣称。

## 父串行执行

在 `C:/Users/38347/.codex/worktrees/render-motion-preview/phymeo`，用现有已安装环境。
每项输出必须全新。三条独立进程总计6次真实capture，不新采参考：

```powershell
$py='C:/Users/38347/AppData/Local/Temp/phymeo-g5-5fdf4ec3-20261123-02/runtime/env/Scripts/python.exe'
$root='C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo'
$d="$root/artifacts/rendering/nov24"
$fixture="$root/rendering/examples/guided_motion_study.py"
& $py -B -m unittest discover -s rendering/examples/preview_edge_rescue -p test_coverage.py -v
& $py -B rendering/examples/preview_edge_rescue/diagnose.py --kind compact --adapter "$root/rendering/examples/preview_compact/runtime.py" --prepared "$d/near120-sealed" --fixture $fixture --reference "$d/thin3-ref" --prior "$d/thin3-near" --device-index 1 --output "$d/edge-near-01"
& $py -B rendering/examples/preview_edge_rescue/diagnose.py --kind compact --adapter "$root/rendering/examples/preview_compact/runtime.py" --prepared "$d/cp4" --fixture $fixture --reference "$d/thin3-ref" --device-index 1 --output "$d/edge-cp4-01"
& $py -B rendering/examples/preview_edge_rescue/diagnose.py --kind sparse --adapter "$root/rendering/examples/preview_sparse/runtime.py" --prepared "$d/vr2" --fixture $fixture --reference "$d/thin3-ref" --prior "$d/thin3-vr2" --device-index 1 --output "$d/edge-vr2-01"
& $py -B rendering/examples/preview_edge_rescue/compare.py "$d/edge-near-01" "$d/edge-cp4-01" "$d/edge-vr2-01" --output "$d/edge-comparison-01.json"
```

作者未执行。若实际来源/传感/已存RGB或69像素绑定不符，保留failed报告，停止因果解释；
不能调mask、跳像素或放宽阈值来得到结论。

每目录保存 `summary.json`、69条完整 `pixels.json`、`evidence.npz`（实际全率几何、
低/全率guide、compact attributes及所选真实线性总色）、原图 `pose-14.png` / `pose-15.png`、
0/255 `exposed-right-finger.png`、实际输入和所有哈希。不会复制既有参考原件。

## 通用GPU补样的候选设计，尚未实现/验收

先依据取证确定fallback或supported错误占比。现compute没有TLAS/material/Ng绑定，
不能仅替换compute就冒出正确同模型首跳。可行路径是保全率primary和低率GI原样，在
低率GI后增加全率但选择性采样的fragment pass：使用当前真实P/Ns/Ng/材料，同套K25
可信门，不可信接收者才按原MIS真正采样；其余保原快速重建。不得按ROI/实例ID决定
采样；只允许几何支撑、局部轮廓/深度/法线判据。直接光、镜面、原传感附件不重写。

该路径至少多一次全率绘制/支撑判断，不能预先承诺低成本。需独立记录救援像素数、
每像素proposal数、总proposal硬界、fallback/overflow行为、pipeline时间及显存；
超界只能明示拒绝或保留当前方案并报告，不得把旧16计数当真实工作量。下一步应先
以闭合室odd/thin和本69像素同模型对照验证，再决定是否值得接入300帧，不在此诊断
中偷改产品。当前未提供shader或声称边缘已修复。
