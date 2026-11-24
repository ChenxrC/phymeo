# Panda 运动预览的软阴影与轮廓抗锯齿

建议在需要观察夹爪外壳和工具轮廓时选择每灯 16 点、开启轮廓覆盖采样的画质档。[同帧实测与适用范围](../../results/surface-quality/README.md)

主表面对每盏面积灯的 16 个分层位置查询当前遮挡。各位置在灯面上固定，接收面或遮挡物移动时会立即更新直接光。在完整距离的一次反弹中，二次命中仍使用每灯 4 点。主表面的预算与低分辨率间接光滤波各自独立。

轮廓使用当前帧的不透明场景加速结构。中心实例 ID 与邻域不同时，在像素内四处查询真实可见表面，取得该表面的局部着色，在场景线性空间按覆盖合成并做显示变换。深度、实例 ID 和有效性使用原始中心采样。该档能处理机械臂与背景、不同连杆和工具之间的边界；同一实例的内部折线及完全漏过中心采样的细小几何还需要其他处理。[轮廓实现和回退条件](../preview_contour_aa/README.md)

## 构建

从[已构建并封存的低分辨率间接光滤波预览包](../preview_lowres_filter/README.md)出发，在仓库根目录运行。沿用其 Python 3.11、CMake、Vulkan SDK 设置。下例中的 `$original` 和 `$inputs` 指向本机本次实测的输入。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
$original = 'artifacts/rendering/nov24/lf100'
$inputs = 'artifacts/rendering/nov24/film-cp4/inputs.json'
$package = 'artifacts/rendering/local-surface-16-aa'
python -B -m preview_surface_quality.build prepare --parent $original --output $package --samples 16 --contours
cmake -S "$package/source/_native" -B "$package/build" -A x64 "-DVULKAN_INCLUDE_DIR=$vkInclude" "-DVULKAN_LIBRARY=$vkLibrary" "-DGLSLANG_VALIDATOR=$glslang"
cmake --build "$package/build" --config Release
python -B -m preview_surface_quality.build seal --package $package
```

省略 `--contours` 可单独启用软阴影。`--samples 8` 是一个更低的灯面预算；本页的设备结果使用 16。准备和拍摄命令分别接受新的输出目录。封存步骤验证原有 GPU 着色器身份、实际编译的主表面采样数及可选轮廓着色器。

## 回放与比较

可将新包直接交给既有的 `preview_compact_rigid_history.preview --prepared`。也可用下面的入口检查与参考白壳近景一致的固定机位，导出实际回放和同步 RGB 采集耗时。设备序号以机器的 Vulkan 枚举为准。

```powershell
python -B -m preview_surface_quality.capture --package $package --source-inputs $inputs --device-index 1 --output artifacts/rendering/local-surface-film --rounds 2 --movie
```

采集结束后会单独核验打开传感器时的同帧 RGB，且这个验证阶段不计入 RGB 性能。程序调用可在新的 Python 进程中先构造 `preview_surface_quality.runtime.SurfaceQualityBackend(package, ...)`，再从已激活的包导入场景与相机类型。其 `open_scene` 与原刚体历史会话使用相同的 `capture(frame, request, presentation=...)`。输出包括原始深度、实例 ID、有效性及已合成的 RGB。

独立标定由 `preview_contour_aa.calibrate`、`preview_contour_aa.motion` 和 `preview_panel_shadow.motion` 运行。`preview_surface_quality.reference` 可用完整距离的 Vulkan 基准 DLL 重建相同 Panda 材质、相机和灯光的四子像素参考，`preview_surface_quality.assess` 比较旧档、软阴影档和组合档的原始 PNG 与传感器。
