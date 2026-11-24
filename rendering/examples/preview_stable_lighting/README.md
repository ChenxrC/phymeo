# Panda 默认运动预览

[完整运动影片](../../../artifacts/rendering/panda-stable-surface/acceptance/stable-motion.mp4) · [原档与新档并排，左旧右新](../../../artifacts/rendering/panda-stable-surface/acceptance/old-new.mp4) · [白壳双灯位对照](../../../artifacts/rendering/panda-stable-surface/acceptance/white-shell-lights.png)

摄影棚、HDR 方向与强度、曝光和 300 个原始姿态沿用同一段真实回放。每个像素用当前着色法线查询预滤波环境光（SH9），在同一点叠加面积灯直接光和材质的镜面响应。环境漫反射只进入最终颜色一次。每盏面积灯采样 16 个位置。不同实例与背景轮廓附近按原有四个子像素的真实可见性合成颜色。

短程环境遮蔽使用当前帧的场景加速结构。每个有效漫反射片元最多查询 10 个随刚体旋转的方向，按最近遮挡物的距离平滑减弱环境光。默认作用距离 25 mm，零半径用于无环境遮蔽对照。真实灯面的阴影来自面积灯各自的可见性。该运动档舍弃一次反弹带来的局部颜色传递。现有低分辨率调度和中间缓冲仍有开销，间接光射线与跨帧颜色混合已关闭。

Panda 白漆部件在场景准备时派生着色法线。原厂资产散列和材质匹配后，才沿平缓、连续且方向一致的共边合并法线。24° 以上折边、开口、UV 缝和非流形连接保留原法线。派生资产使用独立的场景键。传感器仍输出中心采样的深度、实例 ID 和有效性。

## 实测

RTX 4060 Laptop GPU，1280×720，软棚和原运动机位。计时从公共 RGB 采集调用到主机拿到帧，编码与诊断在计时之外。播放器按原轨迹的 30 Hz 播放。

| 检查 | 结果 |
| --- | --- |
| 独立验收，300 帧 | 44.88 FPS，P95 27.75 ms，300 帧均在 33.3 ms 内 |
| 默认拍摄入口，300 帧 | 49.58 FPS，P95 22.39 ms，300 帧均在 33.3 ms 内 |
| 冻结第 60 帧 | 32 次全图 RGB 逐字节相同，也等于连续回放的第 60 帧 |
| 第 51 至 57 帧开爪 | 两指节点间距从 42.4 增至 75.9 mm。1,439 个可确认刚露出的像素与重新从第 57 帧开场的结果相同 |
| 开爪处环境遮蔽 | 指尖邻域 35,866 个像素中有 4,784 个因环境遮蔽变暗。开关遮蔽所得深度、实例 ID、有效性相同 |
| 白壳曲面 | 两个等强侧向灯位下，手部白壳连续深度区域的亮度二阶差分 P95 从 10.00 降到 4.79、从 8.00 降到 5.00（8 位显示码值） |
| 轮廓与传感 | 第 60 帧有 67,879 条轮廓可见性射线，5,213 个像素解析出混合覆盖。第 59–61 帧附加传感请求得到相同 RGB |

[设备验收原始数据](../../../artifacts/rendering/panda-stable-surface/acceptance/report.json) · [默认入口逐帧耗时与轮廓计数](../../../artifacts/rendering/panda-stable-surface/default-entry/timings.json)

## 构建与拍摄

从仓库根目录使用原 Vulkan 工具链、带 NumPy、Pillow 的 Python 和 FFmpeg。先按[软阴影与轮廓构建步骤](../preview_surface_quality/README.md)制成每灯 16 点且开启轮廓采样的父包。构建稳定档复用该包的 CMake 工具链配置、原生 DLL 和 Python 运行时，父包需保留。以下为本次使用的路径。

```powershell
$env:PYTHONPATH = "$PWD/rendering/src;$PWD/rendering/tests;$PWD/rendering/examples"
$parent = 'artifacts/rendering/panda-surface-fix/edge16'
$inputs = 'artifacts/rendering/panda-surface-fix/sequence/inputs.json'
python -B -m preview_stable_lighting.build --parent $parent
python -B -m preview_surface_quality.capture --source-inputs $inputs --output artifacts/rendering/my-stable-film --rounds 1 --movie
```

默认包为 `artifacts/rendering/panda-stable-surface/default`，拍摄机位和布光取自输入记录。显式传 `--package $parent` 可使用原稀疏一次反弹档。复核局部遮蔽、两种灯位的法线和真实开爪时，用同一父包构建零半径对照。以下命令的输出应选择新的目录。

```powershell
$stable = 'artifacts/rendering/panda-stable-surface/default'
$control = 'artifacts/rendering/my-unoccluded-package'
python -B -m preview_stable_lighting.build --parent $parent --output $control --ao-radius-m 0
python -B -m preview_stable_lighting.validate --package $stable --unoccluded-package $control --source-inputs $inputs --old-frames artifacts/rendering/panda-surface-fix/sequence/rigid-history/frames --output artifacts/rendering/my-stable-audit
```
