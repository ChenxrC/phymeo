# Panda 抓取 · 动态渲染

四部无声 MP4 都来自同一次 Phymeo 抓取仿真。先看摄影棚全景，可以看到夹爪下降、夹住红色方块、提升、横移、放回小台并松开。以下片长均为半速播放时长。

| 影片 | 时长 | 观察位置 |
| --- | --- | --- |
| [摄影棚全景](panda-studio.mp4) | 4.5 s | 固定相机拍摄完整 Panda；可观察手臂转动时外壳上移动的高光。 |
| [单灯与摄影棚同步对照](panda-lighting.mp4) | 4.5 s | 左边只有方向光，右边再加入两盏矩形面积灯和中性 HDR 环境光。两边的相机、姿态、材质、曝光和采样完全对应。 |
| [材质场景 · 全景再看近景](panda-materials.mp4) | 9.0 s | 前 4.5 秒看木纹工作台和完整机械臂；后 4.5 秒从夹爪近处重看同一次动作。黑色橡胶台面与黄铜倒角的反光随视角变化。 |
| [方块表面 · 红漆与铜同步对照](panda-surface.mp4) | 4.5 s | 仅替换方块的材质声明，左右使用同一摄影棚和同一运动相机。看方块侧面如何反射工作台和夹指。 |

机器人使用厂家 Panda 视觉网格和本项目的 Phong 到 PBR 转换。红色方块为电介质，线性底色 (0.54, 0.032, 0.017)、粗糙度 0.41；铜色方块为展示用金属，线性底色 (0.93, 0.61, 0.47)、金属度 1、粗糙度 0.18。木纹是按米制板宽布置的程序几何色带。橡胶、钢制支柱、黄铜边框及铜版方块都是渲染饰面。方块质量、抓取轨迹与受力记录沿用同一原始仿真。[厂家资产与材质说明](../../assets/panda/README.md)

## 输入、相机与播放

铜色近景里，指尖附近和白色外壳偶尔会出现细小亮点。高反射表面的间接光在有限采样下仍有噪声，这一版保留了离屏渲染的原始表现。

原始记录共 3,600 步，间隔 0.625 ms，覆盖 2.25 s。从 48 Hz 时间片右端选取最近的实际出步记录，等距选较早者，共 108 个独立姿态。首个采样时刻为 0.020625 s，最后为 2.25 s。回放直接使用原关节坐标以及方块的位置和四元数，渲染期间物理步进为零。影片以 24 fps 编码，观看时钟比源记录慢一倍。材质全景与近景之间有一次换镜，新镜头从轨迹起点重放。[可分发的原始采样子集](../../tests/fixtures/panda_motion/trajectory.json)

完整机器人使用 SAPIEN 3.0.3 路径追踪参考端。每路拍摄使用 768 × 576、256 样本/像素、最多 4 次反弹，关闭降噪。单灯与摄影棚沿用 R1 固定机位的视场。材质片加入小幅运镜，近景跟随方块中心。单机材质片画面外有 48 像素镜头标签，两路对照片另有 2 像素中线。MP4 使用 H.264、YUV420 和快速起播，颜色采用 BT.709 矩阵与 sRGB 传递标记。R1 的高采样原图及自研 Vulkan 适用范围见[双帧实测](../r1/README.md)。

## 复现

在仓库根目录运行。使用 Python 3.11、渲染包的 SAPIEN 可选依赖及 Pillow；编码时需可用的 FFmpeg（含 libx264）和 ffprobe。每次采集使用新的工作目录名。每条命令完成后再执行下一条。

```powershell
$env:PYTHONPATH = (Resolve-Path rendering/src).Path
$work = 'artifacts/rendering/my-panda-movies'
python rendering/examples/panda_movies.py render --work $work --pass single
python rendering/examples/panda_movies.py render --work $work --pass studio
python rendering/examples/panda_movies.py render --work $work --pass materials
python rendering/examples/panda_movies.py render --work $work --pass copper
python rendering/examples/panda_movies.py encode --work $work --output artifacts/rendering/my-panda-mp4
```

工作目录保留各视角原始 PNG、逐帧相机、源记录序号及 RGB 摘要。编码入口检查它们与 GPU 读回一致，再核对灯光对照、材质对照的镜头与采样时刻。成片逐帧解码，检查分辨率、帧数、时间戳及实际片长。[成片和源帧收据](recording.json)

本次在 RTX 4060 Laptop 上完成 540 次曝光。将四部成片实际解码，去掉画面外的标题与分隔线，按对应的原始相机逐帧比较，共核对 756 幅画面。有损视频最差单幅的 RGB8 平均绝对差为 1.70 以内。测试覆盖 7 项轨迹采样与来源校验。[逐片解码比较](decoded-image-check.json) · [四次采集的场景、设备和逐帧记录](captures.zip)
