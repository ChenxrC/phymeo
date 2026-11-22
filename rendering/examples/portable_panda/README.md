# Panda 独立渲染回放示例

把 ZIP 解压到一个新的目录。此包提供一段真实记录的十秒 Panda 抓取、放置和空手回程：300 个原始出步姿态，按 30 Hz 拍摄。播放器只计算运动学和图像，不运行物理求解器，不补帧，也不改变原关节、方块或时钟。

## 准备

先在独立 Python 3.11+ 环境安装本次发布的 `phymeo-rendering` wheel，并安装该 wheel 的 `ibl,textures` 可选依赖；参考拍摄另需 `sapien` 可选依赖。示例不自动安装或下载任何内容。

```text
python -m pip install "./phymeo_rendering-版本-py3-none-any.whl[ibl,textures]"
```

原生 Vulkan 使用与你的系统匹配的 DLL 和同一构建目录内的 SPIR-V。完整 Panda 需要 resident 路径；扫描粗糙度、HDRI 与 PBR Neutral 还要求本次发布的对应可选接口。缺功能会报错，不会偷偷换后端、换材质或关闭阴影。机器人外观及其制造商来源、许可证由已安装 wheel 的 `panda_visual_assets()` 读取，本 ZIP 不再复制机器人 mesh，也不含渲染引擎或求解器源码。

## 先拍一帧

在解压目录运行；输出放到包目录以外的新位置，避免改动固定输入。`-I` 隔离 Python 的环境搜索路径。

```text
python -I replay.py native --library C:/render/build/phymeo_native.dll --frame 60 --output ../panda-frame
python -I replay.py studio --samples 64 --frame 60 --output ../panda-reference
```

原生默认软件 BVH 阴影；已验证支持的 GPU/DLL 可显式加 `--shadow-mode ray_query`。参考调用已安装的 SAPIEN 路径追踪，默认 2×2 分块；`--tiles 1` 可选择完整相机。两者默认使用相同的 720p 相机、真实扫描纹理、HDR 环境和 PBR Neutral。随机采样噪声和估计方法不同，不承诺跨后端逐字节相同。

## 拍完整十秒

```text
python -I replay.py native --library C:/render/build/phymeo_native.dll --all --output ../panda-300
python -I replay.py studio --all --samples 256 --resolution 1080p --output ../panda-300-reference
```

`--camera fixed|wide|detail|sequence` 选择原示例相机；默认 `sequence`。`--frame 0..299` 选择一帧，省略时为 60；`--all` 才拍全部 300 帧。`--device-index` 是原生枚举索引，`--device` 是 SAPIEN 设备名。PNG 位于输出的 `frames/`，原拍摄收据为 `capture.json`，包及已装引擎来源为 `portable-source.json`。完整片的目标播放速率是 30 fps；编码可另用现有视频工具，不计入采集速度。

## 来源与边界

启动时按固定清单检查所有包文件及许可、URDF、轨迹和图片 SHA256；`manifest.json` 的摘要不是数字签名，应从可信发布处核对 ZIP 的 `.sha256`。原 `panda_realtime_film.py`、场景和相机助手保持原逻辑；唯一重定位桥将原 Panda 目录参数显式改为 `None`，选择 wheel 自带、仍由公共加载器校验的完整制造商外观。其余相对布局保留，不依赖原仓库路径。包内不接受附加脚本；将个人输出放在包外。

URDF 的 Apache-2.0 原文、NOTICE 和来源位于 `tests/fixtures/panda_replay/`。原始记录及相位说明在 `tests/fixtures/panda_film/`。大学工作坊 HDR 与 Wood Table 001 的颜色、线性粗糙度扫描来自 Poly Haven，原来源说明在 `examples/materials/NOTICE.md`，CC0 1.0 原文在 `licenses/CC0-1.0.txt`。这些外观不等同于实测机器人材料；原记录的成功标记也不构成新一次物理执行。

## 从受控源码生成本包

维护者使用固定版本源码运行：

```text
python rendering/tools/build_panda_example.py --rendering-root rendering --output NEW-panda-example.zip
```

构建器使用标准库、离线校验固定文件；不会探测 GPU、导入引擎或运行录制。源码外拍摄验证属于独立示例验收，不替代 G5 wheel 验收。
