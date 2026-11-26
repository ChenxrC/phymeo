# 摩尔线程双臂训练场录像

全场有 16 行、16 列，每个工位都有两台独立驱动的机械臂。左半场搬箱，右半场两臂协同推动重 T 形件。桌面的目标轮廓和自由工件来自各自训练环境。录像用同一台透视相机拍摄，依次展示整个场地、两区交界处的工作台，再退回全景。

## 轨迹如何进入画面

两项任务各从已有双臂策略继续独立的 CEM 训练。每项任务运行 2 代，每代在 8 个初态上各评估 8 个候选策略。相同初态的 8 个候选在 MUSA 上成批运行，各自读取双腕 RGB 与本体状态。每条训练轨迹的回报都进入当代的候选评分。场景保留这些实际记录的关节、自由工件和任务目标，不按成功与否挑选。两任务各有自己的策略和优化器。

每个工位绑定一条源轨迹。先用源 XML 和广义坐标还原机械臂、工具与工件的姿态，再把整个工位平移到厂房内。Vulkan 将重复使用的网格上传一次，把每个实例的位置、朝向、材质和缩放送入同一深度缓冲。主相机做透视光栅化，并使用方向光阴影贴图。训练中策略使用的腕部相机独立于这台展示相机。

片中动作按源物理时间的 0.25 倍速播放。每个工位从自己的轨迹偏移开始，播到回合末尾后重置到这个回合的初态。录像同步展示 256 个实际训练回合，设备训练的批大小是 8。

为方便看清动作，前排交界处的两个工位各展示一条完整成功的训练轨迹，进入近景时从接近工件开始播放。只调整这两条轨迹在厂房里的位置和起播时间，全场保留全部候选，包括未完成任务的回合。

## 重现拍摄

训练及场景模块位于 [轨迹采集](../../examples/_bimanual_floor_motion.py)、[共享场景](../../examples/_bimanual_floor_scene.py) 和 [相机与编码](../../examples/bimanual_training_floor_reel.py)。专用 [Vulkan 实例化光栅后端](../../rendering/native/instanced_floor/CMakeLists.txt) 需要 Vulkan 开发头文件、加载库、C++17 编译器与 glslangValidator。

```bash
cmake -S rendering/native/instanced_floor -B build/floor -DCMAKE_BUILD_TYPE=Release
cmake --build build/floor
python -B examples/_bimanual_floor_motion.py index \
  --training "$CARRY_RUN" --training "$PUSH_RUN" --out inputs.json
python -B examples/bimanual_training_floor_reel.py \
  --clips inputs.json --library build/floor/libphymeo_floor.so --device-index 0 \
  --width 1600 --height 900 --chinese-labels --font "$CJK_FONT" --out training-floor
```

可先加 `--preview-only` 检查全景、近景和末帧，正式录制使用另一个输出目录。索引要求训练回合已有 XML、NPZ 和对应的 MUSA 收据。采集新回合时以 `_bimanual_floor_motion.py capture-train` 启动原训练器，并沿用对应任务的策略配置。正式影片要求 256 条互异的训练轨迹和完成更新的当代训练检查点。

## 2026-11-26 实拍结果

[播放完整录像](../../artifacts/s4000-training-floor-256/training-floor.mp4)。开场展示 256 个工位，第 4 秒开始推近，第 13–23 秒停在两项任务交界处，第 23 秒开始退回全景。左侧任务需要双臂抬起并搬运 6 kg 箱体，右侧需要把 T 形件推向桌上的目标轮廓。

MTT S4000 使用 4 倍多重采样，在同一场景绘制 20,418 个网格实例。720 次 Vulkan 提交对应 720 帧，输出为 1600 × 900、24 fps 的 H.264 影片，共 30 秒。逐帧解码后的时间戳与拍摄时间一致。把每一帧与编码前的原图比较，最差一帧的 RGB 通道平均绝对误差约为 3.25，通道取值范围为 0–255。录像文件共 27,968,112 字节，远端与下载副本的 SHA-256 均为 `3784f9b9189c4d38c0af986e9ad6db3fceabaceadd7cc3a1ca43d1ccd00e9a6d`。
