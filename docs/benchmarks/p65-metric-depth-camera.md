# 从物理状态读取深度、法线和形状编号

`ph.DepthCamera(width, height, vertical_fov_rad)` 从给定的碰撞模型和当前几何位置返回四张图。每个像素同时给出沿视线的距离、沿相机前向的距离（单位 m）、世界系法线和碰撞形状索引。用 `image.keys[i]` 取回世界、物体和形状编号。相机向局部 −Z 看，+Y 向上，像素从左上角计。未命中返回无穷远、零法线和 −1。相机可指定世界、姿态与近远截面。

```python
camera = ph.DepthCamera(160, 120, vertical_fov_rad=1.0)
poses = asset.collision_motion(accepted_robot_state.q)
image = camera.capture(asset.collision_model, poses, position_m=eye, orientation_xyzw=view_xyzw)
```

## 独立运动和相机对照

Phymeo 用公开刚体积分器推进自由球，Newton 1.7.0 用 Featherstone 在实际 CUDA 上独立推进同一个球。两边的桌面与胶囊保持静止。初始球心为 (0.11, −0.04, 0.72) m，初速 (1.3, 0.18, 0) m/s，重力 −1 m/s²，步长 1/128 s。各自拍摄第 0、12、24 步的 35×27 图。两侧传感器各读自身的状态。

| 步 | 球心 x（Newton，m） | 最大视线深度差（µm） | 最大前向深度差（µm） | 最大法线分量差 |
|---:|---:|---:|---:|---:|
| 0 | 0.110000 | 5.27 | 4.95 | 0.0000235 |
| 12 | 0.231875 | 12.72 | 12.42 | 0.0000392 |
| 24 | 0.353750 | 7.01 | 6.78 | 0.0000235 |

误差向上取整。三帧共 2,835 个像素，所有原生形状编号和背景均逐格对应。三种形状在每一帧都有可见像素。原始记录保存两端球心和 Newton 原生深度、法线与形状图。首个驱动尝试因输出名与作业配置重名，在仿真开始前拒绝覆盖；实际成功记录使用独立结果文件。[两端驱动](../../tests/reference_newton/robot/depth_camera_observation.py) · [原生 CUDA 逐像素结果](../../artifacts/p65-newton-night-20261105-01/depth-final-cuda.result.json) · [Newton 1.7 传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)

当前 Phymeo 射线查询在 HOST 运行，能读取球、有限胶囊、平面、无圆角三角面/四面体/封闭三角网格/凸包。遇到尚未实现的曲面种类会明确报告。并列距离按注册的稳定编号选取。颜色、材质、光照、畸变、GPU 光栅或光追性能仍待后续验证。[实现](../../src/phymeo/sensors/depth_camera.py) · [几何及隔离回归](../../tests/test_scene_depth_camera.py)
