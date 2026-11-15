# V03 双软球：Newton 外部静态参考报告

日期：2026-11-15。分支：codex/exp。状态：**参考执行完成，预定精度门未通过（FAIL_RECORDED）**。
[预登记计划](../superpowers/plans/2026-11-15-area-pressure-newton-reference.md) · [上批 V00–V02](area-pressure-static-reference-report-20261115.md)。

## 结论与范围

已完成两个不同软硬球的独立等压曲面参考，以及 Newton 1.7.0 CUDA 接触前端的七个固定案例。最细网格的轴向力误差约3.4%，面积误差约6.46%；面积与最后两档收敛未满足原5%门。因此不能把本批记为交叉参考通过或已可正式集成。

这是一个有边界的预研结果：等压界面、软硬交换和零接触行为已具备可观察的外部实现；当前离散精度尚不足。结果本身不能证明 Hydroelastic 模型不适用，也不能直接归因为 Newton 算法缺陷。不同表示的离散误差、面截取与积分误差尚未分离。

没有改动 src/phymeo、MuJoCo-TYX、TYX pin、旧接触或训练接口。没有调用 Newton solver.step。这里的力来自其前端保存的压力×真实承压面积，再在HOST按保存法向积分；不是Newton动力学求解器已经发布的力或运动。codex/dev保持原稳定基线；没有新增训练结果。

## 固定模型与两个实现

两球半径20mm，中心相距39.6mm，总重叠0.4mm。A/B的压力形状参数分别100/300kPa，交换组只交换两值。连续场为 p=E(1-r/R)，Newton显式使用 kh=E/R，margin=gap=0。这里的E不是直接可当作杨氏模量的物理标定结果。

- 独立NumPy：在投影圆盘的三角积分点逐点二分等压曲面；解析坡度给真实曲面面积元和法向。可视化弦面三角形面积不代替积分面积。
- 独立oracle：公共压力参数化两球等压壳层，以固定64点Gauss积分计算面积、轴向力和压力加权中心；不共享前者的二分或网格。
- Newton：既有1.7.0/Warp1.17.0，cuda:0、float32 SDF、关闭contact reduction/pre-prune，MC edge clamp=0。保留原生计数、overflow、全部raw ID/压力/承压面积/编码法向/局部与世界位置、shape pair、fingerprint和可视化曲面。未用渲染三角面积代替真实承压dA。
- 同一连续场、不同离散表示。原始grid/AABB全部保存；请求2mm的实际体素间距为约1.666667mm，1mm/0.5mm档实际约1mm/0.5mm。不能把它们称为相同离散网格。

本地Newton关键源码的SHA保存在每个case的engine.source_files。wheel无VCS来源元数据；上游v1.7.0 tag只是参考，不声称整包与tag逐字一致。
[Newton压力接触前端源码](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/geometry/sdf_hydroelastic.py)描述了压力、承压面积和接触输出的关系。

## 实际执行

父任务唯一串行执行owner；每条项目命令fresh process guard。原两agent只纯作者，已释放。
HOST环境Python3.13.3/NumPy2.4.6/pytest9.1.1；Newton环境Python3.12.13/NumPy2.5.4，RTX4060 Laptop 8GiB、sm89。没有新安装。

1. 首Ruff只发现一处import-block空行，首format-check五个新Python文件需排版；修正后静态检查通过。
2. 源码22621c7b：完整新HOST **43 passed / 0.76s / exit0**。只运行本批两个新测试文件，没有重跑旧14项。
3. 同源独立NumPy七格首次生成成功，原始等压残差/完整积分点均保存。
4. 首Newton运行在首格碰撞后读取层级计数时失败：把list[device array]误当单个array调用.numpy()。原log/失败JSON保留，完成格为0。
5. 修正仅两行层级计数读取，commit feb9f714；受影响文件Ruff/format通过。纯导出测试所覆盖的函数未改，不重复已绿HOST。
6. 修正后的一次完整Newton七格成功，14:55:11.4429至14:55:34.5926+08，exit0。该时间含JIT、设备构造和HOST读回，不是性能比较。
7. 保存结果比较exit1，明确返回STATIC_CROSS_REFERENCE_FAIL；没有第四/第五档、第二次数值修复或参数调整。

## 数值结果

连续场oracle：接触曲面面积 **25.03857323mm²**，轴向合力 **0.01872381272N**。A较软时轴心位于z=19.7mm，交换后19.9mm，轴压均1500Pa。参考曲面压力加权中心z分别19.73323502/19.86676498mm；它不是平面COP。

| 软的一侧 | 请求网格mm | Newton面积mm² | Newton轴向力N | 面积误差 | 力误差 | 承压面数 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2 | 18.896388 | 0.0123358193 | 24.5309% | 34.1169% | 16 |
| A | 1 | 21.805902 | 0.0162254870 | 12.9108% | 13.3430% | 48 |
| A | 0.5 | 23.420845 | 0.0180865076 | 6.4609% | 3.4037% | 196 |
| B | 2 | 18.909797 | 0.0123336748 | 24.4773% | 34.1284% | 16 |
| B | 1 | 21.809756 | 0.0162128481 | 12.8954% | 13.4105% | 48 |
| B | 0.5 | 23.421254 | 0.0180807369 | 6.4593% | 3.4345% | 196 |
| 零重叠 | 0.5 | 0 | 0 | 不适用 | 不适用 | 0 |

独立NumPy最细档面积误差0.253099%、力误差0.0007652%；最后两档面积变化0.720442%、力变化0.010506%，通过原门。等压、轴点和手算/独立积分测试通过。

Newton最细档压力加权中心误差最大约0.000684mm，小于0.2mm。材料交换的最细面积差0.001746%、力差0.031916%、中心镜像误差0.000146mm，通过对应门。实际七次查询均无overflow；共520个非零承压面完整保留。双方同点相反载荷与共同原点力矩相消是导出积分的恒等检查，不冒充独立动态守恒验证。

**未通过项**：最细面积误差约6.46%>5%；Newton最后两档面积变化最大6.8953%、力变化最大10.3308%>5%。粗档在CSV中的COARSE_OBSERVATION只表示完整观察和结构检查，不表示粗档精度合格。

## 如何看图

[交互查看器](../../artifacts/area-pressure-newton-20261115-01/results/index.html) · [28秒说明视频](../../artifacts/area-pressure-newton-20261115-01/results/two-sphere-pressure-comparison.mp4) · [预览图](../../artifacts/area-pressure-newton-20261115-01/results/preview.png)。

1. 左：先认清两个球。A在下、B在上，圆内数字是压力形状参数；绿色箭头表示积分受力方向，位置是预先规定的。
2. 中：看接触界面偏向哪边。青色是独立参考，橙色是Newton点；更软一侧承担更大压入。纵向采用单独刻度，实际放大倍数随画布显示（1440px视口约9.2倍）。
3. 右：看哪里承载。色标固定0–1500Pa，中间压力高；白十字是压力加权中心的俯视位置。圆点是积分点，不是连续覆盖的渲染网格。
4. 切换网格，只改变离散精度；切换软硬，只交换材料；最后一格恰好相切，应无正面积/载荷。视频每格4秒，总28秒，不是28秒动力学。

浏览器七格切换、播放/暂停、三画布检查通过，pageerror为0；仅favicon请求404。视频1440×1000、30fps、840帧，由保存结果渲染，未重跑物理。

## 下一步：停止追网格，继续可否证的功能实验

- 本批V03外部精度资格保留未通过。若要修此项，先在**已有raw数据**中区分SDF表示、等压面/边界截取、dA/压力积分的误差；不直接增加网格档或调E追绿，也暂不移植这个前端到TYX。
- 接触实验下一独立节点为原计划**V04分布摩擦/扭转**：固定圆盘与双盘holdout、真实面积积分、共同原点力矩、反向转动和摩擦功率。先登记正则化律及参数再执行；不能用一个中心力代替分布力偶。此项不需要重复V00–V03。
- 随后V05/V06才验证同一步的状态/载荷共同更新、冲击和卸载；只有静态图不能进入机器人承载验收。
- T-smoke真实语义共享PPO仍是独立主线，不等待本接触前端达到更高精度。本批没有训练更新、Drake执行或runtime集成。

当前Newton路由只对双有限压力场做本批资格；V02刚平面外部对照仍待支持rigid-compliant的参考工具。没有用极大第二模量伪装刚平面。

## 证据与复现

完整本机证据目录：artifacts/area-pressure-newton-20261115-01。命令、guard、raw exit、源前后哈希、首失败、七格输入/原始输出及保存比较均保留。生成大文件仍ignored，不上传Git。

[完整比较JSON](../../artifacts/area-pressure-newton-20261115-01/results/comparison.json) · [CSV](../../artifacts/area-pressure-newton-20261115-01/results/summary.csv)。
JSON SHA256 d1f7989e6955870ed2f06d95af8b320887d754bc245f1e64a23e43a089ebabb0。
MP4 SHA256 20711fc66c4b8e9224ca4965327bda7c993c2039e8854793f66590cc4aeb3e02。

复现入口（使用全新输出路径，命令严格串行；不要求重复本次已完成组）：

```powershell
python -m pytest tests/test_area_pressure_two_sphere.py tests/test_area_pressure_newton_export.py -q -p no:cacheprovider
python -m experiments.area_pressure.run_two_sphere_cases reference --output artifacts/new-v03-reference
.venv-newton-1.7.0/Scripts/python.exe -m experiments.area_pressure.run_two_sphere_cases newton --output artifacts/new-v03-newton
python -m experiments.area_pressure.run_two_sphere_cases compare --reference artifacts/new-v03-reference/reference.json --newton artifacts/new-v03-newton/newton.json --output artifacts/new-v03-comparison
```

当前预定比较应返回exit1；复现时不能把它解释为运行器崩溃，也不能屏蔽失败字段。
