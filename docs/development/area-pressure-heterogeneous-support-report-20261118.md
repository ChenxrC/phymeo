# I3l：四世界连续支撑、加压与离层

2026-11-18，codex/exp 源码267c6e34。2 HOST声明/前置节点和1 CUDA连续窗口全部通过；真实450整批步、1800 world-step、0.45秒，唯一物理运行267.95秒。没有改参数或重跑同源绿窗。dev不变。

## 看什么

[23秒视频](../../artifacts/area-pressure-heterogeneous-support-20261118-01/media-v2/heterogeneous-support.mp4)；[可拖动450步的离线查看器](../../artifacts/area-pressure-heterogeneous-support-20261118-01/media-v2/viewer.html)；[承载峰值图](../../artifacts/area-pressure-heterogeneous-support-20261118-01/media-v2/heterogeneous-support.png)。

四格都是同一批次里的真实物体。先只受重力落下并承重，0.15秒增加1.5N向下载荷，0.30秒改为7N向上拉离。没有在阶段切换时重置或指定坐标。左侧是竖直导轨上的物体和支撑板；右侧俯视显示当前接触面与积分点，颜色是共同求解得到的牵引Fn/dA（Pa），不是输入压力场值。红线是接触Fz，绿线是gap（负值表示本模型的压入），My是关于该物体入步原点的接触矩。

世界41与305共享同一模型但初态不同，因此第一次接触分别出现在第52和26个入步。世界7质量更大、支撑板偏心4mm，因此真实接触力具有偏心矩。世界11是原生球—板接触；几何和接触律不同，只说明异构批次中两条路径可共存，不是材料精度或算法速度公平对比。

## 固定范围与结果

所有运动体是竖直slider，导轨承担横向力和转矩。此案例不是自由物体平衡、机器人抓取或策略训练。CUDA负责动力学，HOST从当前完整压力体/有限刚面生成接触面；无新增摩擦、CCD或FFT材料。

| world | 模型 | 原重力平台均值Fz | 加压平台均值Fz | 最大压入 | 末端gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 41 | .30kg居中面积支撑 | 2.939990N | 4.442981N | 1.110750mm | 7.845279mm |
| 7 | .45kg偏心面积支撑 | 4.383003N | 5.914435N | 1.314331mm | 3.649642mm |
| 305 | .30kg居中，不同入步速度/位置 | 2.942761N | 4.442981N | 1.110750mm | 7.845279mm |
| 11 | 原生球—板 | 不套面积模型门 | 不套面积模型门 | 不作同模型比较 | 7.963378mm |

面积平台目标分别2.943/4.443N和4.4145/5.9145N，两个平台均值力/深度、速度全部满足预定3%/0.002m/s门。三个area world的末40步均零接触力且gap>1mm，native也真实出生并最终分离。各sample保真实dA，面积分别4e-4、3e-4m²；双方同参考点反作用通过，偏心My=-.004Fz通过。

独立逐步标量BE检查最大力误差1.077e-6N、速度误差5.085e-9m/s。另从原初态独立推进完整FP64参考，不用产品状态回填；全轨迹最大位移差9.391e-10m、速度差3.983e-8m/s、力差6.075e-6N。后者是本次测量，不作为更严格新产品精度承诺。

能量按底面相对支撑面的gap作重力势能基准，保外载做功和真实阻尼耗散；累计正残差/初始正能量最大0.1163%，小于预定2%。没有借任意绝对高度常数放大分母。

## 运行与保存

入口 `experiments/area_pressure/heterogeneous_support.py --output ABS/fresh.json.gz`，默认且仅允许CUDA。现有root .venv-training-cuda313可运行。全部模型参数/资产/阶段固定在case_declarations与本批计划，不按结果调节。失败保存已接受前缀；输出必须fresh绝对gzip路径，不覆盖原证据。

离线入口 `experiments/area_pressure/plot_heterogeneous_support.py --input ABS/trajectory.json.gz --output ABS/fresh-directory`，不导入项目求解器、不重跑物理；只接收完整450步。视频46保存帧×2fps=23秒，不插值；HTML全部450步。PNG已目视核，首版标题挤压后仅调整排版并重新离线渲染；保留原版本。浏览器检查0/149/299/449、播放/暂停，最后离层可见；只有favicon 404，没有脚本错误。临时localhost查看服务已关闭。

证据 `artifacts/area-pressure-heterogeneous-support-20261118-01`：06-full 3pass/RAW_EXIT0；11/12三文件Ruff/format通过；10/13两次离线渲染RAW_EXIT0。源执行前后SHA相同，只有renderer之后修排版；模型/测试未再运行。完整trajectory gzip2296177B，SHA256 d3bb265bd0b09fd9f6ff1a2c62a7c424badc454cb30634fd76aa00e735a6e0d0。`checked-summary.json`保存全部门、完整独立FP64参考和逐步误差；render-receipt绑定原轨迹/renderer/三个媒体hash。

下一按[07:10计划](../superpowers/plans/2026-11-18-area-pressure-through-0710.md)完成I3m单次读回多body查询，再I3n可携带压力资产。当前异构语义网络/PPO、面积摩擦、全GPU几何、FFT非局部材料和自由抓持仍未因本例完成。