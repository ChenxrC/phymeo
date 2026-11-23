# 第一轮实际取证结论

父串行执行：8 HOST通过；near/cp4/vr2各实际14→15两次四通道capture、读回及独立
saved-data compare均exit0。作者只读真实JSON，未运行项目。证据根为父仓
`artifacts/rendering/nov24/`；各原报告不修改。

| 证据 | SHA256 |
|---|---|
| edge-comparison-01.json | 1759f171ce208fb3ff38f5b9253782e11f4a41d366d12c04d4443a7ead81702f |
| edge-near-01/summary.json | 0cf0be082ff3a25c2d0190c4653cadf4851244bb92e00742991871e600103d86 |
| edge-cp4-01/summary.json | a83848c8bbd454f72604bbc2ace000503cd3c379bfb1bbcffc76aac5e0c5fff99 |
| edge-vr2-01/summary.json | 016d8d4f9551a59b7a33e9476595f60533b4258cf79209b7b63d3f5784ed6d21 |

固定69像素三档均68个supported、1个geometry fallback；53像素只1个anchor，15个
仅2个，合计只涉及6个真实anchor，FP32门边界不确定为0。单SH回退不是主要误差来源。

| 模式 | 对seed1 RMSE | 对seed2 RMSE | signed Y(seed1) | signed Y(seed2) |
|---|---:|---:|---:|---:|
| near120 compact4 | 0.041142911345 | 0.040759647634 | -0.026756664041 | -0.026072586151 |
| unlimited compact4 | 0.041142911345 | 0.040759647634 | -0.026756664041 | -0.026072586151 |
| old sparse2 | 0.052674349739 | 0.052563287251 | -0.039954275621 | -0.039270197730 |
| full-rate16 + original guided | 0.007561580975 | 0.007714118898 | -0.000821757301 | -0.000137679411 |

这是显示RGB8逆sRGB误差，不是scene radiance。两独立1024spp参考互差RMSE为
0.003671707455，仍是有限参考而非精确真值。

near/cp4的69像素RGB逐字相同，但全图RGB、所选实际scene-linear总色和CPU incoming
不全相同；不能把相同显示码写成物理输运完全相同。两个4-path方案在这组像素均43个
重建incoming为0，52像素signed Y为负、17个为正，主要表现为偏暗。6个被共享anchor
位于x=790，y=478/482/486/490/494/498；前三者中的478/482以及490 incoming为0，
其余三个约0.116中性值。该坐标仅解释已保存现象，绝不成为着色规则。

证据支持的优先方向是：薄几何只有一列少量低样本支持，零/高样本值被复制或共享。
尚不能仅凭这些数据区分每个anchor的有限样本噪声与接收点不同的二次可见性。
120mm与无限程的本组显示相同，局部光程并未在此显示误差上带来可见改善；
old sparse2另有路径数/低率raster差异，不能作为纯局部模型消融。

两捕获无暖身、debug读回另计：pose15 near/cp4/vr2墙钟分别约25.036/17.729/20.294ms。
这些是诊断时刻，不作稳态性能比较或速度保证。后续边缘补样必须独立量测真实工作量。
