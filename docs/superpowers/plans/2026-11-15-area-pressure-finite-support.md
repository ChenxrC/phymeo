# V05 有限支撑边缘：规定姿态参考

日期2026-11-15；分支codex/exp。本批仅R2 prescribed pose参考，前序[V04](../../development/area-pressure-torsion-report-20261115.md)已收口。所有参数在执行前固定；不接求解器、训练或TYX，不称真实滑落。沿[总计划](2026-11-15-area-pressure-contact-visual-cases.md)先完成有限边缘，再V05动态/V06双方运动。

## 模型与范围

本批明确选用有限平面上的局部仿射弹性基础层：gap(x,y)=a*x+b*y+c（正为分离），p=κ*max(-gap,0)。κ=1e7 Pa/m，层深H=2 mm，法向+z为A受力方向。支持平面z=0；接触面为两个CCW凸多边形footprint/support的交集，再裁去gap>0部分。压力与真实投影面积积分，接触点都在z=0；同点向B施反力，同原点计算反矩。
这是声明的有限弹性基础模型，不是已实现Drake/Newton完整Hydroelastic。弹性基础压力与压入关系参考[Drake参数说明](https://drake.mit.edu/doxygen_cxx/group__hydro__params.html)；其真实材料标定与动态稳定性另验。凸多边形逐半平面裁剪采用Sutherland–Hodgman路线，不把轴对齐矩形交集作为通用实现。

只支持有限平面顶面、凸投影足迹和小斜率仿射gap。没有非凸多片、侧面/背面、一般闭体碰撞、自由俯仰或静摩擦求解。当前沉入超过H必须拒绝，不把背面当顶面；合法无交叠时返回空域。κ是压力梯度，不当作Young模量。

输入每polygon 3..16点、严格凸CCW/无重复/无自交，二维坐标绝对值≤1 m；gap斜率hypot(a,b)≤0.1，κ与H正有限且不接受bool；n为整数1..8。固定几何长度容差1e-12 m、退化面积阈值1e-18 m²；只处理浮点重复点/边界，不把它称物理穿透容差或改载荷标定。输出polygon≤40点，原quad上限8192三角形/8193顶点/24576点保留；资源先检查再分配，失败不静默截断。压力非负裁剪属于声明的无粘附律；不把力强行调回mg。

## 主序列与holdout

0.5 kg块，底面40×40 mm、高20 mm；支撑80×80 mm、高10 mm，顶面z=0。g=9.81 m/s²，μ=0.5。
一次标定p0=mg/(0.04²)=3065.625 Pa，δ0=p0/κ=0.3065625 mm。主序列始终保持这个压入量和水平姿态。
位移x_j=0.002*j m，j=0..30（0至60 mm），各运行n=1,4,8三档，共93格。第0格规定零速度，其余规定A速度(+0.02,0,0)m/s、B=0；V04充分滑动库仑律，vs仍填1 mm/s（不影响充分滑动）。块COM=(x_j,0,0.01-δ0)，力矩同时保存共同world原点及由此转换到COM的法向/摩擦/总矩，不能把摩擦高度力臂藏进法向俯仰矩。g/m只用于一次压力标定与显示mg，不推动运动；没有h或真实时间，本批horizontal displacement是画面横轴。

独立解析oracle：w=0.04（x≤0.02），否则w=max(0,0.06-x)；A=0.04*w，Fn=p0*A。正面积COPx=x（x≤0.02）否则(x+0.02)/2，COPy=0；world My=-Fn*COPx；COM法向My=Fn*(x-COPx)。充分滑动Fx=-μFn；COM总矩包括cross((COPx-x,0,-COMz),(Fx,0,Fn))。x=60 mm退化线接触必须面积/力/矩0，COP=None；不补偿压力维持mg。

另外每n三格holdout（3×3=9），总102格：
1. 完全越界x=61 mm，维持原δ/速度，全部载荷0。
2. 居中抬高，gap=+0.1 mm：投影仍有1600 mm²交叠，但active area/载荷0。
3. 仿射倾斜足迹：投影仍40×40 mm，gap=0.02*x-0.0001 m、速度0。这是投影尺寸固定的独立仿射面，不冒充原块刚性转动后的精确尺寸。active x∈[-20,5]mm，area=1000 mm²、Fn=2.5N、COPx=-35/3 mm、world My=7/240 Nm。

通用几何测试另包含：两个居中等尺寸正方形相对yaw45°，交集octagon面积8*(sqrt(2)-1)*r²（r为半边长）；统一平移/转动后的面积、COP、合力/矩等变；三角形/多边形包含、顶点与边相触、支持顶点存储循环移位、输入CW/凹/星形/重复/坏dtype/资源拒绝、零gap/空域/深度越界。旋转时gap斜率和常数必须一起变换。独立测试作者不读取新实现。

## 冻结API与文件

新experiments/area_pressure/finite_support_reference.py：
build_finite_patch(footprint_xy,support_xy,*,gap_coefficients=(a,b,c),stiffness_pa_per_m=1e7,layer_depth_m=.002,subdivisions=1) -> frozen FiniteSupportPatch。
字段qualification='NUMPY_FINITE_AFFINE_LAYER_REFERENCE'、footprint_xy/support_xy/gap_coefficients（owned tuples）、stiffness_pa_per_m/layer_depth_m/subdivisions、overlap_polygon_xy/active_polygon_xy、overlap_area_m2/active_area_m2、quadrature（原SurfaceQuadrature）、pressure_pa、to_dict()。顶面normal+z，三角形内三个二阶精确点，仿射压力与力矩准确积分；fan三角形按n细分，不归一化dA。通用API不含位置/质量/案例标签特判。

新tests/test_area_pressure_finite_support.py：全部独立oracles/负控。
父新增run_finite_support_cases.py、finite_support_viewer.html、报告；旧reference/friction源码与测试不改。root唯一串行执行owner；复用scene实现作者与training-review测试作者，隔离原worktree纯文本，可detached提交；不要新建助手/分支。执行仅新HOST/固定102格，所有import/静态/数值/视频fresh guard，不复跑V00–V04已绿组。

## 验收与解释

实现门：每非零case面积/Fn/world与COM矩绝对误差≤1e-10 SI加相对1e-8，COP≤1e-9 m；这类平面仿射积分是低阶解析验证，不把阈值外推动力学。三档全部报告，最后两档亦满足相同门。作用反作用力/矩≤1e-12 SI；摩擦功率≤1e-12W；零域精确零及COP=None；所有点在两polygon和active半平面内（1e-12长度容差）。所有102格保存完整输入、polygon、quad/dA、pressure、双侧力/矩和预期值，禁止只挑成功帧。
最多一次有依据实现修复；若原声明模型不适用，保留失败，不移动几何/质量/阈值。安装/外部engine/整机训练不在本批。

可视化基于保存数据：俯视有限支撑/块足迹/真实active压力区/COP；侧视块的规定位置（压入可注明放大）；曲线显示随位移的面积、Fn与mg、COP和COM法向俯仰矩；不伪造块翻倒或自由运动。102格选择器/31主序列播放、关键帧MP4与JSON/CSV。明确“PRESCRIBED_POSE / 非自由动力学”，压力色标固定0..5kPa，零域无COP标记。体重只作比较，支撑下降是固定压入的模型输出，真实下落/旋转留给下一动力学批。


执行前域澄清：深度门只检查footprint∩support上的顶点最小gap；support外的端点不参与。active定义为正压力支撑闭包：overlap上最小gap>=0时active polygon/quad/active area为空/0，投影overlap保留；跨零时才裁gap<=0，保留零压边界。全zero-gap不报告正active area。已同步两独立作者，尚无项目执行。

## 2026-11-15 执行收口

数值源码fdf05a42，首次14新HOST/102固定格通过，34组网格比较通过；两处import和格式修正的完整AST与原作者同，无功能修复/数值重跑。102完整结果与媒体共114文件已绑定，详[实际报告](../../development/area-pressure-finite-support-report-20261115.md)。资格V05_REFERENCE_PASS，仅本计划的规定姿态有限顶面域。

本批全部作者及串行执行槽已释放。下一接触节点是V05/V06独立动力学共同发布，不能再以增加静态网格代替真实运动；原h=1/0.5 ms与总计划工程门保留。V03外部失败不改写，TYX集成/训练未在本批执行。
