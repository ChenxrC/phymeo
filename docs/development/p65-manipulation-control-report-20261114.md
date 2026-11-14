# T1a：异构 CUDA 操作诊断与视频（2026-11-14）

本批完成通用诊断控制接口、公共Env的真实CUDA轨迹采集和8画面视频连接。
**执行完整，任务成功0/8；没有共享策略、语义编码模型或optimizer，不能称训练成功。**
原任务、目标、物体初态、质量、摩擦、solver和成功阈值全部保留。

## 实现与验证

- 原作者控制提交69a49c3、视频8dcb987由父正常导入；集成采集产品5a281455，最终视频排版3bed01fd。
- NativeJointPD按joint name解析原qpos/DOF/actuator地址，CPU scratch RNE提供bias，PD输出进入原motor限幅交集；不向目标FREE物体施加额外力或修改状态。
- solve_native_point_ik使用解析Jacobian和固定预算阻尼最小二乘，保留原关节范围和全部未选中q；不收敛仅报告stalled/iteration-limit。
- CPU用于编译、控制诊断和离线渲染运动学；实际物理只走MuJoCo-TYX CUDA。运行中禁止CPU mj_step/step1/step2。
- 12个新HOST节点通过（1.44s）；包括独立native bias、乱序地址、输入不变、限幅、不支持执行器负控、IK独立重投影和不可达残差。
- 原八场景2200个公共Env动作步、17600 world-step完成，h=.001、实际2.2s；采集wall约245.998s。这包含HOST控制、观察和逐步保存，不能作为求解器性能比较。
- 每步保存raw动作、native通道、控制诊断、同入步SuccessFrame/结果、完整公共接触身份/载荷与q/v；221个回放帧含初态、每10步及末态。
- 保存审计核完整步序、8world membership、原source SHA、state/trace SHA、tick/time/dimensions/finite、SuccessFrame入步相位与最终Goal结果，一次通过。
- 所有原已绿套件未重跑；没有安装、build、push或CI操作。

## 真实结果

| world | 构型 / 任务 | 路线 | 本次结果 |
| --- | --- | --- | --- |
| 0 | RRPR / 抓球 | IK收敛，真实闭指抬升 | 第993步两侧法向力均超过.05N，第1449步lifted；离台峰54.513mm，双指峰各1.117N，随后滑落；稳定保持失败 |
| 1 | RRRR / 抓盒 | 低位TCP未收敛 | 保持原机器人参考；未执行抓取，不能判全任务无解 |
| 2 | 5R / 放长块 | 低位TCP未收敛 | 保持原参考；未执行放置 |
| 3 | 6R / 放胶囊 | 低位TCP未收敛 | 保持原参考；未执行放置 |
| 4 | 7R / 放复合块 | 低位TCP未收敛 | 保持原参考；未执行放置 |
| 5 | RR / PushT | 最终推点未收敛 | 保持原参考；未执行推行 |
| 6 | RRR / PushT | 最终推点未收敛 | 保持原参考；未执行推行 |
| 7 | PPPR / PushT | IK收敛，真实推行 | 对象x从.51999998到.53714955m，原目标位置误差42.867mm、yaw误差.10746rad、IoU .19214；未成功 |

world0的1.699s入步仍有两侧约1.094N法向力和零台面支撑，但球在指垫中相对下滑；
2.199s已无指接触并重新落到台面。全部关节均无控制限幅饱和，不能把失败直接归咎于扭矩不够。
现有证据尚不能区分摩擦/接触离散、接触姿态与持握反馈各自影响；下批先查同一保存轨迹的切向力、相对滑移和指垫边缘关系。
world0底部最小值约-6.108mm也原样记录，不隐藏数值穿透。
world7发生约17.15mm的x移动和偏转，既不是未传力，也不是原目标成功。

## 首失败与一次修正

首次Ruff仅2个import/style错误、format4文件需要排版，修复后精确4文件检查通过。原raw/log保留。
09预检只有world0/7路线全部收敛。原机器人指垫z半高40mm；直接追矮物体中心会让pad进入台面。
首物理前仅将机器人TCP接近高度改为max(物体静置中心z,45mm)，给pad5mm台面余量，抬升/放下参考同步；不改物体或Goal。
修正后的六条未收敛路线仍完整记录。没有为获得视频而不断调gain、改场景或重跑CUDA。
首视频的多行指令与状态重叠；仅改为逐行固定纵坐标后重新编码，同源RGB帧hash完全相同，未重跑物理。

## 视频与证据

最终视频：[八世界CUDA诊断](../../artifacts/mujoco-tyx-manipulation-control-20261114-01/video-02/phymeo-eight-manipulation-cuda.mp4)。
1920×1080、30fps、221帧、7.366667s，回放约.2986倍速；不是实时吞吐。
已检查中间、抬升及末帧：全部构型/目标site可见，长指令不重叠，所有Goal均标not met。
画面明确写有Diagnostic controller - NOT a trained policy，六条未收敛路线标ik-incomplete-hold。
保存端点直接重放，不插值或修改物体轨迹，不用CPU推进物理。

证据根：artifacts/mujoco-tyx-manipulation-control-20261114-01。
- 08-host：12passed；12-cuda-eight-worlds：RAW_EXIT0。
- cuda-01/manifest.json、summary.json、control-plan.json、final.checkpoint。
- cuda-01/trace.jsonl.gz、states.jsonl.gz、saved-audit.json、selected-control-phases.json。
- 13-saved-audit、18-saved-phases-visual-frames；video-02/render-receipt.json及三张检查图。
- trace SHA22240140864a820dc90d4fd6f3a4eb1415a6a5105282200b774ef3869194aec7。
- states SHA3e66840fa6739c67b130545f9ec038e1a52923aabe9790c84445d04beb4b3d24。
- video SHA419b0a42216f54263597d63b183d304c273b8b75159901c87f9d5a8b36f741dc。

## 结论和后续

通用控制接口、诊断采集和视频连接可复用；T1的抓/放/PushT成功验收仍未完成。
按[主计划第15节](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)补接触姿态、可达路线与持握诊断，再扩8例/holdout。
本批无新执行owner，两个原作者已释放；没有自动启动训练或新的周期任务。
