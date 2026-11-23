# 已保存 Panda 运动的中文离线交付适配

本目录调用现有 `lighting_preview_delivery/build.py`，复用原300帧 H264 / 30Hz
PTS / 全部指定帧解码与原PNG的实际核验、原播放器。不调用renderer。

```powershell
# workdir C:/Users/38347/.codex/worktrees/render-motion-preview/phymeo
python -m unittest discover -s rendering/examples/preview_sparse_delivery -p test_deliver.py -v
python rendering/examples/preview_sparse_delivery/deliver.py --runs RUN_LOCAL_120MM RUN_UNLIMITED_CP4 --media MEDIA_LOCAL_120MM MEDIA_UNLIMITED_CP4 --scores LOCAL_AUDIT.json UNLIMITED_AUDIT.json --labels "720p 近场120mm" "720p 无限程cp4" --frames 59 60 149 --delivery-tool C:/Users/38347/.codex/worktrees/render-secondary-budget/phymeo/rendering/examples/lighting_preview_delivery/build.py --decoded-mae-max 3 --output NEW_OFFLINE_PAGE
```

参数中的 RUN/MEDIA/SCORE 是实际路径，不会生成或替换原始结果。某一路没有独立
审计时，其 `--scores` 对应项写 `none`，页面明确“未提供绑定的独立画质审计”。
`--labels` 只是人为命名，真实profile、source contract与所有文件SHA保留在原收据；
不能用标签改变实际光程/采样声明。`--decoded-mae-max` 是PNG与H264解码的明确RGB8
媒体比较容限，例值3需实际执行；它不是物理画质门，失败不得篡改原图。

输入须为本批 `preview_sparse_motion` 完整300帧指定影片；可容纳其他分组失败而该组
完整的原报告，但原 `partial_failure` 留在派生收据。两路逐帧实际state/key/time/camera
必须相同，native分辨率相同。detail与sequence不能当同视角相互比较。

输出目录拒绝覆盖，另建 `NEW_OFFLINE_PAGE-bindings` 保存派生交付收据：

- `index.html`：中文说明，真实每轮RGB/all均值FPS、均值/P95/峰值ms、镜头、look及
  原画质审计状态。FPS来自1000/实际capture墙钟均值，播放30Hz不作吞吐承诺。
- `media/lane-0.mp4`、`lane-1.mp4`：原视频逐字硬链接（跨盘则复制），便于整体移动。
- `raw/lane-0-frame-059.png`等：原无损PNG逐字复制，不调色/缩放/插帧。
- 原工具的 `checks.json`、probe、解码帧和日志；`delivery-binding.json` 再绑定
  实际工具SHA、原影片/收据/输入SHA、新HTML和内置媒体SHA。

原工具仍会展示视频解码帧（有损）；新增原PNG区明示两者差别。浏览器实际回放
仍待父验收，本入口不把成功生成HTML称为浏览器通过。原始数据始终不修改。
编码名称以已保存的 `*-source-view/summary.json` 为准：完整借用行只允许增加preset、
替换派生receipt路径/hash；原capture的profile_label、时间字典、300帧身份及PNG顺序
保持逐项相同。source-view及其receipt/inputs必须通过原media的SHA和原source绑定。
因此真实近场`local-120mm`后缀不会被旧通用命名器丢掉；也不接受任意改名或另一场景
影片。`current_generic_profile`只记录当前通用函数的推导结果，不用它重命名历史文件。
