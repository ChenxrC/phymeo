"""Reuse the existing verified media/page tool for two saved motion candidates.

No renderer runs here. Derived receipts preserve original source hashes. Actual
ffprobe/decode/PNG comparison is delegated to lighting_preview_delivery/build.py.
"""
import argparse
from hashlib import sha256
import html
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
FIELDS=("pose_index","frame_key","time_s","state_sha256","camera_sha256")


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def measured_timing(frames,saved):
    values=[f["capture_ms"] for f in frames]
    require(len(values)==300 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values),
            "delivery needs300 actual completed capture durations")
    ordered=sorted(values);mean=sum(values)/len(values)
    actual=dict(count=300,mean_ms=mean,p50_ms=ordered[149],p95_ms=ordered[284],max_ms=ordered[-1],completed_fps_from_mean=1000/mean)
    require(all(k in saved and math.isclose(saved[k],v,rel_tol=1e-12,abs_tol=1e-10) for k,v in actual.items()),
            "saved timing differs from actual per-frame completed wall times")
    return dict(actual,fps=actual["completed_fps_from_mean"])


def same_inputs(a,b):
    require(len(a)==len(b) and all(all(x[k]==y[k] for k in FIELDS) for x,y in zip(a,b,strict=True)),
            "comparison has different actual frames, camera or source times")


def cost_table(lanes):
    rows=[]
    for lane in lanes:
        for group in lane["measured_groups"]:
            t=group["timing"]
            rows.append("<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in (
                lane["label"],group["round"],group["channels"],f'{t["fps"]:.2f}',f'{t["mean_ms"]:.2f}',
                f'{t["p95_ms"]:.2f}',f'{t["max_ms"]:.2f}',lane["camera"]+" / "+lane["look"],lane["quality_status"]))+"</tr>")
    return ('<section><h2>真实捕获成本与审计状态</h2><p>FPS = 1000 / 完成帧墙钟均值。P95为最近秩；'
        'RGB与四通道、两轮分别列出，不以合并值替代。影片按30 FPS播放，播放速度不是实测吞吐。'
        '质量状态保持原审计结论，不能从画面好看或媒体验证成功推出通过。</p>'
        '<div style="overflow:auto"><table><thead><tr><th>方案</th><th>轮</th><th>通道</th><th>实测FPS</th>'
        '<th>均值ms</th><th>P95 ms</th><th>峰值ms</th><th>镜头 / 灯光</th><th>画质审计</th></tr></thead><tbody>'
        +"".join(rows)+"</tbody></table></div></section>")


def load_tool(path,name):
    path=Path(path).resolve(strict=True)
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,"explicit delivery module missing")
    module=importlib.util.module_from_spec(spec)
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(path.parent));spec.loader.exec_module(module)
    finally:sys.path[:]=prior
    return module


def bound_encoded_movie(view,media,row):
    """Bind the encoder's saved name, never reconstruct historical filenames.

    A local-distance suffix can be newer than this consumer. Its original capture
    label, full borrowed receipt and every PNG remain mandatory source evidence.
    """
    require(view.get("schema")=="phymeo-panda-motion-preview-v1" and view.get("status")=="complete"
            and view.get("quick") is False,"unknown or incomplete encoded source view")
    require(len(view.get("runs",[]))==1,"encoded view must identify one designated capture")
    borrowed=view["runs"][0];profile=borrowed.get("preset")
    require(isinstance(profile,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",profile) is not None,
            "unsafe saved media profile")
    require(view.get("requested_presets")==[profile],"encoded view preset declaration differs")
    require({k:v for k,v in borrowed.items() if k not in ("receipt","receipt_sha256","preset")}==
            {k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")},
            "encoded view differs from the original actual capture")
    require("profile_label" not in row or row["profile_label"]==profile,
            "saved encoded profile differs from the actual capture label")
    movies=[m for m in media.get("movies",[]) if m.get("file")=="panda-"+profile+".mp4"]
    require(len(movies)==1,"no unique saved full movie for the bound encoded preset")
    movie=movies[0]
    require(movie["source_png_sha256"]==[f["png_sha256"] for f in row["frames"]],
            "movie source PNG sequence differs")
    return profile,movie,borrowed

def source_lane(root,media,score,label,run):
    root=Path(root).resolve(strict=True);media=Path(media).resolve(strict=True)
    report=read(root/"summary.json")
    require(report["schema"]==run.SCHEMA and report["status"] in ("complete","partial_failure") and not report["quick"],
            "requires a finalized actual full300-state motion report")
    require(digest(root/"inputs.json")==report["inputs_sha256"],"saved inputs changed")
    rows=[x for x in report["rounds"][0]["runs"] if x["name"]==report["film"]]
    require(rows,"no designated actual movie")
    row=next((x for x in rows if x["channels"]=="rgb"),rows[0])
    path=(root/row["receipt"]).resolve(strict=True)
    require(path.is_relative_to(root) and digest(path)==row["receipt_sha256"],"actual movie receipt changed")
    require(read(path)=={k:v for k,v in row.items() if k not in ("receipt","receipt_sha256")},"summary/receipt differs")
    require(row["status"]=="complete" and len(row["frames"])==300,"movie capture is incomplete")
    run.identity_audit(row["frames"],range(300))
    derived_profile=run.media_profile_name(row,report["contract"],camera=report.get("camera","detail"),look=report.get("studio_look","recorded"))
    bound_media=read(media/"media.json")
    require(bound_media["schema"]=="phymeo-panda-motion-preview-media-v1","unknown saved media contract")
    # The media encoder deliberately uses a new immutable view of the real receipt.
    view=media.with_name(media.name+"-source-view")/"summary.json"
    view_doc=read(view)
    require(digest(view)==bound_media["summary_sha256"] and view_doc["source_sparse_summary_sha256"]==digest(root/"summary.json"),
            "video media view is not bound to this real source report")
    require(view_doc["inputs_sha256"]==report["inputs_sha256"] and digest(view.parent/"inputs.json")==report["inputs_sha256"],
            "encoded view input bytes differ from the actual capture")
    profile,movie,borrowed=bound_encoded_movie(view_doc,bound_media,row)
    view_receipt=(view.parent/borrowed["receipt"]).resolve(strict=True)
    require(view_receipt.is_relative_to(view.parent) and digest(view_receipt)==borrowed["receipt_sha256"],
            "encoded view receipt bytes changed")
    require(read(view_receipt)==dict(row,preset=profile),"encoded receipt lost its exact original capture binding")
    video=(media/movie["file"]).resolve(strict=True)
    require(video.is_relative_to(media) and digest(video)==movie["sha256"] and movie["full_decode"] is True,
            "saved movie bytes or full-decode proof differs")
    verification=movie["verification"]
    require(verification["frames"]==300 and verification["contiguous_timestamps"] is True and
            abs(verification["duration_s"]-10)<1e-6 and movie["actual_extent"]==row["extent"],"saved media frame/extent/clock proof differs")
    require(movie["source_png_sha256"]==[f["png_sha256"] for f in row["frames"]],"movie source PNG sequence differs")
    quality="未提供绑定的独立画质审计";quality_source=None
    if score is not None:
        score=Path(score).resolve(strict=True);value=read(score)
        require(value["source_summary_sha256"]==digest(root/"summary.json"),"quality audit belongs to a different capture")
        quality=value["status"];quality_source=dict(file=str(score),sha256=digest(score))
    groups=[]
    for rd in report["rounds"]:
        for value in rd["runs"]:
            if value["name"]==row["name"] and value["status"]=="complete":
                groups.append(dict(round=rd["index"],channels=value["channels"],timing=measured_timing(value["frames"],value["timing"])))
    return dict(label=label,profile=profile,derived_profile=derived_profile,root=root,media=media,receipt=path,video=video,video_sha256=movie["sha256"],
        frames=row["frames"],extent=row["extent"],timing=measured_timing(row["frames"],row["timing"]),measured_groups=groups,
        camera=report.get("camera","detail"),look=report.get("studio_look","recorded"),quality_status=quality,
        quality_source=quality_source,source_status=report["status"],contract=report["contract"],source_sha256=digest(root/"summary.json"),
        source_media_sha256=digest(media/"media.json"),source_receipt_sha256=digest(path),
        source_view_sha256=digest(view),source_view_receipt_sha256=digest(view_receipt))


def link_copy(source,target):
    require(not target.exists(),"new delivery target required")
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    require(digest(source)==digest(target),"copied media bytes differ")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs",nargs=2,type=Path,required=True)
    ap.add_argument("--media",nargs=2,type=Path,required=True)
    ap.add_argument("--scores",nargs=2,default=["none","none"],help="two bound audit paths or literal none")
    ap.add_argument("--labels",nargs=2,default=["候选 A","候选 B"])
    ap.add_argument("--frames",nargs="+",type=int,default=[59,60,149])
    ap.add_argument("--delivery-tool",type=Path,required=True)
    ap.add_argument("--decoded-mae-max",type=float,required=True,help="explicit PNG-to-H264-decode RGB8 MAE check; not physical quality acceptance")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--ffmpeg",default="ffmpeg");ap.add_argument("--ffprobe",default="ffprobe")
    args=ap.parse_args(argv)
    run=load_tool(HERE.parent/"preview_sparse_motion/run.py","delivery_bound_motion")
    lanes=[source_lane(r,m,None if s=="none" else s,label,run) for r,m,s,label in zip(args.runs,args.media,args.scores,args.labels,strict=True)]
    same_inputs(lanes[0]["frames"],lanes[1]["frames"])
    require(lanes[0]["extent"]==lanes[1]["extent"],"this matched-camera page requires identical native extent")
    require(args.frames and len(set(args.frames))==len(args.frames) and all(0<=i<300 for i in args.frames),"registered actual frame index required")
    output=args.output.resolve();bindings=output.with_name(output.name+"-bindings")
    require(not output.exists(),"delivery output must be new");bindings.mkdir(parents=True,exist_ok=False)
    trajectory=bindings/"trajectory.json"
    write(trajectory,dict(source_times_s=[f["time_s"] for f in lanes[0]["frames"]],
        frames=[{k:f[k] for k in FIELDS} for f in lanes[0]["frames"]],
        scope="derived exact shared source identities from two immutable capture receipts; not rewritten physical records",
        source_summary_sha256=[x["source_sha256"] for x in lanes]))
    specification=dict(trajectory_file=str(trajectory),source_times_pointer="/source_times_s",frames=args.frames,lanes=[],
        rois=[dict(label="完整分辨率视频解码帧（有损）",rect=[0,0,*lanes[0]["extent"]])])
    for i,lane in enumerate(lanes):
        receipt=bindings/f"lane-{i}.json"
        write(receipt,dict(delivery=dict(profile=lane["profile"],rgb_mean_ms=lane["timing"]["mean_ms"],
            status=lane["quality_status"],trajectory_sha256=digest(trajectory)),
            provenance={k:lane[k] for k in ("source_sha256","source_media_sha256","source_receipt_sha256","source_view_sha256","source_view_receipt_sha256","quality_source")},
            measured_groups=lane["measured_groups"],source_status=lane["source_status"],camera=lane["camera"],look=lane["look"],
            actual_candidate_contract=lane["contract"],current_generic_profile=lane["derived_profile"],
            profile_scope="saved encoded preset bound to original capture, not a regenerated filename",
            original_capture_report=str(lane["root"]/"summary.json")))
        specification["lanes"].append(dict(label=lane["label"],video=str(lane["video"]),video_sha256=lane["video_sha256"],
            receipt=str(receipt),receipt_fields={k:"/delivery/"+k for k in ("profile","rgb_mean_ms","status","trajectory_sha256")},
            examples=[dict(frame=n,path=str(lane["receipt"].parent/lane["frames"][n]["png"]),
                sha256=lane["frames"][n]["png_sha256"],decoded_rgb8_mae_max=args.decoded_mae_max) for n in args.frames]))
    manifest=bindings/"manifest.json";write(manifest,specification)
    tool=load_tool(args.delivery_tool,"bound_existing_delivery")
    original_page=tool.page
    def chinese_page(data):
        page=original_page(data).replace('<html lang="en">','<html lang="zh-CN">')
        page=page.replace('<h1>Lighting · same-trajectory comparison / 同轨迹光照对照</h1>',
                          '<h1>Panda 同轨迹预览 · 真实画面与耗时</h1>'+cost_table(lanes))
        page=page.replace('</style>','table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #3a495c;text-align:left;font-variant-numeric:tabular-nums}</style>')
        page=page.replace('Play / Pause','播放 / 暂停').replace('◀ Frame','◀ 上一帧').replace('Frame ▶','下一帧 ▶')
        page=page.replace('Motion time / 运动时刻','原档运动时刻').replace('RGB render mean:','RGB 捕获均值：').replace('Receipt status:','画质审计：')
        page=page.replace('Registered original-size crops / 登记原尺寸裁剪','登记视频解码帧 · 保留原像素尺寸')
        return page
    tool.page=chinese_page
    tool.build(manifest,output,ffmpeg=args.ffmpeg,ffprobe=args.ffprobe)
    (output/"media").mkdir();(output/"raw").mkdir()
    page=(output/"index.html").read_text(encoding="utf-8");images=[];owned=[]
    for i,lane in enumerate(lanes):
        target=output/"media"/f"lane-{i}.mp4";link_copy(lane["video"],target)
        page=page.replace(lane["video"].as_uri(),"media/"+target.name)
        owned.append(dict(file=str(target.relative_to(output)),sha256=digest(target)))
        for n in args.frames:
            frame=lane["frames"][n];source=lane["receipt"].parent/frame["png"]
            target=output/"raw"/f"lane-{i}-frame-{n:03d}.png";link_copy(source,target)
            require(digest(target)==frame["png_sha256"],"original lossless PNG changed")
            relative=str(target.relative_to(output)).replace("\\","/")
            images.append('<figure><figcaption>'+html.escape(lane["label"])+f' · 帧 {n} · 原始无损PNG</figcaption><a href="{relative}"><img style="max-width:100%;height:auto" src="{relative}"></a></figure>')
            owned.append(dict(file=relative,sha256=digest(target)))
    page=page.replace('</main>','<h2>原始无损帧图</h2><p>点击查看1:1原图。未调色、未插帧；上方视频为有损编码。0.5–2.3秒可留意夹指；镜头以实际记录为准。</p>'+"".join(images)+'</main>')
    (output/"index.html").write_text(page,encoding="utf-8")
    write(output/"delivery-binding.json",dict(schema="phymeo-sparse-motion-delivery-v1",status="saved_media_and_source_bound",
        helper_sha256=digest(__file__),delegate_file=str(args.delivery_tool.resolve()),delegate_sha256=digest(args.delivery_tool),
        manifest_sha256=digest(manifest),checks_sha256=digest(output/"checks.json"),html_sha256=digest(output/"index.html"),
        owned_assets=owned,browser_execution_verified=False,
        scope="same shared actual inputs; original per-round timings and quality status; no new render or numerical quality acceptance"))
    print(str(output/"index.html"))


if __name__=="__main__":main()