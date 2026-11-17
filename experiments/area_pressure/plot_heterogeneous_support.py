"""Offline Chinese playback of the saved I3l CUDA trajectory, without physics.

Only accepted-input poses, contacts and solved wrenches are drawn. This module
imports no Phymeo/MuJoCo/TYX code. The MP4 repeats no synthesized motion: each
video frame is one saved record; the HTML slider exposes every saved record.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ORDER = (41, 7, 305, 11)
PHASES = (
    (150, "gravity", 0.0, "落下与承重"),
    (300, "loaded", -1.5, "向下加压"),
    (450, "release", 7.0, "向上拉离"),
)
FONT = Path("C:/Windows/Fonts/msyh.ttc")
TITLE = "四个独立世界：落下 → 承重 → 加压 → 拉离"
SCOPE = "CUDA 动力学 + HOST 自动有限面｜竖直导轨承担横向力矩｜非训练策略"
QUALIFICATION = "不同几何与接触律，仅展示混合路径；不是材料精度的公平比较"
LABELS = {
    "soft": "面积压力 · 居中支撑",
    "offset": "面积压力 · 偏心支撑",
    "native": "原生接触 · 球体",
}


def _require(ok, message):
    if not ok:
        raise ValueError("saved support playback: " + message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "nonfinite/non-numeric value")
    return float(value)


def _vector(value, width):
    _require(type(value) is list and len(value) == width, "vector shape differs")
    return [_number(v) for v in value]


def _reject_constant(value):
    raise ValueError("nonfinite JSON token: " + value)


def _load(path):
    path = Path(path)
    _require(path.is_absolute() and path.is_file(), "input must be an existing absolute gzip file")
    _require(path.stat().st_size <= 256 * 1024 * 1024, "compressed input exceeds 256 MiB")
    compressed = path.read_bytes()
    _require(len(compressed) <= 256 * 1024 * 1024, "compressed input exceeds 256 MiB")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
        raw = stream.read(512 * 1024 * 1024 + 1)
    _require(len(raw) <= 512 * 1024 * 1024, "uncompressed input exceeds 512 MiB")
    payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    _require(payload.get("schema") == "heterogeneous-area-support-v1", "unsupported schema")
    _require(
        payload.get("status") == "complete", "failed/incomplete prefixes are not a completed video"
    )
    _require(payload.get("mode") == "cuda", "this gallery requires the saved CUDA run")
    _require(
        payload.get("h") == 0.001 and payload.get("duration") == 0.45, "different fixed time window"
    )
    _require(payload.get("world_order") == list(ORDER), "world order differs")
    records = payload["records"]
    _require(
        type(records) is list and len(records) == 450, "complete 450-record trajectory required"
    )
    _require(set(payload["worlds"]) == set(map(str, ORDER)), "world declarations differ")
    stages = payload["stages"]
    _require(type(stages) is list and len(stages) == 3, "three saved load stages required")
    for saved, (end, _, force, _) in zip(stages, PHASES, strict=True):
        _require(
            _number(saved["end_s"]) == end * 0.001 and _number(saved["force_n"]) == force,
            "saved stage declaration differs",
        )
    panels = []
    for key in ORDER:
        declaration = payload["worlds"][str(key)]
        model_key = declaration["model"]
        _require(model_key in LABELS, "unexpected fixed model")
        model = payload["models"][model_key]
        _require(model["shape"] in ("cube", "sphere"), "unsupported display shape")
        _require(_number(model["body_half_height_m"]) == 0.01, "fixed body size differs")
        names = payload["model_body_names"][model_key]
        _require(names.count("pressure") == names.count("plate") == 1, "ambiguous body identity")
        half = _vector(model["plate_half_size_m"], 3)
        _require(all(v > 0 for v in half), "invalid plate dimensions")
        panels.append(
            dict(
                key=key,
                model=model_key,
                label=LABELS[model_key],
                shape=model["shape"],
                mass=_number(model["mass_kg"]),
                damping=_number(model["damping_ns_m"]),
                plate_half=half,
                center_pressure=None
                if model_key == "native"
                else _number(model["center_pressure_pa"]),
                body_index=names.index("pressure"),
                plate_index=names.index("plate"),
            )
        )
    frames, previous = [], {}
    traction_max = 0.0
    for index, record in enumerate(records):
        end, phase, force, caption = next(p for p in PHASES if index < p[0])
        _require(type(record["step"]) is int and record["step"] == index, "missing/reordered step")
        _require(
            record["phase"] == phase and _number(record["force_n"]) == force, "stage/force mismatch"
        )
        _require(set(record["worlds"]) == set(map(str, ORDER)), "record world set differs")
        arrays = record["arrays"]
        _require(
            len(arrays["entry_body_position"]) == len(ORDER), "body position world count differs"
        )
        rows = []
        for row, panel in enumerate(panels):
            world = record["worlds"][str(panel["key"])]
            before, after = world["input"], world["output"]
            _require(
                type(before["tick"]) is int
                and before["tick"] == index
                and type(after["tick"]) is int
                and after["tick"] == index + 1,
                "tick phase differs",
            )
            tin, tout = _number(before["time"]), _number(after["time"])
            _require(
                0 <= tin < tout
                and abs(tin - index * 0.001) <= 2e-6
                and abs(tout - (index + 1) * 0.001) <= 2e-6,
                "time phase differs",
            )
            for state in (before, after):
                _vector(state["q"], 1)
                _vector(state["v"], 1)
                _require(type(state["act"]) is list and not state["act"], "unexpected activation")
            if panel["key"] in previous:
                _require(before == previous[panel["key"]], "saved input/output state chain differs")
            previous[panel["key"]] = after
            meta = arrays["entry_meta"][row]
            _require(
                len(meta) == 6 and meta[:3] == [1, index, index + 1] and meta[5] == 0,
                "public accepted-input metadata differs",
            )
            clock = _vector(arrays["entry_clock"][row], 4)
            _require(
                clock[0] - clock[1] == tin and clock[2] - clock[3] == tout,
                "public input/output clocks differ",
            )
            positions = arrays["entry_body_position"][row]
            body = _vector(positions[panel["body_index"]], 3)
            plate = _vector(positions[panel["plate_index"]], 3)
            wrench = world["body_wrenches"]["pressure"]
            _require(
                wrench["input_tick"] == index
                and wrench["output_tick"] == index + 1
                and _number(wrench["time_s"]) == tin,
                "wrench is not at input phase",
            )
            _require(
                _vector(wrench["reference_point_world"], 3) == body,
                "wrench reference is not the saved input body origin",
            )
            fz = _vector(wrench["total_force_world"], 3)[2]
            my = _vector(wrench["total_moment_world"], 3)[1]
            samples, triangles = [], []
            for pair in world["area"]:
                points, measures, loads = pair["points"], pair["dA"], pair["normal_force"]
                _require(len(points) == len(measures) == len(loads), "area sample lengths differ")
                for point, measure, load in zip(points, measures, loads, strict=True):
                    point = _vector(point, 3)
                    measure, load = _number(measure), _number(load)
                    _require(measure > 0 and load >= 0, "invalid saved area/load")
                    traction = load / measure
                    _require(math.isfinite(traction), "traction display overflow")
                    samples.append([*point, traction])
                    traction_max = max(traction_max, traction)
            for geometry in world["geometry"]:
                for triangle in geometry["surface"]["triangles"]:
                    vertices = triangle["vertices"]
                    _require(len(vertices) == 3, "triangle shape differs")
                    triangles.append([_vector(v, 3) for v in vertices])
            points = [
                _vector(c["point_world"], 3)
                for c in wrench["contacts"]
                if c["source"] == "native" and c["active"]
            ]
            _require(
                panel["model"] != "native" or not samples, "native world contains area samples"
            )
            active_points = [
                _vector(c["point_world"], 3)
                for c in wrench["contacts"]
                if c["active"] and abs(_vector(c["force_world"], 3)[2]) > 0
            ]
            _require(
                abs(fz) <= 1e-10 or bool(active_points), "nonzero load has no saved active point"
            )
            force_point = active_points[0] if active_points else body
            gap = body[2] - 0.01 - (plate[2] + panel["plate_half"][2])
            rows.append(
                dict(
                    t=tin,
                    body=body,
                    plate=plate,
                    gap_mm=1000 * gap,
                    fz=fz,
                    my=my,
                    force_point=force_point,
                    samples=samples,
                    triangles=triangles,
                    native_points=points,
                )
            )
        frames.append(dict(step=index, t=rows[0]["t"], phase=caption, external=force, rows=rows))
    fmax = max(1.0, max(r["fz"] for f in frames for r in f["rows"]) * 1.12)
    gaps = [r["gap_mm"] for f in frames for r in f["rows"]]
    return dict(
        schema="heterogeneous-support-playback-v1",
        panels=panels,
        frames=frames,
        duration=0.45,
        traction_max=max(1.0, traction_max),
        force_max=fmax,
        gap_min=min(gaps) - 0.3,
        gap_max=max(gaps) + 0.3,
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(compressed).hexdigest(),
    )


def _rgb(value):
    value = min(1.0, max(0.0, value))
    return tuple((a + (b - a) * value) / 255 for a, b in zip((220, 235, 244), (183, 37, 58)))


def _draw(fig, axes, data, index):
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    current = data["frames"][index]
    fig.text(0.5, 0.985, TITLE, fontsize=20, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.937, SCOPE, ha="center", fontsize=12, color="#334155")
    fig.text(
        0.5,
        0.914,
        f"{current['phase']}｜入步 t={current['t']:.3f} s｜广义外载 {current['external']:+g} N",
        ha="center",
        fontsize=16,
        color="#9a3412",
    )
    fig.text(
        0.5,
        0.038,
        "0.45 秒真实仿真 → 23 秒视频；46 个保存帧，无插值。全程曲线淡显，竖线标当前入步。",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.5,
        0.018,
        QUALIFICATION + "；姿态、载荷、几何均来自同一入步。箭头只示意接触力方向。",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    for col, (ax, panel, state) in enumerate(
        zip(axes, data["panels"], current["rows"], strict=True)
    ):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#d9e2ec", lw=1))
        ax.text(
            0.04,
            0.95,
            f"世界 {panel['key']} · {panel['label']}",
            fontsize=15,
            weight="bold",
            va="top",
        )
        material = (
            "无面积牵引"
            if panel["model"] == "native"
            else f"中心场压力 {panel['center_pressure'] / 1000:g} kPa"
        )
        ax.text(0.04, 0.875, f"{panel['mass']:g} kg｜{material}", fontsize=10, color="#64748b")
        size = ax.get_position()
        aspect = (size.width * fig.get_figwidth()) / (size.height * fig.get_figheight())
        sx, sz, base = 6.0, 6.0 * aspect, 0.405

        def xy(x, z):
            return (0.26 + sx * x, base + sz * z)

        body, plate, half = state["body"], state["plate"], panel["plate_half"]
        for x in (-0.025, 0.025):
            ax.plot([xy(x, 0)[0]] * 2, [0.37, 0.80], color="#cbd5e1", lw=2)
        ax.text(0.26, 0.81, "竖直导轨 · 侧视", ha="center", fontsize=9, color="#64748b")
        ax.add_patch(
            Rectangle(
                xy(plate[0] - half[0], plate[2] - half[2]),
                2 * half[0] * sx,
                2 * half[2] * sz,
                facecolor="#64748b",
            )
        )
        if panel["shape"] == "cube":
            ax.add_patch(
                Rectangle(
                    xy(body[0] - 0.02, body[2] - 0.01),
                    0.04 * sx,
                    0.02 * sz,
                    facecolor="#c9e2f4",
                    edgecolor="#25628e",
                    lw=1.7,
                )
            )
        else:
            ax.add_patch(
                Ellipse(
                    xy(body[0], body[2]),
                    0.02 * sx,
                    0.02 * sz,
                    facecolor="#e1d6ef",
                    edgecolor="#765b9c",
                    lw=1.7,
                )
            )
        if abs(state["fz"]) > 1e-10:
            point = state["force_point"]
            x, y = xy(point[0], point[2])
            ax.annotate(
                "",
                xy=(x, y + math.copysign(0.13, state["fz"])),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#b7253a", lw=2),
            )
        ax.text(0.04, 0.33, f"gap {state['gap_mm']:+.3f} mm", fontsize=12, color="#0f766e")
        ax.text(0.50, 0.43, f"Fz  {state['fz']:.3f} N", fontsize=13, color="#b7253a")
        ax.text(0.50, 0.365, f"My  {state['my'] * 1000:+.3f} mN·m", fontsize=12, color="#334155")
        # Top view of the saved finite patch and actual quadrature locations.
        scale = 8.0

        def uv(x, y):
            return (0.72 + scale * (x - plate[0]), 0.67 + scale * aspect * (y - plate[1]))

        ax.add_patch(
            Rectangle(
                uv(plate[0] - half[0], plate[1] - half[1]),
                2 * half[0] * scale,
                2 * half[1] * scale * aspect,
                facecolor="#f1f5f9",
                edgecolor="#94a3b8",
            )
        )
        for triangle in state["triangles"]:
            ax.add_patch(
                Polygon(
                    [uv(v[0], v[1]) for v in triangle],
                    closed=True,
                    fill=False,
                    edgecolor="#718096",
                    lw=0.6,
                )
            )
        for point in state["samples"]:
            ax.scatter(
                *uv(point[0], point[1]),
                s=40,
                color=_rgb(point[3] / data["traction_max"]),
                edgecolors="#9f5260",
                linewidths=0.25,
                zorder=5,
            )
        for point in state["native_points"]:
            ax.scatter(*uv(point[0], point[1]), s=35, color="#765b9c", zorder=5)
        if not state["samples"] and not state["native_points"]:
            ax.text(0.72, 0.67, "当前无接触样本", fontsize=9, ha="center", color="#64748b")
        label = (
            "原生接触点 · 无面积牵引"
            if panel["model"] == "native"
            else "俯视：求解牵引 Fn/dA（Pa）"
        )
        ax.text(0.72, 0.84, label, fontsize=9, ha="center")
        if panel["model"] != "native":
            for stripe in range(40):
                ax.add_patch(
                    Rectangle(
                        (0.59 + stripe * 0.0065, 0.49),
                        0.0065,
                        0.018,
                        facecolor=_rgb(stripe / 39),
                        edgecolor="none",
                    )
                )
            ax.text(0.59, 0.52, "0", fontsize=8)
            ax.text(0.85, 0.52, f"{data['traction_max']:.0f} Pa", fontsize=8, ha="right")
        left, right, low, high = 0.085, 0.91, 0.065, 0.255
        for tick in range(3):
            y = low + (high - low) * tick / 2
            ax.plot([left, right], [y, y], color="#e2e8f0", lw=0.7)
            ax.text(
                left - 0.018,
                y,
                f"{data['force_max'] * tick / 2:.1f}",
                fontsize=8,
                ha="right",
                va="center",
                color="#b7253a",
            )
            gap = data["gap_min"] + (data["gap_max"] - data["gap_min"]) * tick / 2
            ax.text(right + 0.012, y, f"{gap:.1f}", fontsize=8, va="center", color="#0f766e")
        for seconds in (0, 0.15, 0.30, 0.45):
            x = left + (right - left) * seconds / 0.45
            ax.plot([x, x], [low, high], color="#d4dce5", lw=0.8)
            ax.text(x, low - 0.025, f"{seconds:g}", ha="center", va="top", fontsize=8)
        for field, color, minimum, span in (
            ("fz", "#b7253a", 0, data["force_max"]),
            ("gap_mm", "#0f766e", data["gap_min"], data["gap_max"] - data["gap_min"]),
        ):
            xs = [left + (right - left) * frame["t"] / 0.45 for frame in data["frames"]]
            ys = [
                low + (high - low) * (frame["rows"][col][field] - minimum) / span
                for frame in data["frames"]
            ]
            ax.plot(xs, ys, color=color, alpha=0.25, lw=1)
            ax.plot(xs[: index + 1], ys[: index + 1], color=color, lw=1.6)
        cursor = left + (right - left) * current["t"] / 0.45
        ax.plot([cursor, cursor], [low, high], color="#1e293b", lw=1)
        ax.text(left, 0.29, "Fz / N", color="#b7253a", fontsize=9)
        ax.text(right, 0.29, "gap / mm（负值为压入）", color="#0f766e", ha="right", fontsize=9)


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>四世界连续面积接触回放</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172d45;font-family:"Microsoft YaHei",sans-serif}
main{max-width:1600px;margin:auto;padding:14px}canvas{display:block;width:100%;height:auto;background:#f8fafc;border-radius:12px}
nav{display:flex;align-items:center;gap:15px;padding:14px;background:white;position:sticky;top:0;z-index:2;border-radius:10px}
input{flex:1;min-width:80px}button{border:0;border-radius:7px;background:#1c5273;color:white;padding:9px 18px;font:inherit}
small{display:block;padding:12px;line-height:1.7;color:#526477}output{min-width:130px}
</style><main><nav><button id="play">播放</button><input id="seek" type="range" min="0" max="449" step="1" value="0"><output id="time"></output></nav>
<canvas id="view" width="1600" height="1080"></canvas><small>拖动可查看全部 450 个真实入步。播放只切换保存记录，不插值、不重算物理。曲线全程淡色背景，深色为当前之前的记录。<br>接触力与 My 均关于保存的压力体入步原点；导轨承担横向反力与转矩。源压缩文件 SHA256：<span id="sha"></span></small></main>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),C=document.getElementById('view'),g=C.getContext('2d');
const seek=document.getElementById('seek'),button=document.getElementById('play');let timer=null;
document.getElementById('sha').textContent=D.source_sha256;
function text(s,x,y,size=18,color='#243b53',align='left'){g.font=size+'px "Microsoft YaHei",sans-serif';g.fillStyle=color;g.textAlign=align;g.fillText(s,x,y)}
function line(x,y,a,b,color='#cbd5e1',width=1){g.beginPath();g.moveTo(x,y);g.lineTo(a,b);g.strokeStyle=color;g.lineWidth=width;g.stroke()}
function box(x,y,w,h,fill,stroke){g.fillStyle=fill;g.fillRect(x,y,w,h);if(stroke){g.strokeStyle=stroke;g.lineWidth=1;g.strokeRect(x,y,w,h)}}
function circle(x,y,r,color){g.beginPath();g.arc(x,y,r,0,2*Math.PI);g.fillStyle=color;g.fill()}
function color(v){v=Math.max(0,Math.min(1,v));return 'rgb('+[220+(183-220)*v,235+(37-235)*v,244+(58-244)*v].map(Math.round).join(',')+')'}
function arrow(x,y,dy){line(x,y,x,y+dy,'#b7253a',3);g.beginPath();g.moveTo(x,y+dy);g.lineTo(x-6,y+dy-Math.sign(dy)*10);g.lineTo(x+6,y+dy-Math.sign(dy)*10);g.closePath();g.fillStyle='#b7253a';g.fill()}
function panel(p,r,n,index){const ox=10+(n%2)*800,oy=140+Math.floor(n/2)*450,W=780,H=435;
box(ox,oy,W,H,'#fff','#d9e2ec');text('世界 '+p.key+' · '+p.label,ox+20,oy+31,22);
const material=p.model==='native'?'无面积牵引':'中心场压力 '+(p.center_pressure/1000)+' kPa';text(p.mass+' kg｜'+material,ox+20,oy+58,15,'#64748b');
const px=x=>ox+205+x*5500,pz=z=>oy+283-z*5500,b=r.body,t=r.plate,h=p.plate_half;
line(px(-.025),oy+90,px(-.025),oy+306,'#cbd5e1',2);line(px(.025),oy+90,px(.025),oy+306,'#cbd5e1',2);
text('竖直导轨 · 等比例侧视',ox+205,oy+82,13,'#64748b','center');
box(px(t[0]-h[0]),pz(t[2]+h[2]),2*h[0]*5500,2*h[2]*5500,'#64748b');
if(p.shape==='cube')box(px(b[0]-.02),pz(b[2]+.01),220,110,'#c9e2f4','#25628e');else circle(px(b[0]),pz(b[2]),55,'#bba3d3');
if(Math.abs(r.fz)>1e-10)arrow(px(r.force_point[0]),pz(r.force_point[2]),-55*Math.sign(r.fz));
const mx=x=>ox+565+(x-t[0])*5200,my=y=>oy+162-(y-t[1])*5200;
box(mx(t[0]-h[0]),my(t[1]+h[1]),2*h[0]*5200,2*h[1]*5200,'#f1f5f9','#94a3b8');
for(const tri of r.triangles){g.beginPath();tri.forEach((v,j)=>j?g.lineTo(mx(v[0]),my(v[1])):g.moveTo(mx(v[0]),my(v[1])));g.closePath();g.strokeStyle='#718096';g.lineWidth=.7;g.stroke()}
for(const s of r.samples)circle(mx(s[0]),my(s[1]),4,color(s[3]/D.traction_max));for(const s of r.native_points)circle(mx(s[0]),my(s[1]),4,'#765b9c');
if(!r.samples.length&&!r.native_points.length)text('当前无接触样本',ox+565,oy+162,13,'#64748b','center');
text(p.model==='native'?'原生点：无面积牵引':'俯视：求解牵引 Fn/dA（Pa）',ox+565,oy+83,14,'#475569','center');
if(p.model!=='native'){for(let k=0;k<80;k++)box(ox+475+k*2.2,oy+227,2.2,9,color(k/79));text('0',ox+475,oy+222,11);text(D.traction_max.toFixed(0)+' Pa',ox+651,oy+222,11,'#475569','right')}
text('gap '+r.gap_mm.toFixed(3)+' mm',ox+35,oy+322,18,'#0f766e');text('Fz '+r.fz.toFixed(3)+' N',ox+447,oy+275,21,'#b7253a');text('My '+(r.my*1000).toFixed(3)+' mN·m',ox+447,oy+304,18);
const L=ox+60,R=ox+714,T=oy+350,B=oy+405;for(let k=0;k<3;k++){const y=B-(B-T)*k/2;line(L,y,R,y,'#e2e8f0');text((D.force_max*k/2).toFixed(1),L-10,y+4,11,'#b7253a','right');text((D.gap_min+(D.gap_max-D.gap_min)*k/2).toFixed(1),R+8,y+4,11,'#0f766e')}
for(const t0 of [0,.15,.30,.45]){const x=L+(R-L)*t0/.45;line(x,T,x,B,'#d4dce5');text(t0.toString(),x,B+18,11,'#64748b','center')}
for(const [field,col,lo,range] of [['fz','#b7253a',0,D.force_max],['gap_mm','#0f766e',D.gap_min,D.gap_max-D.gap_min]]){for(const [end,alpha] of [[449,.25],[index,1]]){g.beginPath();for(let j=0;j<=end;j++){const f=D.frames[j],x=L+(R-L)*f.t/.45,y=B-(B-T)*(f.rows[n][field]-lo)/range;j?g.lineTo(x,y):g.moveTo(x,y)}g.strokeStyle=col;g.globalAlpha=alpha;g.lineWidth=1.6;g.stroke();g.globalAlpha=1}}
line(L+(R-L)*D.frames[index].t/.45,T,L+(R-L)*D.frames[index].t/.45,B,'#1e293b');text('Fz / N',L,T-9,12,'#b7253a');text('gap / mm（负值为压入）',R,T-9,12,'#0f766e','right');}
function draw(){const i=Number(seek.value),f=D.frames[i];g.clearRect(0,0,C.width,C.height);box(0,0,C.width,C.height,'#f8fafc');text('四个独立世界：落下 → 承重 → 加压 → 拉离',800,41,30,'#1e3a5f','center');text('CUDA 动力学 + HOST 自动有限面｜竖直导轨承担横向力矩｜非训练策略',800,74,18,'#475569','center');text(f.phase+'｜入步 t='+f.t.toFixed(3)+' s｜广义外载 '+(f.external>=0?'+':'')+f.external+' N',800,112,23,'#9a3412','center');D.panels.forEach((p,n)=>panel(p,f.rows[n],n,i));text('不同几何与接触律，非公平材料精度对照。姿态与载荷同入步；箭头长度只示意方向。',800,1062,15,'#475569','center');document.getElementById('time').textContent='step '+i+' / 449';}
seek.addEventListener('input',draw);button.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;button.textContent='播放';return}if(Number(seek.value)===449)seek.value=0;button.textContent='暂停';timer=setInterval(()=>{const next=Number(seek.value)+1;if(next>449){clearInterval(timer);timer=null;button.textContent='播放';return}seek.value=next;draw()},50)});draw();
</script></html>"""


def render(input_path, output):
    output = Path(output)
    _require(
        output.is_absolute() and not output.exists(), "output must be a fresh absolute directory"
    )
    data = _load(input_path)
    _require(FONT.is_file(), "Microsoft YaHei font is missing")
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
            "axes.unicode_minus": False,
            "figure.facecolor": "#f8fafc",
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    selected = [0] + list(range(9, 450, 10))
    peak = max(range(450), key=lambda i: sum(r["fz"] for r in data["frames"][i]["rows"]))
    image_path, video_path, html_path = (
        output / name
        for name in ("heterogeneous-support.png", "heterogeneous-support.mp4", "viewer.html")
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.8), dpi=100)
    fig.subplots_adjust(
        left=0.015, right=0.985, bottom=0.055, top=0.875, wspace=0.025, hspace=0.045
    )
    axes = tuple(axes.flat)
    try:
        _draw(fig, axes, data, peak)
        fig.savefig(image_path, dpi=150)
        with imageio.get_writer(
            video_path,
            format="FFMPEG",
            mode="I",
            fps=2,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
        ) as writer:
            for index in selected:
                for ax in axes:
                    ax.clear()
                for item in list(fig.texts):
                    item.remove()

                _draw(fig, axes, data, index)
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        plt.close(fig)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    html_path.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    receipt = dict(
        schema="heterogeneous-support-render-v1",
        qualification="SAVED_INPUT_POSTPROCESS_ONLY",
        source=data["source"],
        source_sha256=data["source_sha256"],
        records=450,
        input_phase="accepted-step-input",
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        video_record_steps=selected,
        video_fps=2,
        video_seconds=len(selected) / 2,
        simulation_seconds=0.45,
        playback_rule="saved records only; no interpolation",
        png_record_step=peak,
        html_records=450,
        chinese_font=str(FONT),
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),
        matplotlib_version=matplotlib.__version__,
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (image_path, video_path, html_path)
        },
    )
    (output / "render-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.input, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "video_seconds": receipt["video_seconds"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
