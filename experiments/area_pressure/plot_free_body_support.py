"""Offline Chinese views of one complete saved FREE-body support trajectory.

No Phymeo, MuJoCo or TYX imports, physics steps, or interpolated contact frames.
All geometry and loads use accepted-input phase. The final output at 0.150 s
is retained in the source but is not paired with the preceding input force.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ORDER = (41, 7, 305)
FONT = Path("C:/Windows/Fonts/msyh.ttc")
LABELS = ("居中落下", "倾斜支撑", "偏心支撑")
FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
TITLE = "真实自由刚体：落下承载、倾斜支撑与偏心转动"
SCOPE = "CUDA 动力学 + HOST 自动有限接触面｜完整 FREE 六自由度｜无摩擦、非抓持、非训练"
PHASE = "全部几何、载荷与曲线取同一入步；0.150 s 的最终出步不配用上一帧接触力。"


def _require(value, message):
    if not value:
        raise ValueError("FREE support playback: " + message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "finite real number required")
    return float(value)


def _vector(value, size):
    _require(type(value) is list and len(value) == size, "vector shape mismatch")
    return [_number(x) for x in value]


def _reject(value):
    raise ValueError("nonfinite JSON token: " + value)


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _corners(position, rotation, half):
    local = (
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    )
    return [
        [
            position[i] + math.fsum(rotation[i][j] * half[j] * sign[j] for j in range(3))
            for i in range(3)
        ]
        for sign in local
    ]


def _angle(q0, q1):
    # Sign- and scale-invariant orientation distance; the saved poses are not altered.
    cosine = abs(_dot(q0, q1)) / math.sqrt(_dot(q0, q0) * _dot(q1, q1))
    return math.degrees(2 * math.acos(min(1.0, max(0.0, cosine))))


def load_trajectory(path):
    path = Path(path)
    _require(path.is_absolute() and path.is_file(), "input must be an absolute existing file")
    _require(path.stat().st_size <= 256 * 1024 * 1024, "compressed input exceeds 256 MiB")
    compressed = path.read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(512 * 1024 * 1024 + 1)
    _require(len(raw) <= 512 * 1024 * 1024, "decompressed input exceeds 512 MiB")
    payload = json.loads(raw.decode("utf-8"), parse_constant=_reject)
    _require(payload.get("schema") == "free-body-area-support-v1", "unknown schema")
    _require(
        payload.get("status") == "complete" and payload.get("mode") == "cuda",
        "incomplete/failed prefixes cannot produce a completed gallery",
    )
    _require(
        payload.get("h") == 0.001
        and payload.get("duration") == 0.15
        and payload.get("steps") == 150,
        "fixed time window differs",
    )
    _require(
        payload.get("accepted_steps") == payload.get("complete_records") == 150,
        "incomplete acceptance/observation count",
    )
    _require(payload.get("world_order") == list(ORDER), "world order differs")
    names = payload["body_names"]
    _require(names.count("pressure") == names.count("plate") == 1, "ambiguous body identity")
    body_id, plate_id = names.index("pressure"), names.index("plate")
    p = payload["parameters"]
    half = _vector(p["cube_half_size_m"], 3)
    plate_half = _vector(p["plate_half_size_m"], 3)
    _require(
        half == [0.02, 0.02, 0.01] and plate_half == [0.01, 0.01, 0.0025], "fixed dimensions differ"
    )
    records = payload["records"]
    _require(type(records) is list and len(records) == 150, "150 complete records required")
    initial_q = {key: _vector(payload["initial"][str(key)]["q"], 7)[3:] for key in ORDER}
    previous, frames, traction_max = {}, [], 0.0
    for step, record in enumerate(records):
        _require(
            type(record["step"]) is int and record["step"] == step, "missing or reordered step"
        )
        _require(
            record["state_accepted"] is True and record["observation_complete"] is True,
            "partial saved observation",
        )
        _require(set(record["worlds"]) == set(map(str, ORDER)), "world set differs")
        _require(record["readback_calls"] == 1, "expected one bound public readback")
        arrays, rows = record["arrays"], []
        for index, key in enumerate(ORDER):
            row = record["worlds"][str(key)]
            before, after = row["input"], row["output"]
            for state in (before, after):
                _vector(state["q"], 7)
                _vector(state["v"], 6)
                _require(state["act"] == [], "unexpected activation")
            _require(
                type(before["tick"]) is int
                and before["tick"] == step
                and type(after["tick"]) is int
                and after["tick"] == step + 1,
                "tick phase differs",
            )
            tin, tout = _number(before["time"]), _number(after["time"])
            _require(
                abs(tin - step * 0.001) < 2e-6
                and abs(tout - (step + 1) * 0.001) < 2e-6
                and tin < tout,
                "time phase differs",
            )
            if key in previous:
                _require(before == previous[key], "broken accepted-state chain")
            else:
                _require(
                    before == payload["initial"].get(str(key)),
                    "first input differs from saved initial state",
                )
            previous[key] = after
            meta = arrays["entry_meta"][index]
            _require(
                meta[:3] == [1, step, step + 1] and len(meta) == 6 and meta[5] == 0,
                "entry metadata mismatch",
            )
            clock = _vector(arrays["entry_clock"][index], 4)
            _require(
                clock[0] - clock[1] == tin and clock[2] - clock[3] == tout, "entry clock mismatch"
            )
            position = _vector(arrays["entry_body_position"][index][body_id], 3)
            plate = _vector(arrays["entry_body_position"][index][plate_id], 3)
            rotation = [_vector(r, 3) for r in arrays["entry_body_rotation"][index][body_id]]
            plate_rotation = [_vector(r, 3) for r in arrays["entry_body_rotation"][index][plate_id]]
            _require(len(rotation) == len(plate_rotation) == 3, "rotation shape differs")
            _require(position == before["q"][:3], "drawing position is not input body origin")
            receipt = row["receipt"]
            _require(
                receipt["input_q"] == before["q"]
                and receipt["input_v"] == before["v"]
                and _number(receipt["input_time"]) == tin,
                "receipt is not the same input",
            )
            wrench = row["body_wrenches"]["body_origin"]["pressure"]
            _require(
                wrench["input_tick"] == step
                and wrench["output_tick"] == step + 1
                and _number(wrench["time_s"]) == tin,
                "wrench phase mismatch",
            )
            _require(
                _vector(wrench["reference_point_world"], 3) == position,
                "wrench reference differs from input origin",
            )
            force = _vector(wrench["total_force_world"], 3)
            moment = _vector(wrench["total_moment_world"], 3)
            samples, triangles, loads = [], [], []
            for pair in receipt["raw_contact_fields"]["area_pressure"]:
                _require(
                    pair["identity"]["compliant_body"] == "pressure"
                    and pair["identity"]["rigid_body"] == "plate",
                    "unexpected area pair",
                )
                _require(
                    pair["stamp"]["step_index"] == step
                    and _number(pair["stamp"]["sample_time_s"]) == tin,
                    "area phase mismatch",
                )
                points, measures, normal_loads, normals = (
                    pair["points"],
                    pair["dA"],
                    pair["normal_force"],
                    pair["normals"],
                )
                _require(
                    len(points) == len(measures) == len(normal_loads) == len(normals),
                    "area row count mismatch",
                )
                for point, measure, load, normal in zip(
                    points, measures, normal_loads, normals, strict=True
                ):
                    point, normal = _vector(point, 3), _vector(normal, 3)
                    measure, load = _number(measure), _number(load)
                    _require(measure > 0 and load >= 0, "invalid area or normal load")
                    _require(
                        abs(normal[0]) + abs(normal[1]) + abs(normal[2] - 1) <= 1e-5,
                        "this plate CoP display requires parallel +z normals",
                    )
                    traction = load / measure
                    _require(math.isfinite(traction), "traction overflow")
                    traction_max = max(traction_max, traction)
                    samples.append([*point, traction])
                    loads.append((point, load))
            fn = math.fsum(load for _, load in loads)
            _require(
                abs(fn - force[2]) <= 0.002 + 0.002 * abs(force[2]),
                "saved sum Fn and public Fz disagree",
            )
            cop = (
                None
                if fn == 0
                else [math.fsum(point[i] * load for point, load in loads) / fn for i in range(3)]
            )
            for geometry in row["geometry"]:
                for triangle in geometry["surface"]["triangles"]:
                    _require(len(triangle["vertices"]) == 3, "triangle shape differs")
                    triangles.append([_vector(point, 3) for point in triangle["vertices"]])
            rows.append(
                dict(
                    key=key,
                    position=position,
                    rotation=rotation,
                    plate=plate,
                    cube=_corners(position, rotation, half),
                    plate_corners=_corners(plate, plate_rotation, plate_half),
                    samples=samples,
                    triangles=triangles,
                    force=force,
                    moment=moment,
                    fn=fn,
                    fz=force[2],
                    z_mm=1000 * position[2],
                    cop=cop,
                    angle_deg=_angle(initial_q[key], before["q"][3:]),
                )
            )
        _require(
            all(record["worlds"][str(key)]["input"]["time"] == tin for key in ORDER),
            "world clocks differ",
        )
        frames.append(dict(step=step, t=tin, rows=rows))
    for key in ORDER:
        _require(
            previous[key] == payload["final"][str(key)], "final state does not close saved chain"
        )
    chosen = max(range(len(frames)), key=lambda i: math.fsum(r["fn"] for r in frames[i]["rows"]))
    visible = [
        point
        for frame in frames
        for row in frame["rows"]
        for point in row["cube"] + row["plate_corners"]
    ]
    for frame in frames:
        for row in frame["rows"]:
            if row["cop"] is not None:
                visible.append([row["cop"][i] + 0.002 * row["force"][i] for i in range(3)])
    limits = [
        [
            1000 * (min(point[i] for point in visible) - 0.003),
            1000 * (max(point[i] for point in visible) + 0.003),
        ]
        for i in range(3)
    ]
    projected = [
        (0.8660254038 * p[0] + 0.5 * p[1], -0.25 * p[0] + 0.4330127019 * p[1] + 0.8660254038 * p[2])
        for p in visible
    ]
    u0, u1 = min(p[0] for p in projected), max(p[0] for p in projected)
    v0, v1 = min(p[1] for p in projected), max(p[1] for p in projected)
    projection = dict(
        center=((u0 + u1) / 2, (v0 + v1) / 2),
        scale=min(420 / (u1 - u0 + 0.006), 230 / (v1 - v0 + 0.006)),
    )
    return dict(
        schema="free-body-support-playback-v1",
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(compressed).hexdigest(),
        frames=frames,
        duration=0.15,
        chosen_frame=chosen,
        traction_max=max(1.0, traction_max),
        limits_mm=limits,
        projection=projection,
    )


def _draw(data, index, *, video=False):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frame = data["frames"][index]
    fig = plt.figure(figsize=(18, 13), facecolor="#f8fafc")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(3.7, 1.2, 1.2, 1.2),
        left=0.055,
        right=0.97,
        bottom=0.105,
        top=0.79,
        hspace=0.52,
        wspace=0.26,
    )
    fig.text(0.5, 0.985, TITLE, ha="center", va="top", fontsize=20, weight="bold", color="#16324f")
    fig.text(0.5, 0.948, SCOPE, ha="center", fontsize=12, color="#475569")
    timing = "0.150 s → 15 s（100 倍慢放）" if video else "真实窗口 0.150 s"
    fig.text(
        0.5,
        0.914,
        f"{timing} · 当前入步 {frame['t'] * 1000:.1f} ms · 原尺寸/原角度，无运动夸大",
        ha="center",
        fontsize=14,
    )
    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(0, data["traction_max"] / 1000)
    for col, state in enumerate(frame["rows"]):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        for vertices, color, alpha in (
            (state["plate_corners"], "#64748b", 0.7),
            (state["cube"], "#6aadd5", 0.16),
        ):
            polygons = [[[1000 * v for v in vertices[i]] for i in face] for face in FACES]
            ax.add_collection3d(
                Poly3DCollection(
                    polygons, facecolors=color, edgecolors=color, linewidths=1, alpha=alpha
                )
            )
        for triangle in state["triangles"]:
            ax.add_collection3d(
                Poly3DCollection(
                    [[[1000 * v for v in point] for point in triangle]],
                    facecolors="#fff0cf",
                    edgecolors="#ae7950",
                    linewidths=0.6,
                    alpha=0.35,
                )
            )
        if state["samples"]:
            ax.scatter(
                *zip(*[[1000 * v for v in point[:3]] for point in state["samples"]]),
                c=[point[3] / 1000 for point in state["samples"]],
                cmap=cmap,
                norm=norm,
                s=16,
                depthshade=False,
            )
        if state["cop"] is not None:
            point = [1000 * v for v in state["cop"]]
            ax.scatter(*point, color="#172d45", s=30, marker="x", depthshade=False)
            ax.quiver(
                *point,
                *[2 * v for v in state["force"]],
                color="#b42336",
                linewidth=2,
                arrow_length_ratio=0.18,
            )
        ax.scatter(*[1000 * v for v in state["position"]], color="#2563a4", s=18, depthshade=False)
        ax.set(
            xlim=data["limits_mm"][0],
            ylim=data["limits_mm"][1],
            zlim=data["limits_mm"][2],
            xlabel="x / mm",
            ylabel="y / mm",
            zlabel="z / mm",
        )
        ax.set_box_aspect([high - low for low, high in data["limits_mm"]])
        ax.view_init(elev=30, azim=-60)
        ax.tick_params(labelsize=8)
        ax.set_title(f"世界 {state['key']} · {LABELS[col]}", fontsize=15, pad=12)
        ax.text2D(
            0.02,
            0.985,
            f"Σ Fn {state['fn']:.3f} N\n相对初态 {state['angle_deg']:.3f}°",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
        )
        if state["cop"] is None:
            ax.text2D(
                0.02,
                0.04,
                "当前无承载，CoP 未定义",
                transform=ax.transAxes,
                fontsize=10,
                color="#64748b",
            )
        for row, (field, label, color) in enumerate(
            (
                ("z_mm", "质心高度 z / mm", "#0f766e"),
                ("fz", "接触 Fz / N", "#b42336"),
                ("angle_deg", "相对初态角 / °", "#b57616"),
            ),
            start=1,
        ):
            curve = fig.add_subplot(grid[row, col])
            times = [f["t"] * 1000 for f in data["frames"]]
            values = [f["rows"][col][field] for f in data["frames"]]
            curve.plot(times, values, color=color, lw=1.7)
            curve.axvline(frame["t"] * 1000, color="#334155", ls="--", lw=0.9)
            curve.scatter([frame["t"] * 1000], [state[field]], color=color, s=17, zorder=3)
            curve.set(xlim=(0, 150), ylabel=label)
            curve.grid(alpha=0.2)
            curve.tick_params(labelsize=9)
            if row == 3:
                curve.set_xlabel("实际入步时间 / ms")
    color_axis = fig.add_axes((0.36, 0.858, 0.28, 0.012))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation="horizontal")
    color_axis.tick_params(labelsize=8)
    color_axis.set_title("积分点颜色：求解牵引 Fn/dA（kPa），非入步弹性压力", fontsize=10, pad=4)
    fig.text(
        0.5,
        0.035,
        "红箭头：同相位合力，1 N → 2 mm（不是位移）｜黑 ×：载荷加权 CoP｜本夹具质心与 body 原点重合",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.text(0.5, 0.015, PHASE, ha="center", fontsize=10, color="#475569")
    return fig


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FREE 刚体真实载荷回放</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#16324f;font-family:"Microsoft YaHei",sans-serif}main{max-width:1650px;margin:auto;padding:16px}nav{display:flex;gap:14px;align-items:center;background:white;padding:14px;border-radius:10px;position:sticky;top:0;z-index:2}input{flex:1;min-width:60px}button{border:0;border-radius:7px;background:#245879;color:white;padding:9px 16px;font:inherit}output{min-width:180px}canvas{width:100%;height:auto;background:#f8fafc;border-radius:10px;margin-top:10px}p{font-size:14px;line-height:1.7;margin:10px 0;color:#526477}</style>
<main><nav><button id="play">播放</button><input id="seek" type="range" min="0" max="149" step="1"><output id="time"></output></nav><canvas id="view" width="1600" height="1080"></canvas>
<p>150 个原始入步帧，播放约 15 秒；真实模拟 0.150 秒，100 倍慢放，无插值。所有几何/接触力/曲线同为入步；末次出步 0.150 秒不配用上一帧力。<br>蓝色为实际尺寸的 FREE 刚体，灰色为有限支撑。黑 × 为 Σ(Fn·点)/ΣFn，零载荷时 CoP 不定义。力箭头 1 N → 2 mm，颜色是求解牵引 Fn/dA，不是入步弹性压力。<br>本夹具 COM=body 原点；无导轨、无摩擦、非抓持。此查看器只读取已保存轨迹，不执行物理或独立验收。源 SHA256：<span id="sha"></span></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),C=document.getElementById('view'),g=C.getContext('2d'),seek=document.getElementById('seek'),play=document.getElementById('play');let timer=null;
const faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],labels=['居中落下','倾斜支撑','偏心支撑'];
document.getElementById('sha').textContent=D.source_sha256;
function text(s,x,y,size=18,color='#16324f',align='left'){g.fillStyle=color;g.font=size+'px "Microsoft YaHei",sans-serif';g.textAlign=align;g.fillText(s,x,y)}
function line(a,b,color='#cbd5e1',width=1){g.beginPath();g.moveTo(...a);g.lineTo(...b);g.strokeStyle=color;g.lineWidth=width;g.stroke()}
function project(p,col){let u=.8660254038*p[0]+.5*p[1],v=-.25*p[0]+.4330127019*p[1]+.8660254038*p[2];return[290+col*520+D.projection.scale*(u-D.projection.center[0]),335-D.projection.scale*(v-D.projection.center[1])]}
function depth(p){return .4330127019*p[0]-.75*p[1]+.5*p[2]}
function polygon(points,col,fill,stroke){g.beginPath();points.forEach((p,i)=>{let q=project(p,col);if(i)g.lineTo(...q);else g.moveTo(...q)});g.closePath();g.fillStyle=fill;g.fill();g.strokeStyle=stroke;g.lineWidth=1;g.stroke()}
function solid(vertices,col,fill,stroke){faces.map(f=>f.map(i=>vertices[i])).sort((a,b)=>a.reduce((s,p)=>s+depth(p),0)-b.reduce((s,p)=>s+depth(p),0)).forEach(f=>polygon(f,col,fill,stroke))}
function color(t){t=Math.min(1,Math.max(0,t));return`rgb(${255-80*t},${242-209*t},${191-143*t})`}
function arrow(a,b){line(a,b,'#b42336',3);let t=Math.atan2(b[1]-a[1],b[0]-a[0]);line(b,[b[0]-11*Math.cos(t-.4),b[1]-11*Math.sin(t-.4)],'#b42336',3);line(b,[b[0]-11*Math.cos(t+.4),b[1]-11*Math.sin(t+.4)],'#b42336',3)}
function chart(col,top,field,label,color,index){let left=65+col*520,right=510+col*520,low=top+103,high=top+22,values=D.frames.map(f=>f.rows[col][field]);let minimum=Math.min(...values),maximum=Math.max(...values),span=Math.max(maximum-minimum,.01);minimum-=span*.08;maximum+=span*.08;span=maximum-minimum;text(label,left,top+3,16,color);for(let i=0;i<3;i++){let y=low-i*(low-high)/2;line([left,y],[right,y]);text((minimum+i*span/2).toFixed(2),left-8,y+5,12,'#64748b','right')}for(let ms of[0,50,100,150]){let x=left+(right-left)*ms/150;line([x,high],[x,low],'#e2e8f0');text(String(ms),x,low+18,12,'#64748b','center')}g.beginPath();values.forEach((v,i)=>{let x=left+(right-left)*D.frames[i].t/.15,y=low-(low-high)*(v-minimum)/span;if(i)g.lineTo(x,y);else g.moveTo(x,y)});g.strokeStyle=color;g.lineWidth=2;g.stroke();let x=left+(right-left)*D.frames[index].t/.15;line([x,high],[x,low],'#334155',1.3)}
function draw(index){let f=D.frames[index];g.clearRect(0,0,1600,1080);text('真实自由刚体：落下承载、倾斜支撑与偏心转动',800,37,28,'#16324f','center');text('CUDA 动力学 + HOST 当前有限面 · 无导轨 / 无摩擦 · 原尺寸与原角度',800,69,18,'#526477','center');text('积分点颜色：求解牵引 Fn/dA，非入步弹性压力',800,102,16,'#9a3412','center');for(let j=0;j<80;j++){g.fillStyle=color(j/79);g.fillRect(640+j*4,115,4,10)}text('0',628,127,13,'#64748b','right');text((D.traction_max/1000).toFixed(2)+' kPa',970,127,13,'#64748b');f.rows.forEach((r,col)=>{text('世界 '+r.key+' · '+labels[col],290+col*520,169,23,'#16324f','center');text('Σ Fn '+r.fn.toFixed(3)+' N  |  相对初态 '+r.angle_deg.toFixed(3)+'°',290+col*520,200,18,'#475569','center');solid(r.plate_corners,col,'#94a3b8','#475569');r.triangles.forEach(p=>polygon(p,col,'#fff0c9','#ae7950'));solid(r.cube,col,'rgba(106,173,213,.14)','#4b87aa');r.samples.forEach(p=>{let q=project(p,col);g.beginPath();g.arc(q[0],q[1],4,0,Math.PI*2);g.fillStyle=color(p[3]/D.traction_max);g.fill()});if(r.cop){let a=project(r.cop,col),b=project(r.cop.map((v,i)=>v+.002*r.force[i]),col);arrow(a,b);line([a[0]-5,a[1]-5],[a[0]+5,a[1]+5],'#172d45',2);line([a[0]-5,a[1]+5],[a[0]+5,a[1]-5],'#172d45',2)}else{text('无承载，CoP 未定义',290+col*520,442,15,'#64748b','center')}let c=project(r.position,col);g.beginPath();g.arc(c[0],c[1],4,0,Math.PI*2);g.fillStyle='#2563a4';g.fill();text('同尺度固定投影 · 红箭头 1 N → 2 mm',290+col*520,477,14,'#64748b','center');chart(col,515,'z_mm','质心高度 z / mm','#0f766e',index);chart(col,682,'fz','接触 Fz / N','#b42336',index);chart(col,849,'angle_deg','相对初态角 / °','#b57616',index)});text('实际入步时间 / ms（全程曲线；竖线为当前保存帧）',800,1030,17,'#526477','center');document.getElementById('time').textContent='入步 '+(f.t*1000).toFixed(1)+' ms / 150 ms';seek.value=index}
seek.oninput=()=>draw(Number(seek.value));play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放'}else{play.textContent='暂停';timer=setInterval(()=>{let i=Number(seek.value)+1;if(i>=D.frames.length)i=0;draw(i)},100)}};draw(D.chosen_frame);
</script></html>"""


def render(input_path, output_dir, *, summary=None, mp4=False):
    data = load_trajectory(input_path)
    output = Path(output_dir)
    _require(
        output.is_absolute() and not output.exists(), "output must be a fresh absolute directory"
    )
    summary_record = None
    if summary is not None:
        summary = Path(summary)
        _require(
            summary.is_absolute()
            and summary.is_file()
            and summary.stat().st_size <= 32 * 1024 * 1024,
            "summary must be a bounded absolute JSON file",
        )
        raw_summary = summary.read_bytes()
        json.loads(raw_summary.decode("utf-8"), parse_constant=_reject)
        summary_record = dict(
            path=str(summary.resolve()),
            sha256=hashlib.sha256(raw_summary).hexdigest(),
            interpretation="attached independent audit; viewer does not infer PASS",
        )
    _require(FONT.is_file(), "Chinese font not found: " + str(FONT))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
            "axes.unicode_minus": False,
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        schema="free-body-support-render-v1",
        status="started",
        input=data["source"],
        input_sha256=data["source_sha256"],
        summary=summary_record,
        physics_recomputed=False,
        interpolated=False,
        phase="accepted-step-input",
        physical_duration_s=0.15,
        frames=150,
        png_frame=data["chosen_frame"],
        png_selection="first maximum of all-world summed Fn",
        force_arrow_m_per_n=0.002,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={},
    )
    try:
        fig = _draw(data, data["chosen_frame"])
        try:
            fig.savefig(output / "free-body-support.png", dpi=140, facecolor=fig.get_facecolor())
        finally:
            plt.close(fig)
        embedded = json.dumps(
            data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).replace("<", "\\u003c")
        (output / "free-body-support.html").write_text(
            HTML.replace("__DATA__", embedded), encoding="utf-8"
        )
        if mp4:
            import imageio.v2 as imageio
            import numpy as np

            with imageio.get_writer(
                output / "free-body-support.mp4", fps=10, codec="libx264", macro_block_size=16
            ) as writer:
                for index in range(150):
                    fig = _draw(data, index, video=True)
                    try:
                        fig.set_dpi(80)
                        fig.canvas.draw()
                        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
                    finally:
                        plt.close(fig)
            receipt["video"] = dict(frames=150, fps=10, playback_s=15, slowdown=100)
        for path in sorted(output.iterdir()):
            if path.is_file():
                receipt["files"][path.name] = dict(
                    bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        (output / "render-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--mp4", action="store_true")
    args = parser.parse_args()
    render(args.input, args.output_dir, summary=args.summary, mp4=args.mp4)


if __name__ == "__main__":
    main()
