"""Create one offline 3D playback from two saved Panda runs, without resimulating."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo.builders.panda import load_panda
from phymeo.dynamics.rotation import matrix_quaternion, quaternion_matrix


def display_geometry(run):
    """Use the saved physical dimensions, including pre-shape box recordings."""
    geometry = run.get("payload_geometry", {"kind": "box"})
    kind = geometry["kind"]
    if kind == "sphere":
        radius = float(geometry["radius_m"])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("sphere radius must be positive and finite")
        size = np.full(3, 2 * radius)
        if "cube_full_size_m" in run and not np.array_equal(run["cube_full_size_m"], size):
            raise ValueError("saved sphere dimensions disagree with its radius")
    elif kind == "box":
        size = np.asarray(
            geometry.get("full_size_m", run.get("cube_full_size_m", [0.05] * 3)), dtype=float
        )
        if "cube_full_size_m" in run and not np.array_equal(run["cube_full_size_m"], size):
            raise ValueError("saved box dimensions disagree")
    else:
        raise ValueError(f"unknown saved payload geometry {kind!r}")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("expected three positive, finite payload dimensions")
    return {"kind": kind, "size_m": size.tolist()}


def collision_poses(asset, poses):
    shapes = []
    for part in asset.collisions:
        pos, rot = poses[part.link_name]
        if part.shape.kind == "hull":
            p = pos + rot @ part.origin_xyz
            q = matrix_quaternion(rot @ quaternion_matrix(np.array([part.origin_xyzw]))[0])
            shapes.append([p.tolist(), q.tolist()])
        else:
            shapes.append((pos + np.asarray(part.local_vertices) @ rot.T).tolist())
    return shapes


def read_run(asset, path):
    run = json.loads(path.read_text(encoding="utf8"))
    rows = run.pop("trace")
    if len(rows) != run["steps"] or not rows:
        raise ValueError("expected the complete motion trace")
    frames = []
    keys = {part.link_name for part in asset.collisions}
    indices = set(np.linspace(0, len(rows) - 1, min(len(rows), 181), dtype=int))
    registered = "structural_contacts" in rows[0]
    if any(("structural_contacts" in row) != registered for row in rows):
        raise ValueError("incomplete structural contact recording")
    if registered:
        indices.update(i for i, row in enumerate(rows) if row["structural_contacts"])
    jumps = {}
    peak = 0.0
    for i in sorted(indices):
        row = rows[i]
        contacts = []
        pose = row
        if registered:
            pose = (
                rows[i - 1]
                if i
                else dict(
                    t=0.0,
                    q=run["initial_arm_q"],
                    cube=run["initial_payload_position_m"],
                    xyzw=run["initial_payload_xyzw"],
                )
            )
            for contact in row["structural_contacts"]:
                force = np.asarray(contact["force_on_payload_N"], dtype=float)
                point = np.asarray(contact["point_world_m"], dtype=float)
                if any(a.shape != (3,) or not np.isfinite(a).all() for a in (force, point)):
                    raise ValueError("invalid recorded structural force or application point")
                magnitude = np.linalg.norm(force)
                if magnitude > 0:
                    contacts.append([contact["shape"], point.tolist(), force.tolist()])
                    jumps.setdefault("first", len(frames))
                    if magnitude > peak:
                        peak = magnitude
                        jumps["peak"] = len(frames)
        poses = {name: asset.authored_link_pose(name, np.array(pose["q"])) for name in keys}
        shapes = collision_poses(asset, poses)
        frames.append(
            [
                pose["t"],
                shapes,
                pose["cube"],
                pose["xyzw"],
                row["pad_normal_N"],
                row["table_N"],
                contacts,
            ]
        )
    if registered:
        # Include the accepted endpoint. No new force was sampled at this time.
        final = rows[-1]
        poses = {name: asset.authored_link_pose(name, np.array(final["q"])) for name in keys}
        frames.append(
            [
                final["t"],
                collision_poses(asset, poses),
                final["cube"],
                final["xyzw"],
                None,
                None,
                [],
            ]
        )
    return dict(
        frames=frames,
        summary=run,
        payload=display_geometry(run),
        jumps=jumps,
        load_time="entry" if registered else "preceding_step",
    )


def export(loaded, slippery, output, urdf=None):
    asset = load_panda(urdf)
    runs = [read_run(asset, path) for path in (loaded, slippery)]
    parts = []
    for part in asset.collisions:
        if part.shape.kind == "hull":
            rot = quaternion_matrix(np.array([part.origin_xyzw]))[0]
            size = np.ptp((np.array(part.local_vertices) - part.origin_xyz) @ rot, axis=0).tolist()
        else:
            size = part.shape.radius
        parts.append([part.name, part.shape.kind, size])
    vendor = Path(__file__).resolve().parents[1] / "docs/demos/vendor"
    data = json.dumps(dict(parts=parts, runs=runs), separators=(",", ":"), allow_nan=False)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    html = PAGE.replace("__LIB__", (vendor / "three-r160.min.js").read_text(encoding="utf8"))
    html = html.replace("__LICENSE__", (vendor / "three-r160.LICENSE").read_text(encoding="utf8"))
    with output.open("x", encoding="utf8") as stream:
        stream.write(html.replace("__DATA__", data))


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Panda 抓放 | Phymeo</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0e1927;color:#edf4f7;font:16px system-ui}main{padding:18px 4vw}h1{font-size:23px;margin:0 0 6px}p{color:#aabdcf;margin:4px 0}canvas{display:block;width:100%;height:63vh;touch-action:none}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input{width:min(46vw,640px)}button,select{padding:7px;color:#142033}output{font-variant-numeric:tabular-nums}</style>
<main><h1>经典 Panda · 自由载荷抓放</h1><p>七轴、双指与 19 个厂家碰撞代理。拖动旋转视角，滚轮缩放。结构受力箭头从真实作用点沿球所受世界力绘制，长度 5 mm/N，上限 150 mm。</p></main>
<canvas id="view"></canvas><main><nav><select id="run"></select><button id="play">播放 / 暂停</button><button id="first-force">首次结构受力</button><button id="peak-force">峰值结构受力</button><input id="time" type="range" min="0" value="0" step="1"></nav><p><output id="value"></output></p></main>
<!-- __LICENSE__ --><script>__LIB__</script><script>
const data=__DATA__, canvas=document.getElementById("view"), slider=document.getElementById("time"), choice=document.getElementById("run");
const renderer=new THREE.WebGLRenderer({canvas,antialias:true}), scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(39,1,.005,15);
scene.background=new THREE.Color(0x132336); camera.up.set(0,0,1);
scene.add(new THREE.HemisphereLight(0xffffff,0x94a4ba,2)); const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(2,-1,3);scene.add(sun);
const vec=a=>new THREE.Vector3(...a);
function mesh(geometry,color){const m=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color}));scene.add(m);return m;}
const parts=data.parts.map(([name,kind,size])=>kind==="hull"?mesh(new THREE.BoxGeometry(...size),name.endsWith("_pad")?0x32d2b7:0x879db6):mesh(kind==="sphere"?new THREE.SphereGeometry(size,16,12):new THREE.CapsuleGeometry(size,1,6,16),0xb9d2e2));
const payloadGeometry={box:new THREE.BoxGeometry(1,1,1),sphere:new THREE.SphereGeometry(.5,32,20)};
const cube=mesh(payloadGeometry.box,0xf6a74d);
const table=mesh(new THREE.BoxGeometry(1,1,1),0x57718c);
let az=-1.08, el=.35, dist=1.35, last=0, playing=false, target=vec([.20,0,.37]);
for (const run of data.runs) {
  const s=run.summary, option=document.createElement("option");
  option.textContent=(s.engine ?? "Phymeo")+" · "+(run.payload.kind==="sphere"?"球 · ":"方盒 · ")+(s.cube_mass_kg*1000).toFixed(0)+" g · μ "+s.finger_friction+" · 闭合目标 "+((s.closed_width_m ?? .029)*1000).toFixed(1)+" mm";
  if(s.success!==undefined) option.textContent+=s.success?" · 已放置":" · 未完成抓放";
  if(s.first_unresolved_structure) option.textContent+=" · 有未求力结构接触";
  else if(run.jumps.first!==undefined) option.textContent+=" · 结构接触已求力";
  else if(s.other_robot_cube_axis_clearance_m<0) option.textContent+=" · 整窗结构净空未通过";
  choice.appendChild(option);
}
const arrows=[];
function drawForces(contacts){
  while(arrows.length<contacts.length){
    const arrow=new THREE.ArrowHelper(vec([0,0,1]),vec([0,0,0]),.01,0xff795b);
    arrows.push(arrow);scene.add(arrow);
  }
  arrows.forEach((arrow,i)=>{
    arrow.visible=i<contacts.length;
    if(!arrow.visible)return;
    const [,p,f]=contacts[i],direction=vec(f),length=Math.min(.15,.005*direction.length());
    arrow.position.fromArray(p);arrow.setDirection(direction.normalize());
    arrow.setLength(length,Math.min(.012,length*.3),Math.min(.005,length*.15));
  });
}
function jump(which){const i=data.runs[choice.selectedIndex].jumps[which];if(i===undefined)return;playing=false;slider.value=i;draw();}
document.getElementById("first-force").onclick=()=>jump("first");
document.getElementById("peak-force").onclick=()=>jump("peak");
function draw(){const {frames,summary,payload,jumps,load_time}=data.runs[choice.selectedIndex];
document.getElementById("first-force").disabled=jumps.first===undefined;
document.getElementById("peak-force").disabled=jumps.peak===undefined;
cube.geometry=payloadGeometry[payload.kind];cube.scale.fromArray(payload.size_m);
table.scale.fromArray(summary.table_full_size_m);table.position.fromArray(summary.table_center_m);
slider.max=frames.length-1;const [t,shapes,p,q,forces,tbl,contacts]=frames[+slider.value];
shapes.forEach((coords,i)=>{const obj=parts[i],kind=data.parts[i][1];
if(kind==="hull"){obj.position.fromArray(coords[0]);obj.quaternion.fromArray(coords[1]);}
else if(kind==="sphere"){obj.position.fromArray(coords[0]);}
else{const a=vec(coords[0]), b=vec(coords[1]),delta=b.clone().sub(a),length=delta.length();
if(!obj.userData.ready){obj.geometry.dispose();obj.geometry=new THREE.CapsuleGeometry(data.parts[i][2],Math.max(0,length),6,16);obj.userData.ready=true;}
obj.position.copy(a).lerp(b,.5);if(length>0)obj.quaternion.setFromUnitVectors(vec([0,1,0]),delta.divideScalar(length));}});
cube.position.fromArray(p);cube.quaternion.fromArray(q);drawForces(contacts);
const forceText=forces===null?"终态":(load_time==="entry"?"入步载荷　":"前一子步载荷　")+"左指 "+forces[0].toFixed(2)+" N　右指 "+forces[1].toFixed(2)+" N　桌面 "+tbl.toFixed(2)+" N";
document.getElementById("value").textContent=t.toFixed(6)+" s　"+forceText+
  contacts.map(([name,,f])=>"　"+name+" "+Math.hypot(...f).toFixed(3)+" N").join("");
const w=canvas.clientWidth,h=canvas.clientHeight; renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();camera.position.set(target.x+dist*Math.cos(el)*Math.cos(az),target.y+dist*Math.cos(el)*Math.sin(az),target.z+dist*Math.sin(el));camera.lookAt(target);renderer.render(scene,camera);}
choice.onchange=()=>{slider.value=0;playing=false;draw()};slider.oninput=()=>{playing=false;draw()};
document.getElementById("play").onclick=()=>{if(+slider.value===+slider.max)slider.value=0;playing=!playing;last=0;draw()};
let cursor=null;canvas.onpointerdown=e=>{cursor=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!cursor)return;az-=(e.clientX-cursor[0])*.008;el=Math.max(-1.35,Math.min(1.35,el+(e.clientY-cursor[1])*.008));cursor=[e.clientX,e.clientY];draw()};
canvas.onpointerup=canvas.onpointercancel=()=>{cursor=null};canvas.onwheel=e=>{e.preventDefault();dist=Math.max(.35,Math.min(3,dist*Math.exp(e.deltaY*.001)));draw()};
function tick(now){
  if(playing){
    const frames=data.runs[choice.selectedIndex].frames;
    if(!last)last=now-1000*frames[+slider.value][0];
    let i=+slider.value;
    while(i+1<frames.length && frames[i+1][0]<=(now-last)/1000)i++;
    if(i!==+slider.value){slider.value=i;draw();}
    if(i===frames.length-1)playing=false;
  }
  requestAnimationFrame(tick);
}window.onresize=draw;draw();requestAnimationFrame(tick);
</script></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("loaded", "slippery", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()
    export(args.loaded, args.slippery, args.output, args.urdf)
