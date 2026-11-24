"""A compact belt conveyor, two shallow sorting bins and machined workpieces."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from _panda_scene import write_mesh
from phymeo_rendering.assets import Material
from phymeo_rendering.scene import Instance, NodePose, Pose

from .sorting_layout import ITEM_NODES
from .workshop_set import Batch, FINISH, STATIC_NODE, _back_shop, _payload, _room


DRUM_RADIUS = .042
BELT_WIDTH = .222
ROLLERS = ("sort-discharge-roller", "sort-infeed-roller")
MARKERS = tuple(f"sort-belt-mark-{i:02d}" for i in range(16))
PALETTE = {
    **FINISH,
    "belt": ((.045, .068, .078), .01, .76),
    "belt-ink": ((.34, .45, .44), .0, .68),
    "blue-part": ((.032, .252, .665), .44, .31),
    "orange-rim": ((.62, .202, .047), .12, .48),
    "blue-rim": ((.055, .255, .51), .12, .48),
}


def _axles(layout):
    # Actual belt envelope extends one drum radius beyond these two axles.
    return layout.pick_xy[1] - .058, layout.pick_xy[1] + 1.438


def _belt_path(layout, distance):
    """Return Y, Z and the rotation that carries a printed belt mark around the loop."""
    near, far = _axles(layout)
    straight = far - near
    half_wrap = math.pi * DRUM_RADIUS
    period = 2 * (straight + half_wrap)
    s = distance % period
    axle_z = layout.support_z - DRUM_RADIUS
    if s < straight:
        return far-s, layout.support_z, 0.
    if s < straight+half_wrap:
        angle = (s-straight) / DRUM_RADIUS
        return near-DRUM_RADIUS*math.sin(angle), axle_z+DRUM_RADIUS*math.cos(angle), angle
    if s < 2*straight+half_wrap:
        return near+s-straight-half_wrap, axle_z-DRUM_RADIUS, math.pi
    angle = math.pi + (s-2*straight-half_wrap) / DRUM_RADIUS
    return far-DRUM_RADIUS*math.sin(angle), axle_z+DRUM_RADIUS*math.cos(angle), angle


def conveyor_nodes(layout, travel_m):
    """All fixed and belt-drive poses. Positive belt travel advances the upper run toward -Y."""
    if not math.isfinite(travel_m):
        raise ValueError("conveyor travel must be finite")
    x = layout.pick_xy[0]
    near, far = _axles(layout)
    period = 2 * (far-near + math.pi*DRUM_RADIUS)
    def about_x(angle):
        return (math.sin(angle/2), 0., 0., math.cos(angle/2))
    nodes = [NodePose(STATIC_NODE, Pose())]
    for name,y in zip(ROLLERS, (near,far)):
        nodes.append(NodePose(name, Pose((x,y,layout.support_z-DRUM_RADIUS), about_x(travel_m/DRUM_RADIUS))))
    for index,name in enumerate(MARKERS):
        y,z,angle = _belt_path(layout, travel_m + index*period/len(MARKERS))
        nodes.append(NodePose(name, Pose((x,y,z), about_x(angle))))
    return tuple(nodes)


def _belt_skin(batch, layout):
    """Extruded racetrack with tangent top/return runs and smooth drum wraps."""
    near, far = _axles(layout)
    x = layout.pick_xy[0]
    axle_z = layout.support_z - DRUM_RADIUS
    profile = []
    for y, start in ((near,0.),(far,math.pi)):
        for i in range(17):
            a = start + math.pi*i/16
            n = np.array((0., -math.sin(a), math.cos(a)))
            profile.append((np.array((x,y,axle_z)) + DRUM_RADIUS*n, n))
    width = np.array((BELT_WIDTH/2,0.,0.))
    for i,(p,n) in enumerate(profile):
        q,m = profile[(i+1)%len(profile)]
        batch.face((p-width,q-width,q+width,p+width), n+m, (n,m,m,n))
    # The belt is a thin continuous web. An edge annulus exposes the drum face
    # and avoids coincident end caps where the bearings and shafts emerge.
    for sign in (-1,1):
        for i,(p,n) in enumerate(profile):
            q,m = profile[(i+1)%len(profile)]
            outer_p, outer_q = p+sign*width, q+sign*width
            batch.face((outer_p,outer_q,outer_q-.003*m,outer_p-.003*n),(sign,0,0))


def _reader_head(m, layout):
    """Rail-mounted RGB reader aimed down and across the stopped part."""
    x, y = layout.pick_xy
    z = layout.support_z
    eye = np.array((x+.29, y+.06, z+.205))
    back = eye - (x, y, z+.032)
    back /= np.linalg.norm(back)
    right = np.cross((0.,0.,1.), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    def housing(finish, front, rear, width, height):
        rim = [r*right*width/2 + h*up*height/2
               for r,h in ((-1,-1),(1,-1),(1,1),(-1,1))]
        p = [eye+front*back+corner for corner in rim]
        q = [eye+rear*back+corner for corner in rim]
        m[finish].face(p, -back)
        m[finish].face(q, back)
        for i,n in enumerate((-up,right,up,-right)):
            j = (i+1)%4
            m[finish].face((p[i],p[j],q[j],q[i]), n)

    # Dimensions run backward from the optical centre. Even the glass surface
    # lies behind the projection plane of the registered 15 mm-near camera.
    housing("frame", .020, .059, .045, .034)
    housing("amber", .016, .023, .046, .035)
    housing("amber", .054, .060, .046, .035)
    housing("frame", .060, .062, .038, .028)
    for finish,front,rear,radius in (
        ("frame",.0024,.017,.014), ("steel",.0015,.004,.0128),
        ("frame",.0008,.002,.0109), ("blue-glass",.0005,.001,.0068),
    ):
        m[finish].cylinder(eye+front*back, eye+rear*back, radius, 16)
    indicator = eye+.062*back-.011*right+.006*up
    m["pale-glass"].cylinder(indicator, indicator+.0017*back, .0024, 8)

    # Cantilever from the +X rail with the post on the upstream side. All
    # structure crossing back toward the belt is below the upper belt run.
    shoulder = eye+.041*back+.0225*right
    post = shoulder+.043*right
    rail_x, foot_z = x+BELT_WIDTH/2+.022, z-.071
    m["galvanized"].box((rail_x,post[1],foot_z),(.010,.056,.053))
    m["frame"].box(((rail_x+post[0])/2,post[1],foot_z),(post[0]-rail_x,.022,.019))
    m["steel"].cylinder((post[0],post[1],foot_z),post,.0075,10)
    m["frame"].cylinder(shoulder,post+.009*right,.009,10)
    for dy in (-.017,.017):
        m["steel"].cylinder((rail_x+.004,post[1]+dy,foot_z+.010),
                            (rail_x+.009,post[1]+dy,foot_z+.010),.0042,6)
    socket = eye+.062*back+.005*right
    m["steel"].cylinder(socket,socket+.007*back,.0063,10)
    m["rubber"].cylinder(socket+.007*back,socket+.017*back,.0052,10)
    cable = (socket+.017*back, socket+.026*back+.007*right,
             post+(.010,.008,-.021), (post[0]+.008,post[1]+.006,foot_z+.005),
             (rail_x,post[1]+.008,foot_z+.005))
    for a,b in zip(cable,cable[1:]):
        m["rubber"].cylinder(a,b,.0027,8)


def _stand(m, layout):
    near, far = _axles(layout)
    x = layout.pick_xy[0]
    top = layout.support_z
    mid = (near+far)/2
    for sign in (-1,1):
        sx = x + sign*(BELT_WIDTH/2+.013)
        m["paint"].box((sx,mid,top-.068),(.022,far-near+.018,.072))
        m["galvanized"].box((sx,mid,top-.025),(.027,far-near-.032,.014))
        m["frame"].box((sx,mid,.275),(.032,far-near-.26,.038))
        for iy in (near+.139, far-.139):
            m["paint"].box((sx,iy,(top-.066)/2+.012),(.038,.044,top-.095))
            m["steel"].cylinder((sx,iy,.014),(sx,iy,.044),.012,8)
            m["rubber"].cylinder((sx,iy,.001),(sx,iy,.017),.031,10)
    for iy in (near+.139,far-.139):
        m["frame"].box((x,iy,.273),(BELT_WIDTH+.047,.038,.038))
        m["frame"].box((x,iy,top-.110),(BELT_WIDTH+.054,.048,.040))
    # Gearhead and finned motor occupy the outboard side of the infeed axle.
    cx, cy, cz = x+BELT_WIDTH/2+.057, far, top-DRUM_RADIUS
    m["paint"].box((cx,cy,cz),(.084,.075,.082))
    m["frame"].cylinder((cx,cy-.033,cz),(cx,cy-.163,cz),.037,12)
    for i in range(4):
        yy = cy-.056-i*.027
        m["galvanized"].cylinder((cx,yy-.003,cz),(cx,yy+.003,cz),.0395,12)
    m["paint"].box((cx,cy-.097,cz+.043),(.048,.065,.028))
    # A recessed photoeye, mounted at the far side of the pick lane.
    sx, sy = x-BELT_WIDTH/2-.027, layout.pick_xy[1]+.064
    m["steel"].box((sx,sy,top-.003),(.012,.025,.108))
    m["frame"].box((sx+.007,sy,top+.048),(.026,.032,.030))
    m["red-tool"].cylinder((sx+.020,sy,top+.049),(sx+.0225,sy,top+.049),.005,8)
    _belt_skin(m["belt"], layout)
    _reader_head(m, layout)


def _drum():
    drum = {k:Batch() for k in ("steel","frame","amber")}
    end = BELT_WIDTH/2+.026
    drum["steel"].cylinder((-end,0,0),(end,0,0),DRUM_RADIUS-.0036,24)
    for sign in (-1,1):
        a = sign*end
        drum["frame"].cylinder((a,0,0),(a+sign*.003,0,0),.025,16)
        drum["steel"].cylinder((a+sign*.003,0,0),(a+sign*.009,0,0),.012,6)
    # A witness mark on the exposed flange makes indexed rotation visible.
    drum["amber"].box((end+.0006,0,.0305),(.001,.006,.011))
    return drum


def _witness():
    # Printed only along the free belt margins. A carried part rests directly
    # on the belt at support_z throughout the top run.
    batch = Batch()
    for sign in (-1,1):
        points = ((sign*.102,-.002,.00036), (sign*.066,.006,.00036),
                  (sign*.066,.012,.00036), (sign*.102,.004,.00036))
        batch.face(points,(0,0,1))
    return batch


def _receivers(m, layout):
    (ax,ay),(bx,by) = layout.bin_centers
    top = layout.support_z
    midx, midy = (ax+bx)/2+.035, (ay+by)/2
    deck = top-.0075
    # The projecting front edge supports the camera-visible nameplates.
    width, length = .326, abs(ay-by)+.191
    m["benchtop"].box((midx,midy,deck-.012),(width,length,.024))
    m["frame"].box((midx,midy,deck-.041),(width-.025,length-.025,.040))
    for dx in (-.126,.126):
        for dy in (-1,1):
            x, y = midx+dx, midy+dy*(length/2-.028)
            m["galvanized"].box((x,y,deck/2+.005),(.032,.032,deck-.067))
            m["rubber"].cylinder((x,y,.001),(x,y,.033),.028,8)
        m["paint"].box((midx+dx,midy,.26),(.032,length-.045,.032))
    for (x,y),finish in zip(layout.bin_centers, ("orange-rim","blue-rim")):
        # Shallow removable steel bins. The open front lip faces the camera.
        # Usable floor at top; the highest wall is 30 mm above that plane.
        m["galvanized"].box((x,y,top-.00375),(.220,.155,.0075))
        m["frame"].box((x-.108,y,top+.014),(.004,.155,.028))
        m[finish].box((x-.108,y,top+.029),(.005,.155,.002))
        m[finish].box((x+.108,y,top+.007),(.004,.155,.014))
        for sy in (-1,1):
            m["frame"].box((x,y+sy*.0755,top+.010),(.212,.004,.020))
            m[finish].box((x,y+sy*.0755,top+.021),(.212,.004,.002))
        # A longitudinal divider and four pairs of low retaining stops leave
        # room for both the 78 mm blocks and the smaller hexagonal collars.
        m["rubber"].box((x,y,top+.0025),(.004,.118,.005))
        for slot in (x-layout.slot_offset_x, x+layout.slot_offset_x):
            for sy in (-1,1):
                m["rubber"].box((slot,y+sy*.037,top+.002),(.049,.004,.004))


def _hex_payload(size):
    """Regular external hexagon with parallel Y jaws and a bored/chamfered centre."""
    _, width, height = size
    apothem = width/2
    hz = height/2
    bevel = .001
    radius = apothem/math.cos(math.pi/6)
    outer = [radius*np.array((math.cos(i*math.pi/3),math.sin(i*math.pi/3))) for i in range(6)]
    flat = [(1-bevel/apothem)*p for p in outer]
    m = {k:Batch() for k in ("part","cut","bore")}
    hole = .013
    def point(p,z):
        return (p[0],p[1],z)
    for i,a in enumerate(outer):
        j = (i+1)%6
        b = outer[j]
        n = a+b
        n /= np.linalg.norm(n)
        m["part"].face((point(a,-hz+bevel),point(b,-hz+bevel),point(b,hz-bevel),point(a,hz-bevel)),(*n,0.))
        for sign in (-1,1):
            m["cut"].face((point(a,sign*(hz-bevel)),point(b,sign*(hz-bevel)),
                           point(flat[j],sign*hz),point(flat[i],sign*hz)),(*n,sign))
        for k in range(6):
            p = flat[i] + (flat[j]-flat[i])*(k/6)
            q = flat[i] + (flat[j]-flat[i])*((k+1)/6)
            u,v = p/np.linalg.norm(p), q/np.linalg.norm(q)
            for sign in (-1,1):
                m["part"].face((point(p,sign*hz),point(q,sign*hz),point(v*(hole+bevel),sign*hz),
                                point(u*(hole+bevel),sign*hz)),(0,0,sign))
                m["cut"].face((point(u*(hole+bevel),sign*hz),point(v*(hole+bevel),sign*hz),
                               point(v*hole,sign*(hz-bevel)),point(u*hole,sign*(hz-bevel))),
                              (-u[0]-v[0],-u[1]-v[1],2*sign))
            m["bore"].face((point(u*hole,-hz+bevel),point(v*hole,-hz+bevel),
                            point(v*hole,hz-bevel),point(u*hole,hz-bevel)),
                           (-u[0]-v[0],-u[1]-v[1],0),
                           [(-d[0],-d[1],0) for d in (u,v,v,u)])
    return m


def append_sorting_set(scene, directory, layout, kinds):
    """Build separate moving instances for each item; same-class items share all geometry."""
    kinds = tuple(kinds)
    if len(kinds) != len(ITEM_NODES) or any(kind not in ("block","hex") for kind in kinds):
        raise ValueError("expected one block or hex classification for each sorting item")
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    static = {name:Batch() for name in PALETTE}
    _room(static)
    _back_shop(static)
    _stand(static, layout)
    _receivers(static, layout)
    meshes = list(scene.meshes)
    materials = list(scene.materials)
    instances = list(scene.instances)
    identity = max(item.id for item in instances) + 1

    def bind(prefix, batches, nodes, finishes=None):
        nonlocal identity
        for key,batch in batches.items():
            if not batch.triangles:
                continue
            name = prefix+"-"+key
            mesh = write_mesh(directory, name, batch.positions, batch.normals, batch.triangles)
            material = Material(name+"-material", *PALETTE[(finishes or {}).get(key,key)])
            meshes.append(mesh)
            materials.append(material)
            for node in nodes:
                instances.append(Instance(identity, mesh.id, material.id, node))
                identity += 1

    bind(STATIC_NODE, static, (STATIC_NODE,))
    bind("sort-drum", _drum(), ROLLERS)
    bind("sort-witness", {"belt-ink":_witness()}, MARKERS)
    for kind in dict.fromkeys(kinds):
        if kind=="block":
            part = {key:Batch() for key in ("part","cut","bore")}
            _payload(part, layout.payload_size)
        else:
            part = _hex_payload(layout.payload_size)
        nodes = tuple(n for n,k in zip(ITEM_NODES,kinds) if k==kind)
        bind("sort-"+kind, part, nodes, {"part":"blue-part"} if kind=="hex" else None)
    return replace(scene, key=scene.key+"-indexed-sorting-"+"-".join(kinds), meshes=tuple(meshes),
                   materials=tuple(materials), instances=tuple(instances))
