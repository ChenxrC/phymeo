"""Metre-scale manufacturing cell and a bored, two-sided graspable workpiece."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from _panda_scene import write_mesh
from phymeo_rendering.assets import Material
from phymeo_rendering.scene import Instance


STATIC_NODE = 'r1-workshop'
PAYLOAD_NODE = 'r1-workpiece'

# Scene-linear, opaque finishes. The duller back wall keeps the articulated
# parts and the machined orange workpiece legible in the foreground.
FINISH = {
    'floor': ((.21,.245,.25), 0., .70),
    'joint': ((.087,.106,.11), 0., .79),
    'wall': ((.396,.498,.575), 0., .72),
    'dado': ((.185,.256,.305), 0., .66),
    'frame': ((.065,.096,.11), .42, .46),
    'paint': ((.105,.195,.215), .10, .42),
    'drawer': ((.125,.245,.27), .08, .40),
    'steel': ((.46,.50,.50), .78, .32),
    'galvanized': ((.44,.47,.45), .40, .53),
    'benchtop': ((.36,.395,.39), .43, .40),
    'rubber': ((.036,.045,.045), 0., .80),
    'blue-glass': ((.38,.53,.58), .12, .28),
    'pale-glass': ((.56,.66,.65), .09, .31),
    'amber': ((.68,.395,.072), .04, .55),
    'red-tool': ((.48,.092,.042), .05, .49),
    'part': ((.73,.145,.030), .43, .31),
    'cut': ((.65,.55,.42), .80, .28),
    'bore': ((.24,.22,.18), .72, .37),
}


class Batch:
    """One indexed triangle primitive per node and finish."""
    def __init__(self):
        self.positions, self.normals, self.triangles = [], [], []
        self.lookup = {}

    def vertex(self, point, normal):
        p, n = tuple(point), tuple(normal)
        key = p+n
        if key not in self.lookup:
            self.lookup[key] = len(self.positions)
            self.positions.append(p)
            self.normals.append(n)
        return self.lookup[key]

    def face(self, points, outward, normals=None):
        p = np.asarray(points, dtype=float)
        direction = np.asarray(outward, dtype=float)
        direction /= np.linalg.norm(direction)
        ns = np.tile(direction, (len(p),1)) if normals is None else np.asarray(normals)
        if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),direction) < 0:
            p,ns = p[::-1],ns[::-1]
        ids = [self.vertex(point,n) for point,n in zip(p,ns)]
        self.triangles.extend((ids[0],ids[j],ids[j+1]) for j in range(1,len(ids)-1))

    def box(self, center, size):
        c,h = np.asarray(center),np.asarray(size)/2.
        axes = np.eye(3)
        for k in range(3):
            for sign in (-1.,1.):
                n = sign*axes[k]
                u,v = axes[(k+1)%3],axes[(k+2)%3]
                self.face([c+h*(n+a*u+b*v) for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))],n)

    def cylinder(self, a, b, radius, sides=12):
        a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        axis = b-a
        axis /= np.linalg.norm(axis)
        u = np.cross(axis,(1,0,0) if abs(axis[0])<.8 else (0,1,0))
        u /= np.linalg.norm(u)
        v = np.cross(axis,u)
        ns = [math.cos(2*math.pi*i/sides)*u + math.sin(2*math.pi*i/sides)*v for i in range(sides)]
        low, high = [a+radius*n for n in ns], [b+radius*n for n in ns]
        self.face(low,-axis)
        self.face(high,axis)
        for i,n in enumerate(ns):
            j=(i+1)%sides
            self.face([low[i],low[j],high[j],high[i]],n+ns[j],[n,ns[j],ns[j],n])

    def bar_on_wall(self, x, start, end, width, depth):
        """Flat-sided diagonal bar, specified by its two end centres in Y/Z."""
        a,b = np.asarray(start),np.asarray(end)
        d = b-a
        side = np.array([-d[1],d[0]]) * (width/(2*np.linalg.norm(d)))
        polygon = [a-side,b-side,b+side,a+side]
        front = [(x+depth/2,*p) for p in polygon]
        back = [(x-depth/2,*p) for p in polygon]
        self.face(front,(1,0,0))
        self.face(back,(-1,0,0))
        for i in range(4):
            j=(i+1)%4
            v=polygon[j]-polygon[i]
            self.face([back[i],back[j],front[j],front[i]],(0,v[1],-v[0]))


def _room(m):
    box=lambda material,c,s: m[material].box(c,s)
    box('floor',(0,0,-.05),(13,13,.10))
    for x in (-3.,-1.5,1.5,3.):
        box('joint',(x,0,.0006),(.005,10,.001))
    for y in (-3.,-1.5,1.5,3.):
        box('joint',(0,y,.0012),(10,.005,.001))
    # L-shaped painted cell corners leave the central floor and tire paths clear.
    for x,sx in ((-.60,1),(.98,-1)):
        for y,sy in ((-.67,1),(.91,-1)):
            box('amber',(x+.085*sx,y,.0023),(.17,.025,.002))
            box('amber',(x,y+.09*sy,.0023),(.025,.18,.002))

    # Back wall and side return. The cell stays open overhead and on the camera side.
    box('wall',(-3.06,0,1.68),(.12,9.2,3.36))
    box('dado',(-2.989,0,.54),(.022,9.2,1.08))
    box('frame',(-2.97,0,.07),(.03,9.2,.14))
    # Continue past the front-aisle camera. The kick rail, door reveal and
    # inspection window face into the room from the +Y wall.
    box('wall',(1.33,4.18,1.77),(8.76,.12,3.54))
    box('dado',(1.33,4.105,.47),(8.76,.027,.94))
    box('frame',(1.33,4.08,.06),(8.76,.031,.12))
    box('galvanized',(1.33,4.095,.96),(8.76,.022,.032))
    door_x = -.81
    box('frame',(door_x,4.065,1.055),(1.04,.078,2.11))
    box('dado',(door_x,4.012,1.025),(.958,.043,2.05))
    box('galvanized',(door_x,3.987,.18),(.921,.010,.28))
    box('frame',(door_x-.09,3.980,1.55),(.31,.026,.55))
    box('blue-glass',(door_x-.09,3.959,1.55),(.257,.020,.493))
    box('steel',(door_x+.346,3.975,1.02),(.043,.033,.165))
    box('steel',(door_x+.309,3.938,1.048),(.116,.039,.021))
    box('galvanized',(door_x+.313,3.962,1.94),(.195,.069,.057))
    # A surface-mounted isolation switch and conduit sit beside the latch side.
    box('frame',(door_x+.77,4.073,1.23),(.126,.11,.161))
    box('amber',(door_x+.77,4.006,1.23),(.110,.027,.146))
    m['red-tool'].cylinder((door_x+.77,3.990,1.23),(door_x+.77,3.955,1.23),.027,10)
    m['galvanized'].cylinder((door_x+.77,4.067,1.31),(door_x+.77,4.067,2.65),.014,8)
    box('galvanized',(1.33,4.07,2.69),(8.76,.089,.076))
    for y in (-4.25,-2.62,-.90,1.08,2.95,4.12):
        box('galvanized',(-2.94,y,1.66),(.095,.082,3.32))
    # Frosted panes set into dark gaskets, with continuous head, sill and mullions.
    for y0,width in ((-2.47,2.95),(1.92,2.02)):
        box('frame',(-2.92,y0,2.31),(.09,width,1.05))
        count=4
        bay=(width-.07)/count
        for i in range(count):
            yc=y0-width/2+.035+(i+.5)*bay
            for j in range(2):
                box('pale-glass' if (i+j)%3==0 else 'blue-glass',
                    (-2.863,yc,2.08+.465*j),(.018,bay-.030,.433))
        box('galvanized',(-2.84,y0,1.77),(.19,width+.08,.045))

    # Rolled-steel header and two exposed roof joists (web plus flanges).
    for z in (3.12,3.30):
        box('frame',(-2.87,0,z),(.18,9.15,.026))
    box('paint',(-2.87,0,3.21),(.027,9.15,.18))
    for y in (-2.53,2.72):
        for z in (3.28,3.44):
            box('galvanized',(-1.40,y,z),(2.97,.14,.024))
        box('paint',(-1.40,y,3.36),(2.97,.027,.16))
    # Surface-mounted conduit and service drops run behind the tool storage.
    for z in (1.03,1.12):
        m['galvanized'].cylinder((-2.895,-4.,z),(-2.895,3.76,z),.017,8)
    for y in (-1.52,2.50):
        box('frame',(-2.89,y,.72),(.10,.14,.20))
        m['galvanized'].cylinder((-2.90,y,.81),(-2.90,y,1.03),.017,8)


def _tool_head(batch, x, y, z, radius, bore, start, sweep):
    """Forged ring or open jaw, extruded in the wall normal direction."""
    steps=12
    angles=np.radians(np.linspace(start,start+sweep,steps+1))
    directions=[np.array((math.cos(a),math.sin(a))) for a in angles]
    if sweep==360:
        directions[-1]=directions[0]
    def point(depth,r,d):
        return (x+depth,y+r*d[0],z+r*d[1])
    thickness=.0035
    for a,b in zip(directions,directions[1:]):
        for side in (-1,1):
            dx=side*thickness
            batch.face([point(dx,radius,a),point(dx,radius,b),point(dx,bore,b),point(dx,bore,a)],(side,0,0))
        for r,sign in ((radius,1),(bore,-1)):
            normal=(0,*(sign*(a+b)))
            batch.face([point(-thickness,r,a),point(-thickness,r,b),
                        point(thickness,r,b),point(thickness,r,a)],normal,
                       [(0,*(sign*d)) for d in (a,b,b,a)])
    if sweep<360:
        for d,sign in ((directions[0],-1),(directions[-1],1)):
            batch.face([point(-thickness,bore,d),point(-thickness,radius,d),
                        point(thickness,radius,d),point(thickness,bore,d)],
                       (0,-sign*d[1],sign*d[0]))


def _tools(m):
    # The combination spanners hang through their box ends above the counter.
    # Overall lengths range from 151 to 224 mm.
    for y,length,width in ((-.54,.151,.025),(-.464,.186,.032),(-.378,.224,.039)):
        x,z=-2.754,1.78
        ring=width*.40
        head=width*.50
        head_z=z+ring-length+head
        m['steel'].bar_on_wall(x,(y,z-ring*.85),(y,head_z+head*.82),width*.32,.006)
        _tool_head(m['steel'],x,y,z,ring,ring*.55,0,360)
        _tool_head(m['steel'],x,y,head_z,head,head*.61,-40,267)
        m['frame'].cylinder((-2.79,y,z),(-2.744,y,z),.0033,6)

    # The square and caliper use separate hooks and read in different directions.
    m['steel'].bar_on_wall(-2.746,(-.98,1.36),(-.98,1.60),.026,.007)
    m['frame'].bar_on_wall(-2.741,(-.998,1.36),(-.835,1.36),.040,.014)
    m['steel'].bar_on_wall(-2.747,(-.20,1.55),(.046,1.55),.014,.007)
    for y in (-.18,-.115):
        m['steel'].bar_on_wall(-2.742,(y,1.562),(y,1.512),.009,.009)
    m['frame'].box((-2.739,-.11,1.55),(.019,.048,.031))
    m['pale-glass'].box((-2.724,-.11,1.551),(.008,.031,.015))

    m['frame'].box((-2.748,.50,1.53),(.060,.44,.022))
    for y,blade,grip,finish in ((.36,.092,.085,'red-tool'),(.45,.125,.105,'rubber'),
                               (.54,.074,.078,'red-tool'),(.65,.108,.093,'rubber')):
        m['steel'].cylinder((-2.738,y,1.535-blade),(-2.738,y,1.55),.0032,6)
        m[finish].cylinder((-2.738,y,1.544),(-2.738,y,1.544+grip),.014,8)


def _label_holder(m,y,z):
    """Stamped drawer label carrier with a protected, inset paper tab."""
    def rect(batch,x,y0,z0,w,h):
        batch.face([(x,y0+w*i,z0+h*j) for i,j in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))],(1,0,0))
    m['galvanized'].box((-2.102,y,z),(.011,.093,.035))
    rect(m['pale-glass'],-2.095,y,z,.076,.023)
    rect(m['frame'],-2.094,y-.009,z,.041,.0035)


def _back_shop(m):
    box=lambda material,c,s: m[material].box(c,s)
    # Powder-coated drawer cabinet, stainless counter, louver rack and hand tools.
    box('rubber',(-2.47,.05,.09),(.65,2.58,.16))
    box('paint',(-2.47,.05,.52),(.66,2.62,.78))
    drawer_banks = (
        (-.79,((.29,.25),(.50,.14),(.665,.14),(.812,.112))),
        (.055,((.326,.318),(.597,.190),(.797,.150))),
        (.90,((.283,.231),(.476,.119),(.605,.109),(.723,.097),(.824,.081))),
    )
    for yc,drawers in drawer_banks:
        for index,(z,h) in enumerate(drawers):
            box('drawer',(-2.124,yc,z),(.038,.797,h))
            box('frame',(-2.090,yc,z+h*.30),(.035,.35,.027))
            box('galvanized',(-2.064,yc,z+h*.30+.009),(.024,.34,.015))
            if index%2==0:
                _label_holder(m,yc+.291,z+.005)
    box('benchtop',(-2.46,.05,.94),(.78,2.74,.070))

    box('frame',(-2.865,-.03,1.60),(.078,2.46,1.14))
    for z in (1.14,1.28,1.42,1.56,1.70,1.84,1.98,2.10):
        box('dado',(-2.812,-.03,z),(.044,2.36,.106))
    _tools(m)

    # Side parts rack. Each tote has an open top and a low front picking lip.
    for x in (-2.79,-2.18):
        for y in (1.67,3.55):
            box('frame',(x,y,.87),(.042,.042,1.72))
    for z in (.19,.69,1.19,1.66):
        box('galvanized',(-2.485,2.61,z),(.66,1.96,.040))
    for row,z in enumerate((.24,.74,1.24)):
        for bay,y in enumerate((1.99,2.56,3.13)):
            if (row,bay)==(2,0):
                # Two latched instrument cases sit below the top shelf.
                for level in (0,1):
                    zcase=z+level*.106
                    box('frame',(-2.47,y,zcase+.041),(.36,.43,.082))
                    box('dado',(-2.47,y,zcase+.094),(.36,.431,.023))
                    for sign in (-1,1):
                        box('galvanized',(-2.281,y+sign*.135,zcase+.077),(.019,.033,.043))
                continue
            if (row,bay)==(1,2):
                continue
            finish='dado' if (row,bay) in ((0,0),(1,0),(2,2)) else 'drawer'
            h=.128 if (row,bay) in ((0,1),(2,2)) else .206
            zmid=z+h*.5
            box(finish,(-2.48,y,z),(.51,.48,.026))
            box(finish,(-2.72,y,zmid),(.024,.49,h))
            for side in (-1,1):
                box(finish,(-2.48,y+side*.229,zmid),(.50,.023,h))
            box(finish,(-2.236,y,z+.036),(.023,.49,.072))


def _workbench(m,layout):
    box=lambda material,c,s: m[material].box(c,s)
    xmid=(layout.pick_xy[0]+layout.place_xy[0])/2+.145
    ymid=(layout.pick_xy[1]+layout.place_xy[1])/2
    top=layout.support_z-.050
    width, span=.56,1.07
    box('benchtop',(xmid,ymid,top-.018),(width,span,.036))
    box('rubber',(xmid,ymid,top-.039),(width-.026,span-.026,.009))
    dx,dy=width/2-.058,span/2-.061
    for sx in (-1,1):
        for sy in (-1,1):
            x,y=xmid+sx*dx,ymid+sy*dy
            box('paint',(x,y,top/2+.005),(.044,.044,top-.065))
            m['steel'].cylinder((x,y,.015),(x,y,.05),.012,8)
            m['rubber'].cylinder((x,y,.001),(x,y,.021),.038,12)
        box('frame',(xmid+sx*dx,ymid,top-.078),(.040,2*dy,.065))
        box('paint',(xmid+sx*dx,ymid,.27),(.032,2*dy,.040))
    for sy in (-1,1):
        box('frame',(xmid,ymid+sy*dy,top-.078),(2*dx,.040,.065))
        box('paint',(xmid,ymid+sy*dy,.27),(2*dx,.032,.040))
    box('galvanized',(xmid,ymid,.292),(2*dx,2*dy,.013))

    # Both beds share the same calibrated support plane. Pin and screw heads
    # stay below it, clear of the gripper's two Y-facing contact pads.
    for index,(x,y) in enumerate((layout.pick_xy,layout.place_xy)):
        box('rubber',(x,y,top+.003),(.154,.133,.006))
        box('frame',(x,y,top+.017),(.158,.138,.022))
        box('benchtop',(x,y,top+.033),(.146,.125,.010))
        for sign in (-1,1):
            box('steel',(x+sign*.018,y,top+.044),(.014,.061,.012))
            for sy in (-1,1):
                cx,cy=x+sign*.057,y+sy*.042
                m['frame'].cylinder((cx,cy,top+.037),(cx,cy,top+.041),.006,8)
        # Visible bed coding on the camera-facing rim.
        for s in range(index+1):
            box('amber',(x+.0796,y-.024+s*.018,top+.017),(.0015,.009,.012))


def _outline(hx,hy,c):
    return [(-hx+c,-hy),(hx-c,-hy),(hx,-hy+c),(hx,hy-c),
            (hx-c,hy),(-hx+c,hy),(-hx,hy-c),(-hx,-hy+c)]


def _clip_x(poly,cut,keep_left):
    result=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        good_a=(a[0]<=cut) if keep_left else (a[0]>=cut)
        good_b=(b[0]<=cut) if keep_left else (b[0]>=cut)
        if good_a:
            result.append(a)
        if good_a!=good_b:
            t=(cut-a[0])/(b[0]-a[0])
            result.append((cut,a[1]+t*(b[1]-a[1])))
    return result


def _payload(m,size):
    hx,hy,hz=np.asarray(size)/2
    bevel=.0010
    outer=_outline(hx,hy,.004)
    flat=_outline(hx-bevel,hy-bevel,.0035)
    # +/-Y contact planes extend to hz-1 mm. The two horizontal ends and the
    # three actual through-bores establish the orientation in the transfer shot.
    for i,a in enumerate(outer):
        j=(i+1)%len(outer)
        b=outer[j]
        n=(b[1]-a[1],a[0]-b[0],0.)
        m['part'].face([(*a,-hz+bevel),(*b,-hz+bevel),(*b,hz-bevel),(*a,hz-bevel)],n)
        for sign in (-1,1):
            quad=[(*a,sign*(hz-bevel)),(*b,sign*(hz-bevel)),(*flat[j],sign*hz),(*flat[i],sign*hz)]
            edge=np.asarray(quad)
            normal=np.cross(edge[1]-edge[0],edge[2]-edge[0])
            if normal[2]*sign<0:
                normal=-normal
            m['cut'].face(quad,normal)
    seam=.48*hx
    for index,cx in enumerate((-.74*hx,0.,.74*hx)):
        poly=_clip_x(flat,-seam,True) if index==0 else (_clip_x(flat,seam,False) if index==2
             else _clip_x(_clip_x(flat,-seam,False),seam,True))
        ring=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            n=max(2,math.ceil(math.dist(a,b)/.006))
            ring.extend((a[0]+(b[0]-a[0])*k/n,a[1]+(b[1]-a[1])*k/n) for k in range(n))
        radial=[np.array([x-cx,y])/math.hypot(x-cx,y) for x,y in ring]
        r=.010 if index==1 else .0037
        chamfer=.0014 if index==1 else .0008
        for i,v in enumerate(radial):
            j=(i+1)%len(ring)
            w=radial[j]
            def rim(unit, radius, z):
                return (cx+unit[0]*radius,unit[1]*radius,z)
            for s in (-1,1):
                m['part'].face([(*ring[i],s*hz),(*ring[j],s*hz),rim(w,r+chamfer,s*hz),rim(v,r+chamfer,s*hz)],(0,0,s))
                m['cut'].face([rim(v,r+chamfer,s*hz),rim(w,r+chamfer,s*hz),
                               rim(w,r,s*(hz-chamfer)),rim(v,r,s*(hz-chamfer))],(-v[0],-v[1],s))
            m['bore'].face([rim(v,r,-hz+chamfer),rim(w,r,-hz+chamfer),
                rim(w,r,hz-chamfer),rim(v,r,hz-chamfer)],(-v[0],-v[1],0),
                [(-v[0],-v[1],0),(-w[0],-w[1],0),(-w[0],-w[1],0),(-v[0],-v[1],0)])


def append_workshop(scene,directory,layout):
    """Add the fixed room and a movable part with its local origin at its centre."""
    directory=Path(directory)
    directory.mkdir(exist_ok=True,parents=True)
    room={name:Batch() for name in FINISH}
    part={name:Batch() for name in ('part','cut','bore')}
    _room(room)
    _back_shop(room)
    _workbench(room,layout)
    _payload(part,layout.payload_size)
    meshes,materials,instances=list(scene.meshes),list(scene.materials),list(scene.instances)
    identity=max(i.id for i in instances)+1
    for node,batches in ((STATIC_NODE,room),(PAYLOAD_NODE,part)):
        for finish,batch in batches.items():
            if not batch.triangles:
                continue
            name=node+'-'+finish
            mesh=write_mesh(directory,name,batch.positions,batch.normals,batch.triangles)
            material=Material(name+'-material',*FINISH[finish])
            meshes.append(mesh)
            materials.append(material)
            instances.append(Instance(identity,mesh.id,material.id,node))
            identity+=1
    return replace(scene,key=scene.key+'-workshop-transfer',meshes=tuple(meshes),
                   materials=tuple(materials),instances=tuple(instances))
