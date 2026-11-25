"""Sunlit breakfast nook and entry bench for ten life-size household assets."""

from dataclasses import replace
from math import cos, isfinite, pi, radians, sin
from pathlib import Path

import numpy as np

from _eight_studio_set import _mesh
from _manipulation_photo_studio import _Surface
from _panda_scene import look_at
from _panda_textures import IMAGE_DIR, WOOD_SHA256
from galaxea_r1.finish_details import _texture
from phymeo_rendering.assets import Material, TexturedMaterial, TextureMap
from phymeo_rendering.cameras import Camera
from phymeo_rendering.photography import studio_lighting
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene


ROOM_NODE = 'home-room'
ROOM_EXPOSURE = 5.95
TABLE_HEIGHT = .75
BENCH_HEIGHT = .425
CONSOLE_HEIGHT = .70


def _placement(x, y, z, yaw_deg=0):
    a = radians(yaw_deg)*.5
    return Pose((x, y, z), (0., 0., sin(a), cos(a)))


# An object's origin is the middle of its footprint at the lowest vertex.
# Tray, plate and embossed coaster offsets use the upper surface inside each
# placed object's footprint, measured from the source triangles in metres.
HOME_PLACEMENTS = {
    '001_bottle': _placement(-.84, .14, TABLE_HEIGHT, -16),
    '002_bowl': _placement(-.80, -.12, TABLE_HEIGHT, 9),
    '003_plate': _placement(-.53, -.15, TABLE_HEIGHT, 6),
    '005_french-fries': _placement(-.2067, .136, TABLE_HEIGHT+.003213, -12),
    '006_hamburg': _placement(-.5316, -.1351, TABLE_HEIGHT+.002225, 10),
    '008_tray': _placement(-.15, .13, TABLE_HEIGHT, -6),
    '019_coaster': _placement(.07, -.13, TABLE_HEIGHT, 8),
    '021_cup': _placement(.0685, -.1288, TABLE_HEIGHT+.005391, -12),
    '041_shoe': _placement(.51, -.10, BENCH_HEIGHT, -35),
    '046_alarm-clock': _placement(.91, .10, CONSOLE_HEIGHT, -9),
}


def room_lighting():
    preset = studio_lighting()
    return replace(preset, center_m=(-.12, .02, .72), scale_m=1.36,
                   panel_size_scale=.84, panel_strength=.21,
                   environment_map=replace(preset.environment_map,strength=.062))


def _unit(v):
    a = np.asarray(v, float)
    return a/np.linalg.norm(a)


def _rod(s, start, end, r, end_r=None, sides=12):
    """Tapered round dowel. End faces meet the surrounding joinery."""
    start, end = np.asarray(start), np.asarray(end)
    axis = _unit(end-start)
    u = _unit(np.cross(axis, (0, 1, 0) if abs(axis[1]) < .85 else (0, 0, 1)))
    v = np.cross(axis, u)
    end_r = r if end_r is None else end_r
    ring = [cos(2*pi*i/sides)*u + sin(2*pi*i/sides)*v for i in range(sides)]
    for i, n in enumerate(ring):
        m = ring[(i+1) % sides]
        s.quad((start+r*n, start+r*m, end+end_r*m, end+end_r*n), (n,m,m,n))


def _plant(parts, x, y, z, scale):
    def point(dx, dy, dz):
        return np.array((x+scale*dx, y+scale*dy, z+scale*dz))

    pot = parts['clay']
    pot.turned(x, y, [(r*scale, z+h*scale) for r,h in (
        (.075,0),(.084,.012),(.114,.163),(.117,.181),(.100,.181),(.097,.16))], 32)
    for branch in range(9):
        a = branch*2.39996
        spread = .115+.04*(branch % 3)
        height = .28+.05*((branch*5) % 4)
        tip = point(spread*cos(a), spread*sin(a), height)
        base = point(.01*cos(a), .01*sin(a), .155)
        _rod(parts['green'], base, tip, scale*.004, scale*.0018, 6)
        side = np.array((-sin(a), cos(a), 0.))
        for k in range(3):
            stem = base+(tip-base)*(.45+.22*k)
            offset = side*(-1 if k % 2 else 1)
            leafend = stem+scale*(.064*offset+np.array((.034*cos(a),.034*sin(a),.045)))
            transverse = _unit(np.cross(leafend-stem, (0,0,1)))*(.021*scale)
            blade = parts['leaf' if (branch+k) % 3 else 'green']
            for near,far in ((.012,.25),(.25,.53),(.53,.77),(.77,.99)):
                def section(at):
                    middle = (1-at)*stem+at*leafend+np.array((0.,0.,.009*scale*sin(pi*at)))
                    width = transverse*sin(pi*at)**.8
                    return middle-width, middle+width

                left,right = section(near)
                left_end,right_end = section(far)
                verts = (left,right,right_end,left_end)
                normal = _unit(np.cross(right-left,right_end-left))
                blade.quad(verts,(normal,)*4)
                blade.quad(verts[::-1],(-normal,)*4)


def _garden_tree(parts, x, y, height, span, phase):
    """Layered foliage on forked trunks, rooted on the garden side of the fence."""
    root = np.array((x,y,-.15))
    fork = np.array((x+.048*span,y+.025,height*.58))
    _rod(parts['walnut'],root,fork,span*.075,span*.048)
    for j in range(7):
        a = 2*pi*j/7+phase
        dx,dy = cos(a)*span*.49,sin(a)*span*.32
        z = height*(.76+.12*sin(2.7*j+phase))
        c = np.array((x+dx,y+dy,z))
        _rod(parts['walnut'],fork,c,span*.025,span*.012,8)
        radii = np.array((span*.48,span*.37,height*(.17+.035*cos(j))))
        surface = parts[('garden-shadow','garden-mid','garden-sun')[j%3]]
        def shell(u,v):
            n = np.array((cos(v)*cos(u),cos(v)*sin(u),sin(v)))
            rough = 1+.073*sin(3*u+.4*j)*cos(v)+.046*cos(5*u+2*v+phase)*cos(v)
            return c+rough*radii*n, _unit(n/radii)
        # Latitude rings exclude both poles. Each cap uses one exact axial
        # vertex, so the tessellation remains nondegenerate after FP32 packing.
        south = len(surface.points)
        surface.points.append(c-(0.,0.,radii[2]))
        surface.normals.append((0.,0.,-1.))
        for row in range(1,7):
            latitude = -.5*pi+pi*row/7
            for col in range(16):
                point,normal = shell(2*pi*col/16,latitude)
                surface.points.append(point)
                surface.normals.append(normal)
        north = len(surface.points)
        surface.points.append(c+(0.,0.,radii[2]))
        surface.normals.append((0.,0.,1.))
        first, last = south+1, north-16
        for col in range(16):
            nxt=(col+1)%16
            surface.faces.append((south,first+nxt,first+col))
            surface.faces.append((last+col,last+nxt,north))
            for row in range(5):
                base=first+16*row
                surface.faces.extend(((base+col,base+nxt,base+16+nxt),
                                      (base+col,base+16+nxt,base+16+col)))


def _architecture(p):
    wall, trim, oak = p['plaster'], p['cream'], p['oak']
    wall.block((-3.,1.48,-.08), (3.,1.58,.84))
    wall.block((-3.,1.48,2.10), (3.,1.58,3.0))
    wall.block((-3.,1.48,.84), (-1.85,1.58,2.10))
    wall.block((-.22,1.48,.84), (3.,1.58,2.10))
    wall.block((-2.58,-4.,0.), (-2.50,1.48,3.0))
    wall.block((2.56,-4.,0.), (2.64,1.48,3.0))
    trim.block((-2.50,1.455,0.), (2.56,1.48,.091))
    for x in (-1.85,-.22):
        trim.block((x-.026,1.445,.82), (x+.026,1.525,2.13))
    for z in (.84,2.10):
        trim.block((-1.88,1.44,z-.023), (-.19,1.52,z+.023))
    for x in (-1.31,-.77):
        trim.block((x-.012,1.48,.86), (x+.012,1.51,2.08))
    trim.block((-1.83,1.48,1.66), (-.24,1.51,1.68))
    oak.block((-1.91,1.35,.814),(-.16,1.55,.84))
    p['garden-shadow'].block((-4.8,3.68,-.16),(4.,3.78,.57))
    _garden_tree(p,-2.75,4.23,1.31,.82,.4)
    _garden_tree(p,-1.83,3.98,1.12,.67,1.8)
    _garden_tree(p,-.64,3.36,.95,.48,2.9)

    p['floor-base'].block((-2.6,-4.5,-.035),(2.6,1.6,-.005))
    for j in range(24):
        yy = -4.4+j*.25
        offset = .47*(j%3)
        for i in range(6):
            left = max(-2.6, -4.2+i*1.52+offset)
            right = min(2.6, -4.2+(i+1)*1.52+offset-.003)
            if left < right:
                p['floor-warm' if (i+j)%3 else 'floor-light'].block((left,yy,-.005),(right,yy+.248,0.))


def _furniture(p):
    oak, walnut, cream = p['oak'],p['walnut'],p['cream']
    # Solid edge, apron and four splayed legs support the textured dining surface.
    oak.block((-1.03,-.34,.712),(.235,.325,TABLE_HEIGHT-.0005))
    for y in (-.272,.255):
        oak.block((-.96,y,.644),(.165,y+.025,.713))
    for x in (-.947,.151):
        for y in (-.264,.249):
            _rod(oak,(x+(-.034 if x<0 else .034),y+(-.031 if y<0 else .031),.005),
                 (x,y,.714),.020,.030,16)
    # Two dining chairs behind the table: shallow seat, curved spindle back, free space below.
    for x in (-.76,-.045):
        y = .60
        oak.block((x-.158,y-.132,.414),(x+.158,y+.131,.44))
        p['sage'].block((x-.153,y-.128,.440),(x+.153,y+.122,.465))
        for sx in (-1,1):
            for sy in (-1,1):
                _rod(oak,(x+.144*sx,y+.126*sy,.005),
                     (x+.124*sx,y+.108*sy,.435),.012,.018)
            _rod(oak,(x+.130*sx,y+.1,.43),(x+.148*sx,y+.165,.85),.015,.014)
        for i in range(-2,3):
            _rod(oak,(x+i*.045,y+.102,.452),(x+i*.049,y+.160,.817),.006,.007)
        oak.block((x-.161,y+.144,.795),(x+.161,y+.175,.85))

    # Separate physical support heights for the entry bench and the clock cabinet.
    oak.block((.307,-.294,.395),(.702,.238,BENCH_HEIGHT))
    for x in (.329,.680):
        for y in (-.268,.206):
            _rod(walnut,(x,y,.008),(x,y,.403),.014,.018)
    for i in range(7):
        y = -.248+i*.066
        oak.block((.323,y,.118),(.689,y+.039,.137))
    p['sage'].block((.750,-.166,.095),(1.108,.315,.674))
    oak.block((.731,-.184,.674),(1.126,.327,CONSOLE_HEIGHT))
    for x in (.775,1.082):
        for y in (-.141,.285):
            _rod(walnut,(x,y,0.),(x,y,.101),.018)
    p['sage-light'].block((.76,-.180,.117),(1.098,-.166,.43))
    p['sage-light'].block((.76,-.180,.442),(1.098,-.166,.654))
    for z in (.391,.61):
        p['brass'].block((.892,-.196,z),(.967,-.180,z+.008))

    # A narrow shelf under the print and a plant by the window complete the rear wall.
    oak.block((.04,1.28,1.048),(1.07,1.47,1.073))
    for x in (.18,.91):
        walnut.block((x,1.39,.972),(x+.026,1.46,1.051))
    for i,(name,h) in enumerate((('clay',.126),('sage',.149),('linen',.112),('walnut',.133))):
        x=.116+i*.032
        p[name].block((x,1.328,1.073),(x+.029,1.428,1.073+h))
    _plant(p,.86,1.354,1.073,.58)
    _plant(p,-1.65,1.42,.84,.88)

    # Small abstract relief in a real shadow frame rather than a decal on the wall.
    walnut.block((.278,1.430,1.24),(.756,1.47,1.82))
    cream.block((.289,1.414,1.252),(.744,1.431,1.808))
    p['linen'].block((.322,1.410,1.284),(.711,1.415,1.775))
    for name,cx,cz,rx,rz,start,end in (
        ('clay',.516,1.486,.141,.192,0,pi),('sage',.501,1.483,.143,.107,pi,2*pi)):
        surf=p[name]
        for i in range(28):
            a=start+(end-start)*i/28
            b=start+(end-start)*(i+1)/28
            tri=((cx,1.406,cz),(cx+rx*cos(a),1.406,cz+rz*sin(a)),
                 (cx+rx*cos(b),1.406,cz+rz*sin(b)))
            base=len(surf.points)
            surf.points.extend(tri)
            surf.normals.extend(((0.,-1.,0.),)*3)
            surf.faces.append((base,base+1,base+2))


def build_room(directory):
    """Build persistent room geometry. Append object instances on their own nodes."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    finishes = {
        'plaster': ((.88,.88,.85),0.,.93), 'cream': ((.91,.91,.87),0.,.65),
        'floor-base': ((.33,.27,.20),0.,.82), 'floor-warm': ((.50,.40,.30),0.,.60),
        'floor-light': ((.54,.45,.34),0.,.62), 'oak': ((.52,.37,.23),0.,.48),
        'walnut': ((.26,.15,.081),0.,.54), 'sage': ((.24,.36,.29),0.,.59),
        'sage-light': ((.30,.43,.35),0.,.53), 'brass': ((.62,.43,.19),.64,.38),
        'clay': ((.58,.235,.12),0.,.77),'green': ((.115,.233,.092),0.,.65),
        'leaf': ((.235,.363,.127),0.,.57), 'linen': ((.74,.64,.44),0.,.92),
        'garden-shadow': ((.12,.24,.17),0.,.96),
        'garden-mid': ((.20,.36,.24),0.,.94), 'garden-sun': ((.32,.46,.28),0.,.92),
    }
    surfaces = {name:_Surface() for name in finishes}
    _architecture(surfaces)
    _furniture(surfaces)
    meshes,materials,instances = [],[],[]
    for name,surface in surfaces.items():
        key = 'home-'+name
        meshes.append(surface.mesh(directory,key))
        materials.append(Material(key,*finishes[name]))
        instances.append(Instance(len(instances)+1,key,key,ROOM_NODE))
    corners=((-1.03,-.34,TABLE_HEIGHT),(.235,-.34,TABLE_HEIGHT),
             (.235,.325,TABLE_HEIGHT),(-1.03,.325,TABLE_HEIGHT))
    key='home-dining-top'
    meshes.append(_mesh(directory,key,corners,((0.,0.,1.),)*4,((0,1,2),(0,2,3)),
                        uv=((.04,.12),(.96,.12),(.96,.66),(.04,.66))))
    color_map=TextureMap((IMAGE_DIR/'wood_table_001_diff_1k.jpg').as_uri(),WOOD_SHA256)
    materials.append(TexturedMaterial(key,(.92,.89,.86),0.,.54,base_color_map=color_map))
    instances.append(Instance(len(instances)+1,key,key,ROOM_NODE))
    # A far backdrop seen only through the real window. The restrained hue
    # change from horizon to zenith remains continuous as the lens travels.
    key='home-garden-sky'
    sky_corners=((-5.,5.24,-.3),(5.,5.24,-.3),(5.,5.24,3.8),(-5.,5.24,3.8))
    meshes.append(_mesh(directory,key,sky_corners,((0.,-1.,0.),)*4,((0,1,2),(0,2,3)),
                        uv=((.125,.995),(.875,.995),(.875,.005),(.125,.005))))
    s=np.linspace(0,1,256)[:,None,None]
    gradient=(1-s)*np.array((177,213,244))+s*np.array((241,246,247))
    sky_texture=_texture(directory,key,np.tile(np.rint(gradient),(1,4,1)))
    materials.append(TexturedMaterial(key,(1.,1.,1.),0.,.98,base_color_map=sky_texture))
    instances.append(Instance(len(instances)+1,key,key,ROOM_NODE))
    sun=DirectionalLight(tuple(_unit((.48,-.76,-.59))),(.14,.14,.137))
    sky_fill=DirectionalLight(tuple(_unit((.16,.86,-.48))),(.050,.065,.090))
    return VisualScene('breakfast-nook-v2',tuple(meshes),tuple(materials),tuple(instances),(sun,sky_fill))


# Time, optical centre, aim. Curves interpolate in metres with continuous velocity.
_SHOTS = (
    (0.,(.19,-2.57,1.60),(-.005,.055,.985)),
    (4.,(-.05,-2.40,1.60),(-.12,.055,.905)),
    (7.5,(-.83,-.96,1.29),(-.68,.035,.83)),
    (10.8,(-.62,-.50,1.12),(-.54,-.095,.775)),
    (12.4,(-.46,-.56,1.16),(-.40,-.01,.785)),
    (15.3,(-.11,-.52,1.15),(-.105,-.003,.79)),
    (18.,(.20,-.99,1.29),(.07,.015,.75)),
    (21.7,(.73,-.95,1.09),(.70,-.005,.59)),
    (24.,(1.07,-1.06,1.10),(.76,.018,.60)),
    (27.,(.77,-1.98,1.51),(.21,.045,.70)),
    (30.,(.19,-2.57,1.60),(-.005,.055,.985)),
)


def _track(seconds, column):
    i=next((j for j in range(len(_SHOTS)-1) if seconds <= _SHOTS[j+1][0]),len(_SHOTS)-2)
    t0,t1=_SHOTS[i][0],_SHOTS[i+1][0]
    a=(seconds-t0)/(t1-t0)

    def tangent(j):
        if j in (0,len(_SHOTS)-1):
            return np.zeros(3)
        before,after=_SHOTS[j-1],_SHOTS[j+1]
        return (np.asarray(after[column])-before[column])/(after[0]-before[0])

    return ((2*a**3-3*a*a+1)*np.asarray(_SHOTS[i][column])
            +(a**3-2*a*a+a)*(t1-t0)*tangent(i)
            +(-2*a**3+3*a*a)*np.asarray(_SHOTS[i+1][column])
            +(a**3-a*a)*(t1-t0)*tangent(i+1))


def room_camera(frame_key,t,width=1280,height=720):
    """Continuous thirty-second establishing, breakfast, entry and returning shot."""
    if not isfinite(t) or not 0 <= t <= 30:
        raise ValueError('home camera time must be between 0 and 30 seconds')
    eye,aim=_track(t,1),_track(t,2)
    focal=.91*width
    return Camera('home-tour',frame_key,look_at(eye,aim,(0.,0.,1.)),width,height,
                  focal,focal,width/2,height/2,.03,12.)
