"""Sunlit breakfast nook and entry bench for ten life-size household assets."""

from dataclasses import replace
from math import cos, isfinite, pi, radians, sin
from pathlib import Path

import numpy as np
from PIL import Image

from _eight_studio_set import _mesh
from _manipulation_photo_studio import _Surface
from _panda_scene import look_at, pose_matrix
from _panda_textures import IMAGE_DIR, WOOD_SHA256
from galaxea_r1.finish_details import _texture
from phymeo_rendering.assets import Material, TextureChannel, TexturedMaterial, TextureMap
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.photography import studio_lighting
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene


ROOM_NODE = 'home-room'
ROOM_EXPOSURE = 5.95
TABLE_HEIGHT = .75
BENCH_HEIGHT = .425
CONSOLE_HEIGHT = .70
_WINDOW_PLATE = (-6.05,1.95,4.30,-1.,3.50)
_WINDOW_FILE = 'garden_nook_window_1600.jpg'
_WINDOW_HASH = 'ab26d2c8109f4749d5c6cdb7f2e289b501e49a7d5c78dc0e639e3bd07731e2d3'


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
                   # Large extended emitters carry the working illumination.
                   panel_size_scale=1.20, panel_strength=.37,
                   environment_map=replace(preset.environment_map,
                       uri=(IMAGE_DIR/'garden_nook_1k.hdr').as_uri(),
                       sha256='f8c7970483be873487fabc0f07891be7e1551220dba911ea9dadd56f8a336a00',
                       # The original solar disc reaches 120,320 radiance units.
                       # At this gain its strongest channel reaches 78.208.
                       strength=.00065, yaw_degrees=42.))


def _unit(v):
    a = np.asarray(v, float)
    return a/np.linalg.norm(a)


def _slab(surface, low, high, radius, bevel):
    """Rounded plan and a small routed edge on a solid furniture component."""
    x0,y0,z0 = low
    x1,y1,z1 = high
    centers = ((x1-radius,y0+radius,-pi/2), (x1-radius,y1-radius,0),
               (x0+radius,y1-radius,pi/2), (x0+radius,y0+radius,pi))
    profile = []
    for cx,cy,start in centers:
        for i in range(5):
            angle=start+(pi/2)*i/4
            profile.append((np.array((cx,cy)),np.array((cos(angle),sin(angle)))))
    def point(center,normal,z,inset):
        return (*tuple(center+(radius-inset)*normal),z)
    for i,(c0,n0) in enumerate(profile):
        c1,n1=profile[(i+1)%len(profile)]
        for lo,hi,il,ih,nz in ((z0,z0+bevel,bevel,0,-1),
                               (z0+bevel,z1-bevel,0,0,0), (z1-bevel,z1,0,bevel,1)):
            surf_normal0=_unit((*n0,nz))
            surf_normal1=_unit((*n1,nz))
            surface.quad((point(c0,n0,lo,il), point(c1,n1,lo,il),
                          point(c1,n1,hi,ih), point(c0,n0,hi,ih)),
                         (surf_normal0,surf_normal1,surf_normal1,surf_normal0))
        for z,nz in ((z0,-1),(z1,1)):
            face=[((x0+x1)/2,(y0+y1)/2,z),point(c0,n0,z,bevel),point(c1,n1,z,bevel)]
            j=len(surface.points)
            surface.points.extend(face if nz>0 else face[::-1])
            surface.normals.extend(((0.,0.,nz),)*3)
            surface.faces.append((j,j+1,j+2))


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
    _slab(oak,(-1.91,1.35,.814),(-.16,1.55,.84),.013,.0025)
    p['floor'].quad(((-2.6,-4.5,0.),(2.6,-4.5,0.),(2.6,1.6,0.),(-2.6,1.6,0.)),((0.,0.,1.),)*4)


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
        _slab(oak,(x-.158,y-.132,.414),(x+.158,y+.131,.44),.026,.004)
        _slab(p['sage'],(x-.153,y-.128,.440),(x+.153,y+.122,.465),.028,.007)
        for sx in (-1,1):
            for sy in (-1,1):
                _rod(oak,(x+.144*sx,y+.126*sy,.005),
                     (x+.124*sx,y+.108*sy,.435),.012,.018)
            _rod(oak,(x+.130*sx,y+.1,.43),(x+.148*sx,y+.165,.85),.015,.014)
        for i in range(-2,3):
            _rod(oak,(x+i*.045,y+.102,.452),(x+i*.049,y+.160,.817),.006,.007)
        oak.block((x-.161,y+.144,.795),(x+.161,y+.175,.85))

    # Separate physical support heights for the entry bench and the clock cabinet.
    _slab(oak,(.307,-.294,.395),(.702,.238,BENCH_HEIGHT),.024,.0045)
    for x in (.329,.680):
        for y in (-.268,.206):
            _rod(walnut,(x,y,.008),(x,y,.403),.014,.018)
    for i in range(7):
        y = -.248+i*.066
        oak.block((.323,y,.118),(.689,y+.039,.137))
    p['sage'].block((.750,-.166,.095),(1.108,.315,.674))
    _slab(oak,(.731,-.184,.674),(1.126,.327,CONSOLE_HEIGHT),.016,.004)
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

    # Separate print, warm white mount and walnut shadow frame.
    walnut.block((.278,1.430,1.24),(.756,1.47,1.82))
    cream.block((.289,1.414,1.252),(.744,1.431,1.808))


def _map(filename, digest):
    return TextureMap((IMAGE_DIR/filename).as_uri(),digest)


def _tinted_scan(directory, name, filename, pigment, grain):
    """Paint and limewash preserve measured local reflectance variation."""
    with Image.open(IMAGE_DIR/filename) as photo:
        neutral = np.asarray(photo.convert('L'),dtype=np.float32)
    modulation = 1+grain*(neutral/max(1.,float(neutral.mean()))-1)
    return _texture(directory,'home-'+name,np.clip(np.rint(modulation[...,None]*pigment),0,255))


def _room_materials(directory):
    wood='wood_table_001_diff_1k.jpg'
    rough=TextureChannel(_map('wood_table_001_rough_1k.jpg',
                          'a2ca9ae89dfe7b8e2e5e92f0aa43467845417de8a50205b5852ef0d57c7624f9'))
    floor=_map('wood_floor_diff_1k.jpg','2a1c07687b6dbb214c4b9213739c6d92d425f1f0fc8ab3ac157f105d78240789')
    floor_rough=TextureChannel(_map('wood_floor_rough_1k.jpg',
                               '061f1e1293251b2d28c76e3fac1ea9452b0e8648bb8f9b3728e09db386166e5c'))
    result={'floor': (TexturedMaterial('home-floor',(1.,1.,1.),0.,.89,floor,floor_rough),1.7)}
    for name,pigment,grain,roughness in (
            ('oak',(191,159,121),.32,.82),('walnut',(131,99,75),.36,.78),
            ('sage',(147,169,153),.09,.90),('sage-light',(158,179,164),.11,.92)):
        image=_tinted_scan(directory,name,wood,pigment,grain)
        result[name]=(TexturedMaterial('home-'+name,(1.,1.,1.),0.,roughness,image,rough),1.5)
    plaster=_tinted_scan(directory,'limewash','white_plaster_02_diff_1k.jpg',(239,238,232),.06)
    result['plaster']=(TexturedMaterial('home-plaster',(1.,1.,1.),0.,.94,plaster),1.)
    return result


def _mapped_mesh(directory,key,surface,period):
    """Per-face planar charts in metres. Separate UV seams retain the smooth normals."""
    faces=np.asarray(surface.faces,dtype=np.int64)
    xyz=np.asarray(surface.points)[faces]
    normals=np.asarray(surface.normals)[faces]
    cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    major=np.argmax(np.abs(cross),axis=1)
    uv=np.empty((len(faces),3,2))
    for axis,(a,b) in enumerate(((1,2),(0,2),(1,0))):
        mask=major==axis
        uv[mask,:,0]=xyz[mask,:,a]/period+.21
        uv[mask,:,1]=-xyz[mask,:,b]/period+.34
    flat=xyz.reshape(-1,3)
    return _mesh(directory,key,flat,normals.reshape(-1,3),np.arange(len(flat)).reshape(-1,3),
                 uv=uv.reshape(-1,2))


def build_room(directory):
    """Build persistent room geometry. Append object instances on their own nodes."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    finishes = {
        'plaster': ((.88,.88,.85),0.,.93), 'cream': ((.91,.91,.87),0.,.65),
        'floor': ((.50,.40,.30),0.,.60), 'oak': ((.52,.37,.23),0.,.48),
        'walnut': ((.26,.15,.081),0.,.54), 'sage': ((.24,.36,.29),0.,.59),
        'sage-light': ((.30,.43,.35),0.,.53), 'brass': ((.62,.43,.19),.64,.38),
        'clay': ((.58,.235,.12),0.,.77),'green': ((.115,.233,.092),0.,.65),
        'leaf': ((.235,.363,.127),0.,.57), 'linen': ((.74,.64,.44),0.,.92),
    }
    surfaces = {name:_Surface() for name in finishes}
    _architecture(surfaces)
    _furniture(surfaces)
    meshes,materials,instances = [],[],[]
    mapped=_room_materials(directory)
    for name,surface in surfaces.items():
        key = 'home-'+name
        if name in mapped:
            material,period=mapped[name]
            meshes.append(_mapped_mesh(directory,key,surface,period))
            materials.append(material)
        else:
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
    # The same real garden was photographed for the outdoor plate, print and HDR.
    for key,filename,digest,x0,x1,y,z0,z1,color in (
        ('home-window-view',_WINDOW_FILE,_WINDOW_HASH,*_WINDOW_PLATE,(1.,1.,1.)),
        ('home-canopy-print','garden_nook_canopy_print.jpg',
         'b5efe0a41488534c6df8813c3c1e771b0c513c1f82103efcc63e52f43be0e6b4',
         .329,.705,1.412,1.300,1.775,(.95,.925,.89))):
        corners=((x0,y,z0),(x1,y,z0),(x1,y,z1),(x0,y,z1))
        meshes.append(_mesh(directory,key,corners,((0.,-1.,0.),)*4,((0,1,2),(0,2,3)),
                            uv=((.001,.999),(.999,.999),(.999,.001),(.001,.001))))
        materials.append(TexturedMaterial(key,color,0.,.97,_map(filename,digest)))
        instances.append(Instance(len(instances)+1,key,key,ROOM_NODE))
    sun=DirectionalLight(tuple(_unit((.48,-.76,-.59))),(.0035,.0035,.0034))
    sky_fill=DirectionalLight(tuple(_unit((.16,.86,-.48))),(.0012,.0015,.0020))
    return VisualScene('breakfast-nook-v4',tuple(meshes),tuple(materials),tuple(instances),(sun,sky_fill))


def _window_bounds(camera, camera_to_world):
    # Only rays passing through the architectural opening can reach the plate.
    # Project both faces of the wall with the actual (possibly off-axis) lens.
    corners = np.array([(x,y,z) for x in (-1.85,-.22) for y in (1.44,1.58)
                        for z in (.84,2.10)])
    local=(corners-camera_to_world[:3,3])@camera_to_world[:3,:3]
    depth=-local[:,2]
    if np.all(depth<=camera.near_m):
        return None
    if np.any(depth<=camera.near_m):
        return (0,camera.width,0,camera.height)
    px=camera.cx_px+camera.fx_px*local[:,0]/depth
    py=camera.cy_px-camera.fy_px*local[:,1]/depth
    x0=max(0,int(np.floor(px.min()))-2)
    x1=min(camera.width,int(np.ceil(px.max()))+2)
    y0=max(0,int(np.floor(py.min()))-2)
    y1=min(camera.height,int(np.ceil(py.max()))+2)
    return (x0,x1,y0,y1) if x0<x1 and y0<y1 else None


class _BackplateSession:
    """Display-referred exterior plate behind native depth-tested room geometry."""
    def __init__(self,native,scene):
        self.native=native
        self.plate_id,=(i.id for i in scene.instances if i.mesh_id=='home-window-view')
        code=np.arange(256,dtype=np.float32)/255.
        self.decode=np.where(code<=.04045,code/12.92,((code+.055)/1.055)**2.4)
        linear=np.linspace(0.,1.,65536,dtype=np.float32)
        self.encode=np.rint(255*np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)).astype(np.uint8)
        with Image.open(IMAGE_DIR/_WINDOW_FILE) as image:
            self.photo=self.decode[np.asarray(image.convert('RGB'))]
        self.plate_captures=0

    def diagnostics(self):
        data=dict(self.native.diagnostics())
        data['exterior_backplate']={'method':'display-referred photo behind native centre-sample instance visibility',
                                    'image_sha256':_WINDOW_HASH,'composited_captures':self.plate_captures}
        return data

    def capture(self,frame,request):
        if Channel.RGB not in request.channels:
            return self.native.capture(frame,request)
        camera=request.camera
        matrix=pose_matrix(camera.world_pose)
        bounds=_window_bounds(camera,matrix)
        if bounds is None:
            return self.native.capture(frame,request)
        channels=request.channels
        if Channel.INSTANCE_ID not in channels:
            channels=(*channels,Channel.INSTANCE_ID)
        result=self.native.capture(frame,replace(request,channels=channels))
        planes={plane.channel:plane for plane in result.planes}
        x0,x1,y0,y1=bounds
        ids=np.frombuffer(planes[Channel.INSTANCE_ID].data,'<u4').reshape(camera.height,camera.width)
        visible=ids[y0:y1,x0:x1]==self.plate_id
        if visible.any():
            rgb=planes[Channel.RGB]
            data=self._replace_visible(camera,matrix,bounds,visible,rgb.data)
            planes[Channel.RGB]=replace(rgb,data=data)
            self.plate_captures+=1
        return replace(result,planes=tuple(planes[channel] for channel in request.channels))

    def _replace_visible(self,camera,matrix,bounds,visible,raw):
        # Pixel ray/plane intersection rather than a screen-space cutout keeps
        # window parallax identical in crops, supersamples and offset-lens views.
        bx,_,by,_=bounds
        row,col=np.nonzero(visible)
        iy,ix=row+by,col+bx
        sx=(ix.astype(np.float32)+.5-camera.cx_px)/camera.fx_px
        sy=(camera.cy_px-iy.astype(np.float32)-.5)/camera.fy_px
        origin=matrix[:3,3]
        axis=matrix[:3,:3].astype(np.float32)
        ray_y=axis[1,0]*sx+axis[1,1]*sy-axis[1,2]
        x0,x1,y,z0,z1=_WINDOW_PLATE
        t=(y-origin[1])/ray_y
        x=origin[0]+t*(axis[0,0]*sx+axis[0,1]*sy-axis[0,2])
        z=origin[2]+t*(axis[2,0]*sx+axis[2,1]*sy-axis[2,2])
        height,width=self.photo.shape[:2]
        u=np.clip((.001+.998*(x-x0)/(x1-x0))*width-.5,0,width-1.001)
        v=np.clip((.999-.998*(z-z0)/(z1-z0))*height-.5,0,height-1.001)
        uu,vv=u.astype(np.int32),v.astype(np.int32)
        du,dv=(u-uu).astype(np.float32)[:,None],(v-vv).astype(np.float32)[:,None]
        top=self.photo[vv,uu]*(1-du)+self.photo[vv,uu+1]*du
        bottom=self.photo[vv+1,uu]*(1-du)+self.photo[vv+1,uu+1]*du
        sampled=(1-dv)*top+dv*bottom
        dest=bytearray(raw)
        rgb=np.frombuffer(dest,np.uint8).reshape(camera.height,camera.width,3)

        # RGB already contains native subpixel foreground coverage. Keep some of
        # that resolve at mixed silhouettes; never write any foreground-owned sample.
        halo=np.pad(visible,1,mode='edge')
        coverage=(4+halo[row,col+1].astype(np.float32)+halo[row+2,col+1]
                  +halo[row+1,col]+halo[row+1,col+2])*.125
        border=coverage<1
        alpha=coverage[border,None]
        sampled[border]=sampled[border]*alpha+self.decode[rgb[iy[border],ix[border]]]*(1-alpha)
        rgb[iy,ix]=self.encode[np.rint(np.clip(sampled,0.,1.)*65535).astype(np.uint16)]
        return bytes(dest)


def with_home_backplate(session,scene):
    """Wrap a native session once, ahead of the film's AA, focus and lens references."""
    return _BackplateSession(session,scene)


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


def _track(seconds, column, keys=_SHOTS):
    i=next((j for j in range(len(keys)-1) if seconds <= keys[j+1][0]),len(keys)-2)
    t0,t1=keys[i][0],keys[i+1][0]
    a=(seconds-t0)/(t1-t0)

    def tangent(j):
        if j in (0,len(keys)-1):
            return np.zeros(3)
        before,after=keys[j-1],keys[j+1]
        return (np.asarray(after[column])-before[column])/(after[0]-before[0])

    return ((2*a**3-3*a*a+1)*np.asarray(keys[i][column])
            +(a**3-2*a*a+a)*(t1-t0)*tangent(i)
            +(-2*a**3+3*a*a)*np.asarray(keys[i+1][column])
            +(a**3-a*a)*(t1-t0)*tangent(i+1))


_FOCUS = (
    (0.,(-.15,-.025,.79)), (4.,(-.26,-.02,.79)),
    (7.5,(-.75,.00,.83)), (10.8,(-.532,-.135,.773)),
    (11.5,(-.532,-.135,.773)), (12.4,(-.37,-.03,.785)),
    (15.3,(.068,-.129,.793)), (18.,(.12,-.04,.76)),
    (21.7,(.71,-.002,.625)), (24.,(.75,.025,.64)),
    (27.,(.19,.005,.72)), (30.,(-.15,-.025,.79)),
)


def room_focus_target(seconds):
    """World-space focus pull, tracking subject surfaces independently of camera aim."""
    if not isfinite(seconds) or not 0 <= seconds <= 30:
        raise ValueError('home focus time must be between 0 and 30 seconds')
    return tuple(_track(seconds,1,_FOCUS))


def room_camera(frame_key,t,width=1280,height=720):
    """Continuous thirty-second establishing, breakfast, entry and returning shot."""
    if not isfinite(t) or not 0 <= t <= 30:
        raise ValueError('home camera time must be between 0 and 30 seconds')
    eye,aim=_track(t,1),_track(t,2)
    focal=.91*width
    return Camera('home-tour',frame_key,look_at(eye,aim,(0.,0.,1.)),width,height,
                  focal,focal,width/2,height/2,.03,12.)
