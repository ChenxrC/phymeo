"""One metre-scale dining room, kitchen and study for the thirty-object film."""

from dataclasses import replace
from math import cos, isfinite, pi, radians, sin
from pathlib import Path

import numpy as np

from _home_showcase_room import (
    HOME_PLACEMENTS, ROOM_NODE, _mapped_mesh, _placement, _rod, _slab, _track, build_room, room_lighting,
)
from _manipulation_photo_studio import _Surface
from _panda_scene import look_at
from phymeo_rendering.assets import Material, TexturedMaterial
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import Instance, VisualScene


# Rectangle (left, right, front, rear, top). These are the actual support surfaces.
_SUPPORTS = {
    'counter': (-2.38,-1.13,-.43,.435,.885),
    'hob': (-2.31,-1.37,-.002,.391,.892),
    'pantry': (-2.36,-1.14,.77,1.23,1.125),
    'board': (-1.761,-1.589,.866,1.055,1.139),
    'desk': (1.174,2.378,-.385,.344,.765),
    'mat': (1.211,1.675,-.298,.287,.768),
}

# Key, footprint centre, yaw and supporting piece of furniture. Paired items
# occupy separate footprints so that their authored geometry remains inspectable.
_NEW_OBJECTS = (
    ('047_mouse',1.305,-.174,14,'mat'), ('048_stapler',1.526,-.179,-22,'mat'),
    ('050_bell',1.765,.143,-12,'desk'), ('057_toycar',1.963,-.128,-32,'desk'),
    ('060_kitchenpot',-2.055,.177,9,'hob'), ('062_plasticbox',-2.035,-.239,3,'counter'),
    ('071_can',-1.246,.225,-12,'counter'), ('072_electronicscale',-1.72,-.239,-4,'counter'),
    ('073_rubikscube',2.138,.119,17,'desk'), ('075_bread',-1.674,.958,-15,'board'),
    ('076_breadbasket',-1.919,1.012,8,'pantry'), ('077_phone',1.329,.050,-11,'mat'),
    ('078_phonestand',1.543,.142,-10,'mat'), ('081_playingcards',1.774,-.191,11,'desk'),
    ('086_woodenblock',2.218,-.15,-12,'desk'), ('105_sauce-can',-1.406,-.222,10,'counter'),
    ('106_skillet',-1.606,.188,-47,'hob'), ('107_soap',-2.263,-.241,11,'counter'),
    ('110_basket',-2.184,1.005,-6,'pantry'), ('112_tea-box',-1.362,1.003,-8,'pantry'),
)


def collection_lighting():
    """Soft key over the kitchen and a broad opposing fill over the study."""
    return replace(room_lighting(),center_m=(.01,-.17,.81),scale_m=1.43,
                   panel_size_scale=1.29,panel_strength=.43)


def collection_placements(assets):
    """Place every supplied asset at its real size with its original lowest vertex on the support."""
    by_key={asset.key:asset for asset in assets}
    expected=set(HOME_PLACEMENTS)|{row[0] for row in _NEW_OBJECTS}
    if len(by_key)!=len(assets) or set(by_key)!=expected:
        raise ValueError(f'thirty-object layout has missing {sorted(expected-set(by_key))} '
                         f'or extra {sorted(set(by_key)-expected)} asset groups')
    poses=dict(HOME_PLACEMENTS)
    footprints=[]
    for key,x,y,angle,support in _NEW_OBJECTS:
        low,high=np.asarray(by_key[key].bounds,float)
        if low.shape!=(3,) or not np.isfinite((low,high)).all() or np.any(high<=low):
            raise ValueError(f'{key} has invalid metric bounds')
        a=radians(angle)
        rotation=np.array(((cos(a),-sin(a)),(sin(a),cos(a))))
        corners=np.array([(u,v) for u in (low[0],high[0]) for v in (low[1],high[1])])
        corners=corners@rotation.T+(x,y)
        lo,hi=corners.min(axis=0),corners.max(axis=0)
        x0,x1,y0,y1,z=_SUPPORTS[support]
        if lo[0]<x0 or hi[0]>x1 or lo[1]<y0 or hi[1]>y1:
            raise ValueError(f'{key} extends beyond the {support} surface')
        for other,other_lo,other_hi in footprints:
            if np.all(hi>other_lo) and np.all(other_hi>lo):
                raise ValueError(f'{key} and {other} have overlapping footprints')
        footprints.append((key,lo,hi))
        poses[key]=_placement(x,y,float(z-low[2]),angle)
    return poses


def _ring(surface,x,y,z,inner,outer):
    for k in range(48):
        a,b=2*pi*k/48,2*pi*(k+1)/48
        corners=[(x+r*cos(t),y+r*sin(t),z) for r,t in ((inner,a),(outer,a),(outer,b),(inner,b))]
        surface.quad(corners,((0.,0.,1.),)*4)


def _top(surface,support,thickness,radius,bevel):
    x0,x1,y0,y1,z=_SUPPORTS[support]
    _slab(surface,(x0,y0,z-thickness),(x1,y1,z),radius,bevel)


def _kitchen(p):
    _top(p['counterstone'],'counter',.038,.020,.004)
    _top(p['hob'],'hob',.007,.014,.0016)
    for x,y,r in ((-2.055,.177,.116),(-1.606,.188,.103)):
        _ring(p['etched'],x,y,.89208,r,r+.0017)
    for k in range(4):
        x=-1.511+.023*k
        p['etched'].block((x,.027,.89202),(x+.012,.029,.89208))
    # Painted fronts, drawer gaps, recessed plinth and real brass pulls.
    p['walnut'].block((-2.342,-.338,.024),(-1.168,.397,.120))
    p['sage'].block((-2.356,-.372,.110),(-1.154,.408,.849))
    for left,right in ((-2.345,-1.970),(-1.959,-1.555),(-1.544,-1.166)):
        for lo,hi in ((.123,.641),(.650,.826)):
            p['sage-light'].block((left,-.390,lo),(right,-.370,hi))
            center=(left+right)/2
            for dx in (-.050,.050):
                p['brass'].block((center+dx-.004,-.409,hi-.040),
                                 (center+dx+.004,-.388,hi-.032))
            p['brass'].block((center-.058,-.413,hi-.040),(center+.058,-.406,hi-.032))

    _top(p['oak'],'pantry',.028,.014,.003)
    _top(p['walnut'],'board',.014,.017,.002)
    p['oak'].block((-2.334,.805,.527),(-1.168,1.195,.551))
    for x in (-2.319,-1.182):
        for y in (.812,1.189):
            _rod(p['walnut'],(x,y,.016),(x,y,1.100),.015,.021)
    p['oak'].block((-2.335,1.170,.995),(-1.166,1.201,1.100))


def _study(p):
    _top(p['oak'],'desk',.032,.026,.005)
    _top(p['blotter'],'mat',.003,.026,.00075)
    for x in (1.219,2.331):
        for y in (-.327,.290):
            dx=-.019 if x<1.6 else .019
            dy=-.018 if y<0 else .018
            _rod(p['walnut'],(x+dx,y+dy,.007),(x,y,.739),.016,.027)
    p['walnut'].block((1.235,.244,.643),(2.310,.280,.739))
    # A shadow gap separates the shallow writing drawer from its apron.
    p['oak'].block((1.239,-.317,.658),(2.306,-.294,.733))
    p['brass'].block((1.648,-.329,.692),(1.838,-.315,.700))


def build_collection_room(cache):
    """Reuse the breakfast room, window plate and materials; append the kitchen and study."""
    directory=Path(cache).resolve()
    base=build_room(directory)
    extras={m.id:m for m in (
        Material('collection-counterstone',(.72,.695,.65),.04,.51),
        Material('collection-hob',(.024,.032,.032),.12,.27),
        Material('collection-etched',(.34,.39,.37),.08,.53),
        Material('collection-blotter',(.125,.175,.182),0.,.77),
    )}
    materials={m.id:m for m in base.materials}|extras
    surfaces={key:_Surface() for key in ('counterstone','hob','etched','blotter',
                                         'oak','walnut','sage','sage-light','brass')}
    _kitchen(surfaces)
    _study(surfaces)
    meshes,instances=list(base.meshes),list(base.instances)
    for name,surface in surfaces.items():
        key='collection-'+name
        material=materials[key if key in extras else 'home-'+name]
        mesh=(_mapped_mesh(directory,key,surface,1.5) if isinstance(material,TexturedMaterial)
              else surface.mesh(directory,key))
        meshes.append(mesh)
        instances.append(Instance(len(instances)+1,mesh.id,material.id,ROOM_NODE))
    return VisualScene(base.key+'-thirty-v1',tuple(meshes),tuple(materials.values()),tuple(instances),base.lights)


# Time in seconds, optical centre, aim. The camera physically travels between
# full-size stations; shelf and desktop detail do not depend on enlarged assets.
_TOUR=(
    (0.,(.08,-4.21,2.22),(-.015,.12,.83)),
    (1.65,(-.34,-3.79,2.15),(-.28,.13,.86)),
    (3.3,(-1.48,-1.66,1.68),(-1.77,.13,.99)),
    (4.5,(-1.96,-.95,1.52),(-1.94,.036,.975)),
    (6.6,(-1.96,-.32,1.68),(-1.84,.988,1.205)),
    (8.2,(-1.49,-.78,1.53),(-1.54,-.07,.98)),
    (9.6,(-1.05,-1.03,1.42),(-.93,.01,.88)),
    (11.5,(-.64,-.69,1.22),(-.54,-.078,.79)),
    (12.6,(-.68,-.47,1.15),(-.60,.10,.80)),
    (14.,(-.705,.064,.996),(-.758,.748,.710)),
    (14.65,(-.66,.02,1.055),(-.72,.694,.746)),
    (16.,(-.13,-.71,1.22),(-.070,-.012,.802)),
    (18.,(.72,-1.05,1.16),(.69,.006,.605)),
    (19.5,(1.17,-1.24,1.33),(1.14,.005,.771)),
    (21.2,(1.39,-.69,1.22),(1.45,.012,.804)),
    (22.5,(1.58,-.535,1.17),(1.58,.053,.821)),
    (24.7,(2.13,-.68,1.22),(2.02,-.018,.812)),
    (26.3,(1.92,-1.44,1.48),(1.70,.003,.78)),
    (28.55,(.55,-3.49,2.03),(.25,.07,.805)),
    (30.,(.08,-4.21,2.22),(-.015,.12,.83)),
)


COLLECTION_CHAPTERS=(
    (0,'全屋总览'),(135,'灶台与备餐工具'),(198,'早餐架与茶'),(246,'台秤与调料'),(288,'饮料与碗'),
    (345,'早餐餐具'),(420,'木椅靠背'),(480,'杯、杯垫与托盘'),(540,'玄关鞋与钟'),
    (636,'电话与办公用品'),(675,'书桌细节'),(741,'桌游与玩具'),(899,'全屋回望'),
)


def collection_photo_checks():
    return tuple(frame for frame,_ in COLLECTION_CHAPTERS)


def collection_camera(key,t,width=1440,height=810):
    if not isfinite(t) or not 0<=t<=30:
        raise ValueError('collection camera time must be between 0 and 30 seconds')
    eye,aim=_track(t,1,_TOUR),_track(t,2,_TOUR)
    focal=.855*width
    return Camera('home-collection-tour',key,look_at(eye,aim,(0.,0.,1.)),width,height,
                  focal,focal,width/2,height/2,.025,15.)


def collection_focus_target(t):
    if not isfinite(t) or not 0<=t<=30:
        raise ValueError('collection focus time must be between 0 and 30 seconds')
    return tuple(_track(t,2,_TOUR))
