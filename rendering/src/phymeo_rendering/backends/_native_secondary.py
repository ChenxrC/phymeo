"""Bound secondary-light shader contract; no runtime loading or fallback."""
import struct
from . import BackendUnavailableError, UnsupportedRequestError


def secondary_grid(primary, secondary, *, resident, shadow_mode):
    if secondary is None:return primary
    if type(secondary) is not int:
        raise TypeError('secondary_area_light_grid must be an integer or None, not bool')
    if secondary not in (2,4,8):
        raise ValueError('secondary_area_light_grid must be 2, 4 or 8')
    if secondary != primary and not (primary==8 and secondary==2 and resident and shadow_mode=='ray_query'):
        raise UnsupportedRequestError('split area-light budget supports only primary 8 / secondary 2 with resident=True and shadow_mode=ray_query')
    return secondary


def validate_secondary_shader(data):
    """Check named unsigned specialization constant in the actual loaded bytes.

    This is a narrow contract check, not a general SPIR-V validator. Vulkan still
    validates the complete module. Defaults do not inspect any new marker.
    """
    error=BackendUnavailableError('native fragment lacks the compiled primary 8 / secondary 2 light budget contract')
    if len(data)<20 or len(data)%4:raise error
    words=struct.unpack('<'+'I'*(len(data)//4),data)
    if words[0]!=0x07230203:raise error
    names={};specs={};types={};values={};offset=5
    try:
        while offset<len(words):
            length,op=words[offset]>>16,words[offset]&0xffff
            if length==0 or offset+length>len(words):raise error
            args=words[offset+1:offset+length]
            if op==5 and len(args)>=2:
                raw=struct.pack('<'+'I'*(len(args)-1),*args[1:])
                names[args[0]]=raw.split(b'\0',1)[0].decode('utf-8')
            elif op==71 and len(args)==3 and args[1]==1:specs[args[0]]=args[2]
            elif op==21 and len(args)==3:types[args[0]]=args[1:]
            elif op==50 and len(args)==3:values[args[1]]=(args[0],args[2])
            offset+=length
    except (UnicodeError,struct.error) as cause:raise error from cause
    candidates=[identity for identity,number in specs.items() if number==73]
    if len(candidates)!=1:raise error
    identity=candidates[0];value=values.get(identity)
    if names.get(identity)!='PR_SECONDARY_LIGHT_BUDGET' or value is None or types.get(value[0])!=(32,0) or value[1]!=0x0802:
        raise error
    return dict(primary_grid=8,secondary_grid=2,spec_id=73)
