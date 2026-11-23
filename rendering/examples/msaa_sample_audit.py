"""Offline point-sample audit of an already captured MSAA calibration archive.

Usage (no renderer/NumPy/Pillow import and no device calls):
  python -I rendering/examples/msaa_sample_audit.py --run RUN_DIR --output NEW_DIR

This is an attribution experiment, not a replacement for the registered 20%
motion gate. Source geometry, PNG bytes, decoded RGB and sensor bytes are bound
to the original receipts. Nominal point coverage uses exact rational edge tests.
Vulkan requires shared-edge samples to belong to exactly one triangle; it does
not mandate our chosen top/left convention for the outer boundary. Such ties
are separately marked. Unrecorded subPixelPrecisionBits is NEVER filled in.
A separate conservative sensitivity envelope perturbs every projected vertex
by +/-1/16 pixel in each axis (the core minimum 4 fractional bits, not a claim
about this GPU's actual precision or transform arithmetic).
"""
import argparse
import base64
from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

SPEC='https://docs.vulkan.org/spec/latest/chapters/primsrast.html'
LIMITS='https://docs.vulkan.org/spec/latest/chapters/limits.html'
SAMPLES={1:((F(1,2),F(1,2)),),4:tuple((F(x,8),F(y,8)) for x,y in ((3,1),(7,3),(1,5),(5,7)))}
IDENTITY=[0.,0.,0.,1.]


def require(condition,message):
    if not condition: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()


def read(path,limit=64*1024*1024):
    with path.open('rb') as stream: data=stream.read(limit+1)
    require(len(data)<=limit,'input exceeds bounded size: '+str(path))
    return data


def json_read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(read(path),parse_constant=bad,object_pairs_hook=pairs)


def bound_path(root,relative):
    path=(root/relative.replace('\\','/')).resolve()
    require(path.is_relative_to(root),'archive path escapes run directory')
    return path


def png_rgb(raw):
    """Strict bounded RGB8, noninterlaced PNG decoder; all five row filters."""
    require(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG signature')
    offset=8; compressed=bytearray(); dimensions=None; ended=False
    while offset<len(raw):
        require(offset+12<=len(raw),'truncated PNG chunk')
        size=struct.unpack_from('>I',raw,offset)[0]; kind=raw[offset+4:offset+8]
        end=offset+12+size; require(end<=len(raw),'truncated PNG payload')
        payload=raw[offset+8:offset+8+size]
        require(zlib.crc32(kind+payload)&0xffffffff==struct.unpack_from('>I',raw,end-4)[0],'PNG CRC')
        if kind==b'IHDR':
            require(dimensions is None and size==13,'duplicate/invalid IHDR')
            w,h,depth,color,comp,filt,interlace=struct.unpack('>IIBBBBB',payload)
            require(0<w<=1024 and 0<h<=1024 and (depth,color,comp,filt,interlace)==(8,2,0,0,0),'only bounded RGB8 PNG supported')
            dimensions=w,h
        elif kind==b'IDAT': compressed.extend(payload)
        elif kind==b'IEND':
            require(size==0 and end==len(raw),'invalid PNG end'); ended=True; break
        else: require(kind[0]&32,'unsupported critical PNG chunk')
        offset=end
    require(ended and dimensions,'PNG missing header/end')
    w,h=dimensions; stride=w*3; expected=h*(stride+1)
    decoder=zlib.decompressobj(); packed=decoder.decompress(bytes(compressed),expected+1)
    require(len(packed)==expected and decoder.eof and not decoder.unused_data,'PNG decompression size')
    output=bytearray(); previous=bytearray(stride)
    for y in range(h):
        start=y*(stride+1); kind=packed[start]; require(kind<=4,'PNG filter')
        row=bytearray(packed[start+1:start+1+stride])
        for x in range(stride):
            a=row[x-3] if x>=3 else 0; b=previous[x]; c=previous[x-3] if x>=3 else 0
            if kind==1: predictor=a
            elif kind==2: predictor=b
            elif kind==3: predictor=(a+b)//2
            elif kind==4:
                p=a+b-c; distance=(abs(p-a),abs(p-b),abs(p-c))
                predictor=(a,b,c)[distance.index(min(distance))]
            else: predictor=0
            row[x]=(row[x]+predictor)&255
        output.extend(row); previous=row
    return w,h,bytes(output)


def geometry(path):
    doc=json_read(path); require(doc['asset']['version']=='2.0','glTF version')
    require(doc['nodes']==[{'mesh':0}] and len(doc['meshes'])==1,'unsupported glTF node transforms')
    primitive=doc['meshes'][0]['primitives']; require(len(primitive)==1 and primitive[0].get('mode',4)==4,'triangle primitive required')
    require(len(doc['buffers'])==1,'one embedded buffer required')
    uri=doc['buffers'][0]['uri']; prefix='data:application/octet-stream;base64,'
    require(uri.startswith(prefix),'embedded glTF only')
    buffer=base64.b64decode(uri[len(prefix):],validate=True)
    require(len(buffer)==doc['buffers'][0]['byteLength'],'glTF buffer size')
    def accessor(index,component,shape,format_,components):
        item=doc['accessors'][index]; view=doc['bufferViews'][item['bufferView']]
        require(item['componentType']==component and item['type']==shape and 'sparse' not in item,'glTF accessor subset')
        require(view['buffer']==0 and 'byteStride' not in view and item['count']<=100,'glTF view subset')
        start=view.get('byteOffset',0)+item.get('byteOffset',0); count=item['count']*components
        size=struct.calcsize('<'+format_*count)
        require(start+size<=view.get('byteOffset',0)+view['byteLength']<=len(buffer),'glTF accessor bounds')
        values=struct.unpack_from('<'+format_*count,buffer,start)
        return tuple(tuple(values[i:i+components]) for i in range(0,count,components))
    item=primitive[0]
    vertices=accessor(item['attributes']['POSITION'],5126,'VEC3','f',3)
    normals=accessor(item['attributes']['NORMAL'],5126,'VEC3','f',3)
    require(normals==((0.,0.,1.),)*4,'different source normals outside frozen fixture')
    indices=accessor(item['indices'],5123,'SCALAR','H',1)
    indices=tuple(i[0] for i in indices)
    require(len(vertices)==4 and len(indices)==6 and max(indices)<4,'frozen two-triangle quad required')
    return vertices,tuple(tuple(indices[i:i+3]) for i in (0,3))


def cross(a,b,p): return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def triangle_hit(triangle,p):
    a,b,c=triangle
    require(cross(a,b,c)!=0,'degenerate triangle')
    if cross(a,b,c)<0: b,c=c,b
    for u,v in ((a,b),(b,c),(c,a)):
        edge=cross(u,v,p); dx=v[0]-u[0]; dy=v[1]-u[1]
        if edge<0 or (edge==0 and not (dy<0 or (dy==0 and dx>0))): return False
    return True


def sensitivity(poly,p,epsilon=F(1,16)):
    """Conservative interval, including the bilinear vertex perturbation term."""
    stable_inside=True; equality=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        value=cross(a,b,p)
        equality=equality or (value==0 and min(a[0],b[0])<=p[0]<=max(a[0],b[0]) and min(a[1],b[1])<=p[1]<=max(a[1],b[1]))
        dx,dy=b[0]-a[0],b[1]-a[1]; px,py=p[0]-a[0],p[1]-a[1]
        bound=epsilon*(abs(dx)+abs(dy)+2*(abs(px)+abs(py)))+4*epsilon*epsilon
        if value < -bound: return 0,False
        stable_inside=stable_inside and value>bound
    return (1 if stable_inside else None),equality


def srgb8(x): return round(255*(12.92*x if x<=.0031308 else 1.055*x**(1/2.4)-.055))
def linear8(x):
    value=x/255
    return value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4


def model(poly,indices,width,height,samples):
    triangles=tuple(tuple(poly[i] for i in triangle) for triangle in indices)
    count=width*height; rgb=bytearray(count*3); uncertain=bytearray(count); equal=bytearray(count)
    lower=bytearray(count); upper=bytearray(count)
    for y in range(max(0,math.floor(min(p[1] for p in poly))-1),min(height,math.ceil(max(p[1] for p in poly))+1)):
        for x in range(max(0,math.floor(min(p[0] for p in poly))-1),min(width,math.ceil(max(p[0] for p in poly))+1)):
            covered=lo=hi=0; index=y*width+x
            for sx,sy in SAMPLES[samples]:
                point=(x+sx,y+sy); hits=sum(triangle_hit(t,point) for t in triangles)
                require(hits<=1,'shared triangle edge counted twice')
                covered+=bool(hits)
                state,tie=sensitivity(poly,point); equal[index]|=int(tie)
                if state==1: lo+=1; hi+=1
                elif state is None: hi+=1; uncertain[index]=1
            code=srgb8(covered/samples); rgb[index*3:index*3+3]=bytes((code,))*3
            lower[index]=srgb8(lo/samples); upper[index]=srgb8(hi/samples)
    return bytes(rgb),bytes(uncertain),bytes(equal),bytes(lower),bytes(upper)


def project(vertices,record,instance):
    camera=record['camera']; frame=record['frame']
    require(camera['frame_key']==frame['key'],'camera/frame key mismatch')
    camera_pose=camera['world_pose']; nodes={n['node_id']:n['world_pose'] for n in frame['nodes']}
    pose=nodes[instance['node_id']]; local=instance['local_pose']
    require(all(p['orientation_xyzw']==IDENTITY for p in (camera_pose,pose,local)),'audit only supports recorded unrotated fixture')
    require(local['position_m']==[0.,0.,0.],'nonidentity instance local pose')
    result=[]; depths=[]
    for vertex in vertices:
        relative=tuple(F(v)+F(t)-F(c) for v,t,c in zip(vertex,pose['position_m'],camera_pose['position_m']))
        depth=-relative[2]; require(F(camera['near_m'])<depth<F(camera['far_m']),'clip crossing outside audit scope')
        result.append((F(camera['cx_px'])+F(camera['fx_px'])*relative[0]/depth,
                       F(camera['cy_px'])-F(camera['fy_px'])*relative[1]/depth)); depths.append(depth)
    require(len(set(depths))==1 and all(cross(result[i-1],result[i],result[(i+1)%4])>0 for i in range(4)),'flat convex quad required')
    layer=record['projected_layers']; require(len(layer)==1,'single isolated rod layer required')
    require([[float(x),float(y)] for x,y in result]==layer[0][0] and float(depths[0])==layer[0][1] and instance['id']==layer[0][2],'independent projection differs from receipt')
    return tuple(result)


def temporal(frames,reference,roi):
    errors=[]
    for previous,current,r0,r1 in zip(frames,frames[1:],reference,reference[1:]):
        errors.append(sum(abs((current[i]-previous[i])-(r1[i]-r0[i])) for i in roi)/len(roi))
    return {'mae':sum(errors)/len(errors),'per_transition_mae':errors,'transitions':len(errors)}


def self_test():
    quad=((F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1)))
    triangles=((quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3]))
    require(sum(triangle_hit(t,(F(1,2),F(1,2))) for t in triangles)==1,'shared diagonal self-test')
    require(sum(triangle_hit(tuple(reversed(t)),(F(1,2),F(1,2))) for t in triangles)==1,'winding self-test')
    require(not triangle_hit(triangles[0],(F(2),F(1,2))),'outside negative control')
    for width,count in ((F(1,4),1),(F(1,2),2),(F(3,4),3)):
        p=((F(0),F(0)),(width,F(0)),(width,F(1)),(F(0),F(1)))
        rgb,*_=model(p,((0,1,2),(0,2,3)),1,1,4)
        require(rgb==bytes((srgb8(count/4),))*3,'analytic fractional coverage control')
    require(sensitivity(quad,(F(0),F(1,2)))[1],'outer equality marked')
    require(sensitivity(quad,(F(1,2),F(1,2)))[0]==1,'interior stability control')
    return {'shared_diagonal_exactly_once':True,'reversed_winding':True,'outside_negative':True,'fractional_controls_rgb8':[137,188,225],'outer_equality_separate':True}


def audit(run,output):
    run=Path(run).resolve(); output=Path(output).resolve()
    require(not output.exists(),'output must be a new directory')
    controls=self_test(); summary=json_read(run/'summary.json'); ideal=json_read(run/'ideal-coverage.json')
    require(summary['schema']=='phymeo-native-msaa-calibration-v1' and summary['physics_steps']==0,'source schema/phase')
    require(summary['registration']['rod_phases_px']==[i/16 for i in range(16)],'original 16 phases required')
    four=summary['modes']['4']['diagnostics_after_close']
    require(four['device']['resident_msaa']['standard_sample_locations']==1,'standard sample locations not recorded true')
    require(four['msaa']['actual_color_samples']==4 and four['msaa']['resolve_space']=='display-linear then IEC sRGB once','source resolve contract')
    require(summary['scene']['lights']==[{'direction':[0.,0.,-1.],'illuminance':[16.,16.,16.]}],'different lighting outside frozen-rod scope')
    bindings=[]
    def bind(path,expected=None):
        raw=read(path); actual=digest(raw)
        require(expected is None or actual==expected,'SHA mismatch: '+str(path))
        bindings.append({'path':str(path),'sha256':actual,'bytes':len(raw)})
        return raw
    bind(run/'summary.json'); bind(run/'ideal-coverage.json')
    for name,sha in summary['mesh_sha256'].items(): bind(bound_path(run,'geometry/'+name),sha)
    vertices,indices=geometry(run/'geometry/rod.gltf')
    require([list(v) for v in vertices]==summary['authored_vertices']['rod']['vertices'],'decoded FP32 vertices differ')
    instance=next(x for x in summary['scene']['instances'] if x['node_id']=='rod')
    require(instance['material_id']=='white' and instance['mesh_id']=='rod','white rod binding required')
    white=next(m for m in summary['scene']['materials'] if m['id']=='white')
    require(white=={'id':'white','base_color':[1.,1.,1.],'metallic':0.,'roughness':1.},'changed rod material')
    names=[f'rod-{i:02d}' for i in range(16)]
    width,height=summary['registration']['extent']; require((width,height)==(128,96),'frozen extent required')
    references=[[v for row in ideal[name] for v in row] for name in names]
    require(all(len(frame)==width*height for frame in references),'coverage extent')
    roi=[i for i in range(width*height) if any(1e-12<frame[i]<1-1e-12 for frame in references)]
    require(len(roi)==summary['modes']['4']['geometric_motion_region_pixels'],'original ROI mismatch')
    output.mkdir(parents=True); results={}; predictions={}; actuals={}; all_uncertain={}
    for samples in (1,4):
        source=summary['modes'][str(samples)]; frames=[]; measured=[]; predicted=[]; uncertain_frames=[]
        require(source['measured_black_srgb8']==[0.,0.,0.] and source['measured_white_srgb8']==[255.,255.,255.],'nonbinary control colors outside audit scope')
        for name in ('clear','solid',*names):
            record=source['records'][name]
            image=png_rgb(bind(bound_path(run,record['png']),record['png_sha256']))
            require(image[:2]==(width,height),'PNG extent')
            actual=image[2]
            require(digest(actual)==record['request_receipts']['mixed']['sha256']['srgb8_rgb'],'decoded RGB differs from capture receipt')
            require(digest(actual)==record['request_receipts']['rgb']['sha256']['srgb8_rgb'],'RGB-only receipt differs')
            if name in ('clear','solid'):
                expected=0 if name=='clear' else 255
                require(actual==bytes((expected,))*len(actual),'whole-frame binary control changed'); continue
            other=summary['modes']['1']['records'][name]
            require(all(record[k]==other[k] for k in ('frame','camera','projected_layers')),'cross-mode inputs differ')
            poly=project(vertices,record,instance)
            require(record['frame']['time_s']==record['frame']['key']['tick']/60,'frame time binding')
            nominal,uncertain,ties,lower,upper=model(poly,indices,width,height,samples)
            differences=[a-b for a,b in zip(actual,nominal)]
            changed=[i for i in range(width*height) if any(differences[3*i:3*i+3])]
            outside=[i for i in range(width*height) if any(actual[3*i+c]<lower[i]-1 or actual[3*i+c]>upper[i]+1 for c in range(3))]
            for channel,raw_suffix,receipt_key in (('FORWARD_DEPTH','FORWARD_DEPTH','forward_depth_m_f32'),('INSTANCE_ID','INSTANCE_ID','instance_id_u32'),('VALID','VALID','valid_u8')):
                sensor=bind(run/f'samples-{samples}/{name}.{raw_suffix}.raw',record['request_receipts']['mixed']['sha256'][receipt_key])
                require(digest(sensor)==summary['modes']['1']['records'][name]['request_receipts']['mixed']['sha256'][receipt_key],'sensor modes differ')
            prefix=output/f'{samples}-{name}'
            for suffix,raw in (('predicted.rgb8.gz',nominal),('delta.i16le.gz',struct.pack('<'+'h'*len(differences),*differences)),('uncertain.u8.gz',uncertain)):
                (prefix.parent/(prefix.name+'.'+suffix)).write_bytes(gzip.compress(raw,mtime=0))
            frame_info={'name':name,'frame_key':record['frame']['key'],'time_s':record['frame']['time_s'],
                'all_vertices_on_1_16_grid':all((v*16).denominator==1 for p in poly for v in p),
                'actual_rgb_sha256':digest(actual),'nominal_rgb_sha256':digest(nominal),'mismatch_pixels':len(changed),
                'mismatch_bytes':sum(x!=0 for x in differences),'max_abs_byte_difference':max(map(abs,differences)),
                'signed_byte_difference_histogram':dict(sorted(Counter(differences).items())),
                'mismatch_pixels_outside_uncertain':sum(not uncertain[i] for i in changed),
                'outer_equal_pixels':sum(ties),'uncertain_pixels':sum(uncertain),'roi_uncertain_pixels':sum(uncertain[i] for i in roi),
                'outside_sensitivity_rgb_interval_plus_one_lsb':len(outside),
                'first_mismatches':[{'xy':[i%width,i//width],'actual':list(actual[3*i:3*i+3]),'nominal':list(nominal[3*i:3*i+3]),'uncertain':bool(uncertain[i]),'outer_equal':bool(ties[i])} for i in changed[:12]]}
            frames.append(frame_info); uncertain_frames.append(uncertain)
            measured.append([sum(linear8(actual[i+c]) for c in range(3))/3 for i in range(0,len(actual),3)])
            predicted.append([linear8(nominal[i]) for i in range(0,len(nominal),3)])
        actual_metric=temporal(measured,references,roi); nominal_metric=temporal(predicted,references,roi)
        require(abs(actual_metric['mae']-source['motion']['mae'])<1e-12,'original time metric reconstruction mismatch')
        require(all(abs(a-b)<1e-12 for a,b in zip(actual_metric['per_transition_mae'],source['motion']['per_transition_mae'])),'original transition mismatch')
        results[str(samples)]={'frames':frames,'actual_motion':actual_metric,'nominal_motion':nominal_metric,
            'nominal_actual_motion_mae_difference':nominal_metric['mae']-actual_metric['mae'],
            'all_nominal_rgb_bytes_equal':all(f['mismatch_bytes']==0 for f in frames),
            'total_mismatch_pixels_outside_uncertain':sum(f['mismatch_pixels_outside_uncertain'] for f in frames)}
        predictions[samples]=predicted; actuals[samples]=measured; all_uncertain[samples]=uncertain_frames
    stable_roi=[i for i in roi if all(not frame[i] for samples in (1,4) for frame in all_uncertain[samples])]
    report={'schema':'phymeo-msaa-offline-sample-audit-v1','physics_steps':0,'gpu_calls':0,'run':str(run),
        'self_tests':controls,'sources':bindings,'script_sha256':digest(read(Path(__file__))),
        'original_registered_acceptance':summary['acceptance'],'original_motion_improvement':summary['motion_improvement'],
        'roi_pixels':len(roi),'transitions':15,'modes':results,
        'nominal_predicted_reduction':1-results['4']['nominal_motion']['mae']/results['1']['nominal_motion']['mae'],
        'stable_roi_all_frames_pixels':len(stable_roi),
        'stable_roi_metrics':{str(s):{'actual':temporal(actuals[s],references,stable_roi),'nominal':temporal(predictions[s],references,stable_roi)} for s in (1,4)} if stable_roi else None,
        'assumptions':{'standard_locations_source':'recorded standard_sample_locations=1; Vulkan standard table',
            'device_subPixelPrecisionBits':None,'device_subpixel_precision_recorded':False,
            'sensitivity_vertex_epsilon_px':1/16,'sensitivity_scope':'conservative per-coordinate envelope based on core minimum; not measured device rounding or transform-error bound',
            'equal_outer_boundary':'nominal top/left is a convention, not a Vulkan-required outer-edge choice; ties separately reported',
            'shared_diagonal':'exact rational edge tests; complementary ownership exactly once; identical white/depth on both triangles',
            'color_model':'constant display-linear white1/black0 verified by saved controls; no independent BRDF or shadow re-execution',
            'coordinate_model':'exact real projection from recorded FP32 vertices and saved identity rotations/translations; not a replay of GPU floating-point evaluation',
            'quantization':'IEC sRGB with nearest RGB8; interval comparison separately allows one code for arithmetic rounding, never changes original acceptance'},
        'specification_links':[SPEC,LIMITS]}
    report['conclusion']='nominal point-sample model reproduces every saved rod RGB byte' if all(r['all_nominal_rgb_bytes_equal'] for r in results.values()) else 'nominal model has differences; use per-frame uncertainty and intervals before attribution'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); report=audit(args.run,args.output)
    print(json.dumps({k:report[k] for k in ('conclusion','roi_pixels','nominal_predicted_reduction','stable_roi_all_frames_pixels')},indent=2))


if __name__=='__main__': main()
