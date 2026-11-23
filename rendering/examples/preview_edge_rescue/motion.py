"""Small edge adapter over the already registered 10+300 motion/encode helpers.

The helper is loaded by explicit path and hashed. No capture loop or trajectory
is copied, and no old receipt is relabelled as this new backend.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
SCHEMA='phymeo-preview-edge-motion-v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def policy_check(policy):
    edge=policy.get('edge_rescue',{})
    require(edge.get('schema')=='phymeo-preview-edge-rescue-v1','explicit edge contract required')
    require(tuple(policy.get(k) for k in ('tile','paths_per_anchor','area_light_grid','neighbor_candidates','reconstruction_revision'))==(4,4,2,25,2),'unsupported edge compact parent')
    distance=edge.get('distance',{})
    require(distance.get('rescue_radius_m','absent') is None,'rescue must remain full distance')
    if 'local_diffuse' in policy:
        require(policy['local_diffuse'].get('radius_m')==.12 and distance.get('sparse_radius_m')==.12,'local radius contract differs')
    return edge


def profile_name(policy):
    policy_check(policy)
    radius='local120mm' if 'local_diffuse' in policy else 'full'
    return 'edge-t4-p4-g2-k25-r2-'+radius+'-rescue16-full'


def backend_name(policy):
    policy_check(policy);return 'experimental-native-compact-edge-rescue-v1'


def load(path,name):
    path=Path(path).resolve(strict=True);before=list(sys.path)
    try:
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location(name,path)
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    finally:sys.path[:]=before
    return m


def configure(helper):
    run=load(helper,'edge_registered_motion_helper')
    old_adapter=run.adapter;old_check=run.check_backend;old_actual=run.check_actual
    old_binding=run.source_binding;old_load=run.load_runtime
    def kind(policy):policy_check(policy);return 'edge-rescue'
    def require_kind(policy,requested):
        policy_check(policy);require(requested=='edge-rescue','wrong requested candidate kind')
    def runtime(path):
        m=load(path,'edge_motion_runtime')
        m.motion_adapter_dependencies=[dict(file=str(HERE/'build_source.py'),sha256=sha256((HERE/'build_source.py').read_bytes()).hexdigest())]
        return m
    def compact(value,extent,expected,actual):
        w,h=extent
        fixed=dict(abi_version=1,enabled=True,tile=4,full_width=w,full_height=h,
            guide_width=(w+3)//4,guide_height=(h+3)//4,last_capture_rgb=True,rgb_captures=expected,
            sensor_only_captures=0,successful_guide_draws=expected,primary_color_attachments=5,reconstruction_sampled_images=9)
        require(all(value.get(k)==v for k,v in fixed.items()),'edge actual compact extent/draw/9-sampler contract differs')
        require(0<=actual['visible_pixels']<=w*h and 0<=actual['traced_anchor_pixels']<=((w+3)//4)*((h+3)//4),'edge actual geometry counter range differs')
    run.adapter=SimpleNamespace(__file__=__file__,kind=kind,require_kind=require_kind,load_runtime=runtime,
        backend_class=lambda m,p:(policy_check(p) and m.EdgeBackend),backend_name=backend_name,check_compact=compact)
    run.SCHEMA=SCHEMA;run.profile_name=profile_name
    accounting=load(HERE/'build_source.py','edge_motion_accounting')
    def actual(value,policy,expected_captures):
        old_actual(value,policy,expected_captures)
        extra=accounting.validate_counts(value['edge_rescue'])
        require(value.get('total_primary_proposals')==value['primary_proposals']+extra['extra_primary_proposals'],'separate sparse/rescue proposal total differs')
    run.check_actual=actual
    def check(diag,args,policy,name,source,*,expected_captures):
        old_check(diag,args,policy,name,source,expected_captures=expected_captures)
        if name=='sparse':
            compact(diag['compact'],run.base.preset(run.SIZES[args.resolution])['extent'],expected_captures,diag['actual'])
            shader=diag['edge_shader'];require(source['sparse']['shaders'].get(shader['file'])==shader['sha256'],'actual extra-pass shader is not bound')
    run.check_backend=check
    def source(args,prepared,manifest):
        value=old_binding(args,prepared,manifest)
        value['edge_delivery']={'file':str(Path(__file__).resolve()),'sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'registered_helper_file':str(Path(helper).resolve()),'registered_helper_sha256':sha256(Path(helper).read_bytes()).hexdigest(),
            'generic_candidate_slot':'sparse is only the inherited storage slot; actual backend is edge-rescue',
            'distance':policy_check(manifest['contract'])['distance']}
        return value
    run.source_binding=source
    def source_load(args):
        result=old_load(args)
        prepared,manifest=result[1:]
        actual_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in [prepared/'build/phymeo_native.dll',*sorted((prepared/'build').glob('*.spv'))]}
        require(manifest.get('compiled_sha256')==actual_hashes,'seal the completed new edge build before motion capture')
        return result
    run.load_runtime=source_load
    return run


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-helper',type=Path,required=True,help='actual registered preview_sparse_motion/run.py')
    sub=p.add_subparsers(dest='command',required=True)
    cap=sub.add_parser('capture')
    cap.add_argument('--prepared',type=Path,required=True);cap.add_argument('--baseline-library',type=Path,required=True)
    cap.add_argument('--baseline-shaders',type=Path);cap.add_argument('--output',type=Path,required=True)
    cap.add_argument('--device-index',type=int,default=1);cap.add_argument('--validation',action='store_true')
    cap.add_argument('--resolution',choices=('720p','540p','360p'),default='720p')
    cap.add_argument('--camera',choices=('detail','sequence'),default='detail')
    cap.add_argument('--studio-look',choices=('recorded','soft-studio'),default='recorded')
    cap.add_argument('--quick',action='store_true');cap.add_argument('--candidate-only',action='store_true')
    cap.add_argument('--poses',type=int,nargs='+',default=[59,60,149])
    cap.add_argument('--channels',choices=('rgb','all'),nargs='+',default=['rgb','all'])
    cap.add_argument('--rounds',type=int,choices=(1,2),default=2)
    cap.add_argument('--film',choices=('none','sparse','baseline16'),default='sparse',help='sparse names the generic candidate slot, here always edge-rescue')
    enc=sub.add_parser('encode');enc.add_argument('result',type=Path);enc.add_argument('--media-output',type=Path,required=True)
    score=sub.add_parser('score');score.add_argument('result',type=Path)
    score.add_argument('--references',type=Path,nargs='+',required=True);score.add_argument('--output',type=Path,required=True)
    score.add_argument('--baseline-archive',type=Path,help='actual prior baseline summary.json or directory; never recapture or relabel it')
    a=p.parse_args(argv);run=configure(a.motion_helper)
    prior=list(sys.path)
    try:
        sys.path.insert(0,str(a.motion_helper.resolve().parents[1]))
        sys.path.insert(0,str(a.motion_helper.resolve().parent))
        return execute(a,run)
    finally:sys.path[:]=prior


def execute(a,run):
    if a.command=='encode':return run.encode(a)
    if a.command=='score':
        previous=sys.modules.get('run')
        try:
            sys.modules['run']=run
            assess=load(a.motion_helper.parent/'assess.py','edge_registered_assessment')
            archived=load(HERE/'archived_score.py','edge_archived_baseline')
            if a.baseline_archive is not None:return archived.score(a,run,assess)
            references=[archived.summary_path(v).parent for v in a.references]
            return assess.main(['--result',str(a.result),'--references',*[str(v) for v in references],'--output',str(a.output)])
        finally:
            if previous is None:sys.modules.pop('run',None)
            else:sys.modules['run']=previous
    a.runtime=HERE/'runtime.py';a.candidate_backend='edge-rescue'
    a.baseline_library=a.baseline_library.resolve(strict=True)
    a.baseline_shaders=(a.baseline_shaders or a.baseline_library.parent).resolve(strict=True)
    a.output=a.output.resolve();a.poses=sorted(set(run.REGISTERED)|set(a.poses))
    require(len(set(a.channels))==len(a.channels),'duplicate channel request')
    require(not a.candidate_only or a.film!='baseline16','candidate-only has no baseline film')
    run.indices(a.quick,a.poses)
    return run.capture(a)

if __name__=='__main__':main()