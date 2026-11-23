"""Bind an explicitly named prior baseline, then reuse the original ROI reader.

No fake capture receipt, physical frame, image, or GPU timing is written. The
reader sees an in-memory comparison roster whose rows are each verified against
their own untouched source archive; the final report names both provenances.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

IDENTITY=('pose_index','frame_key','time_s','state_sha256','camera_sha256')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='uri'}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def summary_path(value):
    p=Path(value).resolve(strict=True)
    if p.is_dir():p=p/'summary.json'
    require(p.name=='summary.json' and p.is_file(),'baseline/reference must name its actual summary.json or directory')
    return p


def same_frames(a,b):
    require(len(a)==len(b)==300,'archive comparison requires full 300 states')
    require([f['pose_index'] for f in a]==[f['pose_index'] for f in b]==list(range(300)),'original source index sequence differs')
    require(len({json.dumps(f['frame_key'],sort_keys=True) for f in a})==300,'duplicate source frame key')
    for x,y in zip(a,b):require(all(x[k]==y[k] for k in IDENTITY),'archived actual state/camera/time differs')


def baseline_history(h):
    i=h['indirect'];p=h['progressive']
    require(i['mode']=='diffuse_bounce' and i['seed']==0 and i['bounces']==1 and i['samples_per_pixel']==16
        and p['batches']==1 and p['primary_samples_per_pixel']==16,'archive baseline is not actual moving one-hop16 seed0')


def select(report,root,name,reader):
    require(report['rounds'],'missing real capture rounds')
    rows=[r for r in report['rounds'][0]['runs'] if r['name']==name]
    result={}
    for channel in ('rgb','all'):
        candidates=[r for r in rows if r['channels']==channel and r['status']=='complete']
        require(len(candidates)==1,'no unique completed '+name+'/'+channel+' archived group')
        row=candidates[0];_,actual=reader(root,row)
        require(len(actual['warmup_ms'])==10,'archived warmup count differs')
        same_frames(actual['frames'],actual['frames']);result[channel]=row
    return result


def score(args,run,assess):
    output=Path(args.output).resolve();require(not output.exists(),'new assessment output required')
    current=Path(args.result).resolve(strict=True);current_file=current/'summary.json'
    baseline_file=summary_path(args.baseline_archive);baseline_root=baseline_file.parent
    report=read(current_file);old=read(baseline_file)
    require(report['schema']==run.SCHEMA and report['status'] in ('complete','partial_failure'),'current edge capture not finalized')
    require(old['schema']=='phymeo-preview-sparse-motion-v1' and old['status'] in ('complete','partial_failure'),'unsupported archived baseline schema')
    require(not report['quick'] and not old['quick'],'diagnostic montages cannot stand in for registered300')
    assess.audit_configuration(report);assess.audit_configuration(old)
    for root,value in ((current,report),(baseline_root,old)):
        require(digest(root/'inputs.json')==value['inputs_sha256'],'source input archive changed')
    inputs=read(current/'inputs.json');old_inputs=read(baseline_root/'inputs.json')
    for name in ('scene','studio','declaration','actual_full_inputs','area_light_grid','camera','studio_look','trace_sha256'):
        require(clean(inputs.get(name))==clean(old_inputs.get(name)),'saved scene/light/material/color/input declaration differs: '+name)
    native=report['source']['baseline'];old_native=old['source']['baseline']
    require(native['library_sha256']==old_native['library_sha256'] and native['shaders']==old_native['shaders'],'baseline DLL or full SPV model differs')
    original_read=assess.read_row
    base_rows=select(old,baseline_root,'baseline16',original_read)
    current_rows=select(report,current,'sparse',original_read)
    # All four actual groups remain separate. Matching RGB-only groups receive
    # geometry evidence from their own identical same-configuration all group.
    rows=[base_rows['rgb'],base_rows['all'],current_rows['rgb'],current_rows['all']]
    run.compare(rows)
    for a in base_rows.values():
        for b in current_rows.values():same_frames(a['frames'],b['frames'])
        diag=a['backend'];baseline_history(diag)
        require(diag['library_sha256']==old_native['library_sha256'] and
            all(v in old_native['shaders'].values() for v in diag['shader_sha256'].values()),'baseline actual loaded code differs from its archive')
        require(diag['exposure']==8. and diag['display_transform']=='pbr-neutral' and diag['area_light_grid']==2
            and diag['msaa']['actual_color_samples']==1 and diag['reflection']['mode']=='single_bounce','baseline actual color/light policy differs')
        for frame in a['frames']:baseline_history(frame['baseline_history'])
    require(set(base_rows['all']['frames'][0]['sensors'])=={'forward_depth_m_f32','instance_id_u32','valid_u8'},'baseline all-channel evidence incomplete')
    for round_value in report['rounds']:
        complete=[r for r in round_value['runs'] if r['name']=='sparse' and r['status']=='complete']
        for row in complete:
            original_read(current,row)
            same_frames(row['frames'],current_rows[row['channels']]['frames'])
            run.compare_repeat(current_rows[row['channels']],row)
    # Reader adaptation changes only where verified source rows are resolved.
    # No row payload, image or recorded input is modified or synthesized.
    virtual=deepcopy(report)
    virtual['rounds'][0]['runs']=rows
    original_json=assess.read_json
    def read_json(path):
        return virtual if Path(path).resolve()==current_file else original_json(path)
    def read_row(root,row):
        if row['name']=='baseline16':
            require(row==base_rows[row['channels']],'unbound historical baseline row')
            return original_read(baseline_root,row)
        return original_read(current,row)
    refs=[summary_path(x).parent for x in args.references]
    require(len(refs)>=2,'two independent actual reference seeds required')
    assess.read_json=read_json;assess.read_row=read_row
    try:
        assess.main(['--result',str(current),'--references',*[str(r) for r in refs],'--output',str(output)])
    finally:assess.read_json=original_json;assess.read_row=original_read
    result=read(output)
    result['schema']='phymeo-preview-edge-archived-baseline-assessment-v1'
    result['baseline_archive']={'file':str(baseline_file),'sha256':digest(baseline_file),
        'inputs_sha256':old['inputs_sha256'],'source':old['source'],
        'groups':[{k:r[k] for k in ('name','channels','receipt','receipt_sha256','timing')} for r in base_rows.values()],
        'scope':'previously completed real groups; not recaptured in this candidate run; exact300 input/camera/three-sensor hashes matched',
        'verified_states':300,'rgb_all_consistent':True,'full_three_sensor_hashes_exact':True}
    result['candidate_source']={'file':str(current_file),'sha256':digest(current_file),'source':report['source']}
    result['comparison_roster']='in-memory provenance-checked original records; no new baseline capture or changed source receipt'
    result['per_round_timing']=[{'index':r['index'],'groups':[{'name':x['name'],'channels':x['channels'],'timing':x['timing'],
        'phase_timing':x.get('phase_timing'),'identity_audit':x.get('identity_audit')} for x in r['runs'] if x['status']=='complete']} for r in report['rounds']]
    result['timing_comparison_scope']='current candidate per-round timings; prior baseline timings separate, not interleaved or contemporaneous'
    result['reader_provenance']={'file':str(Path(assess.__file__).resolve()),'sha256':digest(assess.__file__),
        'adapter_file':str(Path(__file__).resolve()),'adapter_sha256':digest(__file__)}
    run.save(output,result)
    return result