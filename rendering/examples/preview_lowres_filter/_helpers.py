"""Small standard-library helpers owned by this standalone experiment."""
from hashlib import sha256
import json
from pathlib import Path
import struct


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs)


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source anchor absent or ambiguous: '+old[:100])
    return text.replace(old,new)


def verify(directory):
    root=Path(directory).resolve();manifest=read(root/'source-manifest.json');hashes={};seen=set()
    for raw,expected in manifest['generated_sha256'].items():
        key=raw.replace('\\','/')
        if ':' in key or '\0' in key or any(x in ('','.','..') or x.endswith((' ','.')) for x in key.split('/')):raise ValueError('unsafe source path')
        if key.casefold() in seen:raise ValueError('duplicate source alias')
        seen.add(key.casefold());path=(root/key).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:raise ValueError('source changed: '+key)
        hashes[key]=expected
    manifest['generated_sha256']=hashes;return root,manifest


def markers(blob):
    if len(blob)<20 or len(blob)%4:raise ValueError('invalid SPIR-V')
    w=struct.unpack('<%dI'%(len(blob)//4),blob)
    if w[0]!=0x07230203:raise ValueError('invalid SPIR-V magic')
    names={};ids={};values={};i=5
    while i<len(w):
        n=w[i]>>16;op=w[i]&65535
        if not n or i+n>len(w):raise ValueError('truncated SPIR-V')
        a=w[i+1:i+n]
        if op==5:names[a[0]]=struct.pack('<%dI'%(len(a)-1),*a[1:]).split(b'\0')[0].decode()
        if op==71 and len(a)>=3 and a[1]==1:ids[a[0]]=a[2]
        if op==50:values[a[1]]=list(a[2:])
        i+=n
    return {str(index):dict(name=names.get(target),value=values.get(target)) for target,index in ids.items()}
