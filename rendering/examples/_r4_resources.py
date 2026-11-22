"""Read-only Windows per-PID/per-adapter GPU memory sampling for R4.

CIM runs in one hidden child, never queries whole-device use as process use.
Reported peaks are sampled WDDM dedicated/shared usage, not exact allocation
high-water marks. Vulkan's device LUID binds the adapter; an absent binding is
reported explicitly and never guessed from device ordinal or GPU name.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

_NAME=re.compile(r'^pid_(\d+)_luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_(\d+)$',re.I)


def parse_counter_name(name):
    match=_NAME.fullmatch(name)
    if match is None: raise ValueError('unrecognized GPU process counter name')
    pid,high,low,physical=match.groups()
    luid=int(low,16).to_bytes(4,'little')+int(high,16).to_bytes(4,'little')
    return int(pid),luid.hex(),int(physical)


def summarize_samples(samples,*,pid,luid,start,end,period_s):
    selected=[]; unavailable=[]
    for sample in samples:
        if not start<=sample['received_perf_s']<=end: continue
        if sample.get('status')!='ok':
            unavailable.append(sample.get('error','counter unavailable')); continue
        matched=[]
        for row in sample['rows']:
            try: row_pid,row_luid,physical=parse_counter_name(row['name'])
            except ValueError: continue
            if row_pid==pid and luid is not None and row_luid==luid.lower():
                matched.append((physical,int(row['dedicated_bytes']),int(row['shared_bytes'])))
        if len({item[0] for item in matched})!=len(matched):
            raise ValueError('duplicate PID/adapter physical memory counter')
        if matched:
            selected.append(dict(time_s=sample['received_perf_s'],
                dedicated_bytes=sum(row[1] for row in matched),shared_bytes=sum(row[2] for row in matched)))
    times=[s['time_s'] for s in selected]
    return dict(source='Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory',
        pid=pid,device_luid_hex=luid,adapter_bound=luid is not None,samples=len(selected),
        period_requested_s=period_s,
        max_observed_sample_gap_s=max((b-a for a,b in zip(times,times[1:])),default=None),
        peak_device_memory_bytes=max((s['dedicated_bytes'] for s in selected),default=None),
        peak_shared_memory_bytes=max((s['shared_bytes'] for s in selected),default=None),
        scope='current Python PID on Vulkan LUID; WDDM dedicated usage, not whole GPU or isolated session allocation',
        limitation='sampling can miss shorter peaks; refresh cadence and CIM query overhead exceed requested sleep interval',
        unavailable_reason=('Vulkan device LUID unavailable' if luid is None else
                            ('no matching process/adapter sample during phase' if not selected else None)),
        errors=sorted(set(unavailable)))


class WindowsProcessMemory:
    def __init__(self,period_s=.25):
        if not isinstance(period_s,(int,float)) or isinstance(period_s,bool) or not .05<=period_s<=60:
            raise ValueError('memory sampling interval must be 0.05..60 s')
        self.period_s=float(period_s); self.pid=os.getpid(); self.samples=[]
        self.process=None; self.thread=None; self.error=None

    def start(self):
        if sys.platform!='win32':
            self.error='Windows per-process GPU counter is unavailable on this platform'; return
        script=r'''
$ErrorActionPreference='Stop'
while ($true) {
  try {
    $taskRows=@(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
      Where-Object { $_.Name -match '^pid_PIDVALUE_' } |
      ForEach-Object { @{name=$_.Name; dedicated_bytes=[uint64]$_.DedicatedUsage; shared_bytes=[uint64]$_.SharedUsage} })
    @{status='ok'; queried_utc_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds(); rows=$taskRows} | ConvertTo-Json -Depth 5 -Compress
  } catch {
    @{status='error'; error=$_.Exception.Message; rows=@()} | ConvertTo-Json -Depth 5 -Compress
  }
  Start-Sleep -Milliseconds INTERVALVALUE
}
'''.replace('PIDVALUE',str(self.pid)).replace('INTERVALVALUE',str(round(self.period_s*1000)))
        try:
            self.process=subprocess.Popen(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',script],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except OSError as error:
            self.error=f'cannot start GPU process sampler: {error}'; return
        def read():
            for line in self.process.stdout:
                try:
                    value=json.loads(line); value['received_perf_s']=time.perf_counter()
                    if not isinstance(value.get('rows'),list): raise ValueError('counter rows are not an array')
                    self.samples.append(value)
                except (ValueError,TypeError) as error:
                    self.samples.append(dict(status='error',error=f'{error}: {line.strip()}',rows=[],received_perf_s=time.perf_counter()))
        self.thread=threading.Thread(target=read,name='r4-pid-gpu-memory',daemon=True); self.thread.start()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
            self.thread.join(timeout=5)
            self.process.stdout.close()
            self.process=None

    def report(self,luid,start,end):
        result=summarize_samples(tuple(self.samples),pid=self.pid,luid=luid,start=start,end=end,period_s=self.period_s)
        if self.error: result['errors'].append(self.error)
        return result
