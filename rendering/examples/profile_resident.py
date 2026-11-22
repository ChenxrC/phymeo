"""Separate completed public capture time from resident device work on the same frames."""
import argparse
import cProfile
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter_ns

from panda_realtime_film import ROOT, fixed, load_episode
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath


class ResidentProfile(ct.Structure):
    _fields_ = ([(key, ct.c_uint32) for key in
                 ("profile_abi_version","timestamp_supported","timestamp_valid_bits","tlas_node_count")]
                + [("timestamp_period_ns",ct.c_double)]
                + [(key,ct.c_uint64) for key in
                   ("captures","host_prepare_ns","host_upload_ns","host_record_ns","host_submit_ns",
                    "host_wait_ns","host_readback_ns","gpu_render_ns","gpu_copy_ns","gpu_total_ns",
                    "tlas_upload_bytes","timestamp_samples","reserved")])


def snapshot(session):
    # Example-only instrumentation of the named native implementation.
    # The timed calls below all go through the ordinary checked capture.
    device = session._delegate.device
    get = device._lib.pr_resident_profile
    get.argtypes = [ct.c_void_p, ct.POINTER(ResidentProfile)]
    get.restype = ct.c_int
    value = ResidentProfile()
    if ct.sizeof(value) != 128 or get(device._handle,ct.byref(value)) or value.profile_abi_version != 1:
        raise RuntimeError("resident profile ABI unavailable")
    return {key:getattr(value,key) for key,_ in value._fields_}


def measure(args):
    args.output.mkdir(parents=True,exist_ok=False)
    scene, frames = load_episode(args.output/"geometry",ROOT/"tests/fixtures/panda_film/trace.json")
    backend = NativeBackend(library_path=args.library,device_index=args.device_index,resident=True)
    session = backend.open_scene(scene)
    report = {}
    try:
        for name,channels in (("rgb",(Channel.RGB,)),("all",tuple(Channel))):
            requests = [(frames[i],CaptureRequest(fixed(frames[i],1280,720),RenderPath.RASTER,channels))
                        for i in range(0,len(frames),15)]
            for frame,request in requests[:5]:
                session.capture(frame,request)
            before = snapshot(session)
            times = []
            for frame,request in requests:
                start = perf_counter_ns()
                session.capture(frame,request)
                times.append((perf_counter_ns()-start)/1e6)
            after = snapshot(session)
            if after["captures"]-before["captures"] != len(times):
                raise RuntimeError("capture count disagrees with profile")
            # A separately instrumented pass identifies CPU routines; its
            # invasive timings must not be mixed into the wall-time samples.
            trace = cProfile.Profile()
            frame,request = requests[3]
            trace.runcall(session.capture,frame,request)
            trace.dump_stats(str(args.output/(name+".pstats")))
            entries = sorted(trace.getstats(),key=lambda item:item.inlinetime,reverse=True)[:16]
            report[name] = dict(wall_ms=times,mean_ms=sum(times)/len(times),
                native_mean_ms={key:(after[key]-before[key])/1e6/len(times) for key in before
                                if key.endswith("_ns") and key!="timestamp_period_ns"},
                profile_before=before,profile_after=after,
                python_single_frame_own_time=[dict(function=str(e.code) if isinstance(e.code,str)
                    else e.code.co_name,file=None if isinstance(e.code,str) else e.code.co_filename,
                    line=None if isinstance(e.code,str) else e.code.co_firstlineno,
                    total_calls=e.callcount,own_ms=e.inlinetime*1000) for e in entries])
            print(name,report[name]["mean_ms"],report[name]["native_mean_ms"],flush=True)
    finally:
        session.close()
    report["diagnostics"] = session.diagnostics()
    (args.output/"profile.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--device-index",type=int,default=1)
    measure(parser.parse_args())
