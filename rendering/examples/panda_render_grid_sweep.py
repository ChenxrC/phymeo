"""Additional paired presets for the Panda benchmark; run after its main matrix."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import panda_render_performance as bench

HIGH = dict(bench.CASES[-1][2])
ADDITIONAL = [
    ("balanced", "均衡 · 4× MSAA＋4×4 面光", dict(msaa_samples=4, area_light_grid=4)),
    ("high_grid2", "综合光照 · 2×2 面光", dict(HIGH, area_light_grid=2)),
    ("high_grid4", "综合光照 · 4×4 面光", dict(HIGH, area_light_grid=4)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory of the completed main matrix")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=1)
    args = parser.parse_args()
    out, dll = args.output.resolve(), args.library.resolve()
    original = out / "benchmark.json"
    baseline = json.loads(original.read_text(encoding="utf-8"))
    report = out / "additional.json"
    if report.exists():
        raise FileExistsError(report)
    if (baseline["library_sha256"] != bench._hash(dll) or not baseline.get("still") or
            baseline["source_sha256"] != bench._hash(Path(bench.__file__))):
        raise ValueError("require matching benchmark implementation, DLL and complete main measurements")
    cases = [dict(id=key,label=label,overrides=options,rounds=[],image=f"images/{key}.png")
             for key,label,options in ADDITIONAL]
    with TemporaryDirectory(prefix="phymeo-grid-") as directory:
        scenes,frames,studio,trace = bench.bind_scene(Path(directory)/"geometry")
        inputs = [bench.make_inputs(scene,frames) for scene in scenes]
        bound = bench.film.input_binding(scenes[0], [inputs[0][i] for i in bench.WARMUP+bench.MEASURE],trace)
        if (json.loads(json.dumps(bound["frames"])) != baseline["frames"] or
                bound["source_excerpt_sha256"] != baseline["source_trace_sha256"]):
            raise ValueError("different recorded trajectory or camera")
        previous = baseline["cases"][0]["rounds"][0]
        context = out,dll,args.device_index,scenes,inputs,studio
        for order in (range(len(ADDITIONAL)),reversed(range(len(ADDITIONAL)))):
            for i in order:
                row=cases[i]
                measured,rgb = bench.run_case(ADDITIONAL[i], context)
                for key in ("uuid","sensors_sha256","triangles","geometries"):
                    if measured[key]!=previous[key]:
                        raise ValueError(f"{row['id']} has a different {key}")
                if row["rounds"]:
                    if measured["rgb_sha256"]!=row["rounds"][0]["rgb_sha256"]:
                        raise ValueError("supplement hero pose changed")
                else:
                    Image.frombytes("RGB",bench.SIZE,rgb).save(out/row["image"])
                row["rounds"].append(measured)
                print(f"{row['id']} round {len(row['rounds'])}: {sum(measured['capture_ms'])/len(bench.MEASURE):.2f} ms", flush=True)
        reference_mode = ("independent", "独立种子静态参考", dict(HIGH, indirect_seed=1))
        reference, rgb = bench.run_case(reference_mode, context, still=True)
        if reference["hero_sensor_sha256"] != baseline["still"]["hero_sensor_sha256"]:
            raise ValueError("independent reference uses different geometry")
        ref_path = "images/reference-seed1.png"
        Image.frombytes("RGB",bench.SIZE,rgb).save(out/ref_path)
    bench.save(report,dict(base_sha256=bench._hash(original),source_sha256=bench._hash(Path(__file__)),
        cases=cases, reference=dict(reference,image=ref_path,seed=1,batches=64,spp=1024),
        total_timed_captures=64+sum(len(r["capture_ms"]) for c in cases for r in c["rounds"])))


if __name__=="__main__":
    main()
