"""Select a frozen SPSA output-head scale on four new seeds, then run its independent holdout.

python examples/bimanual_residual_selection.py select --base CEM --network SPSA --out NEW --library VULKAN
python examples/bimanual_residual_selection.py evaluate --base CEM --network SPSA --out NEW --library VULKAN
python examples/bimanual_residual_selection.py replay --base CEM --network SPSA --out NEW --library VULKAN --seed 67001

Use --source-dir for archived CEM sources and --neural-source for archived SPSA code.
The selected-network.json is a normal ResidualNet checkpoint. replay verifies its lineage
and writes film-ready XML, NPZ and JSON under NEW/replays/SEED/.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import bimanual_neural_replay as replay
import bimanual_policy_replay as cem


VALIDATION = (64001, 64002, 64003, 64004)
HOLDOUT = tuple(range(66011, 66019))
SCALES = {"zero": 0., "quarter": .25, "half": .5, "full": 1.}


def _seed_rows(args, trainer, residual, base, config, stage, seed, jobs, contract):
    path = args.out/f"{stage}-{seed}.json"
    labels = [name for name, _ in jobs]
    if path.exists():
        saved = replay._read(path)
        if saved["contract"] != contract:
            raise ValueError(f"existing {path.name} belongs to another protocol")
        rows = saved["episodes"]
    else:
        rows = []
        for i in range(0, len(jobs), args.batch):
            rows.extend(residual.rollout(trainer, base, config, args, seed, jobs[i:i+args.batch]))
    if ([r["condition"] for r in rows] != labels or any(r["seed"] != seed for r in rows)
            or len({r["scene_sha256"] for r in rows}) != 1
            or any(type(r["success"]) is not bool or r["completed_batch_kernels"] <= 0
                   or not np.isfinite(r["return_sum"]) for r in rows)):
        raise ValueError(f"incomplete matched device episodes for {stage} seed {seed}")
    if not path.exists():
        trainer._save(path, dict(contract=contract, episodes=rows))
    return dict(zip(labels, rows, strict=True))


def _summary(rows, key):
    runs = [r[key] for r in rows]
    return dict(successes=sum(r["success"] for r in runs), trials=len(runs),
                mean_return=float(np.mean([r["return_sum"] for r in runs])),
                changed_decisions=sum(r["changed_decisions"] for r in runs),
                action_distinct_seeds=sum(r[key]["action_sha256"] != r["zero"]["action_sha256"] for r in rows),
                paired_return_over_cem=[r[key]["return_sum"]-r["zero"]["return_sum"] for r in rows])


def _winner(rows):
    summary = {key: _summary(rows, key) for key in SCALES}
    eligible = ["zero"]
    for name in tuple(SCALES)[1:]:
        s = summary[name]
        if (s["successes"] >= summary["zero"]["successes"]
                and s["changed_decisions"] > 0 and s["action_distinct_seeds"] > 0):
            eligible.append(name)
    choice = max(eligible, key=lambda k: (summary[k]["successes"], summary[k]["mean_return"], -SCALES[k]))
    return dict(choice=choice, scale=SCALES[choice], mode="cem-fallback" if choice=="zero" else "scaled-spsa",
                eligible=eligible, validation_summary=summary)


def load_selected(directory, residual, trained, contract):
    """Load a derived checkpoint only if frozen validation, weights and source lineage agree."""
    selection, network = directory/"selection.json", directory/"selected-network.json"
    binding = replay._read(directory/"selection-binding.json")
    if binding != {"selection_sha256": replay._sha(selection), "selected_network_sha256": replay._sha(network)}:
        raise ValueError("selected checkpoint and validation receipt differ from the freeze binding")
    report = replay._read(selection)
    if report["contract"] != contract:
        raise ValueError("selected network belongs to a different input model or validation protocol")
    rows = []
    for seed in VALIDATION:
        p = directory/f"validation-{seed}.json"
        if replay._sha(p) != report["validation_sha256"][str(seed)]:
            raise ValueError("frozen validation episodes have changed")
        saved = replay._read(p)
        if (saved["contract"] != contract or [r["condition"] for r in saved["episodes"]] != list(SCALES)
                or any(r["seed"] != seed for r in saved["episodes"])):
            raise ValueError("invalid frozen validation cases")
        rows.append({r["condition"]: r for r in saved["episodes"]})
    decision = _winner(rows)
    if decision != report["decision"]:
        raise ValueError("selected scale differs from its registered validation ranking")
    selected = residual.ResidualNet.load(network, dict(selection_sha256=binding["selection_sha256"],
                                                        protocol=contract, scale=decision["scale"]))
    if (not np.array_equal(selected.head, trained.head*decision["scale"])
            or not np.array_equal(selected.encoder, trained.encoder) or not np.array_equal(selected.bias, trained.bias)):
        raise ValueError("derived weights do not equal the selected frozen-head scaling")
    return selected, decision, binding


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("select", "evaluate", "replay"))
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--network", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--neural-source", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--device-index", type=int)
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--batch", type=int, choices=(1,2,3,4), default=2)
    p.add_argument("--seed", type=int, default=67001, help="new seed for replay only")
    args = p.parse_args(argv)
    path = (args.base/"frozen-policy.json" if args.base.is_dir() else args.base).resolve()
    base, config, updates, bsha, csha = cem._load(path, path.with_name("checkpoint.json"))
    if base["arms"] != "both":
        p.error("select a bilateral frozen base")
    residual, trained, source, original = replay._network(args, base, bsha, csha)
    historical = {s for u in updates for s in u["seeds"]} | set(config["selection_seeds"]) | set(config["holdout_seeds"])
    for field in ("training","development","holdout"):
        historical.update(source[field])
    if historical.intersection(VALIDATION+HOLDOUT):
        raise ValueError("selector partitions overlap source-model partitions")
    _, trainer = cem._sources(config["sources"], path, args.source_dir, base.get("variant","centroid"))
    contract = dict(schema="bimanual-spsa-head-selection-v1", task=base["task"], base_sha256=bsha,
                    checkpoint_sha256=csha, base_contract_sha256=base["contract_sha256"], original=original,
                    sources=config["sources"], residual_source_sha256=source["source_sha256"],
                    selector_source_sha256=replay._sha(Path(__file__)), replay_source_sha256=replay._sha(Path(replay.__file__)),
                    validation=list(VALIDATION), holdout=list(HOLDOUT), scales=SCALES, batch=args.batch,
                    rule="max successes, max mean return, min scale; nonzero scales must match baseline successes and alter actions")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out/"selection-protocol.json"
    if manifest.is_file():
        if replay._read(manifest) != contract:
            raise ValueError("selection protocol differs from the registered one")
    elif args.command == "select":
        args.out.mkdir(parents=True, exist_ok=True)
        if any(args.out.iterdir()):
            p.error("use a fresh selection directory")
        trainer._save(manifest, contract)
    else:
        p.error("selection must be registered and frozen first")

    binding_path = args.out/"selection-binding.json"
    if args.command == "select" and not binding_path.exists():
        jobs = [(k, trained.with_head(s*trained.head)) for k,s in SCALES.items()]
        rows = [_seed_rows(args, trainer, residual, base, config, "validation", s, jobs, contract) for s in VALIDATION]
        decision = _winner(rows)
        report = args.out/"selection.json"
        weights = args.out/"selected-network.json"
        trainer._save(report, dict(contract=contract, decision=decision, validation_sha256={
            str(s): replay._sha(args.out/f"validation-{s}.json") for s in VALIDATION}))
        model_contract = dict(selection_sha256=replay._sha(report), protocol=contract, scale=decision["scale"])
        trained.with_head(trained.head*decision["scale"]).save(weights, model_contract)
        trainer._save(binding_path, dict(selection_sha256=replay._sha(report), selected_network_sha256=replay._sha(weights)))
    if not binding_path.exists():
        p.error("selection has not finished freezing its weights")
    selected, decision, binding = load_selected(args.out, residual, trained, contract)
    print(f"{base['task']}: {decision['mode']}, frozen scale {decision['scale']}", flush=True)
    if args.command == "select":
        return 0
    if args.command == "evaluate":
        initial = residual.ResidualNet.initial()
        jobs = [("zero", trained.with_head(np.zeros_like(trained.head))), ("selected", selected),
                ("untrained_same_scale", initial.with_head(initial.head*decision["scale"]))]
        rows = [_seed_rows(args,trainer,residual,base,config,"holdout",s,jobs,binding) for s in HOLDOUT]
        trainer._save(args.out/"selected-holdout.json", dict(binding=binding, seeds=list(HOLDOUT), decision=decision,
                      conditions={k: _summary(rows,k) for k,_ in jobs}, paired_selected_minus_untrained=[
                          r["selected"]["return_sum"]-r["untrained_same_scale"]["return_sum"] for r in rows]))
        return 0
    if args.seed < 0 or args.seed in historical.union(VALIDATION,HOLDOUT):
        p.error("film replay requires a new seed outside every model-selection and evaluation partition")
    parent = args.out/"replays"
    parent.mkdir(exist_ok=True)
    dest = parent/str(args.seed)
    if dest.exists():
        p.error("this film replay already exists")
    with TemporaryDirectory(dir=parent) as temp:
        root = Path(temp)/str(args.seed)
        root.mkdir()
        provenance = dict(selection=binding, contract=contract, choice=decision["choice"], scale=decision["scale"])
        episode = replay._record(args,args.seed,residual,selected,trainer,base,config,provenance,root)
        receipt = root/"episode.json"
        row = replay._read(receipt)
        row.update(selection_mode=decision["mode"], residual_scale=decision["scale"],
                   algorithm=residual.ARCH if decision["scale"] else "frozen-cem-zero-residual",
                   condition="selected")
        row["job"]["label"] = decision["mode"]
        trainer._save(receipt,row)
        episode["receipt_sha256"] = replay._sha(receipt)
        trainer._save(root/"replay.json", dict(binding=binding, episode=episode, selection_mode=decision["mode"]))
        root.rename(dest)
    print(dest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
