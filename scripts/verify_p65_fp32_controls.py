"""Run only the ten frozen N2 arithmetic controls, without importing the physics solver."""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import tyx as tx

FIXTURE = Path(__file__).resolve().parents[1] / "docs/benchmarks/p65-fp32-reduction-controls.json"
FIXTURE_HASH = "80746deae1c918b1ccfbb0bbf9ff55d1d12e0d57667dc9bee3e39cfd3d5c5b37"
PIN = "99f6df7c0fd8fdeef77f97df83914da27b64b7ee"


@tx.kernel
def reductions(
    count: tx.i32,
    kinds: tx.ndarray(tx.i32, 1),
    x: tx.ndarray(tx.f32, 2),
    words: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        a = x[i, 0]
        b = x[i, 1]
        c = x[i, 2]
        d = x[i, 3]
        if kinds[i] == 0:
            words[i, 0] = tx.fma(a, b, c)
        elif kinds[i] == 1:
            anchor = tx.mul_rn(b, b)
            partial = tx.fma(a, a, anchor)
            words[i, 0] = anchor
            words[i, 1] = partial
            words[i, 2] = tx.fma(c, c, partial)
        elif kinds[i] == 2:
            anchor = tx.mul_rn(a, b)
            naive = tx.fma(-c, d, anchor)
            correction = tx.fma(a, b, -anchor)
            words[i, 0] = anchor
            words[i, 1] = naive
            words[i, 2] = correction
            words[i, 3] = tx.add_rn(naive, correction)
        elif kinds[i] == 3:
            words[i, 0] = tx.mul_rn(a, b)
        else:
            weighted = tx.mul_rn(a, b)
            words[i, 0] = weighted
            words[i, 1] = tx.mul_rn(weighted, c)
            words[i, 2] = tx.fma(weighted, c, d)


@dataclass(frozen=True)
class Case:
    name: str
    kind: int
    operands: tuple
    expected_words: tuple


ONE = 0x3F800000
S = 1 + 2**-23
CASES = (
    Case("positive_denominator", 0, (2**-70, 2**-70, 2**14), (0x46800000,)),
    Case("scale_cannot_cover_both", 0, (2**-100, 2**-100, 2**126), (0x7E800000,)),
    Case("norm_tiny_first", 1, (2**-96, 1, 0), (ONE, ONE, ONE)),
    Case("cross_tiny_second", 2, (1, 1, 2**-96, 2**-48), (ONE, ONE, 0, ONE)),
    Case("fused_residual", 0, (S, S, -(1 + 2**-22)), (0x28800000,)),
    Case(
        "determinant_compensation", 2, (S, S, S, S),
        (0x3F800002, 0xA8800000, 0x28800000, 0),
    ),
    Case("min_normal_tie", 3, (2**-63, 2**-63 - 2**-87), (0x00800000,)),
    Case("raw_subnormal", 4, (1, 2**-70, 2**-70, 1), (0x1C800000, 0x00000200, ONE)),
    Case("false_zero", 0, (2**-80, 2**-70, 0), (0,)),
    Case("overflow", 0, (2**100, 2**100, 0), (0x7F800000,)),
)
FIELDS = (
    ("fma",),
    ("square_y", "plus_square_x", "plus_square_z"),
    ("rounded_ab", "naive_determinant", "ab_rounding_error", "compensated_determinant"),
    ("product",),
    ("weighted_delta", "raw_product", "fused_position"),
)
PUBLIC_INDEX = (0, 2, 3, 0, 1)


def exact_source(case):
    x = tuple(Fraction(v) for v in case.operands)
    if case.kind == 1:
        return sum(v * v for v in x)
    if case.kind == 2:
        return x[0] * x[1] - x[2] * x[3]
    if case.kind == 3:
        return x[0] * x[1]
    if case.kind == 4:
        return x[0] * x[1] * x[2]
    return x[0] * x[1] + x[2]


def classify(word, source):
    magnitude = word & 0x7FFFFFFF
    if magnitude >= 0x7F800000:
        return "reject_nonfinite"
    if magnitude == 0:
        if source:
            return "reject_lost_nonzero"
        return "accept" if word == 0 else "reject_negative_zero"
    if magnitude < 0x00800000:
        return "reject_subnormal"
    return "accept"


def run(mode, cache_root, kinds, inputs):
    output = np.full((len(CASES), 4), -17, dtype=np.float32)
    backend = "cuda" if mode == "cuda" else "cpu"
    before = tx.runtime_stats()
    with ExitStack() as resources:
        runtime = tx.Runtime(backend=backend, num_threads=1, cache_root=cache_root)
        resources.callback(runtime.close)
        resources.enter_context(tx.use_runtime(runtime))
        arrays = [kinds, inputs, output]
        if mode == "cuda":
            devices = []
            for array in arrays:
                device_array = tx.cuda_array(array)
                resources.callback(device_array.free)
                devices.append(device_array)
            reductions(len(CASES), *devices, mode=mode)
            runtime.sync()
            output = devices[-1].to_numpy()
        else:
            reductions(len(CASES), *arrays, mode=mode)
    return output.view(np.uint32), asdict(tx.runtime_stats() - before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    raw = FIXTURE.read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != FIXTURE_HASH:
        raise ValueError("input fixture no longer matches the approved version")
    fixture = json.loads(raw)["cases"]
    if [row["id"] for row in fixture] != [case.name for case in CASES]:
        raise ValueError("fixture membership or order changed")
    dist = distribution("tyx-compute")
    pin = json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]
    if pin != PIN or Path(tx.__file__).resolve() != Path(dist.locate_file("tyx/__init__.py")).resolve():
        raise RuntimeError("unexpected tyx import or revision")
    inputs = np.zeros((len(CASES), 4), np.float32)
    for i, case in enumerate(CASES):
        inputs[i, :len(case.operands)] = case.operands
        if any(Fraction(float(inputs[i, j])) != value for j, value in enumerate(case.operands)):
            raise ValueError(f"input is not represented exactly: {case.name}")
    kinds = np.asarray([case.kind for case in CASES], np.int32)
    words, counters = run(args.mode, args.cache_root, kinds, inputs)
    cases = []
    for i, case in enumerate(CASES):
        actual = tuple(map(int, words[i, :len(case.expected_words)]))
        exact = exact_source(case)
        verdict = classify(actual[PUBLIC_INDEX[case.kind]], exact)
        wanted = {
            "raw_subnormal": "reject_subnormal",
            "false_zero": "reject_lost_nonzero",
            "overflow": "reject_nonfinite",
        }.get(case.name, "accept")
        cases.append({
            "id": case.name,
            "input_words": [f"{w:08x}" for w in inputs[i, :len(case.operands)].view(np.uint32)],
            "source_fraction": str(exact),
            "fields": FIELDS[case.kind],
            "actual_words": [f"{w:08x}" for w in actual],
            "expected_words": [f"{w:08x}" for w in case.expected_words],
            "publication": verdict,
            "matches": actual == case.expected_words and verdict == wanted,
        })
    ok = all(row["matches"] for row in cases)
    if args.mode == "cuda":
        ok = ok and counters["cuda_kernel_launches"] == 1 and counters["cuda_d2h_transfers"] >= 1
    report = {
        "schema": "p65-independent-fp32-controls-run-v1",
        "mode": args.mode,
        "passed": ok,
        "fixture_lf_sha256": FIXTURE_HASH,
        "probe_lf_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "tyx_commit": pin, "tyx_file": tx.__file__, "executable": sys.executable,
        "python": sys.version, "numpy": np.__version__, "tyx_version": dist.version,
        "runtime_counts": counters, "cases": cases,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"{args.mode}: {sum(row['matches'] for row in cases)}/{len(cases)} word records match")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
