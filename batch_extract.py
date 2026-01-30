#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

DATASETS = [
    # ("hypervision", "raw_data/hypervision", "data/hypervision"),
    ("ids2017", "raw_data/ids2017", "data/ids2017"),
]


def find_pairs(input_dir: Path):
    for data_path in sorted(input_dir.glob("*.data")):
        stem = data_path.stem
        label_path = data_path.with_suffix(".label")
        if not label_path.exists():
            print(f"[WARN] missing label for {data_path.name}, expected {label_path.name}", file=sys.stderr)
            continue
        yield data_path, label_path, stem


def run_one(
    py: Path,
    fe_script: Path,
    data_path: Path,
    label_path: Path,
    out_path: Path,
    chunk_size: int,
    max_flow: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(py),
        str(fe_script),
        "--data",
        str(data_path),
        "--label",
        str(label_path),
        "--out",
        str(out_path),
        "--chunk_size",
        str(chunk_size),
        "--max_flow",
        str(max_flow),
    ]

    print(">>", " ".join(cmd))

    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to run FeatureExtractor2.py")
    ap.add_argument("--chunk_size", type=int, default=500_000)
    ap.add_argument("--max_flow", type=int, default=50_000)
    args = ap.parse_args()

    fe_script = Path("FeatureExtractor2.py").resolve()
    py = Path(args.python)
    jobs = [(name, Path(in_dir), Path(out_dir)) for (name, in_dir, out_dir) in DATASETS]
    total = 0
    failed = 0

    for name, in_dir, out_dir in jobs:
        in_dir = in_dir.resolve()
        out_dir = out_dir.resolve()
        if not in_dir.exists():
            print(f"[WARN] input dir not found, skip: {in_dir}", file=sys.stderr)
            continue

        pairs = list(find_pairs(in_dir))
        print(f"\n== Dataset: {name}  pairs={len(pairs)}  in={in_dir}  out={out_dir}\n")

        for data_path, label_path, stem in pairs:
            out_path = out_dir / f"{stem}.npy"

            rc = run_one(
                py=py,
                fe_script=fe_script,
                data_path=data_path,
                label_path=label_path,
                out_path=out_path,
                chunk_size=args.chunk_size,
                max_flow=args.max_flow,
            )
            total += 1
            if rc != 0:
                failed += 1
                print(f"[ERROR] failed rc={rc}: {data_path}", file=sys.stderr)

    print(f"\nDone. total_runs={total} failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
