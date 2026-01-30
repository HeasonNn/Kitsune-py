#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

# (name, input_feature_dir, output_dir)
DATASETS = [
    ("ids2017", "data/ids2017", "results/ids2017"),
    # ("hyervision", "data/hyervision", "results/hyervision"),
]

EXCLUDE_SUFFIXES = (
    ".label.npy",
    ".full.npy",
    "_res.npy",
    "_scores.npy",
)


def list_feature_files(in_dir: Path):
    for p in sorted(in_dir.glob("*.npy")):
        name = p.name
        if any(name.endswith(suf) for suf in EXCLUDE_SUFFIXES):
            continue
        if not (p.parent / (p.name + ".label.npy")).exists():
            print(f"[WARN] missing label sidecar: {p.name}.label.npy (skip)", file=sys.stderr)
            continue
        yield p


def run_one(py: Path, script: Path, x_path: Path, out_path: Path, args) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(py),
        str(script),
        "--data",
        str(x_path),
        "--out",
        str(out_path),
        "--maxAE",
        str(args.maxAE),
        "--FMgrace",
        str(args.FMgrace),
        "--ADgrace",
        str(args.ADgrace),
        "--beta",
        str(args.beta),
        "--batch_size",
        str(args.batch_size),
        "--epochs_ens",
        str(args.epochs_ens),
        "--epochs_out",
        str(args.epochs_out),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    if args.save_full:
        cmd.append("--save")
    if args.n_rows is not None:
        cmd += ["--n_rows", str(args.n_rows)]

    # print(">>", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--script", default="kitsune_offline.py", help="Path to kitsune_offline.py")
    ap.add_argument("--maxAE", type=int, default=10)
    ap.add_argument("--FMgrace", type=int, default=5000)
    ap.add_argument("--ADgrace", type=int, default=50000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--epochs_ens", type=int, default=3)
    ap.add_argument("--epochs_out", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_full", action="store_true", help="Also save <out>.full.npy with NaNs for FM/AD phases")
    ap.add_argument("--n_rows", type=int, default=None, help="Limit rows read from X (useful when tail is unused)")

    args = ap.parse_args()

    py = Path(args.python)
    script = Path(args.script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Not found: {script}")

    total = 0
    failed = 0

    for name, in_dir_s, out_dir_s in DATASETS:
        in_dir = Path(in_dir_s).resolve()
        out_dir = Path(out_dir_s).resolve()

        if not in_dir.exists():
            print(f"[WARN] input dir not found, skip: {in_dir}", file=sys.stderr)
            continue

        files = list(list_feature_files(in_dir))
        print(f"\n== Dataset: {name}  files={len(files)}  in={in_dir}  out={out_dir}\n")

        for x_path in files:
            stem = x_path.stem
            out_path = out_dir / f"{stem}_res.npy"  # overwrite by default
            rc = run_one(py, script, x_path, out_path, args)
            total += 1
            if rc != 0:
                failed += 1
                print(f"[ERROR] rc={rc} file={x_path}", file=sys.stderr)

    print(f"\nDone. total_runs={total} failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
