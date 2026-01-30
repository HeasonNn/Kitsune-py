import sys
import argparse
import subprocess
import numpy as np
import netStat2 as ns2

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Sequence
from numpy.lib.format import open_memmap


TIMESTAMP_UNIT = "ns"
TARGET_DTYPE = np.float32


@dataclass
class LabelSource:
    mode: str  # 'bitstring' | 'lines'
    path: Optional[str] = None
    data: Optional[bytes] = None
    pos: int = 0


class LabelReader:
    def __init__(self, src: LabelSource):
        self.src = src
        self._fh = open(src.path, "r", encoding="utf-8", errors="ignore") if src.mode == "lines" else None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def next_label(self) -> int:
        if self.src.mode == "bitstring":
            b = self.src.data
            pos = self.src.pos
            if pos >= len(b):
                raise RuntimeError("Label bitstring ended before data file.")
            self.src.pos = pos + 1
            return 1 if b[pos] == 49 else 0  # '1' == 49

        while True:
            ln = self._fh.readline()
            if not ln:
                raise RuntimeError("Label file ended before data file.")
            ln = ln.strip()
            if ln:
                break

        return int(ln)


def count_lines(path: str) -> int:
    out = subprocess.check_output(["wc", "-l", path], text=True)
    return int(out.strip().split()[0])


def load_label_source(label_path: str, expected_n: int) -> LabelSource:
    first = ""
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                first = ln
                break

    if not first:
        raise RuntimeError("Label file is empty")

    looks_binary = all(ch in "01" for ch in first[: min(len(first), 1024)])
    if looks_binary and len(first) >= expected_n:
        b = first.encode("ascii", errors="ignore")
        if len(b) < expected_n:
            raise RuntimeError(f"Bitstring label length {len(b)} < expected packets {expected_n}")
        return LabelSource(mode="bitstring", data=b, pos=0)

    return LabelSource(mode="lines", path=label_path)


def parse_data_line(line: str) -> Optional[np.ndarray]:
    arr = np.fromstring(line, sep=" ", dtype=np.int64)
    return arr if arr.size == 8 else None


def stream_valid_chunks(
    data_path: str,
    label_src: LabelSource,
    chunk_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    labels = LabelReader(label_src)
    try:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as fdata:
            while True:
                rows = []
                labs = []

                while len(rows) < chunk_size:
                    ln = fdata.readline()
                    if not ln:
                        break
                    ln = ln.strip()
                    if not ln:
                        continue

                    lab_val = labels.next_label()
                    arr = parse_data_line(ln)
                    if arr is None:
                        continue

                    rows.append(arr)
                    labs.append(lab_val)

                if not rows:
                    break

                yield np.vstack(rows).astype(np.int64, copy=False), np.asarray(labs, dtype=np.int64)
    finally:
        labels.close()


def build_features_batch(extractor: ns2.VectorizedNetStat, data_arr: np.ndarray) -> np.ndarray:
    sip = data_arr[:, 1]
    dip = data_arr[:, 2]
    sp = data_arr[:, 3]
    dp = data_arr[:, 4]
    ts = data_arr[:, 5].astype(np.float64, copy=False)
    proto = data_arr[:, 6]
    plen = data_arr[:, 7].astype(np.float64, copy=False)
    ts_sec = ts * 1e-9
    X = extractor.process_arrays(sip, dip, sp, dp, proto, plen, ts_sec)
    if X.dtype != TARGET_DTYPE:
        X = X.astype(TARGET_DTYPE, copy=False)
    return X


def select_keep_indices_by_flow(
    data_arr: np.ndarray,
    labels: np.ndarray,
    max_flow: int,
    normal_flow_set: set,
) -> np.ndarray:
    n = labels.shape[0]
    keep = np.zeros(n, dtype=bool)

    atk_mask = labels != 0
    keep[atk_mask] = True

    for i in np.flatnonzero(labels == 0):
        key = (
            int(data_arr[i, 1]),
            int(data_arr[i, 2]),
            int(data_arr[i, 3]),
            int(data_arr[i, 4]),
            int(data_arr[i, 6]),
        )

        if key in normal_flow_set:
            keep[i] = True
        elif len(normal_flow_set) < max_flow:
            normal_flow_set.add(key)
            keep[i] = True

    return np.flatnonzero(keep).astype(np.int64, copy=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, help="Output features .npy (X). Labels saved to <out>.label.npy")
    p.add_argument("--chunk_size", type=int, default=500_000)
    p.add_argument("--max_flow", type=int, default=50_000)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    extractor = ns2.VectorizedNetStat(dtype=TARGET_DTYPE)
    F = 20 * int(len(extractor.lambdas))
    N = count_lines(args.data)
    label_src = load_label_source(args.label, expected_n=N)
    if label_src.mode == "bitstring":
        print(f"Label mode: bitstring (len={len(label_src.data)})", file=sys.stderr)

    X_mm = open_memmap(args.out, mode="w+", dtype=TARGET_DTYPE, shape=(N, F))
    y_path = args.out + ".label.npy"
    y_mm = open_memmap(y_path, mode="w+", dtype=np.int64, shape=(N,))

    row = 0
    attack_kept = 0
    normal_flow_set: set = set()
    report_every = 2_000_000
    next_report = report_every
    for data_arr, labels in stream_valid_chunks(args.data, label_src, args.chunk_size):
        X = build_features_batch(extractor, data_arr)
        keep = select_keep_indices_by_flow(data_arr, labels, args.max_flow, normal_flow_set)
        if keep.size == 0:
            continue
        X = X[keep]
        labels = labels[keep]
        attack_kept += int(np.count_nonzero(labels != 0))
        n = X.shape[0]
        if n == 0:
            continue
        if row + n > N:
            n = N - row
            X = X[:n]
            labels = labels[:n]
        X_mm[row : row + n, :] = X
        y_mm[row : row + n] = labels
        row += n
        if row >= next_report:
            if args.max_flow is None:
                print(f"Processed {row}/{N} kept packets...", file=sys.stderr)
            else:
                print(
                    f"Processed {row}/{N} kept packets (attack_kept={attack_kept}, normal_flows_kept={len(normal_flow_set)})...",
                    file=sys.stderr,
                )
            next_report += report_every
    X_mm.flush()
    y_mm.flush()

    if row != N:
        print(
            f"NOTE: wrote {row} rows (preallocated {N}). The file tail is unused due to skipped invalid lines and/or limits.",
            file=sys.stderr,
        )

    if args.max_flow is not None:
        print(
            f"Limit: attack_kept={attack_kept}, normal_flows_kept={len(normal_flow_set)} (max_flow={args.max_flow})",
            file=sys.stderr,
        )

    print(f"Saved features: {args.out} shape=({row},{F}) dtype=float32", file=sys.stderr)
    print(f"Saved labels  : {y_path} shape=({row},) dtype=int64", file=sys.stderr)


if __name__ == "__main__":
    main()
