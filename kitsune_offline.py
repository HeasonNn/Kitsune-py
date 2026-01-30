from __future__ import annotations

import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn

from KitNET.corClust import corClust
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)


@dataclass
class SplitData:
    X_fm: np.ndarray
    X_ad: np.ndarray
    X_ex: np.ndarray
    y_ex: Optional[np.ndarray]


@dataclass
class Normalizers:
    x_min: np.ndarray
    x_max: np.ndarray
    e_min: np.ndarray
    e_max: np.ndarray


class AE(nn.Module):
    def __init__(self, in_dim: int, beta: float = 0.75):
        super().__init__()
        hid = max(1, int(math.ceil(beta * in_dim)))
        self.enc = nn.Linear(in_dim, hid, bias=True)
        self.dec = nn.Linear(hid, in_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.enc(x))
        return self.dec(z)


@torch.no_grad()
def recon_rmse(ae: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    yb = ae(xb)
    mse = torch.mean((yb - xb) ** 2, dim=1)
    return torch.sqrt(mse)


@torch.no_grad()
def score_execution_phase(
    X_ex: np.ndarray,
    ensemble: List[Tuple[List[int], AE]],
    out_ae: AE,
    norms: Normalizers,
    device: str,
    batch_size: int,
) -> np.ndarray:
    print("Scoring execution phase...")
    scores: List[torch.Tensor] = []
    for xb_raw in iter_array_batches(X_ex, batch_size):
        xb_cpu = np.asarray(xb_raw, dtype=np.float32)
        xb_cpu = normalize_np(xb_cpu, norms.x_min, norms.x_max).astype(np.float32, copy=False)
        xb_cpu_t = torch.from_numpy(xb_cpu)
        e_parts = []
        for idxs, ae in ensemble:
            xb = xb_cpu_t[:, idxs].to(device, non_blocking=True)
            e_parts.append(recon_rmse(ae, xb))
        evec_cpu = torch.stack(e_parts, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        evec_cpu = normalize_np(evec_cpu, norms.e_min, norms.e_max).astype(np.float32, copy=False)
        evec = torch.from_numpy(evec_cpu).to(device, non_blocking=True)
        s = recon_rmse(out_ae, evec)
        scores.append(s.detach().cpu())

    return torch.cat(scores, dim=0).numpy().astype(np.float32)


def load_features_and_labels(
    x_path: str, label_path: Optional[str], n_rows: Optional[int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X = np.load(x_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D array, got shape {X.shape}")
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    N, _ = X.shape

    y = None
    cand = x_path + ".label.npy"
    if label_path is None and os.path.exists(cand):
        label_path = cand
    if label_path is not None:
        y = np.load(label_path, mmap_mode="r")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D array, got shape {y.shape}")
        if y.shape[0] < N:
            raise ValueError(f"Label length {y.shape[0]} < feature rows {N}")
    if n_rows is not None:
        if n_rows <= 0 or n_rows > N:
            raise ValueError(f"--n_rows must be in (0, {N}], got {n_rows}")
        N = int(n_rows)
        X = X[:N]
        if y is not None:
            y = y[:N]

    return X, y


def split_data(
    X: np.ndarray,
    y: Optional[np.ndarray],
    FMgrace: int,
    ADgrace: int,
) -> SplitData:
    N = X.shape[0]
    if FMgrace + ADgrace >= N:
        raise ValueError(f"FMgrace+ADgrace must be < N. Got {FMgrace}+{ADgrace} >= {N}")

    X_fm = X[:FMgrace]
    X_ad = X[FMgrace : FMgrace + ADgrace]
    X_ex = X[FMgrace + ADgrace :]

    y_ex = None
    if y is not None:
        y_ex = np.asarray(y[FMgrace + ADgrace :], dtype=np.uint8)

    return SplitData(X_fm=X_fm, X_ad=X_ad, X_ex=X_ex, y_ex=y_ex)


def compute_minmax_np(X_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_min = np.min(X_np, axis=0)
    x_max = np.max(X_np, axis=0)
    rng = x_max - x_min
    rng[rng == 0] = 1.0
    return x_min.astype(np.float32, copy=False), x_max.astype(np.float32, copy=False)


def normalize_np(X_np: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    rng = x_max - x_min
    rng = np.where(rng == 0, 1.0, rng)
    return (X_np - x_min) / rng


def iter_array_batches(X: np.ndarray, bs: int) -> Iterable[np.ndarray]:
    n = X.shape[0]
    for i in range(0, n, bs):
        yield X[i : i + bs]


def train_ae(
    ae: nn.Module,
    data_cpu: torch.Tensor,
    device: str,
    batch_size: int,
    epochs: int,
    lr: float = 1e-3,
) -> None:
    ae.train()
    ds = torch.utils.data.TensorDataset(data_cpu)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="mean")

    for ep in range(epochs):
        tot = 0.0
        cnt = 0
        for (xb,) in dl:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yb = ae(xb)
            loss = loss_fn(yb, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            cnt += xb.size(0)
        print(f"  epoch {ep+1}/{epochs} loss={tot/cnt:.6e}")




def learn_feature_map(X_fm: np.ndarray, d: int, maxAE: int, FMgrace: int) -> List[List[int]]:
    if FMgrace <= 0:
        return [list(range(i, min(i + maxAE, d))) for i in range(0, d, maxAE)]

    fm = corClust(d)
    for x in X_fm:
        fm.update(np.asarray(x, dtype=np.float32))
    return fm.cluster(maxAE)


def train_ensemble_and_output(
    X_ad: np.ndarray,
    groups: List[List[int]],
    beta: float,
    device: str,
    batch_size: int,
    epochs_ens: int,
    epochs_out: int,
) -> Tuple[List[Tuple[List[int], AE]], AE, Normalizers]:
    X_ad_np = np.asarray(X_ad, dtype=np.float32)
    x_min, x_max = compute_minmax_np(X_ad_np)
    X_ad_norm = normalize_np(X_ad_np, x_min, x_max).astype(np.float32, copy=False)
    ensemble: List[Tuple[List[int], AE]] = []
    X_ad_t = torch.from_numpy(X_ad_norm)  # CPU tensor

    print("Training ensemble AEs...")
    for gi, idxs in enumerate(groups):
        ae = AE(len(idxs), beta=beta).to(device)
        data = X_ad_t[:, idxs]
        train_ae(ae, data, device=device, batch_size=batch_size, epochs=epochs_ens)
        ensemble.append((idxs, ae))

    print("Building error vectors for output AE training...")
    errs = []
    for xb_cpu in iter_array_batches(X_ad_norm, batch_size):
        xb_cpu_t = torch.from_numpy(np.asarray(xb_cpu, dtype=np.float32))
        e_parts = []
        for idxs, ae in ensemble:
            xb = xb_cpu_t[:, idxs].to(device, non_blocking=True)
            e_parts.append(recon_rmse(ae, xb))
        errs.append(torch.stack(e_parts, dim=1).detach().cpu())

    E_ad = torch.cat(errs, dim=0)  # [ADgrace, G]
    E_ad_np = E_ad.numpy().astype(np.float32, copy=False)
    e_min, e_max = compute_minmax_np(E_ad_np)
    E_ad_norm = normalize_np(E_ad_np, e_min, e_max).astype(np.float32, copy=False)
    E_ad_t = torch.from_numpy(E_ad_norm)

    print("Training output AE...")
    out_ae = AE(E_ad_t.shape[1], beta=beta).to(device)
    train_ae(out_ae, E_ad_t, device=device, batch_size=batch_size, epochs=epochs_out)

    return ensemble, out_ae, Normalizers(x_min=x_min, x_max=x_max, e_min=e_min, e_max=e_max)


def save_outputs(
    scores: np.ndarray,
    out_path: str,
    N_total: int,
    FMgrace: int,
    ADgrace: int,
    save: bool,
) -> None:
    np.save(out_path, scores)
    print("Saved", out_path + ":", str(scores.shape))
    if save:
        full = np.full((N_total,), np.nan, dtype=np.float32)
        start = FMgrace + ADgrace
        full[start : start + len(scores)] = scores
        full_out = out_path[:-4] + ".full.npy" if out_path.endswith(".npy") else out_path + ".full.npy"
        np.save(full_out, full)
        print("Saved", full_out + ":", str(full.shape))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--n_rows", type=int, default=None)
    p.add_argument("--out", default="rmse_offline.npy")
    p.add_argument("--maxAE", type=int, default=10)
    p.add_argument("--FMgrace", type=int, default=5000)
    p.add_argument("--ADgrace", type=int, default=50000)
    p.add_argument("--beta", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs_ens", type=int, default=3)
    p.add_argument("--epochs_out", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", action="store_true", help="Also save full-length rmse with NaN for FM/AD phases")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    torch.set_float32_matmul_precision("high")

    X, y = load_features_and_labels(args.data, args.label, args.n_rows)
    N, d = X.shape
    print("X:", (N, d), "dtype:", X.dtype, "device:", device, "labels:", (None if y is None else y.shape))

    split = split_data(X, y, args.FMgrace, args.ADgrace)
    groups = learn_feature_map(split.X_fm, d=d, maxAE=args.maxAE, FMgrace=args.FMgrace)
    print("num groups:", len(groups), "group sizes:", [len(g) for g in groups][:10], "...")

    ensemble, out_ae, norms = train_ensemble_and_output(
        split.X_ad,
        groups=groups,
        beta=args.beta,
        device=device,
        batch_size=args.batch_size,
        epochs_ens=args.epochs_ens,
        epochs_out=args.epochs_out,
    )

    scores = score_execution_phase(
        split.X_ex,
        ensemble=ensemble,
        out_ae=out_ae,
        norms=norms,
        device=device,
        batch_size=args.batch_size,
    )

    save_outputs(scores, args.out, N_total=N, FMgrace=args.FMgrace, ADgrace=args.ADgrace, save=args.save)

    if split.y_ex is None:
        raise ValueError("Metrics require labels (provide --label or ensure <data>.label.npy exists)")

    m = min(len(scores), len(split.y_ex))
    y_eval = split.y_ex[:m].astype(np.uint8, copy=False)
    s_eval = scores[:m].astype(np.float64, copy=False)

    auc = roc_auc_score(y_eval, s_eval)
    ap = average_precision_score(y_eval, s_eval)

    precision, recall, thresholds = precision_recall_curve(y_eval, s_eval)
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.where(denom == 0, 0.0, 2.0 * p * r / denom)
    best_idx = int(np.argmax(f1)) if f1.size > 0 else 0
    thr = float(thresholds[best_idx]) if thresholds.size > 0 else float("inf")

    y_pred = (s_eval >= thr).astype(np.uint8)
    p2, r2, f12, _ = precision_recall_fscore_support(y_eval, y_pred, average="binary", zero_division=0)

    print("roc_auc=", float(auc))
    print("average_precision=", float(ap))
    print("best_f1_threshold=", thr)
    print("best_f1=", float(f12))
    print("precision=", float(p2))
    print("recall=", float(r2))


if __name__ == "__main__":
    main()
