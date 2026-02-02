from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs_dir: Path
    fig_dir: Path
    ckpt_dir: Path

    args_json: Path
    history_csv: Path
    summary_json: Path

    def file(self, name: str) -> Path:
        return self.run_dir / name

    def log_file(self, name: str) -> Path:
        return self.logs_dir / name

    def fig_file(self, name: str) -> Path:
        return self.fig_dir / name

    def ckpt_file(self, name: str) -> Path:
        return self.ckpt_dir / name


@dataclass
class RunContext:
    project_name: str
    dataset_stem: str
    base_dir: Path
    index_path: Path

    run_tag: str
    paths: RunPaths

    meta: Dict[str, Any] = field(default_factory=dict)


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    def ensure_dir(self, path: Path) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def write_text(self, path: Path, text: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def write_bytes(self, path: Path, data: bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def write_csv_rows(self, path: Path, rows: List[Dict[str, Any]], *, mode: str, header: bool) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def save_npy(self, path: Path, array: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path: Path) -> bool:
        raise NotImplementedError


class BaseTracker(abc.ABC):
    def __init__(
        self,
        store: ArtifactStore,
        *,
        base_dir: str | Path = "runs",
        index_name: str = "results_index.csv",
        project_name: str = "project",
    ):
        self.store = store
        self.base_dir = Path(base_dir)
        self.index_name = index_name
        self.project_name = project_name

        self._ctx: Optional[RunContext] = None
        self._epoch_rows: List[Dict[str, Any]] = []
        self._epoch_counter: int = 0

    @property
    def ctx(self) -> RunContext:
        if self._ctx is None:
            raise RuntimeError("Tracker has not started. Call start(args) first.")
        return self._ctx

    def start(self, args: Any, *, dataset_stem: Optional[str] = None) -> RunContext:
        ds = dataset_stem or getattr(args, "data_path", None)
        if ds is None:
            dataset_stem = "data"
        else:
            dataset_stem = Path(str(ds)).stem

        run_tag = self.build_run_tag(args=args, dataset_stem=dataset_stem)
        run_dir = self.build_run_dir(args=args, dataset_stem=dataset_stem, run_tag=run_tag)

        paths = self._make_paths(run_dir=run_dir)

        self.store.ensure_dir(paths.run_dir)
        self.store.ensure_dir(paths.logs_dir)
        self.store.ensure_dir(paths.fig_dir)
        self.store.ensure_dir(paths.ckpt_dir)

        index_path = self.base_dir / f"{self.project_name}_{self.index_name}"
        ctx = RunContext(
            project_name=self.project_name,
            dataset_stem=dataset_stem,
            base_dir=self.base_dir,
            index_path=index_path,
            run_tag=run_tag,
            paths=paths,
        )
        self._ctx = ctx
        self._epoch_rows = []
        self._epoch_counter = 0
        self.save_args(args)
        self.on_start(args=args, ctx=ctx)

        return ctx

    def save_args(self, args: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(vars(args)) if hasattr(args, "__dict__") else {"args": str(args)}
        if extra:
            payload.update(extra)
        self.store.write_json(self.ctx.paths.args_json, payload)

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        ctx = self.ctx
        self._epoch_counter += 1
        row = self.format_epoch_row(raw_metrics=metrics, ctx=ctx)

        self._epoch_rows.append(row)
        self.on_epoch_end(ctx=ctx, epoch_row=row)

        policy = self.history_flush_policy(ctx)
        if policy == "every_epoch":
            self.flush_history()
        elif isinstance(policy, int) and policy > 0:
            if len(self._epoch_rows) >= policy:
                self.flush_history(append=True)

    def flush_history(self, *, append: bool = False) -> None:
        if not self._epoch_rows:
            return

        mode = "a" if append and self.store.exists(self.ctx.paths.history_csv) else "w"
        header = not (mode == "a")
        self.store.write_csv_rows(self.ctx.paths.history_csv, self._epoch_rows, mode=mode, header=header)
        self._epoch_rows.clear()

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = self.ctx
        self.flush_history(append=True)
        raw = summary or {}
        formatted = self.format_summary(raw_summary=raw, ctx=ctx)
        self.store.write_json(ctx.paths.summary_json, formatted)

        pol = self.index_policy(ctx)
        if pol != "none":
            row = self.index_row(summary=formatted, ctx=ctx)
            if pol == "append":
                self.store.write_csv_rows(
                    ctx.index_path,
                    [row],
                    mode="a" if self.store.exists(ctx.index_path) else "w",
                    header=not self.store.exists(ctx.index_path),
                )
            elif pol == "overwrite":
                self.store.write_csv_rows(ctx.index_path, [row], mode="w", header=True)

        self.on_finish(ctx=ctx, summary=formatted)

        return formatted

    def save_arrays(self, **arrays: Any) -> None:
        for k, v in arrays.items():
            self.store.save_npy(self.ctx.paths.file(f"{k}.npy"), v)

    def _make_paths(self, *, run_dir: Path) -> RunPaths:
        logs_dir = run_dir / "logs"
        fig_dir = run_dir / "fig"
        ckpt_dir = run_dir / "checkpoints"
        return RunPaths(
            run_dir=run_dir,
            logs_dir=logs_dir,
            fig_dir=fig_dir,
            ckpt_dir=ckpt_dir,
            args_json=run_dir / "args.json",
            history_csv=run_dir / "history.csv",
            summary_json=run_dir / "summary.json",
        )

    def build_run_tag(self, *, args: Any, dataset_stem: str) -> str:
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    def build_run_dir(self, *, args: Any, dataset_stem: str, run_tag: str) -> Path:
        return self.base_dir / f"{self.project_name}_{dataset_stem}" / run_tag

    def format_epoch_row(self, *, raw_metrics: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:
        return dict(raw_metrics)

    def format_summary(self, *, raw_summary: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:
        base = {
            "project": ctx.project_name,
            "dataset": ctx.dataset_stem,
            "run_tag": ctx.run_tag,
            "run_dir": str(ctx.paths.run_dir),
        }
        base.update(raw_summary)
        return base

    def index_row(self, *, summary: Dict[str, Any], ctx: RunContext) -> Dict[str, Any]:
        return dict(summary)

    def on_start(self, *, args: Any, ctx: RunContext) -> None:
        return

    def on_epoch_end(self, *, ctx: RunContext, epoch_row: Dict[str, Any]) -> None:
        return

    def on_finish(self, *, ctx: RunContext, summary: Dict[str, Any]) -> None:
        return

    def history_flush_policy(self, ctx: RunContext) -> str | int:
        return "end"

    def index_policy(self, ctx: RunContext) -> str:
        return "append"
