from __future__ import annotations

import json
import numpy as np
import pandas as pd
import time

from pathlib import Path
from typing import Any, Dict, List
from base_tracker import ArtifactStore, BaseTracker, RunContext


class LocalFSArtifactStore(ArtifactStore):
    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        self.ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def write_text(self, path: Path, text: str) -> None:
        self.ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.ensure_dir(path.parent)
        with open(path, "wb") as f:
            f.write(data)

    def write_csv_rows(self, path: Path, rows: List[Dict[str, Any]], *, mode: str, header: bool) -> None:
        self.ensure_dir(path.parent)
        df = pd.DataFrame(rows)
        df.to_csv(path, mode=mode, header=header, index=False)

    def save_npy(self, path: Path, array: Any) -> None:
        self.ensure_dir(path.parent)
        np.save(path, array)

    def exists(self, path: Path) -> bool:
        return path.exists()


class LocalTracker(BaseTracker):
    def __init__(
        self,
        *,
        base_dir: str | Path = "runs",
        index_name: str = "results_index.csv",
        project_name: str = "project",
        store: ArtifactStore | None = None,
    ):
        super().__init__(
            store=store or LocalFSArtifactStore(),
            base_dir=base_dir,
            index_name=index_name,
            project_name=project_name,
        )

    def build_run_tag(self, *, args: Any, dataset_stem: str) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        parts = [ts]
        for key in ("seed", "suffix", "freeze", "fusion_mode", "structure"):
            v = getattr(args, key, None)
            if v is None or v == "" or v is False:
                continue
            parts.append(f"{key}-{v}")

        return "_".join(parts)

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
        row = dict(summary)
        return row
