import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT.parents[1]
DATASETS_ROOT = EMULATOR_ROOT / "data" / "processed" / "datasets"
DEFAULT_OUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "BACPI"
TASK_PARENTS = {
    "ki": "kinetic_dataset",
    "kd": "binding_affinity_dataset",
    "ec50": "binding_affinity_dataset",
    "ic50": "binding_affinity_dataset",
}
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
DEFAULT_SEEDS = [666, 777, 888]


def task_split_dir(task: str, datasets_root: Optional[str] = None) -> Path:
    if task not in TASK_PARENTS:
        raise ValueError("Unsupported task `%s`. Expected one of %s." % (task, ", ".join(TASK_PARENTS)))
    return Path(datasets_root or DATASETS_ROOT).expanduser().resolve() / TASK_PARENTS[task] / task


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported table format: %s" % path)


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns %s in %s" % (missing, path))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {name: float("nan") for name in ("rmse", "mae", "r2_score", "pearson", "spearman")}

    residual = y_true - y_pred
    total = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / total if total > 0 else 0.0
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = spearman = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman = float(stats.spearmanr(y_true, y_pred).statistic)
    return {
        "rmse": round(float(np.sqrt(np.mean(np.square(residual)))), 6),
        "mae": round(float(np.mean(np.abs(residual))), 6),
        "r2_score": round(r2, 6),
        "pearson": round(pearson, 6),
        "spearman": round(spearman, 6),
    }


def discover_split_jobs(
    split_dir: Path,
    split_groups: Iterable[str] = DEFAULT_SPLIT_GROUPS,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Find train/val/test folders: `<group>/` directly or every `<group>/threshold_*/` on disk."""
    jobs: List[Dict[str, str]] = []
    for split_group in split_groups:
        group_dir = Path(split_dir) / split_group
        if (group_dir / "train.parquet").exists():
            candidates = [(group_dir, "all")]
        else:
            candidates = [
                (child, child.name)
                for child in sorted(group_dir.glob("threshold_*"))
                if child.is_dir() and (thresholds is None or child.name in thresholds)
            ]
        for root, split_name in candidates:
            if all((root / ("%s.parquet" % part)).exists() for part in ("train", "val", "test")):
                jobs.append(
                    {
                        "split_group": split_group,
                        "split_name": split_name,
                        "train_path": str(root / "train.parquet"),
                        "val_path": str(root / "val.parquet"),
                        "test_path": str(root / "test.parquet"),
                    }
                )
    return jobs
