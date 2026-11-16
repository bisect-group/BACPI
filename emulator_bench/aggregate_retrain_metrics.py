import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_OUT_ROOT, TASK_PARENTS

GROUP_COLUMNS = ["task", "split_group", "split_name"]


def collect_runs(out_root: Path, tasks) -> pd.DataFrame:
    rows = []
    for task in tasks:
        for metric_path in sorted((out_root / task).glob("*/*/seed_*/final_results_test.csv")):
            run_dir = metric_path.parent
            split_group, split_name, seed_name = run_dir.parts[-3:]
            summary = json.loads((run_dir / "run_summary.json").read_text())
            row = {
                "task": task,
                "split_group": split_group,
                "split_name": split_name,
                "seed": int(seed_name[len("seed_"):]),
                "best_epoch": summary["best_epoch"],
                **{"%s_size" % name: size for name, size in summary["split_sizes"].items()},
            }
            row.update(pd.read_csv(metric_path).iloc[0].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate BACPI test metrics over seeds (mean and std).")
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_PARENTS), choices=list(TASK_PARENTS))
    args = parser.parse_args()

    runs = collect_runs(args.out_root, args.tasks)
    if runs.empty:
        raise FileNotFoundError("No finished runs under %s" % args.out_root)
    numeric = [c for c in runs.columns if c not in GROUP_COLUMNS + ["seed"]]
    summary = runs.groupby(GROUP_COLUMNS)[numeric].agg(["mean", "std"])
    summary.columns = ["%s_%s" % pair for pair in summary.columns]
    runs.to_csv(args.out_root / "summary_runs.csv", index=False)
    summary.reset_index().to_csv(args.out_root / "summary.csv", index=False)
    print("Wrote %s runs to %s" % (len(runs), args.out_root), flush=True)


if __name__ == "__main__":
    main()
