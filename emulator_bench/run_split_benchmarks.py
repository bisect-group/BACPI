import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.cache_embeddings import build_cache
from emulator_bench.common import (
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    TASK_PARENTS,
    discover_split_jobs,
    task_split_dir,
)

TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def main():
    parser = argparse.ArgumentParser(description="Train BACPI with its original settings for every seed on every default split.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_PARENTS), choices=list(TASK_PARENTS))
    parser.add_argument("--datasets_root", type=str, default=None)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--embeddings_dir", type=Path, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir or args.out_root / "embeddings"
    task_jobs = {
        task: discover_split_jobs(task_split_dir(task, args.datasets_root), args.split_groups, args.thresholds)
        for task in args.tasks
    }
    all_jobs = [job for jobs in task_jobs.values() for job in jobs]
    if not all_jobs:
        raise FileNotFoundError("No split jobs found for tasks %s" % args.tasks)
    build_cache(all_jobs, embeddings_dir, datasets_root=args.datasets_root)

    for task, jobs in task_jobs.items():
        for job in jobs:
            for seed in args.seeds:
                run_dir = args.out_root / task / job["split_group"] / job["split_name"] / ("seed_%s" % seed)
                if (run_dir / "final_results_test.csv").exists() and not args.overwrite:
                    continue
                cmd = [
                    sys.executable, str(TRAIN_SCRIPT),
                    "--train_path", job["train_path"],
                    "--val_path", job["val_path"],
                    "--test_path", job["test_path"],
                    "--embeddings_dir", str(embeddings_dir),
                    "--out_dir", str(run_dir),
                    "--task_name", "%s_%s_%s_seed%s" % (task, job["split_group"], job["split_name"], seed),
                    "--num_epochs", str(args.num_epochs),
                    "--device", args.device,
                    "--num_workers", str(args.num_workers),
                    "--seed", str(seed),
                ]
                if args.amp:
                    cmd.append("--amp")
                if args.skip_smiles_validity_check:
                    cmd.append("--skip_smiles_validity_check")
                print("Running %s" % " ".join(cmd[-14:]), flush=True)
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
