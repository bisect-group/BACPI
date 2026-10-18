import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_OUT_ROOT,
    DEFAULT_SPLIT_GROUPS,
    TASK_PARENTS,
    discover_split_jobs,
    read_table,
    require_columns,
    task_split_dir,
)
from emulator_bench.feature_pipeline import (
    BACPIFeaturizer,
    compound_cache_path,
    filter_invalid_smiles_rows,
    protein_cache_path,
    save_npz_atomic,
)


def collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    """Unique SMILES and sequences over all split files. Rows with invalid SMILES are skipped in memory only."""
    sequences, smiles_values = set(), set()
    for job in jobs:
        for key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[key]))
            require_columns(frame, [sequence_col, smiles_col], Path(job[key]))
            frame, _ = filter_invalid_smiles_rows(frame, smiles_col=smiles_col, source_name=job[key])
            sequences.update(frame[sequence_col].astype(str))
            smiles_values.update(frame[smiles_col].astype(str))
    return sorted(smiles_values), sorted(sequences)


def build_cache(jobs, embeddings_dir: Path, sequence_col: str = "sequence", smiles_col: str = "smiles", datasets_root=None) -> None:
    smiles_values, sequences = collect_unique_values(jobs, sequence_col, smiles_col)
    print("Split jobs: %s | unique compounds: %s | unique proteins: %s" % (len(jobs), len(smiles_values), len(sequences)), flush=True)

    featurizer_path = embeddings_dir / "vocabs" / "featurizer.pkl"
    if featurizer_path.exists():
        featurizer = BACPIFeaturizer.load(featurizer_path)
    else:
        # The vocabulary covers every task and split group on disk, whatever jobs are cached now.
        vocab_jobs = [
            job
            for task in TASK_PARENTS
            for job in discover_split_jobs(task_split_dir(task, datasets_root), DEFAULT_SPLIT_GROUPS)
        ]
        vocab_smiles, vocab_sequences = collect_unique_values(vocab_jobs, sequence_col, smiles_col)
        featurizer = BACPIFeaturizer()
        featurizer.build_from_values(vocab_smiles, vocab_sequences)
        featurizer.save(featurizer_path)

    for values, path_fn, encode in (
        (smiles_values, compound_cache_path, featurizer.encode_compound),
        (sequences, protein_cache_path, featurizer.encode_protein),
    ):
        pending = [value for value in values if not path_fn(embeddings_dir, value).exists()]
        print("Missing cache entries: %s of %s" % (len(pending), len(values)), flush=True)
        for value in pending:
            save_npz_atomic(path_fn(embeddings_dir, value), encode(value))



def main():
    parser = argparse.ArgumentParser(description="Cache BACPI compound and protein features for the EMULaToR splits.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_PARENTS), choices=list(TASK_PARENTS))
    parser.add_argument("--datasets_root", type=str, default=None)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--embeddings_dir", type=Path, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    args = parser.parse_args()

    jobs = [
        job
        for task in args.tasks
        for job in discover_split_jobs(task_split_dir(task, args.datasets_root), args.split_groups)
    ]
    if not jobs:
        raise FileNotFoundError("No split jobs found for tasks %s" % args.tasks)
    build_cache(jobs, args.embeddings_dir or args.out_root / "embeddings", args.sequence_col, args.smiles_col, args.datasets_root)


if __name__ == "__main__":
    main()
