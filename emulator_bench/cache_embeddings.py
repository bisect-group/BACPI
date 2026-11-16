import argparse
import shutil
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
    task_split_dir,
)
from emulator_bench.feature_pipeline import (
    BACPIFeaturizer,
    PackedCache,
    encode_compounds_chunk,
    encode_proteins_chunk,
    build_vocabulary,
    compound_cache_key,
    compound_entry,
    featurizer_path,
    normalize_sequence,
    normalize_smiles_text,
    packed_dir,
    parallel_map,
    protein_cache_key,
    write_packed,
)

DEFAULT_WORKERS = 64


def collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    """Unique SMILES and sequences over all split files. Invalid SMILES are kept and dropped when encoded."""
    sequences, smiles_values = set(), set()
    for job in jobs:
        for key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[key]), [sequence_col, smiles_col])
            sequences.update(frame[sequence_col].astype(str))
            smiles_values.update(frame[smiles_col].astype(str))
    return sorted(smiles_values), sorted(sequences)


def build_cache(
    jobs,
    embeddings_dir: Path,
    sequence_col: str = "sequence",
    smiles_col: str = "smiles",
    datasets_root=None,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Encode the compounds and proteins of `jobs` once, in parallel, into the packed store.

    Without a saved vocabulary, the vocabulary and a new store are built together.
    Otherwise only entries missing from the store are encoded with the frozen vocabulary.
    """
    smiles_values, sequences = collect_unique_values(jobs, sequence_col, smiles_col)
    smiles_values = sorted({normalize_smiles_text(value) for value in smiles_values})
    sequences = sorted({normalize_sequence(value) for value in sequences})
    print("Split jobs: %s | unique compounds: %s | unique proteins: %s" % (len(jobs), len(smiles_values), len(sequences)), flush=True)

    vocab_path = featurizer_path(embeddings_dir)
    compounds, proteins = {}, {}
    if vocab_path.exists():
        featurizer = BACPIFeaturizer.load(vocab_path)
        stored = {kind: set(PackedCache(embeddings_dir, kind).index) if (packed_dir(embeddings_dir) / ("%s.index.npz" % kind)).exists() else set() for kind in ("compounds", "proteins")}
        pending_smiles = [value for value in smiles_values if compound_cache_key(value) not in stored["compounds"]]
    else:
        # The vocabulary covers every task and split group on disk, whatever jobs are cached now.
        vocab_jobs = [
            job
            for task in TASK_PARENTS
            for job in discover_split_jobs(task_split_dir(task, datasets_root), DEFAULT_SPLIT_GROUPS)
        ]
        vocab_smiles, vocab_sequences = collect_unique_values(vocab_jobs, sequence_col, smiles_col)
        featurizer = BACPIFeaturizer()
        compounds = build_vocabulary(featurizer, vocab_smiles, vocab_sequences, smiles_values, workers)
        featurizer.save(vocab_path)
        stored = {"compounds": set(), "proteins": set()}
        shutil.rmtree(packed_dir(embeddings_dir), ignore_errors=True)
        pending_smiles = [value for value in smiles_values if value not in compounds]
    pending_sequences = [value for value in sequences if protein_cache_key(value) not in stored["proteins"]]
    print("Missing cache entries: compounds %s, proteins %s" % (len(compounds) + len(pending_smiles), len(pending_sequences)), flush=True)

    for smiles, encoded in zip(pending_smiles, parallel_map(encode_compounds_chunk, pending_smiles, featurizer, workers, "Encoding compounds")):
        if encoded is not None:
            compounds[smiles] = compound_entry(encoded)
    for sequence, tokens in zip(pending_sequences, parallel_map(encode_proteins_chunk, pending_sequences, featurizer, workers, "Encoding proteins")):
        proteins[sequence] = (tokens,)
    if compounds or proteins or not (packed_dir(embeddings_dir) / "compounds.index.npz").exists():
        write_packed(
            embeddings_dir,
            {
                "compounds": {compound_cache_key(smiles): arrays for smiles, arrays in compounds.items()},
                "proteins": {protein_cache_key(sequence): arrays for sequence, arrays in proteins.items()},
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Cache BACPI compound and protein features for the EMULaToR splits.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_PARENTS), choices=list(TASK_PARENTS))
    parser.add_argument("--datasets_root", type=str, default=None)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--embeddings_dir", type=Path, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    jobs = [
        job
        for task in args.tasks
        for job in discover_split_jobs(task_split_dir(task, args.datasets_root), args.split_groups)
    ]
    if not jobs:
        raise FileNotFoundError("No split jobs found for tasks %s" % args.tasks)
    build_cache(jobs, args.embeddings_dir or args.out_root / "embeddings", args.sequence_col, args.smiles_col, args.datasets_root, args.workers)


if __name__ == "__main__":
    main()
