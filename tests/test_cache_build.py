import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.cache_embeddings import build_cache
from emulator_bench.common import discover_split_jobs, task_split_dir
from emulator_bench.feature_pipeline import (
    BACPIFeaturizer,
    PackedCache,
    build_vocabulary,
    compound_cache_key,
    featurizer_path,
    normalize_sequence,
    packed_dir,
    protein_cache_key,
)

SMILES = ["CCO", "c1ccccc1O", "C", "CC(=O)O", "[Na+].[Cl-]", "N#N", "C1CC1", " CCN ", "not_a_smiles"]
SEQUENCES = ["MKTAYIAK", "mktayiak", "GGSSLLAA", "MKVLAAGG", "AAAAAAAA"]
GROUP = "random_splits_grouped_sequence"


def write_split(root: Path, rows, group: str = GROUP):
    directory = task_split_dir("ki", root) / group
    directory.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        pd.DataFrame(rows).to_parquet(directory / ("%s.parquet" % part))


def store_files(embeddings_dir: Path):
    return {path.name: path.read_bytes() for path in sorted(packed_dir(embeddings_dir).iterdir())}


def test_parallel_vocabulary_and_entries_equal_serial_build():
    serial = BACPIFeaturizer()
    valid = sorted({value.strip() for value in SMILES if value != "not_a_smiles"})
    expected = {smiles: serial.encode_compound(smiles, build=True) for smiles in valid}
    for sequence in sorted({normalize_sequence(value) for value in SEQUENCES}):
        serial.encode_protein(sequence, build=True)

    parallel = BACPIFeaturizer()
    entries = build_vocabulary(parallel, SMILES, SEQUENCES, valid, workers=2)
    assert parallel.to_payload() == serial.to_payload()
    assert list(parallel.fingerprint_dict) == list(serial.fingerprint_dict)
    assert sorted(entries) == valid
    for smiles in valid:
        for got, want in zip(entries[smiles], expected[smiles].values()):
            assert got.dtype == want.dtype and np.array_equal(got, want)


def test_incremental_build_equals_one_shot_build(tmp_path):
    first = [{"smiles": s, "sequence": q} for s, q in zip(SMILES[:5], SEQUENCES)]
    second = [{"smiles": s, "sequence": q} for s, q in zip(SMILES[3:], SEQUENCES[::-1])]
    root = tmp_path / "data"
    write_split(root, first)
    write_split(root, second, "substrate_splits")

    def jobs(*groups):
        return discover_split_jobs(task_split_dir("ki", root), groups)

    steps, one_shot = tmp_path / "steps", tmp_path / "one"
    build_cache(jobs(GROUP), steps, datasets_root=root, workers=2)
    build_cache(jobs("substrate_splits"), steps, datasets_root=root, workers=2)
    build_cache(jobs(GROUP, "substrate_splits"), one_shot, datasets_root=root, workers=2)

    assert BACPIFeaturizer.load(featurizer_path(steps)).to_payload() == BACPIFeaturizer.load(featurizer_path(one_shot)).to_payload()
    assert store_files(steps) == store_files(one_shot)
    compounds, proteins = PackedCache(steps, "compounds"), PackedCache(steps, "proteins")
    assert compound_cache_key("not_a_smiles") not in compounds.index
    assert compounds.compound(compounds.index[compound_cache_key("CCN")])["compounds"].shape[0] == 10
    assert proteins.tokens(proteins.index[protein_cache_key("mktayiak")]).shape[0] == len("MKTAYIAK")
    assert not packed_dir(steps).with_name("packed_v2.tmp").exists()
