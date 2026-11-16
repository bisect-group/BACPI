import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench.cache_embeddings import build_cache
from emulator_bench.common import discover_split_jobs, task_split_dir

SMILES = ["CCO", "c1ccccc1O", "CC(=O)O", "C1CC1", "CCN", "CCCl", "CC(C)O", "c1ccncc1"]
SEQUENCES = ["MKTAYIAK", "GGSSLLAA", "MKVLAAGG", "AAKLLVVG", "MMKKTTAA"]
GROUP = "random_splits_grouped_sequence"


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    rng = np.random.default_rng(0)
    splits = root / "splits"
    splits.mkdir()
    for part, n in (("train", 41), ("val", 12), ("test", 12)):
        frame = pd.DataFrame(
            {
                "smiles": rng.choice(SMILES, n),
                "sequence": rng.choice(SEQUENCES, n),
                "log10_value": rng.normal(size=n),
            }
        )
        frame.to_parquet(splits / ("%s.parquet" % part))
    data = root / "data"
    directory = task_split_dir("ki", data) / GROUP
    directory.mkdir(parents=True)
    for part in ("train", "val", "test"):
        (directory / ("%s.parquet" % part)).write_bytes((splits / ("%s.parquet" % part)).read_bytes())
    embeddings = root / "embeddings"
    build_cache(discover_split_jobs(task_split_dir("ki", data), [GROUP]), embeddings, datasets_root=data, workers=2)
    return splits, embeddings


def train(fixture, out_dir, epochs, num_workers, resume=False):
    splits, embeddings = fixture
    command = [sys.executable, str(ROOT / "emulator_bench/train_single_target_tvt.py")]
    for part in ("train", "val", "test"):
        command += ["--%s_path" % part, str(splits / ("%s.parquet" % part))]
    command += [
        "--embeddings_dir", str(embeddings), "--out_dir", str(out_dir), "--device", "cpu", "--no-compile",
        "--num_epochs", str(epochs), "--batch_size", "8", "--num_workers", str(num_workers), "--step_size", "1",
        "--comp_dim", "16", "--prot_dim", "16", "--latent_dim", "16", "--window", "2",
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}, capture_output=True)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resumed_run_equals_uninterrupted_run(fixture, tmp_path, num_workers):
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    train(fixture, straight, 3, num_workers)
    train(fixture, resumed, 1, num_workers)
    train(fixture, resumed, 3, num_workers, resume=True)

    columns = ["epoch", "lr", "train_loss", "val_rmse"]
    pd.testing.assert_frame_equal(pd.read_csv(straight / "logfile.csv")[columns], pd.read_csv(resumed / "logfile.csv")[columns], check_exact=True)
    for name in ("pred_label_val.csv", "pred_label_test.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(straight / name), pd.read_csv(resumed / name), check_exact=True)
    a, b = (torch.load(d / "checkpoint_last.pt", map_location="cpu", weights_only=False) for d in (straight, resumed))
    assert a["epoch"] == b["epoch"] == 3 and a["best_epoch"] == b["best_epoch"] and a["best_rmse"] == b["best_rmse"]
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key])
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert torch.equal(a["train_generator_state"], b["train_generator_state"])
