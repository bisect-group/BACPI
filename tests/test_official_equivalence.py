"""BACPIv2 must equal the official BACPI model (code/model.py) in fp64: predictions, loss and gradients."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

BACPI = Path(__file__).resolve().parents[1]
REPO_ROOT = BACPI.parents[1]
EMBEDDINGS = REPO_ROOT / "outputs/baselines/BACPI/embeddings"
VAL_SPLIT = REPO_ROOT / "data/processed/datasets/kinetic_dataset/ki/random_splits_grouped_sequence/val.parquet"
# verify_equivalence.sh sets EMULATOR_SYNTHETIC_FIXTURE so every checkout runs the same data.
HAS_REAL_DATA = not os.environ.get("EMULATOR_SYNTHETIC_FIXTURE") and (EMBEDDINGS / "packed_v2/compounds.index.npz").exists() and VAL_SPLIT.exists()
# Without the private cache and split, the test builds a packed cache from these synthetic rows.
SYNTHETIC_SMILES = [
    "CCO", "c1ccccc1O", "CC(=O)O", "CCN", "C1CC1", "CC(C)C", "c1ccncc1", "CCOC(=O)C",
    "CC(=O)Nc1ccc(O)cc1", "OC1CCCCC1", "CCS", "c1ccc2ccccc2c1", "CN(C)C=O", "ClCCl", "CC#N", "OCC(O)CO",
]  # fmt: skip
SYNTHETIC_SEQUENCES = ["MKTAYIAKQRQISFVKSHFSRQ", "GGSSLLAAKVEE", "MKVLAAGGHHPLE", "AAAAKKKLLMMTTYYWW"] * 4
BATCH_SIZE, NUM_BATCHES, MAX_SEQUENCE_LENGTH = 4, 4, 350
PRED_RTOL, LOSS_RTOL, GRAD_RTOL = 1e-10, 1e-10, 1e-9


def synthetic_cache(root):
    from emulator_bench.cache_embeddings import build_cache
    from emulator_bench.common import discover_split_jobs, task_split_dir

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"smiles": SYNTHETIC_SMILES, "sequence": SYNTHETIC_SEQUENCES, "log10_value": rng.normal(size=16)}
    )
    directory = task_split_dir("ki", root / "data") / "random_splits_grouped_sequence"
    directory.mkdir(parents=True)
    for part in ("train", "val", "test"):
        frame.to_parquet(directory / f"{part}.parquet")
    jobs = discover_split_jobs(task_split_dir("ki", root / "data"), ["random_splits_grouped_sequence"])
    build_cache(jobs, root / "embeddings", datasets_root=root / "data", workers=2)
    return root / "embeddings", directory / "val.parquet"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    sys.path[:0] = [str(BACPI), str(BACPI / "code")]
    from emulator_bench.dataset import BACPIDataset, bacpi_collate_fn
    from emulator_bench.feature_pipeline import (
        BACPIFeaturizer,
        PackedCache,
        compound_cache_key,
        featurizer_path,
        protein_cache_key,
    )
    from emulator_bench.modeling_v2 import BACPIModelConfig, build_model
    from model import BACPI as OfficialBACPI

    embeddings, val_split = (EMBEDDINGS, VAL_SPLIT) if HAS_REAL_DATA else synthetic_cache(tmp_path_factory.mktemp("bacpi"))
    compounds, proteins = PackedCache(embeddings, "compounds"), PackedCache(embeddings, "proteins")
    frame = pd.read_parquet(val_split).iloc[:3000]
    keep = (
        frame["sequence"].str.len().le(MAX_SEQUENCE_LENGTH)
        & frame["smiles"].map(lambda s: compound_cache_key(s) in compounds.index)
        & frame["sequence"].map(lambda s: protein_cache_key(s) in proteins.index)
    )
    frame = frame[keep].iloc[: BATCH_SIZE * NUM_BATCHES]
    assert len(frame) == BATCH_SIZE * NUM_BATCHES
    dataset = BACPIDataset(frame, compounds, proteins, "smiles", "sequence", "log10_value")
    batches = [
        bacpi_collate_fn([dataset[j] for j in range(i, i + BATCH_SIZE)])
        for i in range(0, len(dataset), BATCH_SIZE)
    ]
    featurizer = BACPIFeaturizer.load(featurizer_path(embeddings))

    config = BACPIModelConfig()  # the training defaults, which are the official ones
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        official = OfficialBACPI("affinity", featurizer.n_atom_tokens, featurizer.n_amino_tokens, config)
        ours = build_model(featurizer.n_atom_tokens, featurizer.n_amino_tokens, config)
        state = {}
        for key, value in official.state_dict().items():
            if key.startswith("gat_layer_"):  # gat_layer_N.* -> gat_layers.N.*
                key = key.replace("gat_layer_", "gat_layers.", 1)
            if key.endswith(".a"):  # the (2D, 1) attention vector is split into its two halves
                half = value.shape[0] // 2
                state[key + "_left"], state[key + "_right"] = value[:half].clone(), value[half:].clone()
            else:
                state[key] = value
        ours.load_state_dict(state, strict=True)
        yield official, ours, batches
    finally:
        torch.set_default_dtype(previous)


def _run(model, batch, seed):
    """Forward and backward with the training loss; returns predictions, loss and gradients."""
    model.zero_grad()
    torch.manual_seed(seed)
    pred = model(
        batch["compounds"],
        batch["compound_mask"].double(),
        batch["adjacencies"],
        batch["proteins"],
        batch["protein_mask"].double(),
        batch["fingerprint"].double(),
    )
    loss = F.mse_loss(pred, batch["targets"].double())
    loss.backward()
    grads = {}
    for name, p in model.named_parameters():
        name = name.replace("gat_layer_", "gat_layers.", 1)
        if name.endswith(".a"):  # official attention vector -> the halves our model holds
            grads[name + "_left"], grads[name + "_right"] = p.grad.chunk(2)
        else:
            grads[name] = p.grad
    return pred.detach(), loss.detach(), {n: g.clone() for n, g in grads.items()}


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


@pytest.mark.parametrize("train_mode", [False, True], ids=["eval", "train_dropout"])
def test_matches_official(setup, train_mode, capsys):
    official, ours, batches = setup
    official.train(train_mode)
    ours.train(train_mode)
    worst = {"pred": 0.0, "loss": 0.0, "grad": 0.0}
    worst_grad = ""
    for i, batch in enumerate(batches):
        p_o, l_o, g_o = _run(official, batch, 100 + i)
        p_f, l_f, g_f = _run(ours, batch, 100 + i)
        assert set(g_o) == set(g_f)
        worst["pred"] = max(worst["pred"], _rel(p_f, p_o))
        worst["loss"] = max(worst["loss"], _rel(l_f, l_o))
        total = torch.cat([g.flatten() for g in g_o.values()]).norm()
        for n in g_o:
            # Tensors with a gradient near zero get a scale floor.
            err = float((g_f[n] - g_o[n]).norm() / g_o[n].norm().clamp(min=1e-6 * total))
            if err > worst["grad"]:
                worst["grad"], worst_grad = err, n
    with capsys.disabled():
        print(f"\n[{'train' if train_mode else 'eval'}] max relative diffs: {worst} (worst grad: {worst_grad})")
    assert worst["pred"] < PRED_RTOL
    assert worst["loss"] < LOSS_RTOL
    assert worst["grad"] < GRAD_RTOL, worst_grad
