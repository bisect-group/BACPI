import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.modeling_v2 import BACPIModelConfig, build_model
from emulator_bench.train_single_target_tvt import evaluate_loader, forward_batch, train_one_epoch

CPU = torch.device("cpu")


def make_batches(sizes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for n in sizes:
        atoms, residues = 7 + n, 12 + n
        adjacency = torch.rand(n, atoms, atoms, generator=generator) > 0.5
        batches.append(
            {
                "compounds": torch.randint(1, 20, (n, atoms), generator=generator),
                "compound_mask": torch.ones(n, atoms),
                "adjacencies": adjacency | torch.eye(atoms, dtype=torch.bool),
                "fingerprint": torch.rand(n, 1024, generator=generator),
                "proteins": torch.randint(1, 20, (n, residues), generator=generator),
                "protein_mask": torch.ones(n, residues),
                "targets": torch.randn(n, 1, generator=generator),
            }
        )
    return batches


def make_model():
    torch.manual_seed(0)
    return build_model(20, 20, BACPIModelConfig(comp_dim=16, prot_dim=16, latent_dim=16, window=2))


def test_train_loss_is_sample_weighted_mean_of_step_losses():
    batches = make_batches([4, 3, 2])
    model = make_model()
    reference = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-2)
    total = 0.0
    for batch in batches:
        reference_optimizer.zero_grad()
        loss = F.mse_loss(forward_batch(reference, batch), batch["targets"])
        loss.backward()
        reference_optimizer.step()
        total += loss.item() * batch["targets"].shape[0]
    mean_loss = train_one_epoch(model, batches, optimizer, CPU)
    assert mean_loss == pytest.approx(total / 9, rel=1e-6)


def test_eval_output_equals_per_batch_path():
    batches = make_batches([4, 3, 2], seed=1)
    model = make_model().eval()
    with torch.inference_mode():
        expected = torch.cat([forward_batch(model, b) for b in batches]).reshape(-1).numpy()
    y_true, y_pred, _ = evaluate_loader(model, batches, CPU)
    assert (y_pred == expected).all()
    assert (y_true == torch.cat([b["targets"] for b in batches]).reshape(-1).numpy()).all()
