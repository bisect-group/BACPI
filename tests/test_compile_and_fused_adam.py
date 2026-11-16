"""torch.compile vs eager, fused Adam vs the authors' Adam, and hyperparameters vs code/main.py."""

import ast
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

BACPI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACPI))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulator_bench.modeling_v2 import BACPIModelConfig, build_model  # noqa: E402
from emulator_bench.train_single_target_tvt import forward_batch  # noqa: E402
from test_train_loop import make_batches  # noqa: E402

CUDA = pytest.mark.gpu
SIZES = [4, 3, 5, 2]  # different shapes per batch exercise dynamic=True


def make_model(dtype, device, dropout=0.1):
    torch.manual_seed(0)
    config = BACPIModelConfig(comp_dim=16, prot_dim=16, latent_dim=16, window=2, dropout=dropout)
    return build_model(20, 20, config).to(device=device, dtype=dtype)


def negative_control(model):
    """With EQUIV_NEGATIVE_CONTROL=1 (only the gpu tests run then) the candidate's initial weights are scaled by 1.05: the GPU test must then fail."""
    if os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1":
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(1.05)
    return model


def to_batches(dtype, device):
    floats = ("compound_mask", "fingerprint", "protein_mask", "targets")
    return [
        {k: v.to(device, dtype) if k in floats else v.to(device) for k, v in b.items()}
        for b in make_batches(SIZES)
    ]


def run(model, batch):
    model.zero_grad()
    loss = F.mse_loss(forward_batch(model, batch), batch["targets"])
    loss.backward()
    return loss.detach(), {
        n.removeprefix("_orig_mod."): p.grad.clone() for n, p in model.named_parameters()
    }


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def compare_compiled(dtype, device, rtol, train_mode):
    # dropout is random, so the train-mode check uses dropout 0 and keeps the training code path
    eager = make_model(dtype, device, dropout=0.0 if train_mode else 0.1).train(train_mode)
    model = make_model(dtype, device, dropout=0.0 if train_mode else 0.1).train(train_mode)
    compiled = torch.compile(negative_control(model), dynamic=True)
    for batch in to_batches(dtype, device):
        loss_e, grads_e = run(eager, batch)
        loss_c, grads_c = run(compiled, batch)
        assert rel(loss_c, loss_e) < rtol
        total = torch.cat([g.flatten() for g in grads_e.values()]).norm()
        for name, grad in grads_e.items():
            # tensors with a gradient near zero (softmax-shift biases) get a scale floor
            err = (grads_c[name] - grad).norm() / grad.norm().clamp(min=1e-6 * total)
            assert err < rtol, name


@pytest.mark.parametrize("train_mode", [False, True], ids=["eval", "train"])
def test_compiled_equals_eager_on_cpu_fp64(train_mode):
    # fp64: only reassociation noise is allowed
    compare_compiled(torch.float64, torch.device("cpu"), 1e-9, train_mode)


@CUDA
@pytest.mark.parametrize("train_mode", [False, True], ids=["eval", "train"])
def test_compiled_equals_eager_on_gpu_fp32(train_mode):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # fp32 kernel fusion changes summation order; 1e-4 is about 1000 ulps of relative norm error
    compare_compiled(torch.float32, torch.device("cuda"), 1e-4, train_mode)


def adam_trajectory(device, dtype, **kwargs):
    model = make_model(dtype, device, dropout=0.0).train()
    if kwargs:
        negative_control(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, weight_decay=0, amsgrad=True, **kwargs
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    losses = []
    for step, batch in enumerate(to_batches(dtype, device) * 2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(forward_batch(model, batch), batch["targets"])
        loss.backward()
        optimizer.step()
        if step % 2:
            scheduler.step()
        losses.append(float(loss))
    return (
        losses,
        torch.cat([p.detach().flatten() for p in model.parameters()]),
        optimizer.param_groups[0]["lr"],
    )


def check_fused(device, dtype, rtol):
    # reference is the authors' call: Adam(lr, weight_decay=0, amsgrad=True), default implementation
    losses_o, params_o, lr_o = adam_trajectory(device, dtype)
    losses_f, params_f, lr_f = adam_trajectory(device, dtype, fused=True)
    assert lr_f == lr_o
    assert torch.tensor(losses_f) == pytest.approx(torch.tensor(losses_o), rel=rtol)
    assert rel(params_f, params_o) < rtol


def test_fused_adam_equals_authors_adam_on_cpu_fp64():
    # same maths in fp64; only operation order differs
    check_fused(torch.device("cpu"), torch.float64, 1e-9)


@CUDA
def test_fused_adam_equals_authors_adam_on_gpu_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 8 steps at lr 5e-4 move parameters by about 4e-3; 1e-5 relative norm error is far below that
    check_fused(torch.device("cuda"), torch.float32, 1e-5)


def argparse_defaults(path):
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            name = node.args[0].value.lstrip("-")
            defaults.update(
                {
                    name: kw.value.value
                    for kw in node.keywords
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant)
                }
            )
    return defaults


def call_constants(path, function):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == function:
            return {
                kw.arg: kw.value.value
                for kw in node.keywords
                if isinstance(kw.value, ast.Constant)
            }, [a.value for a in node.args if isinstance(a, ast.Constant)]


def test_wrapper_hyperparameters_equal_original_defaults():
    original = argparse_defaults(BACPI / "code/main.py")
    wrapper = argparse_defaults(BACPI / "emulator_bench/train_single_target_tvt.py")
    names = [
        "lr", "step_size", "gamma", "batch_size", "num_epochs", "gat_dim", "num_head", "dropout",
        "alpha", "comp_dim", "prot_dim", "latent_dim", "window", "layer_cnn", "layer_out",
    ]  # fmt: skip
    for name in names:
        assert wrapper[name] == original[name], name
    config = BACPIModelConfig()
    for name in BACPIModelConfig.__dataclass_fields__:
        assert getattr(config, name) == original[name], name
    saved = json.loads((BACPI / "emulator_bench/default_hparams_original.json").read_text())[
        "defaults"
    ]
    assert saved == {n: original[n] for n in names}


def test_wrapper_optimizer_and_scheduler_equal_original():
    orig_kw, _ = call_constants(BACPI / "code/main.py", "Adam")
    wrap_kw, _ = call_constants(BACPI / "emulator_bench/train_single_target_tvt.py", "Adam")
    assert (orig_kw["weight_decay"], orig_kw["amsgrad"]) == (0, True)
    assert (wrap_kw["weight_decay"], wrap_kw["amsgrad"]) == (0, True)
    orig_sched, _ = call_constants(BACPI / "code/main.py", "StepLR")
    assert (orig_sched["step_size"], orig_sched["gamma"]) == (
        10,
        0.5,
    )  # the wrapper defaults are checked above


K = 10  # the production gap must stay within 10x the fp32 noise (1e-6 weight perturbation) gap


def noise_control_trajectory(variant, device, epochs=3):
    """Per-step training loss and final eval RMSE of the original path, its noise control or the production path."""
    from emulator_bench.train_single_target_tvt import evaluate_loader

    model = make_model(torch.float32, device, dropout=0.0)
    if variant == "production":
        negative_control(model)
    if variant == "noise":
        generator = torch.Generator(device).manual_seed(123)
        with torch.no_grad():
            for p in model.parameters():
                p += 1e-6 * p.abs() * torch.randn(p.shape, generator=generator, device=device)
    production = variant == "production"
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0, amsgrad=True, fused=production)
    run_model = torch.compile(model, dynamic=True) if production else model
    batches = to_batches(torch.float32, device)
    losses = []
    run_model.train()
    for _ in range(epochs):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(forward_batch(run_model, batch), batch["targets"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    _, _, metrics = evaluate_loader(run_model, batches, device)
    return torch.tensor(losses, dtype=torch.float64), metrics["rmse"]


@CUDA
def test_production_fp32_compiled_fused_within_noise_control():
    """The production path (fp32, torch.compile, fused Adam; no --amp) against the original eager fp32 path.

    Same model, batches and order for 3 epochs of 4 batches, torch default TF32 settings (cuDNN convolutions use
    TF32, matmuls do not) as in production, dropout 0. The fp32 noise control perturbs the initial weights by 1e-6
    (relative). The per-step loss gap and the final RMSE gap must stay within K times the noise gap.
    Step 0 (forward before any update) is judged like every other step.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    (ref, ref_rmse), (noise, noise_rmse), (prod, prod_rmse) = (
        noise_control_trajectory(v, device) for v in ("original", "noise", "production")
    )
    tiny = 1e-7 * ref.abs().max()
    gap, floor = (prod - ref).abs(), (noise - ref).abs()
    final_gap, final_floor = abs(prod_rmse - ref_rmse), abs(noise_rmse - ref_rmse)
    tiny0 = 1e-7 * ref[0].abs()
    gap0, floor0 = float(gap[0]), float(floor[0])
    print(
        f"bacpi production: max gap={gap.max():.3e}, max floor={floor.max():.3e}, "
        f"ratio={gap.max() / max(floor.max(), tiny):.2f}, K={K}, step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}, "
        f"final gap={final_gap:.3e}, final floor={final_floor:.3e}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap.max() <= K * max(floor.max(), tiny)
    assert final_gap <= K * max(final_floor, 1e-7 * abs(ref_rmse))
