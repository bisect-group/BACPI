"""torch.compile vs eager, fused Adam vs the authors' Adam, and hyperparameters vs code/main.py."""

import ast
import json
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

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
SIZES = [4, 3, 5, 2]  # different shapes per batch exercise dynamic=True


def make_model(dtype, device, dropout=0.1):
    torch.manual_seed(0)
    config = BACPIModelConfig(comp_dim=16, prot_dim=16, latent_dim=16, window=2, dropout=dropout)
    return build_model(20, 20, config).to(device=device, dtype=dtype)


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
    compiled = torch.compile(model, dynamic=True)
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
