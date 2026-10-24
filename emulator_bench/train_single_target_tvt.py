import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import append_csv_row, read_table, regression_metrics, require_columns, resolve_amp_dtype, save_json, set_seed
from emulator_bench.dataset import BACPIDataset, create_loader
from emulator_bench.feature_pipeline import BACPIFeaturizer, PackedCache, compound_cache_key, featurizer_path, filter_invalid_smiles_rows, protein_cache_key
from emulator_bench.modeling_v2 import BACPIModelConfig, build_model


def autocast_context(device: torch.device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def move_batch_to_device(batch, device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_batch(model, batch):
    return model(
        batch["compounds"],
        batch["compound_mask"],
        batch["adjacencies"],
        batch["proteins"],
        batch["protein_mask"],
        batch["fingerprint"],
    )


def evaluate_loader(model, loader, device, amp_dtype=None):
    model.eval()
    preds, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, amp_dtype):
                prediction = forward_batch(model, batch)
            preds.append(prediction.float())
            truths.append(batch["targets"].float())
    y_pred = torch.cat(preds).cpu().numpy().reshape(-1)
    y_true = torch.cat(truths).cpu().numpy().reshape(-1)
    return y_true, y_pred, regression_metrics(y_true, y_pred)


def train_one_epoch(model, loader, optimizer, device, amp_dtype=None, scaler=None):
    model.train()
    total_loss, total_samples = torch.zeros((), dtype=torch.float64, device=device), 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            prediction = forward_batch(model, batch)
        loss = F.mse_loss(prediction.float(), batch["targets"].float())
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        n = int(batch["targets"].shape[0])
        total_loss += loss.detach().double() * n
        total_samples += n
    return total_loss.item() / max(1, total_samples)


def ensure_split_cache(compounds: PackedCache, proteins: PackedCache, frame: pd.DataFrame, smiles_col: str, sequence_col: str):
    for values, packed, key_fn, kind in (
        (frame[smiles_col].astype(str).unique(), compounds, compound_cache_key, "compound"),
        (frame[sequence_col].astype(str).unique(), proteins, protein_cache_key, "protein"),
    ):
        missing = [value for value in values if key_fn(value) not in packed.index]
        if missing:
            raise FileNotFoundError(
                "%s %s caches are missing from the packed cache. Run emulator_bench/cache_embeddings.py first."
                % (len(missing), kind)
            )


def prepare_split_frame(split_path: Path, args, compounds: PackedCache):
    frame = read_table(split_path)
    require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], split_path)
    rows_dropped = 0
    if not args.skip_smiles_validity_check:
        frame, stats = filter_invalid_smiles_rows(frame, smiles_col=args.smiles_col, source_name=str(split_path), packed=compounds)
        rows_dropped = stats["rows_dropped"]
        if rows_dropped:
            print("Dropped %s rows with invalid SMILES from %s (file unchanged)" % (rows_dropped, split_path), flush=True)
    if frame.empty:
        raise ValueError("No valid rows in %s." % split_path)
    return frame.reset_index(drop=True), rows_dropped


def main():
    parser = argparse.ArgumentParser(description="Train BACPI with its original settings on one explicit train/val/test split.")
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, required=True)
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--task_name", type=str, default="bacpi")

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    # Original defaults from code/main.py. Adam(amsgrad=True, weight_decay=0), MSE loss, StepLR.
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)

    parser.add_argument("--gat_dim", type=int, default=50)
    parser.add_argument("--num_head", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--comp_dim", type=int, default=80)
    parser.add_argument("--prot_dim", type=int, default=80)
    parser.add_argument("--latent_dim", type=int, default=80)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--layer_cnn", type=int, default=3)
    parser.add_argument("--layer_out", type=int, default=3)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Mixed precision: bf16 on Ampere or newer, fp16 on older GPUs. The original code trains in fp32")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile the model with dynamic shapes. Same maths, fewer kernel launches")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype, precision = resolve_amp_dtype(device) if args.amp else (None, "fp32")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    compounds, proteins = PackedCache(args.embeddings_dir, "compounds"), PackedCache(args.embeddings_dir, "proteins")
    prepared = {name: prepare_split_frame(getattr(args, "%s_path" % name), args, compounds) for name in ("train", "val", "test")}
    frames = {name: frame for name, (frame, _) in prepared.items()}
    rows_dropped_invalid_smiles = {name: dropped for name, (_, dropped) in prepared.items()}
    for frame in frames.values():
        ensure_split_cache(compounds, proteins, frame, args.smiles_col, args.sequence_col)
    featurizer = BACPIFeaturizer.load(featurizer_path(args.embeddings_dir))

    pin_memory = device.type == "cuda"
    loaders = {
        name: create_loader(
            BACPIDataset(frame, compounds, proteins, args.smiles_col, args.sequence_col, args.target_col),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers if name == "train" else min(2, args.num_workers),
            pin_memory=pin_memory,
            seed=args.seed,
        )
        for name, frame in frames.items()
    }
    sizes = {name: len(frame) for name, frame in frames.items()}
    print("Split sizes: %s" % sizes, flush=True)

    config = BACPIModelConfig(
        **{key: getattr(args, key) for key in BACPIModelConfig.__dataclass_fields__}
    )
    model = build_model(featurizer.n_atom_tokens, featurizer.n_amino_tokens, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    run_model = torch.compile(model, dynamic=True) if args.compile else model

    started = time.time()
    best_rmse, best_epoch = float("inf"), 0
    best_path = args.out_dir / "bestmodel.pth"
    log_path = args.out_dir / "logfile.csv"
    log_path.unlink(missing_ok=True)
    for epoch in range(1, args.num_epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row["train_loss"] = train_one_epoch(run_model, loaders["train"], optimizer, device, amp_dtype, scaler)
        _, _, val_metrics = evaluate_loader(run_model, loaders["val"], device, amp_dtype)
        row["val_rmse"] = val_metrics["rmse"]
        if val_metrics["rmse"] < best_rmse:
            best_rmse, best_epoch = val_metrics["rmse"], epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)
        scheduler.step()
        row["elapsed_seconds"] = round(time.time() - started, 3)
        append_csv_row(log_path, row)
        print("epoch %s train_loss=%.4f val_rmse=%.4f" % (epoch, row["train_loss"], row["val_rmse"]), flush=True)

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    results = {}
    for name in ("train", "val", "test"):
        y_true, y_pred, metrics = evaluate_loader(run_model, loaders[name], device, amp_dtype)
        results[name] = metrics
        pd.DataFrame([metrics]).to_csv(args.out_dir / ("final_results_%s.csv" % name), index=False)
        if name != "train":
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(args.out_dir / ("pred_label_%s.csv" % name), index=False)

    save_json(
        args.out_dir / "run_summary.json",
        {
            "task_name": args.task_name,
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - started, 3),
            "best_epoch": best_epoch,
            "best_val_rmse": best_rmse,
            "split_sizes": sizes,
            "rows_dropped_invalid_smiles": rows_dropped_invalid_smiles,
            "precision": precision,
            "final_metrics": results,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )


if __name__ == "__main__":
    main()
