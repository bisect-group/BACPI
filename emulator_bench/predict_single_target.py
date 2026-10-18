import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import read_table, require_columns
from emulator_bench.dataset import BACPIDataset, CacheStore, create_loader
from emulator_bench.feature_pipeline import BACPIFeaturizer, compound_cache_path, filter_invalid_smiles_rows, protein_cache_path
from emulator_bench.modeling_v2 import BACPIModelConfig, build_model
from emulator_bench.train_single_target_tvt import evaluate_loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved BACPI checkpoint on one split file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_path", type=Path, required=True)
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    featurizer = BACPIFeaturizer.load(args.embeddings_dir / "vocabs" / "featurizer.pkl")
    frame = read_table(args.split_path)
    require_columns(frame, [args.sequence_col, args.smiles_col, args.target_col], args.split_path)
    frame, _ = filter_invalid_smiles_rows(frame, smiles_col=args.smiles_col, source_name=str(args.split_path))

    dataset = BACPIDataset(
        frame,
        CacheStore(lambda smiles: compound_cache_path(args.embeddings_dir, smiles)),
        CacheStore(lambda sequence: protein_cache_path(args.embeddings_dir, sequence)),
        args.smiles_col,
        args.sequence_col,
        args.target_col,
    )
    loader = create_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", seed=0)
    model = build_model(featurizer.n_atom_tokens, featurizer.n_amino_tokens, BACPIModelConfig()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])

    y_true, y_pred, metrics = evaluate_loader(model, loader, device)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(args.out_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(args.out_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
