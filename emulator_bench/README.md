# emulator_bench

Multi-seed evaluation of BACPI with its original settings on the default EMULaToR splits.
The original `code/` is not changed.

## Tasks and splits

| Task | Split parent |
|---|---|
| `ki` | `data/processed/datasets/kinetic_dataset/ki` |
| `kd`, `ec50`, `ic50` | `data/processed/datasets/binding_affinity_dataset/<task>` |

Split folders per task: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`,
`uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`.
Thresholds are read from disk. Each folder needs `train.parquet`, `val.parquet`, `test.parquet`
with columns `sequence`, `smiles`, `log10_value`.

Split files are never written. Rows with invalid SMILES are dropped in memory only.
The count per split is saved in `run_summary.json` as `rows_dropped_invalid_smiles`.

SMILES go to RDKit exactly as in the original `code/data_process.py` (whitespace stripped, no canonicalisation).
The vocabularies follow the original dictionaries: ids are assigned in first-encounter order.
The original encounters train rows in file order, one dataset at a time. Here the encounter order is the
sorted unique SMILES and sequences over all four tasks and all default splits (train, val and test).
The vocabulary is built once, so it does not depend on `--tasks` or on which tasks were cached first.

## Original settings

From `code/main.py`: Adam (`amsgrad=True`, no weight decay), `lr=5e-4`, StepLR (step 10, gamma 0.5),
batch size 16, 20 epochs, MSE loss, fp32, no early stopping, no gradient clipping.
The epoch with the lowest validation RMSE gives the test metrics.
Architecture: `gat_dim=50`, `num_head=3`, `dropout=0.1`, `alpha=0.1`, `comp_dim=prot_dim=latent_dim=80`,
`window=5`, `layer_cnn=3`, `layer_out=3`.
Compound features: RDKit graph fingerprint ids and a 1024-bit Morgan fingerprint (radius 2, chirality).
Protein features: amino-acid 3-grams.

Seeds: 666, 777, 888.

## Scripts

- `cache_embeddings.py`: caches compound and protein features once, in one shared cache for all tasks.
- `train_single_target_tvt.py`: trains one split and one seed.
- `run_split_benchmarks.py`: builds the cache, then runs every seed on every split of every task.
  Finished runs are skipped unless `--overwrite` is set.
- `aggregate_retrain_metrics.py`: writes `summary_runs.csv` and `summary.csv` (mean and std over seeds).
- `predict_single_target.py`: evaluates a saved `bestmodel.pth` on one split file.
- `model_v2.py`, `modeling_v2.py`: memory-efficient BACPI. It gives the same outputs as `code/model.py`.

## Output layout

Default `--out_root`: `outputs/baselines/BACPI`.

- `embeddings/`: feature cache and vocabularies
- `<task>/<split_group>/<split_name>/seed_<seed>/`: `bestmodel.pth`, `logfile.csv`,
  `final_results_{train,val,test}.csv` (rmse, mae, r2_score, pearson, spearman),
  `pred_label_{val,test}.csv`, `run_summary.json`

Random and time splits use `<split_name> = all`.

## Commands

See `../commands.txt`. Use the mldb conda env: `conda run -n mldb python ...`.
