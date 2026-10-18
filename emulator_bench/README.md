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
The train loader uses `--num_workers`. The validation and test loaders use at most 2 workers.

## Scripts

- `cache_embeddings.py`: caches compound and protein features once, in one shared cache for all tasks.
- `train_single_target_tvt.py`: trains one split and one seed.
- `run_split_benchmarks.py`: builds the cache, then runs every seed on every split of every task.
  Finished runs are skipped unless `--overwrite` is set. `--gpus 0 1 --runs_per_gpu 2` runs several trainings at once, one GPU each.
  `--amp` turns on mixed precision (bf16, or fp16 on pre-Ampere GPUs). Default is fp32.
- `aggregate_retrain_metrics.py`: writes `summary_runs.csv` and `summary.csv` (mean and std over seeds).
- `predict_single_target.py`: evaluates a saved `bestmodel.pth` on one split file.
- `model_v2.py`, `modeling_v2.py`: memory-efficient BACPI. It gives the same outputs as `code/model.py` up to float rounding.
  The protein convolutions run as banded matrix products (cuDNN above length 8192). Training uses fused Adam on CUDA.

## Output layout

Default `--out_root`: `outputs/baselines/BACPI`.

- `embeddings/`: feature cache `compounds_v2/`, `proteins_v2/` and the frozen vocabulary `vocabs_v2/`.
  `packed_v2/` holds the same entries as flat memory-mapped files. Training and prediction read only these files.
  `cache_embeddings.py` rebuilds `packed_v2/` after it adds entries. It also builds it for a cache that has only npz files, so run it once on such a cache.
  Valid SMILES have entries in `packed_v2/`, so later runs parse only SMILES that are not in it.
  Cache keys cover the featurizer settings. Change `CACHE_VERSION` in `feature_pipeline.py` to force a rebuild.
- `<task>/<split_group>/<split_name>/seed_<seed>/`: `bestmodel.pth`, `logfile.csv`,
  `final_results_{train,val,test}.csv` (rmse, mae, r2_score, pearson, spearman),
  `pred_label_{val,test}.csv`, `run_summary.json`

Random and time splits use `<split_name> = all`.

## Commands

See `../commands.txt`. Run the commands from the `baselines/BACPI` folder. Use the mldb conda env: `conda run -n mldb python ...`.
