# BACPI wrapper report

## Method
BACPI (Bioinformatics 2022) predicts compound-protein interaction and binding affinity.
The compound is an RDKit graph (atom-environment fingerprint ids, adjacency with self-loops) plus a 1024-bit Morgan fingerprint.
The protein is a sequence of amino-acid 3-gram tokens.
A 3-head graph attention network encodes the compound. A 3-layer 2D CNN encodes the protein.
Four bidirectional attention layers join both sides. A linear layer on the outer product gives the affinity.
Training uses MSE loss with Adam. The paper holds out 10% of train as validation and keeps the best-validation epoch.

## Original default settings
| Setting | Value | Source |
|---|---|---|
| Optimiser | Adam, amsgrad=True, weight_decay=0 | code/main.py:44 (paper p.5: "Adam") |
| lr | 5e-4 | code/main.py:28 |
| Scheduler | StepLR(step 10, gamma 0.5), hard-coded | code/main.py:45 |
| Batch size | 16 | code/main.py:30 |
| Epochs | 20 | code/main.py:31 |
| Loss | MSE | code/main.py:40 |
| Architecture | gat_dim 50, heads 3, dropout 0.1, alpha 0.1, comp/prot/latent dim 80, window 5, cnn layers 3, out layers 3 | code/main.py:33-43 |
| Model selection | lowest dev RMSE over epochs, report test at that epoch | code/main.py:78-100 |
| Early stopping, grad clip, AMP | none (fp32) | code/main.py |
| Features | RDKit AddHs graph, Morgan r=2 1024 bits chirality, 3-grams | code/data_process.py |

Paper and code agree. The paper gives no numbers for lr, batch size or epochs, so the code defaults apply.

## What the wrapper does now
- `run_split_benchmarks.py` takes `--tasks` (ki kd ec50 ic50), finds the default split folders on disk, builds one shared feature cache, and trains every seed (default 666 777 888) on every split.
- Ki uses `kinetic_dataset/ki`. Kd, EC50 and IC50 use `binding_affinity_dataset/<task>`.
- Split folders found on disk: random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits, enzyme_sequence_splits/threshold_*, substrate_splits/threshold_*.
  Current thresholds: ki 0.05 / 0.545, kd 0.01 / 0.39, ec50 0.01 / 0.525, ic50 0.01 / 0.63 (enzyme / substrate).
- Output goes to `--out_root/<task>/<split_group>/<split_name>/seed_<seed>/` (default `outputs/baselines/BACPI`). Nothing is written next to the split files.
- `aggregate_retrain_metrics.py` writes `summary_runs.csv` and `summary.csv` (mean and std over seeds; rmse, mae, r2_score, pearson, spearman; same metric key as the other wrappers).

## Files
Removed: `tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `modeling.py` (unused original-model builder).
Rewritten: `common.py`, `dataset.py`, `train_single_target_tvt.py`, `run_split_benchmarks.py`, `cache_embeddings.py`, `aggregate_retrain_metrics.py`, `predict_single_target.py`, `modeling_v2.py`, `default_hparams_original.json` (Optuna key removed), `emulator_bench/README.md`, `../commands.txt`.
Changed: `model_v2.py` (init fix below), `../Plan.md` (Optuna lines removed).
Original `code/` is not touched.

## Bugs fixed
- Split files were rewritten in place by default (`--clean_splits_in_place`). The option is gone. Invalid-SMILES rows are dropped in memory only.
- Results were written inside the split folders. They now go under `--out_root`.
- The train script defaulted to no validation (`val_every=0`), so it kept the last epoch. It now validates every epoch and keeps the best-RMSE epoch, as the original does.
- `model_v2.py` initialised the GAT attention vectors `a_left`/`a_right` as two (D,1) tensors. The original uses one (2D,1) tensor, which gives a different Xavier scale. v2 now initialises (2D,1) and splits it. With the same seed, v2 and the original model have identical weights.
- MAE and R2 were missing from the metrics. They are now computed in `regression_metrics`.
- Hard-coded `/home/adhil/github/BACPI` and old `baselines/BACPI/<task>` paths removed. Paths come from the repo layout or arguments.
- Stale defaults removed (`random_splits`, `enzyme_structure_splits`, `conformer_cosine_splits`, `--threshold_*` hard values).

## Deviations from the original and why behaviour is unchanged
- Features are cached once per unique compound and protein (same featurisation code path as original, vocabularies built from all splits like the original dict built over the whole dataset). SMILES go to RDKit as given (whitespace stripped), like `code/data_process.py`. They are not canonicalised, because the original features depend on atom order. RDKit parsing is only the validity check for dropping rows.
- `BACPIv2` replaces the repeat_interleave GAT scoring with broadcast addition. Checked on CPU: max output difference vs `code/model.py` is 2.3e-9 (output scale about 1e-2), same seed gives identical initial weights only when both models use the same vocabulary size. The wrapper table is larger, so initial weights differ (see Vocabulary).
- Batches: plain random shuffle with `torch.Generator(seed)`, last partial batch kept, as in the original. The old length-bucketed sampler is removed because it changed batch composition.
- The collate function pads to the batch maximum, as `batch_pad` does. Adjacency is stored as bool (model tests `adj > 0`).
- DataLoader workers and pinned memory are used. They do not change values.
- Mixed precision is off by default (the original is fp32). `--amp` enables bf16 autocast. TF32 and `cudnn.benchmark` flags are removed.
- Vocabulary: ids are assigned in first-encounter order like the original `defaultdict` dictionaries. The encounter order is the sorted unique SMILES and sequences of all four tasks and all default split groups (train, val, test). The vocabulary is built once in `featurizer.pkl` and does not depend on `--tasks` or on which tasks were cached first. The original encounters only the train and test rows of one dataset, in file order, so exact ids and table size differ from a per-dataset original run. The larger table changes the random init stream, so the same seed gives different initial weights than an original run (0 of 84 tensors equal). With weights copied from the original (embedding rows permuted by the id map), the forward pass differs by 1.9e-9. Three-epoch training matches: loss diff 2.4e-7, val RMSE diff 4.9e-7, test prediction diff 4.8e-7, same best epoch.
- Invalid-SMILES rows are dropped in memory per split. The counts are saved in `run_summary.json` as `rows_dropped_invalid_smiles` (train, val, test).
- Seed handling: the original split train into train/dev inside `main.py` and used the dataset's own test. Here the three split files come from EMULaToR.

## Speed-ups
- Protein convolutions: each single-channel 11 x 11 conv in `BACPIv2.prot_cnn` runs as banded matrices and one `addmm` on an unfolded tensor (`banded_conv`). Above length 8192 the model uses the cuDNN conv. Parameters and `state_dict` keys are unchanged, so old checkpoints load.
- Optimiser: Adam uses `fused=True` on CUDA.
- Both changes are numerically equivalent but not bit-identical to the original conv and Adam. They change the order of float operations. Adam amplifies this over many steps.
- The original run is itself not reproducible run to run in default mode (cuDNN and atomic-add kernels). Two identical runs give different losses.
- Features: `packed_v2/` stores all cache entries in flat memory-mapped files. Processes share them through the page cache, so each process no longer loads every npz entry. Batch tensors are bitwise equal to the npz loader, in the same order. Run `cache_embeddings.py` once on a cache that has only npz files to build the pack.
- SMILES validity: SMILES with a packed entry are valid and are not parsed again. Others still go to RDKit. The kept rows are identical.
- Validation and test loaders use at most 2 workers. The train loader and its shuffle order are unchanged.

## Commands
Run from `baselines/BACPI` with the mldb env.
```
python emulator_bench/run_split_benchmarks.py --tasks ki kd ec50 ic50 --seeds 666 777 888 --device cuda:0
python emulator_bench/aggregate_retrain_metrics.py
```
To use several GPUs, start one command per task with `CUDA_VISIBLE_DEVICES=<n> ... --tasks <task>`.
Run the shared cache first (`python emulator_bench/cache_embeddings.py`) before starting several tasks in parallel, so two processes do not write the cache at the same time.

## Smoke test
Env: `/home/adhil/.conda/envs/mldb`, CPU only (load average was above 150, so it was slow).
Data: 64-row copies of ki and kd (random_splits_grouped_sequence and one enzyme_sequence_splits threshold) in the scratchpad `bench_smoke/BACPI/tiny_src`.
Command: `run_split_benchmarks.py --tasks ki kd --datasets_root <tiny> --out_root <scratch out> --seeds 666 --num_epochs 2 --device cpu`.
Result: 4 runs finished, all outputs written, `aggregate_retrain_metrics.py` wrote summary files.
Real split discovery returns the expected folders for all four tasks. No file under `EMULaToR/data` was written.
Re-run after the fixes (ki cached first, then kd run; a fresh cache of kd alone gives the same `featurizer.pkl` hash): one test row had an invalid SMILES and `rows_dropped_invalid_smiles.test` is 1 in its summary. Aggregation shows `r2_score_mean` and `r2_score_std`. Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/BACPI/out2`.
Model equivalence check (v1 vs v2) ran on CPU as described above.

## Open issues
- Precision: the paper appendix says bf16 for binding runs. The original code is fp32, so the default is fp32. Use `--amp` to follow the paper's bf16.
- The vocabulary includes val rows and all four tasks, so it is larger than any per-dataset original vocabulary. Unused embedding rows do not change predictions, but they change the random init stream. Same-seed init therefore differs from an original run. With copied weights the training trajectory matches (3 epochs, val RMSE diff 4.9e-7).
- Full-size training time was not measured. The original batch size of 16 is small, so large splits (ki, kd) will be slow.
- `git status` shows untracked `bacpi.pdf`, as before.
- The vocabulary in `vocabs_v2/featurizer.pkl` is frozen on first build: bump `CACHE_VERSION` or delete the embeddings folder after rebuilding datasets. The vocabulary is global over all tasks and splits (ids and table size differ from the original per-dataset dictionary; no systematic effect).
