<p align="center">
  <img src="nave_logo.svg" alt="NAVE Conformer Wave Logo" width="600">
</p>

# NAVE - Normalized, Adaptive Conformer for Whale Vocalization-Event Detection

A clean implementation of a conformer model for BioDCASE 2026 Task 2 (3-class
Antarctic baleen-whale sound-event detection: BMABZ / D / BP).

- **N**ormalized -- fixed PCEN channel (per-bin AGC; recovers background-buried D downsweeps)
- **A**daptive -- frequency-dynamic (FDY) convolutions in the stem
- Conformer backbone (macaron FFN / RoPE MHSA / wide depthwise conv)

## Files

The training pipeline is **self-contained in four files**. Two evaluation scripts
build on those four: `nave_evaluate.py` (in-sample, tune-on-all) and
`nave_evaluate_loso.py` (leave-one-site-out, the reported protocol). Two shell scripts
drive the ablation ladder and the kernel sweep.

| file | role |
|------|------|
| `nave_config.py`   | single source of truth for all constants + class helpers |
| `nave_features.py` | 4-ch STFT + PCEN front end (parameter-free) |
| `nave_model.py`    | `NAVE` architecture + native checkpoint loader |
| `nave_train.py`    | training entry point **and** the full data / post-processing / metrics / threshold-tuning pipeline |
| `nave_evaluate.py` | *eval* -- single-checkpoint **in-sample** evaluation (one inference pass, thresholds tuned on all three dev sites) |
| `nave_evaluate_loso.py` | *eval* -- **leave-one-site-out** evaluation with per-class breakdown and `mean (std)` over seeds (the reported metric) |
| `run_ablation.sh`     | *helper* -- runs the 5-rung ablation ladder for one seed (train + 60s eval per rung) |
| `run_kernel_sweep.sh` | *helper* -- sweeps the conformer depthwise kernel for one seed (train + 60s eval per kernel) |

`nave_train.py` imports only `nave_config`, `nave_features`, `nave_model` and
third-party packages -- the entire data loader, per-epoch negative resampling,
event-level post-processing, metrics, per-class threshold tuner, EMA, optimiser
and validation paths are bundled inside it. Both eval scripts reuse those helpers
via `from nave_train import …` (and `nave_evaluate_loso.py` also reuses the inference /
threshold path from `nave_evaluate.py`).

## Requirements

Python >= 3.10 and:

```bash
pip install torch numpy pandas scipy soundfile tqdm
```

`tqdm` is optional (a no-op fallback is used if missing). Parquet caching of the
file manifest / annotations is used when a parquet engine (e.g. `pyarrow`) is
installed and silently skipped otherwise.

## Data layout

Point NAVE at the BioDCASE development set -- either export
`NAVE_DATA_ROOT=/path/to/2026_BioDCASE_development_set` (no source edit needed) or
edit `DATA_ROOT` in `nave_config.py`:

```
DATA_ROOT/
  train/      annotations/{dataset}.csv   audio/{dataset}/*.wav
  validation/ annotations/{dataset}.csv   audio/{dataset}/*.wav
```

Audio is 250 Hz mono WAV; annotation CSVs carry `start_datetime`, `end_datetime`,
`annotation` (the 7 fine call types). The site lists live in `nave_config.py`:
8 `TRAIN_DATASETS` for training, 3 `VAL_DATASETS`
(`casey2017` / `kerguelen2014` / `kerguelen2015`) held out for evaluation. The
three dev sites are **never** used in training -- the model is trained once on the
8 training sites; LOSO (below) is an evaluation-time protocol over the three dev
sites.

## Recipe (all in `nave_config.py`)

STFT SR=250 / N_FFT=256 / hop=5 (129 bins, 20 ms). 4 channels [demeaned |S|,
cos phi, sin phi, PCEN]. FDY on filterbank+feat0 (basis 4). d_model 128, 4 heads,
4 layers, ffn x4, dropout 0.1, **depthwise conv k=153** (RF ~3.06 s, the reported
kernel; now the config default). RAdam const LR 5e-5, wd 1e-3, EMA 0.999, 40
epochs, batch 32, neg-ratio 1.0, per-epoch negative resampling. Post: 500 ms
median smooth -> tuned per-class thresholds -> 0.5 s merge gap -> 0.5-30 s
duration filter.

**Eval tile length.** `EVAL_SEGMENT_S` defaults to 30, but the reported numbers
use **60 s** tiling -- pass `--segment-s 60` to `nave_evaluate.py` / `nave_evaluate_loso.py`
(the two `.sh` scripts already do). Everything downstream (metrics, threshold
tuner) is identical; only the eval tiling changes.

## Usage

```bash
# train one seed (full NAVE; k=153 is the default now)
CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42 --tune-workers 20

# in-sample evaluation of a checkpoint at the reported 60 s tiling
CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt \
    --segment-s 60 --workers 13

# leave-one-site-out evaluation (the reported metric)
CUDA_VISIBLE_DEVICES=0 python nave_evaluate_loso.py runs/nave_s42_*/nave_best.pt \
    --segment-s 60 --workers 20
```

Each training run writes `runs/nave_s<seed>_<tag>_<timestamp>/nave_best.pt` (best
tuned macro) and `nave_epoch_NN.pt`. Every checkpoint stores `model_state_dict`,
the tuned per-class `thresholds`, `macro_f1`, `epoch`, `seed`, and the ablation
`config` (so both eval scripts rebuild the matching architecture on load).

`nave_evaluate.py` runs one forward pass over `cfg.VAL_DATASETS`, tunes per-class
thresholds on **all three** dev sites, and prints the tuned thresholds plus both
macro-F1 reductions (`A` = mean of the three per-class F1s; `B` = F1 of the
per-class mean P and mean R). This is the optimistic in-sample number.

## Leave-one-site-out (`nave_evaluate_loso.py`)

The reported metric. The model is trained once (on the 8 training sites); LOSO
guards against **threshold** overfitting across the dev sites. For each checkpoint
it collects the dev-site probabilities once (cached under `runs/nave_loso_cache/`),
then for each of the three dev sites: tunes per-class thresholds on the *other
two* and scores the *held-out* one. It pools tp/fp/fn across the three folds into
pooled per-class P/R/F1 and the macro F1, and also reports the in-sample
(tune-on-all) F1 and the `gap` (in-sample - LOSO).

```bash
# ablation ladder over several seeds -> per-rung mean (std)
python nave_evaluate_loso.py --ladder 0 1 1337 666 --segment-s 60 --workers 20

# single seed with per-FOLD per-class detail
python nave_evaluate_loso.py --ladder 666 --segment-s 60 --per-site

# an explicit checkpoint (any tag)
python nave_evaluate_loso.py runs/nave_s666_pcen1_fdy1_k129_*/nave_best.pt --segment-s 60
```

`--ladder <seeds…>` resolves the five ablation rungs (below) for each seed and
prints a per-rung `mean (std)` table (sample std, ddof=1). Missing rungs are
reported and skipped. Add `--no-cache` to force re-inference.

## Helper scripts

Both scripts run **sequentially on one GPU** (set `CUDA_VISIBLE_DEVICES` in
front), train each configuration, then evaluate it at 60 s as soon as it finishes.
A failed run is skipped and the loop continues. Run them from the repository root
(the folder with `nave_train.py`) inside `tmux`. W&B logging is on by default
(`WANDB=0` to disable; needs
`WANDB_ENTITY` / `WANDB_API_KEY` in the env, else it is skipped).

### `run_ablation.sh <seed>` -- ablation ladder

Walks the 5-rung ladder for one seed, adding one component at a time. Every rung
passes its kernel explicitly, so it is independent of the config default:

| rung | train flags | tag |
|------|-------------|-----|
| `base`      | `--no-pcen --no-fdy --conv-kernel 31` | `pcen0_fdy0_k31` |
| `+fdy`      | `--no-pcen --conv-kernel 31`          | `pcen0_fdy1_k31` |
| `+pcen`     | `--conv-kernel 31`                    | `pcen1_fdy1_k31` |
| `+k65`      | `--conv-kernel 65`                    | `pcen1_fdy1_k65` |
| `full_k129` | `--conv-kernel 129`                   | `pcen1_fdy1_k129` |

```bash
CUDA_VISIBLE_DEVICES=0 bash run_ablation.sh 1     # seed 1 on GPU 0
CUDA_VISIBLE_DEVICES=1 bash run_ablation.sh 0     # seed 0 on GPU 1 (parallel)
```

Env overrides: `PY=` (interpreter), `WORKERS=` (tuner workers, default 20),
`EVAL_SEG=` (eval tile seconds, default 60), `WANDB=0`.

> The ladder's top rung is the k129 "full" model (an ablation point). The reported
> **k153** model is a separate kernel-sweep result -- train it with
> `run_kernel_sweep.sh <seed> 153` (or a bare `nave_train.py --seed <seed>`, since
> k153 is now the default).

### `run_kernel_sweep.sh <seed> [k1 k2 …]` -- receptive-field sweep

Full NAVE (PCEN + FDY on); only the conformer depthwise kernel varies, isolating
the receptive-field effect. Kernels must be **odd** (even values are skipped);
receptive field = kernel x 0.02 s.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42 49 81 113 145   # round 1
CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42 129 153         # k129 vs the reported k153
CUDA_VISIBLE_DEVICES=0 bash run_kernel_sweep.sh 42                 # default round-1 set (49 81 113 145)
```

Same env overrides as above.

## Reproducing the paper numbers

The reported results are **5-seed leave-one-site-out** over the three dev sites,
with event counts **pooled across the three folds** (matching the official
challenge scorer's pooling). This is what `nave_evaluate_loso.py` computes -- the model is
trained once per seed on the 8 training sites, and LOSO rotates which dev site is
held out for threshold tuning. `nave_evaluate.py`'s in-sample (tune-on-all) number
is optimistic and is **not** the headline.

1. **Train the seeds.** Full NAVE at k153 is the default, so per seed:
   `CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed <s> --tune-workers 20`
   (or `run_ablation.sh <s>` for the ablation-ladder checkpoints, and
   `run_kernel_sweep.sh <s> 153` for the k153 checkpoint).
2. **LOSO-evaluate at 60 s.**
   - Ablation ladder with `mean (std)` over seeds:
     `python nave_evaluate_loso.py --ladder <s1 s2 s3 s4 s5> --segment-s 60 --workers 20`
     (covers `base … full_k129`).
   - Headline **k153**: the `--ladder` table stops at k129, so aggregate the five
     k153 checkpoints -- either add a `k153` rung to `LADDER` in `nave_evaluate_loso.py`, or
     pass the five `…pcen1_fdy1_k153…/nave_best.pt` paths (note: explicit
     checkpoints are reported per-run, not auto-aggregated).
3. Aggregate as `mean (std)` (sample std, ddof=1).

### Reported results (macro F1, 5-seed LOSO, event counts pooled across folds)

> This is the NAVE ablation ladder as it appears in the paper's Table 2 --
> transcribed here for convenience, not regenerated by the helper scripts above.

| configuration | macro F1 |
|---------------|:--------:|
| base (no PCEN, no FDY, k31) | 0.328 |
| + FDY | 0.376 |
| + PCEN | 0.399 |
| + k65 | 0.463 |
| full, k129 | 0.492* |
| **full, k153 (reported)** | **0.518** |

*Derived from the paper text (0.518 − 0.026, p=0.036); the paper's Table 2 has no
standalone k=129 row.

- Headline: **macro F1 0.518 (0.016)**, k=153, 5 seeds, LOSO pooled;
  **2,705,787** parameters. This beats both paper baselines (Whale-VAD 0.377,
  WhaleVAD-BPN 0.475).
- The kernel curve has its knee at RF ~1.3 s (k65) and plateaus from k129 onward;
  k153 is the adopted kernel. k=129 -> 153 adds 4 x 128 x 24 = 12,288 params.
- A **fold-averaged** NAVE row (P/R averaged across folds, then F1 recomputed)
  reaches macro F1 **0.510**, reported for a like-for-like comparison against
  WhaleVAD-BPN (0.475), whose paper aggregates across folds rather than pooling.

**Seeds used:** `⟨fill in your exact 5 seeds⟩` (each `.sh` invocation and each
`--ladder` entry is one seed; the `nave_evaluate_loso.py` examples above use `0 1 1337 666`).

## Load a checkpoint

```python
from nave_model import NAVE
model = NAVE()
ckpt = model.load_checkpoint("runs/nave_s42_.../nave_best.pt")   # dict with thresholds, etc.
model.eval()
```

`nave_train.py` also exposes the reusable building blocks (`NAVEFeatureExtractor`
is in `nave_features.py`; the data loader `build_val_segments` / `WhaleDataset` /
`collate_fn`, the post-processing `postprocess_predictions`, the metric
`compute_metrics`, and the tuner `tune_thresholds_per_class`) if you want to write
your own evaluation or ensemble script on top -- import them straight from
`nave_train`. `nave_evaluate.py` and `nave_evaluate_loso.py` are exactly that pattern.

## Logging

Training progress -- per-epoch train/val loss, per-class P/R/F1, the tuned
per-class thresholds and the running best macro F1 -- is printed to stdout.
W&B logging is optional (`--wandb`, entity/project from the env or CLI).
