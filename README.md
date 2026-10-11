# Mamba for Deep-Learning Side-Channel Attack (DLSCA)

This folder contains the code for profiled side-channel attacks on AES with a
**bidirectional Mamba** (selective state-space) model, compared against an
**MLP** baseline. Models are trained on power traces to predict the AES S-box
output of one key byte (identity leakage model, 256 classes), and the attacks
are evaluated with **key rank** and **guessing entropy (GE)**.

Two datasets are used:

| Dataset | Target byte | Input | Profiling / validation traces |
|---|---|---|---|
| ASCAD v1, fixed key | 2 | full 700-sample trace | 40 000 / 10 000 |
| ChipWhisperer (CW-Target, device A) | 0 | 100-sample window around sample 1320 | 40 000 / 10 000 |

## Folder structure

```
code/
├── network/
│   ├── mamba.py              # MAMBA_model: Conv1d embedding + forward & backward Mamba blocks + linear head
│   └── mlp.py                # MLP_model: 3 hidden SELU layers (baseline)
├── utilities/
│   ├── profile.py            # AES S-box, label creation, Hamming weight, SNR, points of interest
│   └── attack.py             # key scoring, key rank, guessing entropy (+ convergence)
├── mamba-profile-ascadv1-fixed.py   # train Mamba on ASCAD v1 fixed key
├── mamba-attack-ascadv1-fixed.py    # attack ASCAD v1 with a trained Mamba checkpoint
├── mlp-profile-ascadv1-fixed.py     # train MLP on ASCAD v1 fixed key
├── mlp-attack-ascadv1-fixed.py      # attack ASCAD v1 with a trained MLP checkpoint
├── chipwh-mamba-profile.py          # train Mamba on ChipWhisperer
├── chipwh-mamba-attack.py           # attack ChipWhisperer with Mamba
├── chipwh-mlp-profile.py            # train MLP on ChipWhisperer
├── chipwh-mlp-attack.py             # attack ChipWhisperer with MLP
├── plot-ge-convergance.py           # GE vs. number of attack traces, MLP vs. Mamba
├── plot-train-metrics.py            # validation GE per epoch, MLP vs. Mamba
├── experiments-final/               # checkpoints (.ckpt), CSV training logs, GE results
└── plots-final/                     # figures used in the report
```

## Requirements

- Linux with an NVIDIA GPU and CUDA. `mamba-ssm` has no CPU or macOS build, so
  the Mamba scripts will not run without a GPU. The MLP scripts run on CPU.
- Python 3.10 or newer.

Install:

```bash
python -m venv .venv
source .venv/bin/activate

# 1. Install PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch

# 2. Install the remaining dependencies
pip install -r requirements.txt
```

If `mamba-ssm` or `causal-conv1d` fails to build, install them after PyTorch
with `pip install causal-conv1d mamba-ssm --no-build-isolation`.

TensorFlow is only used for `to_categorical` (one-hot labels). If you prefer
not to install it, replace that call with
`torch.nn.functional.one_hot(torch.as_tensor(labels).long(), 256).numpy()`.

## Data

The datasets are not part of the repository.

**ASCAD v1 (fixed key).** Download `ASCAD.h5` from the
[ANSSI ASCAD repository](https://github.com/ANSSI-FR/ASCAD) (ATMega8515 fixed-key
database), then rename and place it so that, from inside `code/`, the path is:

```
code/data/ascadv1-fixed.h5
```

**ChipWhisperer.** The scripts expect NumPy archives (traces, plaintexts and
keys, in that order) one level above `code/`:

```
data/chip_whisperer/seeded-train/device_A/dataset_device_A.npy
data/chip_whisperer/seeded-test/device_A/dataset_device_A.npy
```

To use other locations, change `FILE_PATH` / `FILE_PATH_TRAIN` /
`FILE_PATH_TEST` at the top of each script.

## Usage

Run all scripts from inside the `code/` folder, because the imports
(`network`, `utilities`) and all data/output paths are relative to it.
Some scripts save plots to `plots/`, so create it first:

```bash
cd code
mkdir -p plots plots-final
```

### 1. Profiling (training)

```bash
python mamba-profile-ascadv1-fixed.py
python mlp-profile-ascadv1-fixed.py
python chipwh-mamba-profile.py
python chipwh-mlp-profile.py
```

Each script trains for 100 epochs (seed `83545`). After every epoch, it computes
the validation guessing entropy with 1 000 traces and logs it as `val_ge_mean`.
A checkpoint is saved for every epoch in the matching `experiments-final/...`
folder, and metrics are written by the CSV logger to `logs/version_N/metrics.csv`.

### 2. Attack

Set the checkpoint path in the `load_from_checkpoint(...)` call of the attack
script to the epoch you want to evaluate, then run:

```bash
python mamba-attack-ascadv1-fixed.py
python mlp-attack-ascadv1-fixed.py
python chipwh-mamba-attack.py
python chipwh-mlp-attack.py
```

The attack scripts print the key rank and the mean/median GE, save a GE
convergence plot to `plots-final/`, and write a GE-vs-traces CSV
(`avvge-*.csv`) to the experiment folder.

### 3. Comparison plots

```bash
python plot-ge-convergance.py   # GE convergence, MLP vs. Mamba
python plot-train-metrics.py    # validation GE per epoch, MLP vs. Mamba
```

Both scripts have the input CSVs for ASCAD commented out and the ChipWhisperer
ones active; switch them in the `__main__` block.

## Main hyperparameters

| | Mamba | MLP |
|---|---|---|
| Architecture | Conv1d (k=3, stride=3) → 2 forward + 2 backward Mamba blocks, `d_model=64` → linear | 3 × 128 units, SELU |
| Optimizer | AdamW, lr 5e-4, weight decay 0.01 | AdamW, lr 5e-4, weight decay 0.01 |
| Batch size | 768 (ASCAD), 512 (ChipWhisperer) | 768 / 512 |
| Epochs | 100 | 100 |
| Preprocessing | `StandardScaler` fitted on profiling traces | same |

## Known issues

- `mamba-attack-ascadv1-fixed.py` loads a checkpoint from
  `mamba-d64-l2-only-forw-encoder/`, which is not in the repository. Point it
  to a checkpoint in `experiments-final/experiments-ascadv1-fixed-key/mamba-d64-l2/`
  instead (for example `mamba-epoch=14-val_ge_mean=0.92.ckpt`).
- `mamba-profile-ascadv1-fixed.py` writes to a `...-four-forw-encoder/` folder,
  not to `mamba-d64-l2/`. Change `PATH_MAMBA` if you want the default location.
- In `plot-ge-convergance.py`, the ChipWhisperer CSV paths are swapped between
  `mlp_csv` and `mamba_csv`. These CSVs are created by the ChipWhisperer attack
  scripts; because `PATH_MLP` in `chipwh-mlp-attack.py` has no trailing `/`, the
  MLP file is written as `experiments-chipwh/mlp-128avvge-chipwh-mlp.csv`.
- `experiments-final/` holds about 670 MB of checkpoints. Clone with
  `--depth 1` to keep the download small.
