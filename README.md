# ER-PFL

Implementation of **Stratified attribution alignment for cross-corpus reliability of federated Parkinsonian voice classification.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Use PC-GITA, ItalianPVS and NeuroVoz sustained-vowel recordings under their releases. The primary cohort excludes the designated young Italian control stratum before splitting. Export eGeMAPS vectors as `x: (N,88)`, `y`, globally unique participant `group`, `corpus`, numeric `age`, `sex` (0/1, unknown -1) and `split: train/val/test`. LOCO keeps the test corpus outside training. Recording probabilities are averaged within participant. Optional audio extraction uses `pip install -e ".[audio]"` and `python -m saa_fedpd.extract --manifest data/voice.csv --output data/voice.npz`; the CSV has `audio` plus the listed metadata columns. Apply corpus-specific sustained-vowel eligibility before extraction.

```bash
python -m saa_fedpd prepare --input data/voice.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m saa_fedpd train --config config.json --output runs/full
python -m saa_fedpd run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m saa_fedpd evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `fedavg`, `fedprox`, `personalized`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m saa_fedpd prepare --input data/voice.npz --config config.json --output runs/fedavg --variant fedavg
python -m saa_fedpd train --config config.json --output runs/fedavg --variant fedavg
python -m saa_fedpd run --config config.json --output runs/fedavg --variant fedavg
python -m saa_fedpd evaluate --config config.json --output runs/fedavg --variant fedavg
```

## Outputs

All main outputs are written below `runs/full/`.

`learning.csv` records validation participant AUROC and communication size. `model.pt` is the global model; the personalized variant also writes observed-client models. `predictions.npz` contains recording probabilities, attribution and deletion diagnostics. `metrics.csv` adds participant AUROC/AUPRC/Brier, explanation drift and deletion fidelity. Combining LOCO predictions through the probe command gives macro/worst-corpus AUROC, evidence-site AUC and age-probe AUC.

```bash
python -m saa_fedpd plot --config config.json --output runs/full
```
