# ER-PFL

Implementation of **Stratified attribution alignment for cross-corpus reliability of federated Parkinsonian voice classification.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use PC-GITA, ItalianPVS and NeuroVoz sustained-vowel recordings under their releases. The primary cohort excludes the designated young Italian control stratum before splitting. Export eGeMAPS vectors as `x: (N,88)`, `y`, globally unique participant `group`, `corpus`, numeric `age`, `sex` (0/1, unknown -1) and `split: train/val/test`. LOCO keeps the test corpus outside training. Recording probabilities are averaged within participant. Optional audio extraction uses `pip install -e ".[audio]"` and `python -m saa_fedpd.extract --manifest data/voice.csv --output data/voice.npz`; the CSV has `audio` plus the listed metadata columns. Apply corpus-specific sustained-vowel eligibility before extraction.

## Workflow

```bash
python -m saa_fedpd prepare --input data/voice.npz
python -m saa_fedpd train
python -m saa_fedpd run
python -m saa_fedpd evaluate
python -m saa_fedpd plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`local_epochs` defines each communication round's local work. `proximal_weight` regularizes server-bound copies; `alignment_weight` aligns diagnosis/sex/age strata after warm-up. `personal_weight` defines the observed-client penalty. `protocol=loco` enforces unseen-corpus access. `random_deletions` sizes the matched deletion reference. `epochs` is communication rounds and validation selects the server checkpoint. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`learning.csv` records validation participant AUROC and communication size. `model.pt` is the global model; the personalized variant also writes observed-client models. `predictions.npz` contains recording probabilities, attribution and deletion diagnostics. `metrics.csv` adds participant AUROC/AUPRC/Brier, explanation drift and deletion fidelity. Combining LOCO predictions through the probe command gives macro/worst-corpus AUROC, evidence-site AUC and age-probe AUC.

## Method and comparisons

Source preprocessing winsorizes at training percentiles, median-imputes and standardizes. Shared MLPs have 512/128 hidden layers and dropout. Alignment differentiates absolute gradient-times-input through local model parameters. Prototypes are transmitted only for strata with at least five participants; aggregation uses participant counts. Test-corpus heads, prototypes and calibrators are never fitted. Global alignment and observed-client personalization remain distinct.

Controls: `full`, `fedavg`, `fedprox`, `personalized`. Choose `--variant` and use a separate training/output directory.

```bash
python -m saa_fedpd.compare --inputs "runs/*/metrics.csv" --reference full --metrics participant_auroc participant_auprc participant_brier --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

## Cross-corpus probes

```bash
python -m saa_fedpd.probe --predictions runs/loco_a/predictions.npz runs/loco_b/predictions.npz runs/loco_c/predictions.npz --output runs/probes.csv
```

Each prediction archive includes corpus, age and participant metadata. Site and age probes use cross-fitted logistic regression. Personalization is restricted to observed corpora; LOCO uses the global checkpoint.

