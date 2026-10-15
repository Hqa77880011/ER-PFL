import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--output', default='runs/probes.csv')
    args = parser.parse_args()
    records, features = [], []
    for file in args.predictions:
        d = np.load(file, allow_pickle=False)
        group = d['target_group'].astype(str)
        for subject in np.unique(group):
            selected = group == subject
            records.append(dict(corpus=d['target_corpus'][selected][0], y=d['target_y'][selected][0],
                                age=d['target_age'][selected][0], probability=d['probability'][selected, 1].mean()))
            features.append(d['attribution'][selected].mean(0))
    frame = pd.DataFrame(records)
    target = frame.corpus.to_numpy()
    folds = min(5, int(frame.corpus.value_counts().min()))
    if folds < 2:
        raise ValueError('Need at least two participants per corpus')
    cv = StratifiedKFold(folds, shuffle=True, random_state=13)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probability = cross_val_predict(model, np.array(features), target, cv=cv, method='predict_proba')
    if probability.shape[1] == 2:
        site_auc = roc_auc_score(target, probability[:, 1])
    else:
        site_auc = roc_auc_score(target, probability, multi_class='ovr', average='macro')
    outcomes = frame.groupby('corpus').apply(lambda g: roc_auc_score(g.y, g.probability)).to_numpy()
    age_probability = cross_val_predict(model, frame[['age']], frame.y, cv=StratifiedKFold(folds, shuffle=True, random_state=13), method='predict_proba')[:, 1]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(macro_loco_auroc=outcomes.mean(), worst_corpus_auroc=outcomes.min(),
                       evidence_site_auc=site_auc, age_probe_auc=roc_auc_score(frame.y, age_probability))]).to_csv(out, index=False)


if __name__ == '__main__':
    main()
