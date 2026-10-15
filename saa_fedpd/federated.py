import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .model import Model


def train(c, out):
    torch.manual_seed(c['seed'])
    d = dict(np.load(out / 'data.npz', allow_pickle=False))
    required = {'corpus', 'age', 'sex'}
    if not required <= set(d):
        raise ValueError('Federated data additionally requires corpus,age,sex')
    source = d['split'] == 'train'
    heldout = d['split'] == 'test'
    if c['protocol'] == 'loco' and set(d['corpus'][source]) & set(d['corpus'][heldout]):
        raise ValueError('LOCO test corpus must be absent from training')
    x, y = torch.tensor(d['x'], dtype=torch.float32), torch.tensor(d['y'], dtype=torch.long)
    _, unique_ids = np.unique(d['group'][source], return_index=True)
    median_age = np.median(d['age'][source][unique_ids])
    sex = np.where(d['sex'] < 0, 2, d['sex'])
    strata = d['y'] * 6 + sex * 2 + (d['age'] > median_age).astype(int)
    model = Model(c, x.shape[1:])
    prototypes = {}
    curve, best, stale = [], -np.inf, 0
    for round_id in range(c['epochs']):
        snapshot = {k: v.detach().clone() for k, v in model.named_parameters()}
        clients, weights, summaries = [], [], []
        for corpus in np.unique(d['corpus'][source]):
            ids = np.flatnonzero(source & (d['corpus'] == corpus))
            local = copy.deepcopy(model)
            local.train()
            optimizer = torch.optim.AdamW(local.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
            for _ in range(c['local_epochs']):
                output = local.net(x[ids])
                loss = F.cross_entropy(output, y[ids])
                if c['variant'] != 'fedavg':
                    loss += c['proximal_weight'] / 2 * sum((parameter - snapshot[name]).square().sum() for name, parameter in local.named_parameters())
                if round_id > 0 and c['variant'] in ['full', 'personalized']:
                    attribution = local.attribution(x[ids], create_graph=True)
                    total_participants = len(np.unique(d['group'][ids]))
                    for stratum in np.unique(strata[ids]):
                        selected = strata[ids] == stratum
                        participant_count = len(np.unique(d['group'][ids][selected]))
                        if participant_count < 5 or int(stratum) not in prototypes:
                            continue
                        mean = attribution[selected].mean(0)
                        normalized = mean / mean.sum().clamp_min(1e-8)
                        loss += c['alignment_weight'] * participant_count / total_participants * (normalized - prototypes[int(stratum)]).square().sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            local.eval()
            attribution = local.attribution(x[ids]).detach()
            summary = {}
            for stratum in np.unique(strata[ids]):
                selected = strata[ids] == stratum
                participants = len(np.unique(d['group'][ids][selected]))
                if participants >= 5:
                    summary[int(stratum)] = (attribution[selected].mean(0), participants)
            summaries.append(summary)
            clients.append(local.state_dict())
            weights.append(len(np.unique(d['group'][ids])))
        weights = np.array(weights) / np.sum(weights)
        state = model.state_dict()
        for key in state:
            state[key] = sum(float(weight) * client[key] for weight, client in zip(weights, clients))
        model.load_state_dict(state)
        prototypes = {}
        for stratum in set().union(*(s.keys() for s in summaries)):
            items = [s[stratum] for s in summaries if stratum in s]
            mean = sum(value * count for value, count in items) / sum(count for _, count in items)
            prototypes[stratum] = mean / mean.sum().clamp_min(1e-8)
        model.eval()
        all_attr = model.attribution(x[source]).detach().mean(0)
        model.source_attribution.copy_(all_attr)
        validation = d['split'] == 'val'
        with torch.no_grad():
            probability = model.net(x[validation]).softmax(-1)[:, 1].numpy()
        v = pd.DataFrame(dict(group=d['group'][validation], y=d['y'][validation], probability=probability)).groupby('group').mean(numeric_only=True)
        score = roc_auc_score(v.y, v.probability)
        curve.append(dict(round=round_id, validation_auroc=score, communicated_values=sum(len(s) * x.shape[1] for s in summaries)))
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), out / 'model.pt')
        else:
            stale += 1
        if stale >= c['patience']:
            break
    pd.DataFrame(curve).to_csv(out / 'learning.csv', index=False)
    if c['variant'] == 'personalized':
        for corpus in np.unique(d['corpus'][source]):
            ids = np.flatnonzero(source & (d['corpus'] == corpus))
            personal = copy.deepcopy(model).train()
            optimizer = torch.optim.AdamW(personal.parameters(), lr=c['learning_rate'])
            global_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
            for _ in range(c['local_epochs']):
                loss = F.cross_entropy(personal.net(x[ids]), y[ids])
                loss += c['personal_weight'] / 2 * sum((parameter - global_parameters[name]).square().sum() for name, parameter in personal.named_parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.save(personal.state_dict(), out / f'client_{corpus}.pt')
