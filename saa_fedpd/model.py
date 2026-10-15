import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn


class Model(nn.Module):
    def __init__(self, c, shape):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(nn.Linear(shape[0], 512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512, 128),
                                 nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 2))
        self.register_buffer('source_attribution', torch.zeros(shape[0]))
        self.compute_diagnostics = False

    def attribution(self, x, create_graph=False):
        with torch.enable_grad():
            value = x.detach().clone().requires_grad_(True)
            score = self.net(value)
            gradient = torch.autograd.grad((score[:, 1] - score[:, 0]).sum(), value, create_graph=create_graph)[0]
            return (value * gradient).abs()

    def forward(self, batch):
        x = batch['x'].float()
        result = dict(logits=self.net(x))
        if self.compute_diagnostics:
            attribution = self.attribution(x).detach()
            count = max(1, round(.1 * x.shape[1]))
            indices = attribution.topk(count, dim=1).indices
            deleted = x.clone().scatter_(1, indices, 0)
            base = result['logits'].softmax(-1)[:, 1]
            difference = (self.net(deleted).softmax(-1)[:, 1] - base).abs()
            random = torch.zeros_like(base)
            for _ in range(self.c['random_deletions']):
                indices = torch.rand_like(x).topk(count, dim=1).indices
                random += (self.net(x.clone().scatter_(1, indices, 0)).softmax(-1)[:, 1] - base).abs()
            result.update(attribution=attribution, deletion=difference,
                          random_deletion=random / self.c['random_deletions'],
                          source_attribution=self.source_attribution[None].expand(len(x), -1))
        return result


def extra_metrics(d, c):
    groups = d['target_group'].astype(str)
    frame = pd.DataFrame(dict(group=groups, y=d['target_y'], probability=d['probability'][:, 1]))
    participant = frame.groupby('group').agg(y=('y', 'first'), probability=('probability', 'mean'))
    result = dict(participant_auroc=roc_auc_score(participant.y, participant.probability),
                  participant_auprc=average_precision_score(participant.y, participant.probability),
                  participant_brier=brier_score_loss(participant.y, participant.probability))
    if 'attribution' in d:
        result['explanation_drift'] = 1 - spearmanr(d['source_attribution'][0], d['attribution'].mean(0)).statistic
        result['deletion_fidelity'] = float(d['deletion'].mean())
        result['random_deletion_change'] = float(d['random_deletion'].mean())
    return result
