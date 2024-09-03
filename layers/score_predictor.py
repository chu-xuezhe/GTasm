from typing import Callable

import torch
import torch.nn as nn
from torch.nn.modules.module import T


class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores):
        super().__init__()
        self.mlp1 = nn.Linear(3 * in_features, hidden_edge_scores)
        self.mlp2 = nn.Linear(hidden_edge_scores, 1)

    def score_edges(self, edges):
        print("scoreing...")
        data = torch.cat([edges.src['x'], edges.dst['x'], edges.data['e']], dim=1)
        h = self.mlp1(data)
        h = torch.relu(h)
        score = self.mlp2(h)
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.score_edges)
            return graph.edata['score']
