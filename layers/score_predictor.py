from typing import Callable

import torch
import torch.nn as nn
from torch.nn.modules.module import T


class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores):
        super().__init__()
        self.W1 = nn.Linear(3 * in_features, hidden_edge_scores)
        self.W2 = nn.Linear(hidden_edge_scores, 1)

    def apply_edges(self, edges):
        print("score_predictor apply_edges")
        data = torch.cat([edges.src['x'], edges.dst['x'], edges.data['e']], dim=1)
        print("-----------------------------------")
        print(edges.dst['x'].shape)
        print(edges.src['x'].shape)
        print(data.shape)
        h = self.W1(data)
        h = torch.relu(h)
        score = self.W2(h)
        return {'score': score}

    def forward(self, graph, x, e):
        print("score_predictor forward")
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
