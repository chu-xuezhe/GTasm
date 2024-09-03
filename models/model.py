import torch
import torch.nn as nn

import dgl
import layers
from layers.layer import GT_processor




class MyTransformerModelNet(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers,
                 hidden_edge_scores, batch_norm, nb_pos_enc):
        super().__init__()

        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)  # 1 16
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)  # 16
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features)
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features)

        self.graph_transformer = GT_processor(num_layers, hidden_features)


        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):

        pe = self.linear1_node(pe)
        pe = torch.relu(pe)
        pe = self.linear2_node(pe)  # [n,256]
        print(f'pe.shape:{pe.shape}')
        x=pe

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        x,e=self.graph_transformer(graph,x,e)

        scores_t = self.predictor(graph, x, e)
        return scores_t
