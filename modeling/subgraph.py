from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, max_pool
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os


class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        x, edge_index = sub_data.x, sub_data.edge_index
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        assert out_data.x.shape[0] % sub_data.time_step_len[0] == 0

        out_data.x = out_data.x / out_data.x.norm(dim=0)
        return out_data

        # node_feature, _ = torch.max(x, dim=0)
        # # l2 noramlize node_feature before feed it to global graph
        # node_feature = node_feature / node_feature.norm(dim=0)
        # return node_feature

# %%


class GraphLayerProp(MessagePassing):
    def __init__(self, in_channels, hidden_unit=64):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x, edge_index):
        x = self.mlp(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return torch.cat([x, aggr_out], dim=1)
