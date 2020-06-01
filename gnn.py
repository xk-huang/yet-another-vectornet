# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
# %%

__author__ = "xiaoke huang"
__email__ = "xiaokehuang@foxmail.com"


def get_fc_edge_index(num_nodes):
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        edge_index = np.hstack((edge_index, np.vstack((from_, to_))))
    return edge_index.astype(np.int64)


class HGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_subgraph_layres=3, num_global_graph_layer=1, subgraph_width=64, global_graph_width=64):
        super(HGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layres, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape)
        self.traj_pred_mlp = TrajPredMLP(
            self.polyline_vec_shape, out_channels, global_graph_width)

    def forward(self, data):
        """
        args: 
            data in 
                polyline_features: (np.ndarray), vstack[(xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?), (xs, ys, zs, xe, ye, ze, polyline_id)],
                traj_id2mask: Dict[int, int],
                lane_id2mask: Dict[int, int]
        """
        # [FIXME]: low speed due to
        #   1. unbatchify
        #   2. unable to compute each polyline vector set in parallel
        #   3. `edge_index` generating has high-time cost

        all_in_features = data['POLYLINE_FEATURES'].values[0]
        traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
        add_len = data['TARJ_LEN'].values[0]
        agent_id = 0
        assert all_in_features[agent_id][
            4] == 1, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

        # compute in subgraph
        polyline_features = []

        for id_, mask_ in traj_mask.items():
            data_ = all_in_features[mask_[0]:mask_[1]]
            edge_index_ = get_fc_edge_index(data_.shape[0])
            sub_data = Data(x=torch.from_numpy(data_),
                            edge_index=torch.from_numpy(edge_index_))

            subgraph_node_features = self.subgraph(sub_data)
            # print(f"{id_}'s max: {subgraph_node_features.max()}")
            # print(f"polyline feature shape: {subgraph_node_features.shape}")
            polyline_features.append(subgraph_node_features)

        for id_, mask_ in lane_mask.items():
            data_ = all_in_features[mask_[0]+add_len: mask_[1]+add_len]
            edge_index_ = get_fc_edge_index(data_.shape[0])
            sub_data = Data(x=torch.from_numpy(data_),
                            edge_index=torch.from_numpy(edge_index_))

            subgraph_node_features = self.subgraph(sub_data)
            # print(f"{id_}'s max: {subgraph_node_features.max()}")
            # pdb.set_trace()
            # print(f"polyline feature shape: {subgraph_node_features.shape}")
            polyline_features.append(subgraph_node_features)

        polyline_features = torch.stack(polyline_features)
        assert not torch.isnan(polyline_features.max()
                               ).any(), "nan in polyline features"
        polyline_features.retain_grad()  # not sure about stack
        # print(f"polyline features shape: {polyline_features.shape}")

        # compute in global interaction graph
        node_features = self.self_atten_layer(polyline_features)
        # print(polyline_features.shape)
        # print(self.traj_pred_mlp)

        # predict with mlp (predict only one trajecotry)
        out = self.traj_pred_mlp(node_features[agent_id])

        return out

# %%


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, in_channels)
        self.k_lin = nn.Linear(in_channels, in_channels)
        self.v_lin = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # print(x.shape)
        # print(self.q_lin)
        q_x = self.q_lin(x)
        k_x = self.k_lin(x)
        v_x = self.v_lin(x)
        atten = F.softmax(torch.matmul(
            q_x, k_x.t() / 1+int(np.sqrt(self.in_channels))), dim=1)
        return torch.matmul(atten, v_x)


class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)


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
            sub_data: torch_geometric.data.Data(x, edge_index)
        """
        x, edge_index = sub_data.x, sub_data.edge_index
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        node_feature, _ = torch.max(x, dim=0)
        # l2 noramlize node_feature before feed it to global graph
        node_feature = node_feature / node_feature.norm(dim=0)
        return node_feature

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
        # print(x.shape)
        x = self.mlp(x)
        aggr_x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        # print(x)
        # print(aggr_x)
        assert x.shape == aggr_x.shape, "aggr shape not the same."

        return torch.cat([x, aggr_x], dim=1)

    def message(self, x_j):
        return x_j

    def update(self, x):
        return x


# %%
if __name__ == "__main__":
    def get_data_path_ls(dir_):
        return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

    # data preparation
    DIR = 'input_data'
    data_path_ls = get_data_path_ls(DIR)

    # hyper parameters
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, out_channels = 7, 60
    show_every = 10

    # get model
    model = HGNN(in_channels, out_channels).to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    # overfit the small dataset
    # [FIXME]: unparallel batch processing
    model.train()
    for epoch in range(epochs):
        accum_loss = .0
        loss = None
        np.random.shuffle(data_path_ls)
        for sample_id, data_p in enumerate(data_path_ls):
            data = pd.read_pickle(data_p)
            y = data['GT'].values[0].reshape(-1).astype(np.float32)
            y = torch.from_numpy(y)

            # No Batch stuffs
            # optimizer.zero_grad()
            # out = model(data)
            # loss = F.mse_loss(out, y)
            # accum_loss += loss.item()
            # loss.backward()
            # optimizer.step()

            if sample_id % batch_size == 0:
                if loss:
                    accum_loss += loss.item()

                    loss /= batch_size
                    loss.backward()
                    optimizer.step()
                    loss = None
                optimizer.zero_grad()

            out = model(data)
            sample_loss = F.mse_loss(out, y)
            loss = sample_loss if loss is None else sample_loss + loss
        if sample_id % batch_size == 0:
            if loss:
                accum_loss += loss.item()

                loss /= batch_size
                loss.backward()
                optimizer.step()
                loss = None
            optimizer.zero_grad()

        scheduler.step()
        print(
            f"loss at epoch {epoch}: {accum_loss / len(data_path_ls):.3f}, lr{optimizer.state_dict()['param_groups'][0]['lr']: .3f}")

    # eval result on the identity dataset
    model.eval()
    data_path_ls = get_data_path_ls(DIR)

    with torch.no_grad():
        accum_loss = .0
        for sample_id, data_p in enumerate(data_path_ls):
            print(f"sample id: {sample_id}")
            data = pd.read_pickle(data_p)
            y = data['GT'].values[0].reshape(-1).astype(np.float32)
            y = torch.from_numpy(y)

            out = model(data)
            loss = F.mse_loss(out, y)

            accum_loss += loss.item()
            print(f"loss for sample {sample_id}: {loss.item():.3f}")
            show_predict_result(data, out, y, data['TARJ_LEN'].values[0])
            plt.show()

    print(f"eval overall loss: {accum_loss / len(data_path_ls):.3f}")
    # print(y)

    # edge_index = torch.tensor(
    #     [[1, 2, 0, 2, 0, 1],
    #         [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    # x = torch.tensor([[3, 1, 2], [2, 3, 1], [1, 2, 3]], dtype=torch.float)
    # y = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    # data = Data(x=x, edge_index=edge_index, y=y)

    # data = pd.read_pickle('./input_data/features_4791.pkl')
    # all_in_features_, y = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].reshape(-1).astype(np.float32)
    # traj_mask_, lane_mask_ = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
    # y = torch.from_numpy(y)
    # in_channels, out_channels = all_in_features_.shape[1], y.shape[0]
    # print(f"all feature shape: {all_in_features_.shape}, gt shape: {y.shape}")
    # print(f"len of trajs: {traj_mask_}, len of lanes: {lane_mask_}")

# %%


# %%
