# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

# %%
from modeling.predmlp import TrajPredMLP
from modeling.selfatten import SelfAttentionLayer
from modeling.subgraph import SubGraph
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, max_pool
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.data import Data, DataLoader
from utils.viz_utils import show_predict_result
from dataset import GraphDataset
import sys
sys.path.append('..')
# %%
# %%


class HGNN(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, num_subgraph_layres=3, num_global_graph_layer=1, subgraph_width=64, global_graph_width=64, traj_pred_mlp_width=64):
        super(HGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layres, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, data):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        # from pdb import set_trace
        # set_trace()
        pred = self.traj_pred_mlp(out[:, [0]].squeeze(1))
        return pred


# %%
if __name__ == "__main__":
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, out_channels = 8, 60
    show_every = 10
    os.chdir('..')
    # get model
    model = HGNN(in_channels, out_channels).to(device)

    dataset = GraphDataset('.')
    data_iter = DataLoader(dataset, batch_size=batch_size)
    for data in data_iter:
        out = model(data)
# %%
    '''
    def get_data_path_ls(dir_):
        return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

    # data preparation
    DIR = 'input_data'
    data_path_ls = get_data_path_ls(DIR)

    # hyper parameters
    epochs = 100
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
    # print(f"len of trajs: {traj_mask_}, len of lanes: {lane_mask_}")'''

# %%


# %%
