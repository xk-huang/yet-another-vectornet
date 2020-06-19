# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from modeling.vectornet import HGNN
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from dataset import GraphDataset, GraphData
from torch_geometric.data import DataLoader
from utils.config import INTERMEDIATE_DATA_DIR
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from pprint import pprint
from utils.eval import get_eval_metric_results

# %%
DATASET_DIR = os.path.join(INTERMEDIATE_DATA_DIR,
                           'forecasting_sample_intermediate')
SEED = 13
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
decay_lr_factor = 0.7
decay_lr_every = 20
lr = 0.005
in_channels, out_channels = 8, 60
show_every = 10
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # hyper parameters

    ds = GraphDataset(DATASET_DIR).shuffle()
    train_loader = DataLoader(ds, batch_size=batch_size)
    val_loader = DataLoader(ds, batch_size=batch_size)

    model = HGNN(in_channels, out_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    model = model.to(device=device)

    # overfit the small dataset
    model.train()
    for epoch in range(epochs):
        acc_loss = .0
        for data in train_loader:
            data = data.to(device)
            y = data.y.view(-1, out_channels).to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, y)
            loss.backward()
            acc_loss += batch_size * loss.item()
            optimizer.step()
        print(
            f"loss at epoch {epoch}: {acc_loss / len(ds):.3f}, lr{optimizer.state_dict()['param_groups'][0]['lr']: .3f}")

    # eval result on the identity dataset
    # model.eval()
    # with torch.no_grad():
    #     accum_loss = .0
    #     from utils.viz_utils import show_pred_and_gt
    #     for sample_id, data in enumerate(train_loader):
    #         data = data.to(device)
    #         gt = data.y.view(-1, out_channels).to(device)
    #         # optimizer.zero_grad()
    #         out = model(data)
    #         loss = F.mse_loss(out, gt)
    #         accum_loss += batch_size * loss.item()
    #         print(f"loss for sample {sample_id}: {loss.item():.3f}")

    #         for i in range(gt.size(0)):
    #             pred_y = out[i].numpy().reshape((-1, 2)).cumsum(axis=0)
    #             y = gt[i].numpy().reshape((-1, 2)).cumsum(axis=0)
    #             show_pred_and_gt(pred_y, y)
    #             plt.show()
    #     print(f"eval overall loss: {accum_loss / len(ds):.3f}")

    # eval metrics
    get_eval_metric_results(model, train_loader, device, out_channels,
                            max_n_guesses, horizon, miss_threshold)


# %%
