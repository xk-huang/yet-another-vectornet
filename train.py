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
from dataset import GraphDataset
from torch_geometric.data import DataLoader
from utils.eval import get_eval_metric_results
from tqdm import tqdm
import torch_geometric.nn as nn

# %%
TRAIN_DIR = os.path.join('interm_data', 'train_intermediate')
VAL_DIR = os.path.join('interm_data', 'val_intermediate')
gpus = [2, 3]
torch.cuda.set_device(f'cuda:{gpus[0]}')
SEED = 13
epochs = 25
device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
batch_size = 4096 * len(gpus)
decay_lr_factor = 0.3
decay_lr_every = 5
lr = 0.001
in_channels, out_channels = 8, 60
show_every = 10
val_every = 1
small_dataset = False
end_epoch = 0
save_dir = 'trained_params'
best_minade = float('inf')
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0


#%%
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        'val_minade': val_minade
        }
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    return checkpoint_path['end_epoch']

#%%
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # hyper parameters

    train_data = GraphDataset(TRAIN_DIR).shuffle()
    val_data = GraphDataset(VAL_DIR)
    if small_dataset:
        train_loader = DataLoader(train_data[:1000], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data[:200], batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

    model = HGNN(in_channels, out_channels).to(device)
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    # model = model.to(device=device)

    # overfit the small dataset
    model.train()
    for epoch in range(epochs):
        acc_loss = .0
        scheduler.step()
        if epoch < end_epoch:
            continue
        for data in train_loader:
            data = data.to(device)
            y = data.y.view(-1, out_channels)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, y)
            loss.backward()
            acc_loss += batch_size * loss.item()
            optimizer.step()
        print(
            f"loss at epoch {epoch}: {acc_loss / len(train_loader):.3f}, lr{optimizer.state_dict()['param_groups'][0]['lr']: .3f}")
        if (epoch+1) % val_every == 0:
            metrics = get_eval_metric_results(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
            curr_minade = metrics["minADE"]
            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade)

    # eval result on the identity dataset
    metrics = get_eval_metric_results(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
    curr_minade = metrics["minADE"]
    if curr_minade < best_minade:
        best_minade = curr_minade
        save_checkpoint(save_dir, model, optimizer, -1, best_minade)
    # model.eval()
    # from utils.viz_utils import show_pred_and_gt
    # with torch.no_grad():
    #     accum_loss = .0
    #     for sample_id, data in enumerate(train_loader):
    #         data = data.to(device)
    #         gt = data.y.view(-1, out_channels).to(device)
    #         optimizer.zero_grad()
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


# %%
