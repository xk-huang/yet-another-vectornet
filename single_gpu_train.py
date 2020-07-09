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
import time
from typing import List

# %%
TRAIN_DIR = os.path.join('interm_data', 'train_intermediate')
VAL_DIR = os.path.join('interm_data', 'val_intermediate')
SEED = 13
epochs = 50
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
batch_size = 4096
decay_lr_factor = 0.3
decay_lr_every = 10
lr = 0.001
in_channels, out_channels = 8, 60
show_every = 10
val_every = 5
small_dataset = False
end_epoch = 0
save_dir = 'trained_params'
best_minade = float('inf')
date = f"200630.epochs{epochs}.lr_decay{decay_lr_factor}.decay_every{decay_lr_every}.lr{lr}"
global_step = 0
# checkpoint_dir = os.path.join('trained_params', 'epoch_6.valminade_3.796.pth')
checkpoint_dir = None
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0


#%%
#%%
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
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
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"xkhuang"}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


#%%
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # hyper parameters

    train_data = GraphDataset(TRAIN_DIR).shuffle()
    val_data = GraphDataset(VAL_DIR)
    if small_dataset:
        train_loader = DataLoader(train_data[:1000], batch_size=batch_size)
        val_loader = DataLoader(val_data[:200], batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)

    model = HGNN(in_channels, out_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    if checkpoint_dir:
        load_checkpoint(checkpoint_dir, model, optimizer)

    # overfit the small dataset
    model.train()
    for epoch in range(epochs):
        print(f"start training at epoch:{epoch}")
        acc_loss = .0
        num_samples = 1
        start_tic = time.time()
        for data in train_loader:
            if epoch < end_epoch: break
            if isinstance(data, List):
                y = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
            else:
                data = data.to(device)
                y = data.y.view(-1, out_channels)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, y)
            loss.backward()
            acc_loss += batch_size * loss.item()
            num_samples += y.shape[0]
            optimizer.step()
            global_step += 1
            if (global_step + 1) % show_every == 0:
                print( f"epoch {epoch} step {global_step}： loss:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"finished epoch {epoch}: loss:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        
        if (epoch+1) % val_every == 0 and (not epoch < end_epoch):
            print("eval as epoch:{epoch}")
            metrics = get_eval_metric_results(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
            curr_minade = metrics["minADE"]
            print(f"minADE:{metrics['minADE']:3f}, minFDE:{metrics['minFDE']:3f}, MissRate:{metrics['MR']:3f}")

            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date)
                
    # eval result on the identity dataset
    metrics = get_eval_metric_results(model, val_loader, device, out_channels, max_n_guesses, horizon, miss_threshold)
    curr_minade = metrics["minADE"]
    if curr_minade < best_minade:
        best_minade = curr_minade
        save_checkpoint(save_dir, model, optimizer, -1, best_minade, date)


# %%
