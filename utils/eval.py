#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-06-18 22:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import torch
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from pprint import pprint


def get_eval_metric_results(model, data_loader, device, out_channels, max_n_guesses, horizon, miss_threshold):
    """
    ADE, FDE, and Miss Rate
    """
    forecasted_trajectories, gt_trajectories = {}, {}
    seq_id = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            y = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
            out = model(data)
            for i in range(gt.size(0)):
                pred_y = out[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
                y = gt[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
                forecasted_trajectories[seq_id] = [pred_y]
                gt_trajectories[seq_id] = y
                seq_id += 1
        metric_results = get_displacement_errors_and_miss_rate(
            forecasted_trajectories, gt_trajectories, max_n_guesses, horizon, miss_threshold
        )
        pprint(metric_results)
        return metric_results

def eval_loss():
    raise NotImplementedError("not finished yet")
    model.eval()
    from utils.viz_utils import show_pred_and_gt
    with torch.no_grad():
        accum_loss = .0
        for sample_id, data in enumerate(train_loader):
            data = data.to(device)
            gt = data.y.view(-1, out_channels).to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, gt)
            accum_loss += batch_size * loss.item()
            print(f"loss for sample {sample_id}: {loss.item():.3f}")

            for i in range(gt.size(0)):
                pred_y = out[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                y = gt[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                show_pred_and_gt(pred_y, y)
                plt.show()
        print(f"eval overall loss: {accum_loss / len(ds):.3f}")
