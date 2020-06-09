#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
# %%
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import root_dir, lane_radius, obj_radius, SAVE_DIR
# %matplotlib inline


if __name__ == "__main__":
    afl = ArgoverseForecastingLoader(root_dir)
    am = ArgoverseMap()
    for name in afl.seq_list:
        afl_ = afl.get(name)
        path, name = os.path.split(name)
        name, ext = os.path.splitext(name)

        agent_feature, obj_feature_ls, lane_feature_ls = compute_feature_for_one_seq(
            afl_.seq_df, am, 20, lane_radius, obj_radius, viz=True, mode='nearby')
        df = encoding_features(agent_feature, obj_feature_ls, lane_feature_ls)
        print(pd.DataFrame(df['POLYLINE_FEATURES'].values[0]).describe())
        save_features(df, name, SAVE_DIR)


# %%
