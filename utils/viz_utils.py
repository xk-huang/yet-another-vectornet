from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import color_dict


def show_doubled_lane(polygon):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    xs, ys = polygon[:, 0], polygon[:, 1]
    plt.plot(xs, ys, '--', color='grey')


def show_traj(traj, type_):
    """
    args: ndarray in shape of (n, 2)
    returns:
    """
    plt.plot(traj[:, 0], traj[:, 1], color=color_dict[type_])
