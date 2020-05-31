# %%
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
%matplotlib inline

# %%
RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
VELOCITY_THRESHOLD = 1.0
# Number of timesteps the track should exist to be considered in social context
EXIST_THRESHOLD = (15)
# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = (13)
color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
lane_radius = 5
obj_radius = 20
root_dir = '../data/forecasting_sample/data/'


def parse_args() -> Any:
    pass


def compute_velocity(track_df: pd.DataFrame) -> List[float]:
    """Compute velocities for the given track.

    Args:
        track_df (pandas Dataframe): Data for the track
    Returns:
        vel (list of float): Velocity at each timestep

    """
    x_coord = track_df["X"].values
    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values
    vel_x, vel_y = zip(*[(
        x_coord[i] - x_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        y_coord[i] - y_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]

    return vel


def get_is_track_stationary(track_df: pd.DataFrame) -> bool:
    """Check if the track is stationary.

    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[int(len(vel) / 2)]
    return True if threshold_vel < VELOCITY_THRESHOLD else False


def fill_track_lost_in_middle(
        track_array: np.ndarray,
        seq_timestamps: np.ndarray,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Handle the case where the object exited and then entered the frame but still retains the same track id. It'll be a rare case.

    Args:
        track_array (numpy array): Padded data for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        raw_data_format (Dict): Format of the sequence
    Returns:
        filled_track (numpy array): Track data filled with missing timestamps

    """
    curr_idx = 0
    filled_track = np.empty((0, track_array.shape[1]))
    for timestamp in seq_timestamps:
        filled_track = np.vstack((filled_track, track_array[curr_idx]))
        if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
            curr_idx += 1
    return filled_track


def pad_track(
        track_df: pd.DataFrame,
        seq_timestamps: np.ndarray,
        obs_len: int,
        raw_data_format: Dict[str, int],
) -> np.ndarray:
    """Pad incomplete tracks.

    Args:
        track_df (Dataframe): Dataframe for the track
        seq_timestamps (numpy array): All timestamps in the sequence
        obs_len (int): Length of observed trajectory
        raw_data_format (Dict): Format of the sequence
    Returns:
            padded_track_array (numpy array): Track data padded in front and back

    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values

    # start and index of the track in the sequence
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates
    # if self.PADDING_TYPE == "REPEAT"
    padded_track_array = np.pad(track_vals,
                                ((start_idx, obs_len - end_idx - 1),
                                    (0, 0)), "edge")
    if padded_track_array.shape[0] < obs_len:
        padded_track_array = fill_track_lost_in_middle(
            padded_track_array, seq_timestamps, raw_data_format)

    # Overwrite the timestamps in padded part
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    return padded_track_array


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


def get_halluc_lane(centerlane, city_name):
    """
    return left & right lane based on centerline
    args:
    returns:
        doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
    """
    if centerlane.shape[0] <= 1:
        raise ValueError('shape of centerlane error.')

    half_width = LANE_WIDTH[city_name] / 2
    rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
    halluc_lane_1, halluc_lane_2 = np.empty(
        (0, centerlane.shape[1]*2)), np.empty((0, centerlane.shape[1]*2))
    for i in range(centerlane.shape[0]-1):
        st, en = centerlane[i][:2], centerlane[i+1][:2]
        dx = en - st
        norm = np.linalg.norm(dx)
        e1, e2 = rotate_quat @ dx / norm, rotate_quat.T @ dx / norm
        lane_1 = np.hstack(
            (st + e1 * half_width, centerlane[i][2], en + e1 * half_width, centerlane[i+1][2]))
        lane_2 = np.hstack(
            (st + e2 * half_width, centerlane[i][2], en + e2 * half_width, centerlane[i+1][2]))
        # print(halluc_lane_1, )
        halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
        halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
    return halluc_lane_1, halluc_lane_2


def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, has_attr=False, mode='nearby', **kwargs):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
    returns:
        list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    '''

    lane_feature_ls = []
    if mode == 'nearby':
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
            query_x, query_y, city_name, lane_radius)

        for lane_id in nearby_lane_ids:
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)

            centerlane = am.get_lane_segment_centerline(lane_id, city_name)
            # normalize to last observed timestamp point of agent
            centerlane[:, :2] -= norm_center
            halluc_lane_1, halluc_lane_2 = get_halluc_lane(
                centerlane, city_name)

            if has_attr:
                raise NotImplementedError()

            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id])
    elif mode == 'rect':
        raise NotImplementedError()
    else:
        raise ValueError(f"{mode} is not in {'rect', 'nearby'}")

    return lane_feature_ls
    # polygon = am.get_lane_segment_polygon(lane_id, city_name)
    # h_len = polygon.shape[0]
    # polygon = np.hstack(
    #     (polygon, is_intersection * np.ones((h_len, 1)), traffic_control * np.ones((h_len, 1))))
    # polygon_ls.append(polygon)


def get_nearby_moving_obj_feature_ls(agent_df, traj_df, obs_len, seq_ts, norm_center):
    """
    args:
    returns: list of list, (doubled_track, object_type, timestamp, track_id)
    """
    obj_feature_ls = []
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    p0 = np.array([query_x, query_y])
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if remain_df['OBJECT_TYPE'].iloc[0] == 'AGENT':
            continue

        if len(remain_df) < EXIST_THRESHOLD or get_is_track_stationary(remain_df):
            continue

        xys = None
        if len(remain_df) < obs_len:
            paded_nd = pad_track(remain_df, seq_ts, obs_len, RAW_DATA_FORMAT)
            xys = np.array(paded_nd[:, 3:5])
        else:
            xys = remain_df[['X', 'Y']].values
        p1 = xys[obs_len-1]
        if np.linalg.norm(p0 - p1) > obj_radius:
            continue

        xys -= norm_center  # normalize to last observed timestamp point of agent
        xys = np.hstack((xys[:-1], xys[1:]))
        ts = remain_df["TIMESTAMP"].values
        ts = (ts[:-1] + ts[1:]) / 2

        obj_feature_ls.append(
            [xys, remain_df['OBJECT_TYPE'].iloc[0], ts, track_id])
    return obj_feature_ls


def get_agent_feature_ls(agent_df, obs_len, norm_center):
    """
    args:
    returns: 
        list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
    """
    xys, gt_xys = agent_df[["X", "Y"]].values[:obs_len], agent_df[[
        "X", "Y"]].values[obs_len:]
    xys -= norm_center  # normalize to last observed timestamp point of agent
    gt_xys -= norm_center  # normalize to last observed timestamp point of agent
    xys = np.hstack((xys[:-1], xys[1:]))

    ts = agent_df['TIMESTAMP'].values[:obs_len]
    ts = (ts[:-1] + ts[1:]) / 2

    return [xys, agent_df['OBJECT_TYPE'].iloc[0], ts, agent_df['TRACK_ID'].iloc[0], gt_xys]
#######################################################
#######################################################
#######################################################


# %%
def compute_feature_for_one_seq(traj_df: pd.DataFrame, am: ArgoverseMap, obs_len: int = 20, lane_radius: int = 5, obj_radius: int = 10, viz: bool = False) -> List[List]:
    """
    return lane & track features
    args:
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    """
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    seq_len = seq_ts.shape[0]
    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [
        None] * 7
    # agent traj & its start/end point
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        if obj_type == 'AGENT':
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # prune points after "obs_len" timestamp
    traj_df = traj_df[traj_df['TIMESTAMP'] <
                      agent_df['TIMESTAMP'].values[obs_len]]

    assert (np.unique(traj_df["TIMESTAMP"].values).shape[0]
            == obs_len), "Obs len mismatch"

    # search nearby lane from the last observed point of agent
    # [!polygon_ls]
    lane_feature_ls = get_nearby_lane_feature_ls(am,
                                                 agent_df, obs_len, city_name, lane_radius, norm_center)

    # search nearby moving objects from the last observed point of agent
    obj_feature_ls = get_nearby_moving_obj_feature_ls(
        agent_df, traj_df, obs_len, seq_ts, norm_center)
    # get agent features
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)

    # vis
    if viz:
        for features in lane_feature_ls:
            show_doubled_lane(
                np.vstack((features[0][:, :2], features[0][-1, 3:5])))
            show_doubled_lane(
                np.vstack((features[1][:, :2], features[1][-1, 3:5])))
        for features in obj_feature_ls:
            show_traj(
                np.vstack((features[0][:, :2], features[0][-1, 2:])), features[1])
        show_traj(np.vstack(
            (agent_feature[0][:, :2], agent_feature[0][-1, 2:])), agent_feature[1])

        plt.plot(agent_x_end - query_x, agent_y_end - query_y, 'o',
                 color=color_dict['AGENT'], markersize=7)
        plt.plot(0, 0, 'x', color='blue', markersize=4)
        plt.plot(start_x-query_x, start_y-query_y,
                 'x', color='blue', markersize=4)
        plt.show()

    return [agent_feature, obj_feature_ls, lane_feature_ls]


def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.We rotate the coordinate system based on the heading of the target vehicle at the last observed location.
    """
    assert gt.shape == (30, 2), f"{gt.shape} is wrong"
    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
    # import pdb
    # pdb.set_trace()
    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

    return offset_gt


def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls):
    """
    args:
        agent_feature_ls:
            list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    returns:
        pd.DataFrame of (
            polyline_features: vstack[(xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?), (xs, ys, zs, xe, ye, ze, polyline_id)]
            offset_gt: incremental offset from agent's last obseved point,
            traj_id2mask: Dict[int, int]
            lane_id2mask: Dict[int, int]
        )
        where obejct_type = {0 - others, 1 - agent}

    """
    polyline_id = 0
    traj_id2mask, lane_id2mask = {}, {}
    gt = agent_feature[-1]
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 7))

    # encoding agent feature
    pre_traj_len = traj_nd.shape[0]
    agent_len = agent_feature[0].shape[0]
    # print(agent_feature[0].shape, np.ones(
    # (agent_len, 1)).shape, agent_feature[2].shape, (np.ones((agent_len, 1)) * polyline_id).shape)
    agent_nd = np.hstack((agent_feature[0], np.ones(
        (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((traj_nd, agent_nd))
    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1

    # encoding obj feature
    for obj_feature in obj_feature_ls:
        obj_len = obj_feature[0].shape[0]
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))

        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]
        l_lane_nd = np.hstack(
            (lane_feature[0], np.ones((l_lane_len, 1)) * polyline_id))
        assert l_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]
        r_lane_nd = np.hstack(
            (lane_feature[1], np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(gt)
    # print(traj_id2mask)
    # print(lane_id2mask)
    polyline_features = np.vstack((traj_nd, lane_nd))
    data = [[polyline_features, offset_gt, traj_id2mask, lane_id2mask]]
    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK"]
    )


def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )


# %%
if __name__ == "__main__":
    afl = ArgoverseForecastingLoader(root_dir)
    am = ArgoverseMap()
    for name in afl.seq_list:
        afl_ = afl.get(name)
        path, name = os.path.split(name)
        name, ext = os.path.splitext(name)

        agent_feature, obj_feature_ls, lane_feature_ls = compute_feature_for_one_seq(
            afl_.seq_df, am, 20, lane_radius, obj_radius, viz=True)
        df = encoding_features(agent_feature, obj_feature_ls, lane_feature_ls)
        save_features(df, name)

# # %%


# for name in afl.seq_list:
#     afl_ = afl.get(name)
#     path, name = os.path.split(name)
#     name, ext = os.path.splitext(name)
#     agent_feature, obj_feature_ls, lane_feature_ls = compute_feature_for_one_seq(
#         afl_.seq_df, am, 20, lane_radius, obj_radius, viz=True)
#     df = encoding_features(agent_feature, obj_feature_ls, lane_feature_ls)
#     save_features(df, name)
#     break

# # %%
# afl = ArgoverseForecastingLoader(root_dir)
# am = ArgoverseMap()
# PADDING_LEN = 5
# # %%
# for afl_ in afl:
#     compute_feature_for_one_seq(
#         afl_.seq_df, am, 20, lane_radius, obj_radius, viz=True)

# %%
