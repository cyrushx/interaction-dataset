#!/usr/bin/env python

try:
    import lanelet2
    use_lanelet2_lib = True
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details.\n" + \
             "Recommended to intall through ROS with python 2.\n" + \
             "Do not forget to run _source YOUR_ROS_WS/devel/setup.bash_ before running."
    warnings.warn(string)
    use_lanelet2_lib = False

import argparse
import glob
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils import dataset_reader

# use our own visualize scripts (linked from cvm)
from utils import visualize

def filter_by_distance(pos, centerlines, max_distance = 50):
    pos = np.array(pos)
    result = []
    for centerline in centerlines:
        cl_np = np.array(centerline)
        dist_sq = np.sum((cl_np - pos) ** 2, axis=1)
        if np.min(dist_sq) <= max_distance ** 2:
            result.append(centerline)
    return result

def filter_by_rel_goal_distance(pos_start, pos_end, centerlines):
    result = []
    for centerline in centerlines:
        centerline_end = centerline[-1]

        dist_start = (pos_start[0] - centerline_end[0]) ** 2 + (pos_start[1] - centerline_end[1]) ** 2
        dist_end = (pos_end[0] - centerline_end[0]) ** 2 + (pos_end[1] - centerline_end[1]) ** 2

        if dist_start > dist_end:
            result.append(centerline)
    return result

if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--track_length", type=int, default=50, help="Length of track to extract and visualize")
    parser.add_argument("--past_horizon", type=int, default=20, help="Length of observations")
    parser.add_argument("--future_horizon", type=int, default=30, help="Prediction horizon")
    parser.add_argument("--track_limit", type=int, default=10, help="Maximum of tracks to extract (0 means no limit)")
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")

    # check folders and files
    error_string = ""
    tracks_dir = "../recorded_trackfiles"
    maps_dir = "../maps"
    lanelet_map_ending = ".osm"
    lanelet_map_file = maps_dir + "/" + args.scenario_name + lanelet_map_ending
    scenario_dir = tracks_dir + "/" + args.scenario_name
    track_file_prefix = "vehicle_tracks_"
    track_file_ending = ".csv"
    track_file_name = scenario_dir + "/" + track_file_prefix + str(args.track_file_number).zfill(3) + track_file_ending

    visualize_dir = "../visualize_files"
    if not os.path.exists(visualize_dir):
        os.mkdir(visualize_dir)
    visualize_scenario_dir = visualize_dir + "/" + args.scenario_name
    if not os.path.exists(visualize_scenario_dir):
        os.mkdir(visualize_scenario_dir)

    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Did not find map file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Did not find lanelet map file \"" + lanelet_map_file + "\"\n"
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    # obtain map
    lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
    lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
    print("Loading map...")
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
    laneletmap = lanelet2.io.load(lanelet_map_file, projector)

    # obtain centerline
    centerlines = []
    for ll in laneletmap.laneletLayer:
        cl_points = [[pt.x, pt.y] for pt in ll.centerline]
        centerlines.append(cl_points)

    file_list=glob.glob(os.path.join(scenario_dir,'*.csv'))
    file_list.sort()
    print('Extract and visualize {} files'.format(len(file_list)))

    track_length = args.track_length
    past_horizon = args.past_horizon
    future_horizon = args.future_horizon
    assert past_horizon + future_horizon == track_length, 'Track length is not equal to past + future horizons.'

    track_limit = args.track_limit
    extract_count = 0
    for track_file_name in file_list:
        # load the tracks
        track_fname = os.path.split(track_file_name)[-1]
        print("Loading tracks: " + track_fname)
        track_dictionary = dataset_reader.read_tracks(track_file_name)

        # extract data for each track id
        for track_id in track_dictionary:
            track = track_dictionary[track_id]
            agent_type = track.agent_type
            agent_length, agent_width = track.length, track.width
            ts_first = track.time_stamp_ms_first
            ts_last = track.time_stamp_ms_last
            ts_delta = ts_last - ts_first
            dt_ms = 100
            print("Visualize track id: {} with size {} (start {}, end {})".format(track_id,
                                                                                  ts_delta // (track_length * dt_ms),
                                                                                  ts_first,
                                                                                  ts_last))

            for track_start in tqdm.tqdm(range(ts_first, ts_last, track_length * dt_ms)):
                if track_start + track_length * dt_ms > ts_last:
                    break

                track_x = [track.motion_states[ts].x for ts in range(track_start,
                                                                     track_start + track_length * dt_ms,
                                                                     dt_ms)]
                track_y = [track.motion_states[ts].y for ts in range(track_start,
                                                                     track_start + track_length * dt_ms,
                                                                     dt_ms)]
                track_t_ms = [track.motion_states[ts].time_stamp_ms for ts in range(track_start,
                                                                                    track_start + track_length * dt_ms,
                                                                                    dt_ms)]
                # visualize data
                fig, ax = plt.subplots()
                filtered_centerlines = filter_by_distance([track_x[past_horizon], track_y[past_horizon]],
                                                          centerlines,
                                                          max_distance=25)
                # TODO: find closest lane w/ normal distance and perform dfs to prune irrelevant lanes
                filtered_centerlines_final = filter_by_rel_goal_distance([track_x[0], track_y[0]],
                                                                         [track_x[past_horizon], track_y[past_horizon]],
                                                                         filtered_centerlines)
                visualize.draw_lanes(filtered_centerlines, linewidth=0.5)
                visualize.draw_lanes(filtered_centerlines_final, linewidth=2.0)

                visualize.draw_traj(track_x[:past_horizon], track_y[:past_horizon],
                    marker='.', color='c', alpha=1, markersize=3, zorder=15)
                visualize.draw_traj(track_x[past_horizon:], track_y[past_horizon:],
                    marker='.', color='m', alpha=1, markersize=3, zorder=15)

                # for prediction in predictions:
                #     pred = np.array(prediction)
                #     draw_traj(pred[:,0], pred[:,1],
                #         marker='.', color='0.7', alpha=1, markersize=2, zorder=15)

                ax.annotate('b', (track_x[0], track_y[0]))
                ax.annotate('f', (track_x[-1], track_y[-1]))

                plt.axis('equal')
                plt.plot()

                fname = os.path.join(visualize_scenario_dir, '{0}_{1:04d}_{2:08d}.png'.format(track_fname.split('.')[0],
                                                                                              track_id,
                                                                                              track_start))
                title = 'Track id: {0}, start ms: {1}, type: {2}, size: {3:.2f}x{4:.2f}'.format(track_id,
                                                                                                track_start,
                                                                                                agent_type,
                                                                                                agent_length,
                                                                                                agent_width)
                plt.title(title)
                fig.tight_layout()
                plt.savefig(fname, dpi=600)
                plt.close(fig)

                extract_count += 1
                if track_limit > 0 and extract_count > track_limit:
                    break

            if track_limit > 0 and extract_count > track_limit:
                break

        if track_limit > 0 and extract_count > track_limit:
            break
