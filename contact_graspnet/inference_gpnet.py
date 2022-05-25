import os
import sys
import argparse
import numpy as np
import quaternion
import time
import glob
import cv2
import mayavi.mlab as mlab

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
import visualization_utils
# to load GPNet exr files
import Imath
import OpenEXR
import matplotlib.pyplot as plt


def load_depth(exrpath):
    # gpnet data utility
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    return np.reshape(rgb[0], (Size[1], Size[0]))


def read_camera_parameter(camera_info_file, view):
    # gpnet data utility
    data = np.load(camera_info_file)[view]
    K = data['calibration_matrix'].reshape(3, 3)
    pose = np.eye(4)
    pose[0:3, 3] = data['position']
    pose[0:3, 0:3] = quaternion.as_rotation_matrix(quaternion.from_float_array(data['orientation']))

    # transform camera pose from opengl to opencv (flip y and z)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return K, pose


def remove_ground_plane_from_pc(pc, camera_pose, z_threshold=0.002):
    # plot_pc(pc)
    pc_ext = np.concatenate([pc, np.ones((len(pc), 1))], axis=1).T  # 4, N
    pc_world = camera_pose @ pc_ext
    pc_world = pc_world[:3, :].T  # N, 3
    # plot_pc(pc_world)
    above_z_idx = np.nonzero(pc_world[:, 2] > z_threshold)
    # plot_pc(pc_world[above_z_idx])
    return pc[above_z_idx]


def plot_pc(point_cloud, use_mayavi=True):
    print(f'displaying point cloud: {point_cloud.shape}')
    if use_mayavi:
        mlab.figure('point cloud')
        mlab.view(azimuth=180, elevation=180, distance=0.2)
        visualization_utils.plot_coordinates(np.zeros(3), np.eye(3, 3))
        mlab.points3d(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], mode='point')
        mlab.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = point_cloud[:, 0]
        ys = point_cloud[:, 1]
        zs = point_cloud[:, 2]
        ax.scatter(xs, ys, zs, c='r', marker='o')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()


def inference(
    global_config, 
    checkpoint_dir, 
    data_dir, 
    test_shapes,  # list of shapes to test
    view,  # view to test
    forward_passes=1,
    remove_ground_plane=False,
    max_z=1.2,
    sub_dir=None):

    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    result_dir = os.path.join('results-gpnet', sub_dir or '', f'view{view}')
    os.makedirs(result_dir, exist_ok=True)

    summary = {}

    # Process test shapes
    for curr_shape in test_shapes:
        # curr_shape is name of current tested shape
        exrpath = os.path.join(data_dir, 'images', curr_shape, f'render{view}Depth0001.exr')
        camera_info_file = os.path.join(data_dir, 'images', curr_shape, 'CameraInfo.npy')
        
        depth = load_depth(exrpath)
        camera_matrix, camera_pose = read_camera_parameter(camera_info_file, view)
        pc_full, pc_colors = depth2pc(depth, camera_matrix, rgb=None)
        keep_idx = pc_full[:, 2] < max_z
        pc_full = pc_full[keep_idx]
        print(f'** shape: {curr_shape}, view: {view}, pc_full: {pc_full.shape} (after applying z_max={max_z})')

        if remove_ground_plane:
            pc_full = remove_ground_plane_from_pc(pc_full, camera_pose)
            # plot_pc(pc_full)
            print(f'** shape: {curr_shape}, view: {view}, pc_full: {pc_full.shape} (after removing ground plane)')

        print('Generating Grasps...')
        pc_segments = {}
        local_regions = False
        filter_grasps = False
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
            sess, pc_full, pc_segments=pc_segments, local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        key = -1
        grasps = pred_grasps_cam[key]
        score = scores[key]
        len_before_filter = len(grasps)

        # only keep grasps with score above 0.5
        confident_preds = score > 0.5
        grasps = grasps[confident_preds]
        score = score[confident_preds]
        len_after_filter = len(grasps)
        summary[curr_shape] = {
            'before': len_before_filter,
            'after': len_after_filter
        }

        # arrange in ascending order by score
        order = np.argsort(-score)
        grasps = grasps[order]
        score = score[order]

        print(f'Grasps: {len_before_filter} total, {len_after_filter} with score above 0.5')
        print('- transforming grasps to world coordinates')
        grasps = camera_pose @ grasps

        print('- changing grasp frame convention from hand to TCP-based')
        hand_to_ee_offset = 0.103  # offset along z axis
        offsets = grasps[:, :3, 2] * hand_to_ee_offset
        grasps[:, :3, 3] += offsets
        grasps[:, :3, 1:3] *= -1  # swap the y and z axes

        # need to save centers, quaternions and score
        centers = grasps[:, :3, 3]
        quats = quaternion.as_float_array(quaternion.from_rotation_matrix(grasps[:, :3, :3]))
        save_file = os.path.join(result_dir, f'{curr_shape}.npz')
        np.savez(save_file, centers=centers, quaternions=quats, scores=score)
        print('- saved predictions to', save_file)
        # Visualize results
        # visualization_utils.visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    for key, val in summary.items():
        print(f'{val["after"]}/{val["before"]} - {key}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    # gpnet data
    parser.add_argument('--data_dir', nargs="*", type=str, default='/data/datasets/GPNet_release_data', help='path to gpnet data')
    parser.add_argument('--sub_dir', type=str, default=None)
    parser.add_argument('--view', type=int, default=0, help='view to test on')
    parser.add_argument('--remove_ground_plane', action='store_true', default=False, help='will remove ground plane from point clouds')

    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))
    
    # load gpnet test shapes list
    fn = os.path.join(FLAGS.data_dir[0], 'test_set.csv')
    assert os.path.exists(fn), "cannot find gpnet test_set.csv"
    file = open(fn, 'r')
    lines = file.readlines()
    test_shapes = []
    for line in lines:
        shape = line.strip().split('.')[0]
        if shape == '4eefe941048189bdb8046e84ebdc62d2':
            # this shape is present in the .csv file but not in the actual data root
            continue
        test_shapes.append(shape)
    file.close()
    print(f'testing {len(test_shapes)} shapes')
    
    inference(
        global_config, 
        FLAGS.ckpt_dir, 
        FLAGS.data_dir[0], 
        test_shapes,  # list of shapes to test
        FLAGS.view,  # view to test
        forward_passes=FLAGS.forward_passes,
        remove_ground_plane=FLAGS.remove_ground_plane,
        sub_dir=FLAGS.sub_dir
    )


