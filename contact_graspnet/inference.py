import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
from scipy.spatial.transform import Rotation as scipy_rot

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from shapenetsem_data import PointCloudReader, transform_grasps_hand_to_TCP, make_canonical_orientation

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=False, skip_border_objects=False, filter_grasps=False, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    # data loader
    pcreader = PointCloudReader(
        dataset_folder=global_config['DATA']['data_path'],
        batch_size=1,
        split='test',
        n_points=None,
        in_world_coords=True
    )
    num_test_samples = len(pcreader.shapes)

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
    
    os.makedirs('results', exist_ok=True)
    result_dir = 'results/epoch500'
    os.makedirs(result_dir, exist_ok=True)

    # Process example test scenes
    for test_shape_idx in [0]:  # range(num_test_samples):
        shape = pcreader.shapes[test_shape_idx]
        for view in [0]:
            view_dir = os.path.join(result_dir, f'view{view}')
            os.makedirs(view_dir, exist_ok=True)

            pc, _, _ = pcreader.get_scene_batch(test_shape_idx, view=view)
            print(f'loaded test shape {shape} view {view} with pc {pc.shape}')

            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, _ = \
                grasp_estimator.predict_scene_grasps(sess, pc, pc_segments={}, local_regions=local_regions,
                                                     filter_grasps=filter_grasps, forward_passes=forward_passes)

            # convert grasps correspondingly
            # pred_grasps_TCP = transform_grasps_hand_to_TCP(pred_grasps_cam)  # transform from panda_hand to TCP
            # pred_grasps = make_canonical_orientation(pred_grasps_TCP)
            pred_grasps = pred_grasps_cam

            # for direct inspection
            np.savez('results/predictions_{}_v{}2.npz'.format(shape, view), pred_grasps=pred_grasps, scores=scores,
                     contact_pts=contact_pts, pc_cam=pc.squeeze())

            all_grasps_path = os.path.join(view_dir, shape + '.npz')
            rots = scipy_rot.from_matrix(pred_grasps[:, 0:3, 0:3])
            quats = rots.as_quat()  # x, y, z, w
            quats = quats[:, [3, 0, 1, 2]]  # wxyz

            centers = pred_grasps[:, 0:3, 3]
            widths = np.zeros(shape=(len(pred_grasps)))
            np.savez(all_grasps_path, widths=widths, centers=centers, quaternions=quats, scores=scores)

            # Visualize results
            # visualize_grasps(pc, pred_grasps_cam, scores, plot_opencv_cam=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--data_path', type=str, default=None, help='Grasp data root dir')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

