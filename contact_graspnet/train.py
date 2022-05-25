from genericpath import exists
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm

CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'models'))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'utils'))

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import config_utils
from shapenetsem_data import PointCloudReader, load_scene_contacts, center_pc_convert_cam
from summaries import build_summary_ops, build_file_writers
from tf_train_ops import load_labels_and_losses, build_train_op
from contact_grasp_estimator import GraspEstimator

def train(global_config, log_dir):
    """
    Trains Contact-GraspNet

    Arguments:
        global_config {dict} -- config dict
        log_dir {str} -- Checkpoint directory
    """

    contact_infos = load_scene_contacts(global_config['DATA']['data_path'], split='train', max_num_grasps=None)
    
    num_train_samples = len(contact_infos)
    print('using %s meshes' % num_train_samples)

    pcreader = PointCloudReader(
        dataset_folder=global_config['DATA']['data_path'],
        batch_size=global_config['OPTIMIZER']['batch_size'],
        split='train',
        n_points=None,  # i.e. all point clouds will be given in original, varying size
    )

    with tf.Graph().as_default():
        
        # Build the model
        grasp_estimator = GraspEstimator(global_config)
        ops = grasp_estimator.build_network()
        
        # contact_tensors = load_contact_grasps(contact_infos, global_config['DATA'])
        
        loss_ops = load_labels_and_losses(grasp_estimator, contact_infos, global_config)

        ops.update(loss_ops)
        ops['train_op'] = build_train_op(ops['loss'], ops['step'], global_config)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=1)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Log summaries
        summary_ops = build_summary_ops(ops, sess, global_config)

        # Init/Load weights
        grasp_estimator.load_weights(sess, saver, log_dir, mode='train')

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        file_writers = build_file_writers(sess, log_dir)

    ## define: epoch = arbitrary number of views of every training scene
    cur_epoch = sess.run(ops['step']) // num_train_samples
    for epoch in range(cur_epoch, global_config['OPTIMIZER']['max_epoch']):
        log_string('**** EPOCH %03d ****' % (epoch))
        
        sess.run(ops['iterator'].initializer)
        epoch_time = time.time()
        step = train_one_epoch(sess, ops, summary_ops, file_writers, pcreader)
        log_string('trained epoch {} in: {}'.format(epoch, time.time()-epoch_time))

        if epoch % 10 == 0:
            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=step, write_meta_graph=False)
            log_string("Model saved in file: %s" % save_path)

            # there is some bug in validation, have no clue what, skip for now
            # eval_time = time.time()
            # eval_validation_scenes(sess, ops, summary_ops, file_writers, pcreader)
            # log_string('evaluation time: {}'.format(time.time()-eval_time))

def train_one_epoch(sess, ops, summary_ops, file_writers, pcreader):
    """ ops: dict mapping from string to tf ops """
    
    log_string(str(datetime.now()))
    loss_log = np.zeros((10,7))
    get_time = time.time()

    num_batches = len(pcreader.shapes)
    for batch_idx in range(num_batches):
        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=batch_idx)

        # OpenCV OpenGL conversion
        cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)

        feed_dict = {ops['pointclouds_pl']: batch_data, ops['cam_poses_pl']: cam_poses,
                     ops['scene_idx_pl']: scene_idx, ops['is_training_pl']: True}

        step, summary, _, loss_val, dir_loss, bin_ce_loss, \
        offset_loss, approach_loss, adds_loss, adds_gt2pred_loss, scene_idx, print_op = sess.run([ops['step'], summary_ops['merged'], ops['train_op'], ops['loss'], ops['dir_loss'],
                                                                                        ops['bin_ce_loss'], ops['offset_loss'], ops['approach_loss'], ops['adds_loss'], 
                                                                                        ops['adds_gt2pred_loss'], ops['scene_idx'], ops['print_op']], feed_dict=feed_dict)
        assert scene_idx[0] == scene_idx
        
        loss_log[batch_idx%10,:] = loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
        
        if (batch_idx+1)%10 == 0:
            file_writers['train_writer'].add_summary(summary, step)
            f = tuple(np.mean(loss_log, axis=0)) + ((time.time() - get_time) / 10., )
            log_string('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % f)
            get_time = time.time()
            
    return step

def eval_validation_scenes(sess, ops, summary_ops, file_writers, pcreader, max_eval_objects=500):
    """ ops: dict mapping from string to tf ops """
    """ simple validation: using last 20 training samples for evaluation, only view 0 """
    n_test_samples = 20

    is_training = False
    log_string(str(datetime.now()))
    loss_log = np.zeros((min(n_test_samples, max_eval_objects), 7))

    # resets accumulation of pr and auc data
    sess.run(summary_ops['pr_reset_op'])
    
    for batch_idx in np.arange(min(n_test_samples, max_eval_objects)):
        scene_idx = len(pcreader.shapes) - 1 - batch_idx
        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=scene_idx, view=0)

        # OpenCV OpenGL conversion
        cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)

        feed_dict = {ops['pointclouds_pl']: batch_data, ops['cam_poses_pl']: cam_poses,
                     ops['scene_idx_pl']: scene_idx, ops['is_training_pl']: False}

        scene_idx, step, loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss, pr_summary,_,_,_ = sess.run([ops['scene_idx'], ops['step'], ops['loss'], ops['dir_loss'], ops['bin_ce_loss'],
                                                                                                        ops['offset_loss'], ops['approach_loss'], ops['adds_loss'], ops['adds_gt2pred_loss'],
                                                                                                        summary_ops['merged_eval'], summary_ops['pr_update_op'], 
                                                                                                        summary_ops['auc_update_op']] + [summary_ops['acc_update_ops']], feed_dict=feed_dict)
        assert scene_idx[0] == len(pcreader.shapes) - 1 - batch_idx, f'{scene_idx[0]} == {len(pcreader.shapes) - 1 - batch_idx}'
        # todo: for some reason this fires. scene_idx[0] is 0, whereas the other idx is 194 (as it should be)... why?
        
        loss_log[batch_idx,:] = loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss

    file_writers['test_writer'].add_summary(pr_summary, step)
    f = tuple(np.mean(loss_log, axis=0))
    log_string('mean val loss: %f \t mean val dir loss: %f \t mean val ce loss: %f \t mean off loss: %f \t mean app loss: %f \t mean adds loss: %f \t mean adds_gt2pred loss: %f' % f)

    return step

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/contact_graspnet', help='Checkpoint dir')
    parser.add_argument('--data_path', type=str, default=None, help='Grasp data root dir')
    parser.add_argument('--max_epoch', type=int, default=None, help='Epochs to run')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    ckpt_dir = FLAGS.ckpt_dir
    if not os.path.exists(ckpt_dir): 
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            ckpt_dir = os.path.join(BASE_DIR, ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('cp {} {}'.format(os.path.join(CONTACT_DIR, 'contact_graspnet.py'), ckpt_dir)) # bkp of model def
    os.system('cp {} {}'.format(os.path.join(CONTACT_DIR, 'train.py'), ckpt_dir)) # bkp of train procedure

    LOG_FOUT = open(os.path.join(ckpt_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    global_config = config_utils.load_config(ckpt_dir, batch_size=FLAGS.batch_size, max_epoch=FLAGS.max_epoch, 
                                          data_path= FLAGS.data_path, arg_configs=FLAGS.arg_configs, save=True)
    
    log_string(str(global_config))
    log_string('pid: %s'%(str(os.getpid())))

    train(global_config, ckpt_dir)

    LOG_FOUT.close()
