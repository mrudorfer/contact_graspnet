import os

import numpy as np
import trimesh.transformations as tra


def load_scene_contacts(dataset_folder, split='train'):
    """
    Load contact grasp annotations from acronym scenes

    Arguments:
        dataset_folder {str} -- folder with ShapeNetSem-8 data

    Keyword Arguments:
        split {str} -- 'train' or 'test'

    Returns:
        list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
    """

    assert split == 'test' or split == 'train', f'unknown split {split}'

    split_file = os.path.join(dataset_folder, f'{split}_set.csv')
    with open(split_file, 'r') as f:
        shapes = f.read().splitlines()

    contact_infos = []
    for i, shape in enumerate(shapes):
        # original contact info had:
        #     contact_info = {'scene_contact_points': npz['scene_contact_points'],
        #                     'obj_paths': npz['obj_paths'],
        #                     'obj_transforms': npz['obj_transforms'],
        #                     'obj_scales': npz['obj_scales'],
        #                     'grasp_transforms': npz['grasp_transforms']}
        # however, we only need contact points and grasp transforms of successful grasps
        # obj not needed, we load our own point clouds and don't render them

        # print(i, shape)
        annotations_folder = os.path.join(dataset_folder, 'annotations/candidate')
        contact_points = np.load(os.path.join(annotations_folder, f'{shape}_contact.npy'))
        grasp_centers = np.load(os.path.join(annotations_folder, f'{shape}_c.npy'))
        quaternions = np.load(os.path.join(annotations_folder, f'{shape}_q.npy'))

        # use only successful grasps
        sim_result = np.load(os.path.join(dataset_folder, f'annotations/simulateResult/{shape}.npy'))
        contact_points = contact_points[sim_result]
        grasp_centers = grasp_centers[sim_result]
        quaternions = quaternions[sim_result]

        # print('contact points', contact_points.shape)
        # print('centers', grasp_centers.shape)
        # print('quats', quaternions.shape)  # wxyz

        grasps = matrix_from_pos_quat(grasp_centers, quaternions)
        grasps = transform_grasps_TCP_to_hand(grasps)
        # print('grasps', grasps.shape)

        contact_info = {
            'scene_contact_points': contact_points,
            'grasp_transforms': grasps
        }
        contact_infos.append(contact_info)

    return contact_infos


def matrix_from_pos_quat(grasp_centers, quaternions):
    grasps = tra.quaternion_matrix(quaternions)
    grasps[:, 0:3, 3] = grasp_centers
    return grasps


def transform_grasps_hand_to_TCP(grasps, canonical_orientation=False):
    """ grasps: (n, 4, 4) """
    if canonical_orientation:
        raise NotImplementedError('need to implement this for rule-based evaluation')

    # transform from their gripper coordinate system https://github.com/NVlabs/6dof-graspnet/issues/8 to our
    # tool-center-point oriented system:
    #   1. apply offset in z axis of 10.3cm
    #   2. flip z and y axes
    grasps[:, 0:3, 3] += 0.103 * grasps[:, 0:3, 2]
    grasps[:, 0:3, 1] *= -1
    grasps[:, 0:3, 2] *= -1

    return grasps


def transform_grasps_TCP_to_hand(grasps):
    """ grasps: (n, 4, 4) """
    # this is actually the same formula, because z-axis is flipped
    grasps[:, 0:3, 3] += 0.103 * grasps[:, 0:3, 2]
    grasps[:, 0:3, 1] *= -1
    grasps[:, 0:3, 2] *= -1

    return grasps


if __name__ == '__main__':
    load_scene_contacts('/home/martin/datasets/ShapeNetSem-8/', split='test')

