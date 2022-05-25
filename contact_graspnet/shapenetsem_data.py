import os
import math

import OpenEXR
import Imath
import cv2
import numpy as np
import trimesh.transformations as tra

from data import regularize_pc_point_count


def load_scene_contacts(dataset_folder, split='train', max_num_grasps=None):
    """
    Load contact grasp annotations from acronym scenes

    Arguments:
        dataset_folder {str} -- folder with ShapeNetSem-8 data

    Keyword Arguments:
        split {str} -- 'train' or 'test'
        max_num_grasps {int} -- maximum number of grasp annotations to load

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

        if max_num_grasps is not None and grasp_centers.shape[0] > max_num_grasps:
            idcs = np.random.choice(grasp_centers.shape[0], max_num_grasps, replace=False)
            contact_points = contact_points[idcs]
            grasp_centers = grasp_centers[idcs]
            quaternions = quaternions[idcs]

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


def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud

    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds
    :returns: (cam_poses, batch_data) converted
    """
    # okay, let's don't do anything right here. it's a mess...
    return cam_poses, batch_data

    # OpenCV OpenGL conversion
    for j in range(len(cam_poses)):
        cam_poses[j, :3, 1] = -cam_poses[j, :3, 1]
        cam_poses[j, :3, 2] = -cam_poses[j, :3, 2]
        cam_poses[j] = inverse_transform(cam_poses[j])

    pc_mean = np.mean(batch_data, axis=1, keepdims=True)
    batch_data[:, :, :3] -= pc_mean[:, :, :3]
    cam_poses[:, :3, 3] -= pc_mean[:, 0, :3]

    return cam_poses, batch_data


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


class PointCloudReader:
    """
    Class to load the views (=scene point clouds).
    Does not support any augmentation.

    Arguments:
        dataset_folder {str} -- ShapeNetSem-8 root folder
        batch_size {int} -- number of rendered point clouds per-batch (1)
        split {str} -- 'train' or 'test'
        in_world_coords {bool} -- if true, point clouds are transformed in world coordinates, else in cam coords
    """
    def __init__(self, dataset_folder, batch_size=1, split='train', n_points=None, in_world_coords=False):
        assert split == 'test' or split == 'train', f'unknown split {split}'
        assert batch_size == 1

        split_file = os.path.join(dataset_folder, f'{split}_set.csv')
        with open(split_file, 'r') as f:
            shapes = f.read().splitlines()

        self.shapes = shapes
        self.batch_size = batch_size
        self.split = split
        self.n_points = n_points
        self.in_world_coords = in_world_coords
        self.images_dir = os.path.join(dataset_folder, 'images')

    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False, view=None):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})
            view {int} -- index of view (default: {None}, i.e. random)

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        assert not return_segmap
        assert not save

        if scene_idx is None:
            scene_idx = np.random.randint(0, len(self.shapes))
        shape = self.shapes[scene_idx]
        cam_info_path = os.path.join(self.images_dir, shape, 'CameraInfo.npy')
        cam_info = self.read_camera_info(cam_info_path)

        if view is None:
            camera_pose, view = self.random_camera_view(cam_info)
        else:
            camera_pose = cam_info['view%d' % view]

        pc = self.get_point_cloud_from_cam_info(shape, camera_pose, view)
        if self.n_points is not None:
            pc = regularize_pc_point_count(pc, self.n_points)
        # if pc.shape[0] > self.max_points:
        #     idcs = np.random.choice(pc.shape[0], self.max_points, replace=False)
        #     pc = pc[idcs]

        # pc = self.prune_and_normalize(pc)  # perhaps don't mess with the scale, as this might be done in CGN?
        # also we are in cam coordinates now, so the pruning and normalisation might not be meaningful

        if self.in_world_coords:
            cam_pose = np.eye(4)
        else:
            cam_pos, cam_quat = camera_pose[0], camera_pose[1]
            cam_pose = tra.quaternion_matrix(cam_quat)
            cam_pose[0:3, 3] = cam_pos
        cam_poses = cam_pose[None, :]  # add dim for batch size  # todo: openCV or OpenGL cam pose??
        # currently our z-axis is pointing away from the origin/scene, y-axis upwards
        batch_data = pc[None, :]  # add dim for batch size

        return batch_data, cam_poses, scene_idx

    @staticmethod
    def read_camera_info(camera_info_file):
        camera_info_array = np.load(camera_info_file)
        cam_info = {}
        for item in camera_info_array:
            cam_info[item['id'].decode()] = (item['position'], item['orientation'], item['calibration_matrix'])
        return cam_info

    @staticmethod
    def random_camera_view(cameraInfoDict=None):
        view_num = len(cameraInfoDict)
        view = np.random.choice(view_num, 1)[0]
        return cameraInfoDict['view%d' % (view)], view

    def get_point_cloud_from_cam_info(self, shape, camera_pose, view, shelf_th=0.002):
        """
        Retrieve unit cube normalized point cloud from camera info.
        @param shape: ??
        @param camera_pose: ??
        @param view: ??
        @return: the normalized pc
        """
        ca_loc = camera_pose[0]
        ca_ori = camera_pose[1]
        intrinsic = camera_pose[2].reshape(3, 3)

        exr_img = os.path.join(self.images_dir, shape, 'render%dDepth0001.exr' % view)
        File = OpenEXR.InputFile(exr_img)
        PixType = Imath.PixelType(Imath.PixelType.FLOAT)
        DW = File.header()['dataWindow']
        Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
        rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
        depth = np.reshape(rgb[0], (Size[1], Size[0]))

        org_size = depth.shape
        depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)

        # convert it to point cloud and remove inf/nan points, remove plane, crop, ...
        if self.in_world_coords:
            pc = self.Depth2PointCloud(depth, intrinsic, ca_ori, ca_loc, org_size=org_size).transpose()
            # remove the undefined and outlier points
            inf_idx = (pc != pc) | (np.abs(pc) > 100)
            pc[inf_idx] = 0.0

            # need to prune the shelf --> remove all points below a certain threshold on the z axis
            above_shelf_indexes = np.nonzero(pc[:, 2] > shelf_th)[0]
            pc = pc[above_shelf_indexes]

            # also prune everything that is not roughly within a bounding box (as plane removal fails for far away pts)
            tensor_x = pc[:, 0]
            tensor_y = pc[:, 1]
            tensor_z = pc[:, 2]
            del_idx = (tensor_x < -0.22 / 2) | (tensor_x > 0.22 / 2) | (tensor_y < -0.22 / 2) | (
                        tensor_y > 0.22 / 2) | (
                              tensor_z > 0.22)
            pc = pc[del_idx == False]

            return pc
        else:
            # this is a very memory-intensive hack to remove the ground plane but retrieve the pc in cam coordinates
            cam_pc = self.Depth2PointCloud(depth, intrinsic, org_size=org_size).transpose()
            pc = self.Depth2PointCloud(depth, intrinsic, ca_ori, ca_loc, org_size=org_size).transpose()

            # remove the undefined and outlier points
            inf_idx = (pc != pc) | (np.abs(pc) > 100)
            cam_pc[inf_idx] = 0.0

            # need to prune the shelf --> remove all points below a certain threshold on the z axis
            above_shelf_indexes = np.nonzero(pc[:, 2] > shelf_th)[0]
            cam_pc = cam_pc[above_shelf_indexes]
            pc = pc[above_shelf_indexes]

            # also prune everything that is not roughly within a bounding box (as plane removal fails for far away pts)
            tensor_x = pc[:, 0]
            tensor_y = pc[:, 1]
            tensor_z = pc[:, 2]
            del_idx = (tensor_x < -0.22 / 2) | (tensor_x > 0.22 / 2) | (tensor_y < -0.22 / 2) | (tensor_y > 0.22 / 2) | (
                    tensor_z > 0.22)
            cam_pc = cam_pc[del_idx == False]

            return cam_pc

    def Depth2PointCloud(self, dmap, K, orientation=None, position=None, mask=None, org_size=None):
        '''
        K: (3, 3)
        orientation: (4,) quaternion (w, x, y, z)
        position: (3,)
        return (3, n)
        '''

        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        heigh, width = int(dmap.shape[0]), int(dmap.shape[1])
        [x, y] = np.meshgrid(np.arange(0, width), np.arange(0, heigh))
        # print('x shape:', x.ravel().shape)
        # print('y shape:', y.shape)
        if org_size is not None:
            org_h = org_size[0]
            org_w = org_size[1]
        else:
            org_h = heigh
            org_w = width

        cx = cx * width / org_w
        cy = cy * heigh / org_h

        x3 = (x - cx) * dmap * 1.0 / fx * org_w / width
        y3 = (y - cy) * dmap * 1.0 / fy * org_h / heigh
        z3 = dmap

        if mask is not None:
            y_idx, x_idx = mask.nonzero()
            x3 = x3[y_idx, x_idx]
            y3 = y3[y_idx, x_idx]
            z3 = z3[y_idx, x_idx]
        # print('x3:', x3.shape)
        # 3, n
        pc = np.stack([x3.ravel(), -y3.ravel(), -z3.ravel()], axis=0)

        if orientation is not None and position is not None:
            ex = self.QuaternionToMatrix(orientation, position)
            # 4, n
            pc_one = np.concatenate([pc, np.ones((1, pc.shape[1]))], 0)
            # 3, 4 x 4, n
            # ex_inv = np.linalg.inv(ex)
            # ex_inv = ex
            pc_one = ex @ pc_one
            pc = pc_one[:3, :]
        return pc

    def QuaternionToMatrix(self, quaternion, translation, first_w=True):
        # return 4,4
        qw, qx, qy, qz = quaternion

        if not first_w:
            qx, qy, qz, qw = quaternion

        n = 1.0 / math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx *= n
        qy *= n
        qz *= n
        qw *= n

        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz
        mat[0, 1] = 2.0 * qx * qy - 2.0 * qz * qw
        mat[0, 2] = 2.0 * qx * qz + 2.0 * qy * qw

        mat[1, 0] = 2.0 * qx * qy + 2.0 * qz * qw
        mat[1, 1] = 1.0 - 2.0 * qx * qx - 2.0 * qz * qz
        mat[1, 2] = 2.0 * qy * qz - 2.0 * qx * qw

        mat[2, 0] = 2.0 * qx * qz - 2.0 * qy * qw
        mat[2, 1] = 2.0 * qy * qz + 2.0 * qx * qw
        mat[2, 2] = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy

        mat[:3, 3] = translation
        mat[3, 3] = 1

        return mat

    def prune_and_normalize(self, tensor):
        """
        Scale the GPNet data to a unit dimension.
        Moreover, it prunes some points out of the bounds
        @param tensor: the tensor to be normalized
        @return:
        """
        tensor_x = tensor[:, 0]
        tensor_y = tensor[:, 1]
        tensor_z = tensor[:, 2]
        del_idx = (tensor_x < -0.22 / 2) | (tensor_x > 0.22 / 2) | (tensor_y < -0.22 / 2) | (tensor_y > 0.22 / 2) | (
                tensor_z > 0.22)
        tensor = tensor[del_idx == False]
        tensor = tensor / np.array([0.22 / 2, 0.22 / 2, 0.22])
        return tensor


if __name__ == '__main__':
    import burg_toolkit as burg
    dataset_dir = '/home/martin/datasets/ShapeNetSem-8/'
    split = 'test'
    # contacts = load_scene_contacts(dataset_dir, split=split)
    pcreader = PointCloudReader(dataset_dir, split=split)

    for view in [0, 1, 2, 3]:
        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=0, view=view)
        pc = batch_data[0]
        cam_pose = cam_poses[0]
        # frame = burg.visualization.create_frame(size=0.1, pose=cam_pose)
        burg.visualization.show_geometries([pc])




