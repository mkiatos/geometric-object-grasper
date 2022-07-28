import open3d as o3d
import numpy as np
import cv2
import os
import shutil
import pickle
import yaml

from gog.utils import pybullet_utils


def get_pointcloud(depth, seg, intrinsics):
    """
    Creates a point cloud from a depth image given the camera intrinsics parameters.

    Parameters
    ----------
    depth: np.array
        The input image.
    intrinsics: np.array(3, 3)
        Intrinsics parameters of the camera.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud of the scene.
    """
    depth = depth
    width, height = depth.shape
    c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - intrinsics[0, 2]) / intrinsics[0, 0], 0)
    y = np.where(valid, z * (r - intrinsics[1, 2]) / intrinsics[1, 1], 0)
    pcd = np.dstack((x, y, z))

    # colors = np.zeros((seg.shape[0], seg.shape[1], 3))
    # colors[:, :, 0] = seg / np.max(seg)
    # colors[:, :, 1] = seg / np.max(seg)
    # colors[:, :, 2] = np.zeros((seg.shape[0], seg.shape[1]))

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))
    # point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

    return point_cloud


def get_aligned_point_cloud(color, depth, seg, configs, bounds, pixel_size, plot=False):
    """
    Returns the scene point cloud aligned with the center of the workspace.
    """
    full_point_cloud = o3d.geometry.PointCloud()
    for color, depth, seg, config in zip(color, depth, seg, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        point_cloud = get_pointcloud(depth, seg, intrinsics)

        transform = pybullet_utils.get_camera_pose(config['pos'],
                                                   config['target_pos'],
                                                   config['up_vector'])
        point_cloud = point_cloud.transform(transform)

        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
                                                       max_bound=np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]))
        point_cloud = point_cloud.crop(crop_box)

        full_point_cloud += point_cloud

    # full_point_cloud.estimate_normals()
    if plot:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([full_point_cloud, mesh_frame])
    return full_point_cloud


def get_fused_heightmap(obs, configs, bounds, pix_size):
    point_cloud = get_aligned_point_cloud(obs['color'], obs['depth'], obs['seg'], configs, bounds, pix_size)
    xyz = np.asarray(point_cloud.points)

    # Compute heightmap size
    heightmap_size = np.round(((bounds[1][1] - bounds[1][0]) / pix_size,
                               (bounds[0][1] - bounds[0][0]) / pix_size)).astype(int)

    height_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)

    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        idx_x = int(np.floor((x + bounds[0][1]) / pix_size))
        idx_y = int(np.floor((y + bounds[1][1]) / pix_size))

        if 0 < idx_x < heightmap_size[0] - 1 and 0 < idx_y < heightmap_size[1] - 1:
            if height_grid[idx_y][idx_x] < z:
                height_grid[idx_y][idx_x] = z

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(height_grid)
    # ax[1].imshow(seg_grid)
    # plt.show()

    return cv2.flip(height_grid, 1)


def sample_distribution(prob, rng, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    rand_ind = rng.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        # Create the log directory
        if os.path.exists(log_dir):
            print('Directory ', log_dir, 'exists, do you want to remove it? (y/n)')
            answer = input('')
            if answer == 'y':
                shutil.rmtree(log_dir)
                os.mkdir(log_dir)
            else:
                exit()
        else:
            os.mkdir(log_dir)

    def log_data(self, data, filename):
        pickle.dump(data, open(os.path.join(self.log_dir, filename), 'wb'))

    def log_yml(self, dict, filename):
        with open(os.path.join(self.log_dir, filename + '.yml'), 'w') as stream:
            yaml.dump(dict, stream)
