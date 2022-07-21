import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import open3d as o3d

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
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


def rgb2bgr(rgb):
    """
    Converts a rgb image to bgr

    Parameters
    ----------
    rgb : np.array
        The rgb image

    Returns
    -------
    np.array:
        The image in bgr format
    """
    h, w, c = rgb.shape
    bgr = np.zeros((h, w, c), dtype=np.uint8)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    def get_intrinsic_matrix(self):
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def get_focal_length(self):
        return self.fx, self.fy

    def get_principal_point(self):
        return self.cx, self.cy

    def back_project(self, p, z):
        x = (p[0] - self.cx) * z / self.fx
        y = (p[1] - self.cy) * z / self.fy
        return np.array([x, y, z])


def min_max_scale(x, range, target_range):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    range_min = range[0] * np.ones(x.shape)
    range_max = range[1] * np.ones(x.shape)
    target_min = target_range[0] * np.ones(x.shape)
    target_max = target_range[1] * np.ones(x.shape)

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)


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


def get_reconstructed_surface(partial_df, partial_normals, known_regions,
                              pred_df, pred_normals, epsilon=0.03):
    # Match up input
    # resolution = partial_df.shape[0]
    # for x, y, z in itertools.product(*map(range, (resolution,
    #                                               resolution,
    #                                               resolution))):
    #     if known_regions[x, y, z] == 0:
    #         pred_df[x, y, z] = partial_df[x, y, z]
    #         pred_normals[:, x, y, z] = partial_normals[:, x, y, z]

    partial_occupancy = np.zeros(pred_df.shape, dtype=np.float32)
    partial_occupancy[np.abs(partial_df) < epsilon] = 1.0
    # Compute 0-isosurface
    occupancy = np.zeros(pred_df.shape, dtype=np.float32)
    occupancy[np.abs(pred_df) < epsilon] = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    v = occupancy
    x, y, z = v.nonzero()
    ax.scatter(x, y, z, zdir='z', s=1, c='red')

    v = partial_occupancy
    x, y, z = v.nonzero()
    ax.scatter(x, y, z, zdir='z', s=1, c='blue')

    ax.auto_scale_xyz([0, 32], [0, 32], [0, 32])
    plt.show()


def sample_surface(obb, name):
    box_points = np.asarray(obb.get_box_points())
    box_centroid = np.mean(box_points, axis=0)

    # combinations = [[0, 2, 1], [0, 1, 3], [0, 2, 3],
    #                 [4, 5, 6], [1, 7, 6], [2, 5, 7]]

    combinations = [[0, 2, 1]]
    # proximal: 0, 2, 1
    # distal: 0, 1, 3
    # palm 4, 5, 6

    surface_pts = []
    surface_normals = []
    for k in range(len(combinations)):
        p1 = box_points[combinations[k][0]]
        p2 = box_points[combinations[k][1]]
        p3 = box_points[combinations[k][2]]

        # These two vectors are in the plane
        v1 = p2 - p1
        d1 = np.linalg.norm(v1)
        v1 /= np.linalg.norm(v1)
        v2 = p3 - p1
        d2 = np.linalg.norm(v2)
        v2 /= np.linalg.norm(v2)

        normal = np.cross(v1, v2)

        centroid = (p2 + p3) / 2.0
        if (box_centroid - centroid).dot(normal) > 0:
            p2 = box_points[combinations[k][2]]
            p3 = box_points[combinations[k][1]]

            # These two vectors are in the plane
            v1 = p2 - p1
            d1 = np.linalg.norm(v1)
            v1 /= np.linalg.norm(v1)
            v2 = p3 - p1
            d2 = np.linalg.norm(v2)
            v2 /= np.linalg.norm(v2)

            normal = np.cross(v1, v2)

        leaf = min(d1, d2) / 4.0

        pts = []
        normals = []
        for i in range(1, int(d1/leaf)):
            for j in range(1, int(d2/leaf)-1):
                p = p1 + i * leaf * v1 + j * leaf * v2
                pts.append(p)
                normals.append(normal)

    surf_pts = {'points': pts, 'normals': normals}
    pickle.dump(surf_pts, open(name, 'wb'))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    return point_cloud