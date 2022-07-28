"""
3D data processing tools
"""

import open3d as o3d
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pylab as p


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """
    def __init__(self, width, height, fx, fy, cx, cy, z_near=0.01, z_far=10):

        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.z_near = z_near
        self.z_far = z_far

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    @classmethod
    def set_projection(cls, intrinsics_matrix, z_range, image_size):
        fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
        cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
        return cls(image_size[1], image_size[0], fx, fy, cx, cy, z_range[0], z_range[1])

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

    def __str__(self):
        return "fx: {}, fy: {}, cx:{}, cy:{}. width:{}, height:{}".format(self.fx, self.fy, self.cx,
                                                                          self.cy, self.width, self.height)


class PointCloud(o3d.geometry.PointCloud):
    def __init__(self):
        super(PointCloud, self).__init__()

    @staticmethod
    def from_depth(depth, intrinsics):
        """
        Creates a point cloud from a depth image given the camera
        intrinsics parameters.

        Parameters
        ----------
        depth: np.array
            The input image.
        intrinsics: PinholeCameraIntrinsics object
            Intrinsics parameters of the camera.

        Returns
        -------
        o3d.geometry.PointCloud
            The point cloud of the scene.
        """
        depth = depth
        height, width = depth.shape
        c, r = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
        valid = (depth > 0)
        z = np.where(valid, depth, 0)
        x = np.where(valid, z * (c - intrinsics[0, 2]) / intrinsics[0, 0], 0)
        y = np.where(valid, z * (r - intrinsics[1, 2]) / intrinsics[1, 1], 0)
        pcd = np.dstack((x, y, z))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))


class VoxelGrid:
    def __init__(self, resolution):
        self.resolution = resolution
        self.voxelized_pcd = o3d.geometry.PointCloud()
        self.leaf_size = 0
        self.min_pt = []

    def voxelize(self, points, normals, leaf_size=None, min_pt=None):

        if leaf_size is None:
            min_pt = np.min(points, 0)
            max_pt = np.max(points, 0)
            leaf_size = max(max_pt[0] - min_pt[0],
                            max_pt[1] - min_pt[1],
                            max_pt[2] - min_pt[2]) / (self.resolution - 1)
            leaf_size = round(leaf_size, 4)

        self.leaf_size = leaf_size
        self.min_pt = min_pt

        # Generate occupancy grid
        occupancy_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        normals_grid = np.zeros((3, self.resolution, self.resolution, self.resolution), dtype=np.float32)
        for i in range(points.shape[0]):
            pt = points[i]

            # Compute voxel's coordinates
            x = int((pt[0] - min_pt[0]) / leaf_size)
            y = int((pt[1] - min_pt[1]) / leaf_size)
            z = int((pt[2] - min_pt[2]) / leaf_size)

            if x >= self.resolution or x < 0 or \
               y >= self.resolution or y < 0 or \
               z >= self.resolution or z < 0:
                continue

            if np.isnan(normals[i]).any():
                print('Nan')

            occupancy_grid[x, y, z] = 1.0
            normals_grid[:, x, y, z] += normals[i]

        pts = []
        for x, y, z in itertools.product(*map(range, (self.resolution,
                                                      self.resolution,
                                                      self.resolution))):
            if occupancy_grid[x, y, z] == 1.0:
                pt = np.array([x, y, z]) * leaf_size + min_pt
                pts.append(pt)

                # Normalize normals grid
                normals_grid[:, x, y, z] /= np.linalg.norm(normals_grid[:, x, y, z])

        self.voxelized_pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
        self.voxelized_pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))

        distance_field, known_regions = self.compute_sdf(occupancy_grid, normals_grid)
        # direction_angles = np.arccos(normals_grid) / np.pi

        out_volume = np.zeros((5, self.resolution, self.resolution, self.resolution))
        out_volume[0] = distance_field
        out_volume[1] = known_regions
        out_volume[2:5] = normals_grid

        return out_volume

    def compute_sdf(self, occupancy, normals):
        surface_pts = np.argwhere(occupancy == 1)
        surface_normals = np.zeros((surface_pts.shape[0], 3))
        for i in range(len(surface_pts)):
            p = surface_pts[i]
            surface_normals[i, :] = normals[:, p[0], p[1], p[2]]

        # diagonal = np.sqrt(2) * self.resolution

        df = np.zeros(occupancy.shape)
        known_regions = np.zeros(occupancy.shape)
        for x, y, z in itertools.product(*map(range, (self.resolution, self.resolution, self.resolution))):
            if occupancy[x, y, z] == 1.0:
                continue

            # Compute the projection on surface points
            p = np.array([x, y, z])
            dist = np.linalg.norm(surface_pts - p, axis=1)
            min_id = np.argmin(dist)

            # Compute distance field
            df[x, y, z] = dist[min_id]

            # Compute sign bit
            ds = (p - surface_pts[min_id]).astype(float)
            ds /= np.linalg.norm(ds)

            if np.dot(ds, surface_normals[min_id]) < 0:
                known_regions[x, y, z] = 1

        return df, known_regions


class VoxelGrid2:
    def __init__(self):
        self.voxel_size = 0
        self.min_bound = []
        self.max_bound = []
        self.occupancy_grid = []
        self.normals_grid = []

    def generate_sdf(self, occupancy, normals):
        if len(self.occupancy_grid) == 0:
            raise Exception

        surface_pts = np.argwhere(occupancy == 1)
        surface_normals = np.zeros((surface_pts.shape[0], 3))
        for i in range(len(surface_pts)):
            p = surface_pts[i]
            surface_normals[i, :] = normals[:, p[0], p[1], p[2]]

        diagonal = np.sqrt(2) * self.resolution

        df = np.zeros(occupancy.shape, dtype=np.float32)
        known_regions = np.zeros(occupancy.shape, dtype=np.float32)
        for x, y, z in itertools.product(*map(range, (self.resolution,
                                                      self.resolution,
                                                      self.resolution))):
            if occupancy[x, y, z] == 1.0:
                continue

            # Compute the projection on surface points
            p = np.array([x, y, z])
            dist = np.linalg.norm(surface_pts - p, axis=1)
            min_id = np.argmin(dist)

            # Compute distance field
            df[x, y, z] = np.min(dist) / diagonal

            # Compute sign bit
            ds = (p - surface_pts[min_id]).astype(float)
            if np.linalg.norm(ds) != 0:
                ds /= np.linalg.norm(ds)

            if np.dot(ds, surface_normals[min_id]) < 0:
                known_regions[x, y, z] = 1

        return df, known_regions

    def voxelize(self, points, normals, voxel_size, min_bound, resolution):
        # Generate occupancy grid
        self.occupancy_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        self.normals_grid = np.zeros((3, resolution, resolution, resolution), dtype=np.float32)
        for i in range(points.shape[0]):
            pt = points[i]

            # Compute voxel's coordinates
            x = int((pt[0] - min_bound[0]) / voxel_size)
            y = int((pt[1] - min_bound[1]) / voxel_size)
            z = int((pt[2] - min_bound[2]) / voxel_size)

            if x >= resolution or x < 0 or y >= resolution or y < 0 or z >= resolution or z < 0:
                continue

            self.occupancy_grid[x, y, z] = 1.0
            self.normals_grid[:, x, y, z] += normals[i]

        vox_pts = []
        vox_normals = []
        for x, y, z in itertools.product(*map(range, (resolution, resolution, resolution))):
            if self.occupancy_grid[x, y, z] == 1.0:
                pt = np.array([x, y, z]) * voxel_size + min_bound
                vox_pts.append(pt)

                self.normals_grid[:, x, y, z] /= np.linalg.norm(self.normals_grid[:, x, y, z])
                vox_normals.append(self.normals_grid[:, x, y, z])

        return {'points': vox_pts, 'vox_normals': vox_normals}

    def create_from_point_cloud(self, point_cloud, resolution):
        points = point_cloud['points']
        normals = point_cloud['normals']

        # Compute min and max bound
        min_bound = np.min(points, 0)
        max_bound = np.max(points, 0)
        voxel_size = max(max_bound[0] - min_bound[0],
                         max_bound[1] - min_bound[1],
                         max_bound[2] - min_bound[2]) / (resolution - 1)
        # Compute voxel size
        voxel_size = round(voxel_size, 4)

        return self.voxelize(points, normals, voxel_size, min_bound, resolution)

    def create_from_point_cloud_within_bounds(self, point_cloud, voxel_size, min_bound, max_bound):
        points = point_cloud['points']
        normals = point_cloud['normals']

        resolution = (max_bound - min_bound / voxel_size)

        return self.voxelize_2(points, normals, voxel_size, min_bound, resolution)

    def get_fused_volume(self):
        distance_field, known_regions = self.generate_sdf(self.occupancy_grid, self.normals_grid)
        direction_angles = np.arccos(self.normals_grid) / np.pi

        resolution = distance_field.shape[0]
        fused_volume = np.zeros((5, resolution, resolution, resolution))
        fused_volume[0] = distance_field
        fused_volume[1] = known_regions
        fused_volume[2:5] = direction_angles
        return fused_volume


def get_reconstructed_surface(pred_sdf, pred_normals, leaf_size, min_pt, x=None, epsilon=0.02):
    if x is not None:
        pred_sdf, pred_normals = match_up_input(pred_df=pred_sdf, pred_normals=pred_normals,
                                                partial_df=x[0], partial_normals=x[2:], known_regions=x[1])

    n_grid = np.cos(pred_normals * np.pi) / np.linalg.norm(np.cos(pred_normals * np.pi))

    # Compute 0-isosurface
    occupancy = np.zeros(pred_sdf.shape, dtype=np.float32)
    occupancy[np.abs(pred_sdf) < epsilon] = 1.0

    points = []
    normals = []
    resolution = occupancy.shape[0]
    for x, y, z in itertools.product(*map(range, (resolution,
                                                  resolution,
                                                  resolution))):
        local_grid = occupancy[max(0, x - 1):min(resolution, x + 2),
                               max(0, y - 1):min(resolution, y + 2),
                               max(0, z - 1):min(resolution, z + 2)]

        if occupancy[x, y, z] == 1.0 and len(np.argwhere(local_grid == 1)) > 1:

            pt = np.array([x, y, z]) * leaf_size + min_pt
            points.append(pt)

            # Normalize normals grid
            normal = np.cos(pred_normals[:, x, y, z] * np.pi) / np.linalg.norm(np.cos(pred_normals[:, x, y, z] * np.pi))
            normals.append(normal)

    # if plot:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     v = occupancy
    #     x, y, z = v.nonzero()
    #     ax.scatter(x, y, z, zdir='z', s=1, c='red')
    #
    #     v = partial_occupancy
    #     x, y, z = v.nonzero()
    #     ax.scatter(x, y, z, zdir='z', s=1, c='blue')
    #
    #     ax.auto_scale_xyz([0, 32], [0, 32], [0, 32])
    #     plt.show()

    return points, normals


def match_up_input(partial_df, pred_df, partial_normals, pred_normals, known_regions):
    resolution = partial_df.shape[0]
    for x, y, z in itertools.product(*map(range, (resolution,
                                                  resolution,
                                                  resolution))):
        if known_regions[x, y, z] == 0:
            pred_df[x, y, z] = partial_df[x, y, z]
            pred_normals[:, x, y, z] = partial_normals[:, x, y, z]
    return pred_df, pred_normals



