"""
3D data processing tools
"""

import open3d as o3d
import numpy as np
import itertools
import matplotlib.pyplot as plt


def from_depth(depth, camera_intrinsics):
    """
    Creates a point cloud from a depth image given the camera
    intrinsics parameters.

    Parameters
    ----------
    depth: np.array
        The input image.
    camera_intrinsics: PinholeCameraIntrinsics object
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
    x = np.where(valid, z * (c - camera_intrinsics.cx) / camera_intrinsics.fx, 0)
    y = np.where(valid, z * (r - camera_intrinsics.cy) / camera_intrinsics.fy, 0)
    pcd = np.dstack((x, y, z))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))


def assign_labels(point_cloud, seg):
    class_ids = np.unique(seg)
    max_label = class_ids.max()
    color_ids = plt.get_cmap("tab20")(class_ids / (max_label if max_label > 0 else 1))

    labels = seg.flatten()

    colors = []
    for j in range(labels.shape[0]):
        color_id = [i for i, x in enumerate(class_ids) if x == labels[j]]
        colors.append(np.array(color_ids[color_id, 0:3][0]))

    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return point_cloud


def euclidean_clustering(point_cloud, eps, min_points=10, plot=False):
    """
    Performs euclidean clustering with DBSCAN algorithm

    Parameters
    __________
    point_cloud: o3d.geometry.PointCloud
        the input point cloud
    eps: float
        The maximum distance between two samples for
        one to be considered as in the neighborhood of the other.
    min_points: int
        The number of samples (or total weight) in a neighborhood
        for a point to be considered as a core point.
    plot: boolean
        Visualize the clusters with different color

    Returns
    -------
    list of o3d.geometry.PointCloud:
        the generated clusters
    """
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points,
                                                 print_progress=True))
    cluster_ids = np.unique(labels)
    clusters = []
    for i in range(len(cluster_ids)):
        if clusters_ids[i] == -1:
            continue
        ids = np.where(labels == cluster_ids[i])[0]
        clusters.append(point_cloud.select_by_index(indices=ids))

    if plot:
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([point_cloud, frame])

    return clusters


class VoxelGrid:
    def __init__(self, resolution):
        self.resolution = resolution
        self.map = {}

    def voxelize(self, point_cloud, leaf_size=0.01):
        grid = np.zeros(self.resolution)
        colors = np.zeros((self.resolution[0], self.resolution[1],
                           self.resolution[2], 3))

        for i in range(np.asarray(point_cloud.points).shape[0]):
            pt = point_cloud.points[i]
            x = int((pt[0] + 0.25) / leaf_size)
            y = int((pt[1] + 0.25) / leaf_size)
            z = int(pt[2] / leaf_size)

            if x >= self.resolution[0] or x < 0 or \
               y >= self.resolution[1] or y < 0 or \
               z >= self.resolution[2] or z < 0:
                continue

            grid[x, y, z] = 1.0

            colors[x, y, z] = point_cloud.colors[i]

        points = []
        vox_colors = []
        for x, y, z in itertools.product(*map(range, (self.resolution[0], self.resolution[1], self.resolution[2]))):
            if grid[x, y, z] != 1.0:
                continue
            pt = np.array([(x * leaf_size) - 0.25, (y * leaf_size) - 0.25, (z * leaf_size)])
            points.append(pt)
            vox_colors.append(np.array(colors[x, y, z]))

        voxelized_point_cloud = o3d.geometry.PointCloud()
        voxelized_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
        voxelized_point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(vox_colors))
        voxelized_point_cloud.estimate_normals()

        return voxelized_point_cloud

# class VoxelGrid:
#     def __init__(self, resolution):
#         self.resolution = resolution
#         self.map = {}
#
#     def voxelize(self, points, normals):
#         min_pt = np.min(points, 0)
#         max_pt = np.max(points, 0)
#         leaf_size = max(max_pt[0] - min_pt[0],
#                         max_pt[1] - min_pt[1],
#                         max_pt[2] - min_pt[2]) / (self.resolution - 1)
#         leaf_size = round(leaf_size, 4)
#
#         occupancy_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
#         normals_grid = np.zeros((3, self.resolution, self.resolution, self.resolution), dtype=np.float32)
#         for i in range(points.shape[0]):
#             pt = points[i]
#
#             # Compute voxel's coordinates
#             x = int((pt[0] - min_pt[0]) / leaf_size)
#             y = int((pt[1] - min_pt[1]) / leaf_size)
#             z = int((pt[2] - min_pt[2]) / leaf_size)
#
#             if x >= self.resolution or x < 0 or \
#                y >= self.resolution or y < 0 or \
#                z >= self.resolution or z < 0:
#                 continue
#
#             if np.isnan(normals[i]).any():
#                 print('Nan')
#
#             occupancy_grid[x, y, z] = 1.0
#             normals_grid[:, x, y, z] += normals[i]
#             # normals_grid[:, x, y, z] /= np.linalg.norm(normals_grid[:, x, y, z])
#
#         for x, y, z in itertools.product(*map(range, (self.resolution,
#                                                       self.resolution,
#                                                       self.resolution))):
#             if occupancy_grid[x, y, z] == 1.0:
#                 normals_grid[:, x, y, z] /= np.linalg.norm(normals_grid[:, x, y, z])
#
#         distance_field, known_regions = self.compute_sdf(occupancy_grid, normals_grid)
#         direction_angles = np.arccos(normals_grid) / np.pi
#
#         out_volume = np.zeros((5, self.resolution, self.resolution, self.resolution))
#         out_volume[0] = distance_field
#         out_volume[1] = known_regions
#         out_volume[2:5] = direction_angles
#
#         return out_volume
#         #     key = x + self.resolution * (y + self.resolution*z)
#         #     if key not in self.map:
#         #         self.map[key] = []
#         #     self.map[key].append(i)
#         #
#         # voxelized_pts = []
#         # voxelized_normals = []
#         # for key in self.map:
#         #     centroid = np.mean(points[self.map[key]], 0)
#         #     normal = np.sum(normals[self.map[key]], 0)
#         #     normal /= np.linalg.norm(normal)
#         #
#         #     voxelized_pts.append(centroid)
#         #     voxelized_normals.append(normal)
#
#     def compute_sdf(self, occupancy, normals):
#         surface_pts = np.argwhere(occupancy == 1)
#         surface_normals = np.zeros((surface_pts.shape[0], 3))
#         for i in range(len(surface_pts)):
#             p = surface_pts[i]
#             surface_normals[i, :] = normals[:, p[0], p[1], p[2]]
#
#         df = np.zeros(occupancy.shape, dtype=np.float32)
#         known_regions = np.zeros(occupancy.shape, dtype=np.float32)
#         for x, y, z in itertools.product(*map(range, (self.resolution,
#                                                       self.resolution,
#                                                       self.resolution))):
#             if occupancy[x, y, z] == 1.0:
#                 continue
#
#             # Compute the projection on surface points
#             p = np.array([x, y, z])
#             dist = np.linalg.norm(surface_pts - p, axis=1)
#             min_id = np.argmin(dist)
#
#             # Compute distance field
#             df[x, y, z] = np.min(dist) / self.resolution
#
#             # Compute sign bit
#             ds = (p - surface_pts[min_id]).astype(float)
#             if np.linalg.norm(ds) != 0:
#                 ds /= np.linalg.norm(ds)
#
#             if np.dot(ds, surface_normals[min_id]) < 0:
#                 known_regions[x, y, z] = 1
#
#         return df, known_regions
