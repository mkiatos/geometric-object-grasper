import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import numpy as np

from gog import cameras
from gog.utils.utils import get_pointcloud
from gog.utils import pybullet_utils
from gog.utils.o3d_tools import VoxelGrid

from gog.grasp_sampler import GraspSampler
from gog.shape_completion import ShapeCompletionNetwork
from gog.grasp_optimizer import SplitPSO



class QualityMetric:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ShapeComplementarityMetric(QualityMetric):
    def __init__(self):
        self.w = 0.5
        self.a = 0.5

    def compute_shape_complementarity(self, hand, target, radius=0.03):

        # Find nearest neighbors for each hand contact
        kdtree = KDTree(target['points'])
        dists, nn_ids = kdtree.query(hand['points'])

        shape_metric = 0.0
        for i in range(hand['points'].shape[0]):

            # Remove duplicates or contacts with nn distance greater than a threshold
            duplicates = np.argwhere(nn_ids == nn_ids[i])[:, 0]
            if np.min(dists[duplicates]) != dists[i] or dists[i] > radius:
                continue

            # Distance error
            e_p = np.linalg.norm(hand['points'][i] - target['points'][nn_ids[i][0]])

            # Alignment error
            c_n = hand['normals'][i] # for TRO is -hand['normals']
            e_n = c_n.dot(target['normals'][nn_ids[i][0]])

            shape_metric += e_p + self.w * e_n

        return - shape_metric / len(hand['points'])

    @staticmethod
    def compute_collision_penalty(hand, collisions):
        if collisions.shape[0] == 0:
            return 0.0

        # Find nearest neighbors for each collision
        kdtree = KDTree(hand['points'])
        dists, nn_ids = kdtree.query(collisions)

        e_col = 0.0
        for i in range(collisions.shape[0]):
            e_col += np.linalg.norm(collisions[i] - hand['points'][nn_ids[i][0]])
        return e_col

    def __call__(self, hand, target, collisions):
        return self.compute_shape_complementarity(hand, target) + \
               self.a * self.compute_collision_penalty(hand, collisions)


class GraspingPolicy:
    def __init__(self, robot_hand, params):
        self.robot_hand = robot_hand
        self.config = cameras.RealSense.CONFIG[0]
        self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.01, 0.3]])  # workspace limits

        self.grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])
        # self.vae = ShapeCompletionNetwork(input_dim=[40, 40, 40], latent_dim=512)
        # self.grasp_optimizer = SplitPSO(robot_hand, params=params['grasp_optimization'])

    def state_representation(self, obs, plot=True):
        intrinsics = np.array(self.config['intrinsics']).reshape(3, 3)
        point_cloud = get_pointcloud(obs['depth'][0], obs['seg'][0], intrinsics)
        point_cloud.estimate_normals()

        # Transform point cloud w.r.t. world frame
        transform = pybullet_utils.get_camera_pose(self.config['pos'],
                                                   self.config['target_pos'],
                                                   self.config['up_vector'])
        point_cloud = point_cloud.transform(transform)

        # Keep only the points that lie inside the robot's workspace
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([self.bounds[0, 0],
                                                                           self.bounds[1, 0],
                                                                           self.bounds[2, 0]]),
                                                       max_bound=np.array([self.bounds[0, 1],
                                                                           self.bounds[1, 1],
                                                                           self.bounds[2, 1]]))
        point_cloud = point_cloud.crop(crop_box)

        # Orient normals to align with camera direction
        point_cloud.orient_normals_to_align_with_direction(-transform[0:3, 2])

        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
        point_cloud.paint_uniform_color([0.0, 0.1, 0.9])

        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([point_cloud])
        return point_cloud

    def get_samples(self, point_cloud):
        grasp_candidates = self.grasp_sampler.sample(point_cloud, plot=True)
        return grasp_candidates

    def reconstruct(self, candidate):
        voxel_grid = VoxelGrid(resolution=[32, 32, 32])
        voxelized_point_cloud = voxel_grid.voxelize(point_cloud=candidate['enclosing_pts'])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([voxelized_point_cloud, mesh_frame])

    def optimize(self):
        pass

    def predict(self):
        pass
