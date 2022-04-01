import numpy as np
import open3d as o3d

from gog import cameras
from gog.grasp_sampler import GraspSampler
from gog.utils.utils import get_pointcloud
from gog.utils import pybullet_utils


class GraspingPolicy:
    def __init__(self, robot_hand, params):
        self.robot_hand = robot_hand
        self.config = cameras.RealSense.CONFIG[0]
        self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.01, 0.3]])  # workspace limits

        self.grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])

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

        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.visualization.draw_geometries([point_cloud])
        return point_cloud

    def get_samples(self, point_cloud):
        grasp_candidates = self.grasp_sampler.sample(point_cloud, plot=True)
        return grasp_candidates

    def reconstruct(self):
        pass

    def optimize(self):
        pass
