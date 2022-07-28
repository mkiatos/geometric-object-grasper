import open3d as o3d
import numpy as np
import torch

from gog import cameras
from gog.utils.utils import get_pointcloud
from gog.utils import pybullet_utils
from gog.utils.o3d_tools import VoxelGrid, get_reconstructed_surface
from gog.grasp_sampler import GraspSampler
from gog.shape_completion import ShapeCompletionNetwork
from gog.grasp_optimizer import SplitPSO


class GraspingPolicy:
    def __init__(self, robot_hand, params):
        self.robot_hand = robot_hand
        self.config = cameras.RealSense.CONFIG[0]
        self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.01, 0.3]])  # workspace limits

        self.grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])

        self.vae = ShapeCompletionNetwork(input_dim=[5, 32, 32, 32], latent_dim=512).to('cuda')
        self.vae.load_state_dict(torch.load(params['vae']['model_weights']))
        print('Shape completion model loaded....!')
        self.vae.eval()

        self.optimizer = SplitPSO(robot_hand, params['power_grasping']['optimizer'])

    def seed(self, seed):
        self.grasp_sampler.seed(seed)
        self.optimizer.seed(seed)

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

        # Downsample point cloud (no need for dense point cloud)
        # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
        point_cloud.paint_uniform_color([0.0, 0.1, 0.9])

        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])
        return point_cloud

    def get_candidates(self, point_cloud, plot=False):
        grasp_candidates = self.grasp_sampler.sample(point_cloud, plot=plot)
        return grasp_candidates

    def reconstruct(self, candidate, plot=False):
        def normalize(grid):
            diagonal = np.sqrt(2) * grid.shape[1]

            normalized_grid = grid.copy()
            normalized_grid[0] /= diagonal
            normalized_grid[1] = normalized_grid[1]
            normalized_grid[2:] = np.arccos(grid[2:]) / np.pi
            return normalized_grid
        points = np.asarray(candidate['enclosing_pts'].points)
        normals = np.asarray(candidate['enclosing_pts'].normals)

        voxel_grid = VoxelGrid(resolution=32)
        partial_grid = voxel_grid.voxelize(points, normals)
        partial_grid = normalize(partial_grid)

        x = torch.FloatTensor(partial_grid).to('cuda')
        x = x.unsqueeze(0)
        pred_sdf, pred_normals, _, _ = self.vae(x)

        pts, normals = get_reconstructed_surface(pred_sdf=pred_sdf.squeeze().detach().cpu().numpy(),
                                                 pred_normals=pred_normals.squeeze().detach().cpu().numpy(),
                                                 x=x.squeeze().detach().cpu().numpy(),
                                                 leaf_size=voxel_grid.leaf_size,
                                                 min_pt=voxel_grid.min_pt)

        rec_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        rec_point_cloud.normals = o3d.utility.Vector3dVector(normals)

        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rec_point_cloud.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([rec_point_cloud, mesh_frame])

        return rec_point_cloud

    def optimize(self, init_preshape, point_cloud, plot=False):
        opt_preshape = self.optimizer.optimize(init_preshape, point_cloud, plot)
        return opt_preshape

